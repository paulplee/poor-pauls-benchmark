"""
PPB — Multi-Turn Conversational Memory & Coherence (Phase 7).

Two evaluation modes are supported:

* **``longmemeval_s`` (default)** — uses the ``longmemeval_s`` split of
  ``xiaowu0162/longmemeval-cleaned``.  Each case carries a long
  multi-turn ``session_history`` followed by a follow-up ``question``
  whose ground-truth answer is recorded in the dataset.  We measure
  whether the model can recover the answer after consuming the entire
  history.  Cases whose history exceeds the model's measured VRAM
  cliff are skipped (counted under ``cases_skipped_context``).
  Scoring uses the optional judge LLM when ``judge_model_path`` is set,
  otherwise falls back to a normalised lowercase substring match.

* **``quick``** — runs the 80 two-turn questions from
  ``lmsys/mt_bench_human_judgments`` and asks the judge to rate the
  final response on a 1–10 helpfulness/coherence scale.  A judge model
  is REQUIRED in this mode.

Public API
----------
* ``run_multiturn(llm, model_config, suite_config, judge_llm=None) -> dict``
  Pure scorer.  Both ``llm`` (the model under test) and ``judge_llm``
  (when provided) must already be loaded ``llama_cpp.Llama`` instances.
  When ``judge_llm`` is ``None`` and ``judge_model_path`` is set in
  ``suite_config``, the judge is loaded for the duration of the call
  and disposed before returning.  When ``judge_llm`` is supplied (e.g.
  by ``ppb.py`` reusing the Phase 6 judge during ``all`` mode), it is
  used directly and NOT disposed.

* ``run_multiturn_for_model(generator_path, judge_model_path=None,
  *, suite_config, ...) -> dict``
  Convenience wrapper that loads the generator (and judge, if any),
  runs the evaluation, and disposes of any locally-loaded ``Llama``
  instances.  Used by ``ppb.py`` when ``multiturn`` is the only
  qualitative phase requiring a judge.

The module returns the four canonical multi-turn keys of the
``qualitative`` block:

* ``memory_accuracy``       — ``longmemeval_s`` only (else ``None``).
* ``mt_bench_score``        — ``quick`` only (else ``None``).
* ``cases_evaluated``       — number of cases actually scored.
* ``cases_skipped_context`` — cases skipped because the session
  history exceeded the measured VRAM cliff (``longmemeval_s`` only).
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("ppb")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_SIZE = 50
DEFAULT_GEN_MAX_TOKENS = 100
DEFAULT_GEN_TEMPERATURE = 0.0
DEFAULT_QUICK_GEN_MAX_TOKENS = 512
DEFAULT_QUICK_GEN_TEMPERATURE = 0.7
DEFAULT_JUDGE_MAX_TOKENS = 16

LONGMEMEVAL_REPO = "xiaowu0162/longmemeval-cleaned"
LONGMEMEVAL_SPLIT = "longmemeval_s"
MTBENCH_REPO = "lmsys/mt_bench_human_judgments"

VALID_MODES = ("longmemeval_s", "quick")


# ---------------------------------------------------------------------------
# llama-cpp call helpers (defensive against output shape variation)
# ---------------------------------------------------------------------------


def _llm_complete(llm: Any, prompt: str, *, max_tokens: int, temperature: float) -> str:
    """Call ``llm(prompt, ...)`` and return the response text (or "")."""
    try:
        out = llm(prompt, max_tokens=max_tokens, temperature=temperature)
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("[multiturn] llm call failed: %s", exc)
        return ""
    try:
        return out["choices"][0]["text"]
    except Exception:
        return ""


_INT_RE = re.compile(r"-?\d+")


def _first_int(text: str) -> int | None:
    if not text:
        return None
    m = _INT_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def _count_tokens(llm: Any, text: str) -> int:
    """Count tokens in *text* using the loaded model's tokenizer."""
    try:
        toks = llm.tokenize(text.encode("utf-8") if isinstance(text, str) else text)
        return len(toks)
    except Exception:
        # Rough fallback: ~4 chars per token.
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Session history rendering
# ---------------------------------------------------------------------------


def _render_session_history(history: Any) -> str:
    """Render a LongMemEval ``session_history`` payload as plain dialogue.

    LongMemEval stores ``session_history`` as a list of sessions, where
    each session is itself a list of ``{role, content}`` turns.  For
    robustness we also accept a single flat list of turns or a string.
    """
    if isinstance(history, str):
        return history.strip()
    if not isinstance(history, list):
        return ""

    turns: list[str] = []

    def _add(turn: Any) -> None:
        if isinstance(turn, dict):
            role = str(turn.get("role") or turn.get("from") or "user").strip()
            content = str(turn.get("content") or turn.get("value") or "").strip()
            if content:
                turns.append(f"{role.capitalize()}: {content}")
        elif isinstance(turn, str):
            txt = turn.strip()
            if txt:
                turns.append(txt)

    for item in history:
        if isinstance(item, list):
            for t in item:
                _add(t)
        else:
            _add(item)

    return "\n".join(turns)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _exact_match(response: str, answer: str) -> bool:
    """Normalised lowercase substring match."""
    if not response or not answer:
        return False
    return answer.strip().lower() in response.strip().lower()


_LME_JUDGE_PROMPT = (
    "Does the following response correctly answer the question '{question}'? "
    "The correct answer is '{answer}'. Reply YES or NO only.\n\n"
    "Response: {response}\n\nAnswer:"
)


def _judge_match(judge: Any, question: str, answer: str, response: str) -> bool:
    raw = _llm_complete(
        judge,
        _LME_JUDGE_PROMPT.format(question=question, answer=answer, response=response),
        max_tokens=DEFAULT_JUDGE_MAX_TOKENS,
        temperature=0.0,
    )
    if not raw:
        return False
    return "yes" in raw.strip().lower()[:5]


_MTBENCH_JUDGE_PROMPT = (
    "Given the conversation below, rate the final response on a scale of 1 "
    "to 10 for helpfulness and coherence. Reply with only a single integer."
    "\n\n{full_conversation}\n\nScore:"
)


# ---------------------------------------------------------------------------
# Mode 1: LongMemEval
# ---------------------------------------------------------------------------


def _load_longmemeval(sample_size: int) -> list[dict[str, Any]]:
    """Load up to *sample_size* cases from the ``longmemeval_s`` split."""
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ppb_multiturn requires the `datasets` library. "
            "Install with: pip install datasets"
        ) from exc

    log.info("[multiturn] loading %s/%s …", LONGMEMEVAL_REPO, LONGMEMEVAL_SPLIT)
    ds = load_dataset(LONGMEMEVAL_REPO, split=LONGMEMEVAL_SPLIT)
    cases: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        if i >= sample_size:
            break
        if not isinstance(row, dict):
            continue
        cases.append(
            {
                "session_history": row.get("session_history") or row.get("haystack"),
                "question": row.get("question") or "",
                "answer": str(row.get("answer") or ""),
            }
        )
    return cases


def _run_longmemeval(
    llm: Any,
    judge_llm: Any | None,
    *,
    sample_size: int,
    vram_cliff_tokens: int | None,
) -> dict[str, Any]:
    cases = _load_longmemeval(sample_size)
    total = len(cases)
    print(
        f"\n[multiturn] LongMemEval: {total} case(s) "
        f"(judge={'yes' if judge_llm is not None else 'no'}, "
        f"vram_cliff={vram_cliff_tokens})",
        flush=True,
    )

    n_correct = 0
    n_evaluated = 0
    n_skipped = 0

    for i, case in enumerate(cases, start=1):
        history = _render_session_history(case["session_history"])
        question = (case["question"] or "").strip()
        answer = (case["answer"] or "").strip()
        if not question:
            continue

        prompt = (
            f"{history}\n\nUser: {question}\nAssistant:"
            if history
            else f"User: {question}\nAssistant:"
        )

        if vram_cliff_tokens is not None:
            tok_count = _count_tokens(llm, prompt)
            if tok_count > vram_cliff_tokens:
                n_skipped += 1
                print(
                    f"  ⚠ [{i}/{total}] skipped: session history "
                    f"({tok_count} tok) exceeds measured VRAM cliff "
                    f"({vram_cliff_tokens} tok)",
                    flush=True,
                )
                continue

        t0 = time.time()
        response = _llm_complete(
            llm,
            prompt,
            max_tokens=DEFAULT_GEN_MAX_TOKENS,
            temperature=DEFAULT_GEN_TEMPERATURE,
        )
        if judge_llm is not None:
            match = _judge_match(judge_llm, question, answer, response)
            method = "judge"
        else:
            match = _exact_match(response, answer)
            method = "exact"

        n_evaluated += 1
        if match:
            n_correct += 1

        print(
            f"  ✓ [{i}/{total}] LongMemEval: match={match} method={method} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

    memory_accuracy = (n_correct / n_evaluated) if n_evaluated > 0 else None
    print(
        f"[multiturn] memory_accuracy="
        f"{memory_accuracy if memory_accuracy is None else f'{memory_accuracy:.3f}'} "
        f"(evaluated={n_evaluated}, skipped_context={n_skipped})",
        flush=True,
    )

    return {
        "memory_accuracy": memory_accuracy,
        "mt_bench_score": None,
        "cases_evaluated": n_evaluated,
        "cases_skipped_context": n_skipped,
    }


# ---------------------------------------------------------------------------
# Mode 2: MT-Bench quick
# ---------------------------------------------------------------------------


def _load_mtbench() -> list[dict[str, Any]]:
    """Load 80 two-turn questions from MT-Bench, deduplicated by question_id."""
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ppb_multiturn requires the `datasets` library. "
            "Install with: pip install datasets"
        ) from exc

    log.info("[multiturn] loading %s …", MTBENCH_REPO)
    ds = load_dataset(MTBENCH_REPO, split="human")
    seen: set[Any] = set()
    cases: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        if not isinstance(row, dict):
            continue
        qid = row.get("question_id")
        # When ``question_id`` is None for multiple rows, fall back to a
        # per-row synthetic key so we don't silently collapse them all
        # into a single "seen" bucket.
        dedup_key = qid if qid is not None else f"__idx_{idx}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        body = row.get("question_body") or row.get("conversation_a") or []
        # ``question_body`` is typically a list of two user-turn strings.
        turns: list[str] = []
        if isinstance(body, list):
            for t in body:
                if isinstance(t, str):
                    turns.append(t)
                elif isinstance(t, dict):
                    txt = str(t.get("content") or t.get("value") or "").strip()
                    if txt:
                        turns.append(txt)
        if len(turns) < 2:
            continue
        cases.append(
            {
                "question_id": qid,
                "turn1": turns[0],
                "turn2": turns[1],
            }
        )
        if len(cases) >= 80:
            break
    return cases


def _run_mtbench(llm: Any, judge_llm: Any) -> dict[str, Any]:
    cases = _load_mtbench()
    total = len(cases)
    print(
        f"\n[multiturn] MT-Bench quick: {total} two-turn question(s)",
        flush=True,
    )

    scores: list[int] = []

    for i, case in enumerate(cases, start=1):
        t0 = time.time()
        turn1 = case["turn1"]
        turn2 = case["turn2"]

        prompt1 = f"User: {turn1}\nAssistant:"
        resp1 = _llm_complete(
            llm,
            prompt1,
            max_tokens=DEFAULT_QUICK_GEN_MAX_TOKENS,
            temperature=DEFAULT_QUICK_GEN_TEMPERATURE,
        )

        prompt2 = (
            f"User: {turn1}\nAssistant: {resp1.strip()}\nUser: {turn2}\nAssistant:"
        )
        resp2 = _llm_complete(
            llm,
            prompt2,
            max_tokens=DEFAULT_QUICK_GEN_MAX_TOKENS,
            temperature=DEFAULT_QUICK_GEN_TEMPERATURE,
        )

        full_conversation = (
            f"User: {turn1}\nAssistant: {resp1.strip()}\n"
            f"User: {turn2}\nAssistant: {resp2.strip()}"
        )

        raw = _llm_complete(
            judge_llm,
            _MTBENCH_JUDGE_PROMPT.format(full_conversation=full_conversation),
            max_tokens=DEFAULT_JUDGE_MAX_TOKENS,
            temperature=0.0,
        )
        score = _first_int(raw)
        if score is not None:
            score = max(1, min(10, score))
            scores.append(score)
            score_str = str(score)
        else:
            score_str = "none"

        qid = case.get("question_id", i)
        print(
            f"  ✓ [{i}/{total}] MT-Bench q{qid}: score={score_str} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

    mt_bench_score = (sum(scores) / len(scores)) if scores else None
    print(
        f"[multiturn] mt_bench_score="
        f"{mt_bench_score if mt_bench_score is None else f'{mt_bench_score:.3f}'} "
        f"(evaluated={len(scores)}/{total})",
        flush=True,
    )

    return {
        "memory_accuracy": None,
        "mt_bench_score": mt_bench_score,
        "cases_evaluated": len(scores),
        "cases_skipped_context": 0,
    }


# ---------------------------------------------------------------------------
# Public scorer
# ---------------------------------------------------------------------------


def run_multiturn(
    llm: Any,
    model_config: dict[str, Any] | None = None,
    suite_config: dict[str, Any] | None = None,
    judge_llm: Any | None = None,
) -> dict[str, Any]:
    """Score the model on multi-turn memory / coherence.

    Returns a dict with keys ``memory_accuracy``, ``mt_bench_score``,
    ``cases_evaluated``, ``cases_skipped_context`` — these slot directly
    into the canonical ``qualitative`` block.

    Parameters
    ----------
    llm:
        Already-loaded ``llama_cpp.Llama`` for the model under test.
    model_config:
        Optional dict.  Recognised key: ``vram_cliff_tokens`` —
        when set, LongMemEval cases whose rendered prompt exceeds this
        token count are skipped.
    suite_config:
        Suite TOML ``[qualitative]`` block (or equivalent dict).
        Recognised keys:

        * ``multiturn_mode`` — ``"longmemeval_s"`` (default) or ``"quick"``.
        * ``multiturn_sample_size`` — cap on cases (default 50; ignored
          in ``quick`` mode where the 80 MT-Bench questions are fixed).
        * ``judge_model_path`` — optional GGUF path; required in
          ``quick`` mode.
        * ``multiturn_n_ctx``, ``multiturn_judge_n_ctx``,
          ``multiturn_n_gpu_layers``, ``multiturn_judge_n_gpu_layers``
          — only consulted when this function loads a judge locally
          (i.e. ``judge_llm`` is ``None`` and ``judge_model_path`` is
          set).
    judge_llm:
        Optional already-loaded judge ``Llama``.  When provided, this
        function will NOT dispose of it — the caller (typically
        ``ppb.py`` during ``all`` mode) retains ownership.
    """
    suite_config = suite_config or {}
    model_config = model_config or {}

    mode = str(suite_config.get("multiturn_mode") or "longmemeval_s").strip()
    if mode not in VALID_MODES:
        raise ValueError(
            f"multiturn: unknown multiturn_mode={mode!r}; "
            f"expected one of {VALID_MODES}."
        )

    sample_size = int(suite_config.get("multiturn_sample_size", DEFAULT_SAMPLE_SIZE))
    judge_model_path = suite_config.get("judge_model_path") or None

    if mode == "quick" and judge_llm is None and not judge_model_path:
        raise ValueError("MT-Bench quick mode requires judge_model_path in suite TOML.")

    # ── Optionally instantiate a local judge (and arrange to dispose it).
    locally_loaded_judge: Any | None = None
    if judge_llm is None and judge_model_path:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ppb_multiturn requires llama-cpp-python. "
                "Install with: pip install llama-cpp-python"
            ) from exc
        judge_path = Path(str(judge_model_path)).expanduser()
        if not judge_path.exists():
            raise FileNotFoundError(
                f"multiturn: judge_model_path does not exist: {judge_path}"
            )
        judge_n_ctx = int(suite_config.get("multiturn_judge_n_ctx", 4096))
        judge_n_gpu_layers = int(
            suite_config.get(
                "multiturn_judge_n_gpu_layers",
                suite_config.get("n_gpu_layers", -1),
            )
        )
        print(
            f"[multiturn] loading judge {judge_path.name} "
            f"(n_ctx={judge_n_ctx}, n_gpu_layers={judge_n_gpu_layers})",
            flush=True,
        )
        locally_loaded_judge = Llama(
            model_path=str(judge_path),
            n_ctx=judge_n_ctx,
            n_gpu_layers=judge_n_gpu_layers,
            verbose=False,
        )
        judge_llm = locally_loaded_judge

    try:
        if mode == "quick":
            assert judge_llm is not None  # guarded above
            return _run_mtbench(llm, judge_llm)
        # Default: longmemeval_s
        vram_cliff_tokens = model_config.get("vram_cliff_tokens")
        if vram_cliff_tokens is not None:
            try:
                vram_cliff_tokens = int(vram_cliff_tokens)
            except (TypeError, ValueError):
                vram_cliff_tokens = None
        return _run_longmemeval(
            llm,
            judge_llm,
            sample_size=sample_size,
            vram_cliff_tokens=vram_cliff_tokens,
        )
    finally:
        if locally_loaded_judge is not None:
            del locally_loaded_judge


# ---------------------------------------------------------------------------
# Convenience wrapper: handles full Llama lifecycle for ppb.py
# ---------------------------------------------------------------------------


def run_multiturn_for_model(
    generator_path: Path,
    judge_model_path: Path | str | None = None,
    *,
    suite_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    n_ctx: int = 8192,
    n_gpu_layers: int = -1,
    verbose: bool = False,
    reuse_judge_llm: Any | None = None,
) -> dict[str, Any]:
    """Load the generator (and judge if needed), run multi-turn, dispose.

    When ``reuse_judge_llm`` is provided (e.g. by ``ppb.py`` during
    ``all`` mode to share the Phase 6 judge), it is passed through to
    ``run_multiturn`` and NOT disposed by this function.
    """
    try:
        from llama_cpp import Llama
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ppb_multiturn requires llama-cpp-python. "
            "Install with: pip install llama-cpp-python"
        ) from exc

    suite_config = dict(suite_config or {})
    if judge_model_path and "judge_model_path" not in suite_config:
        suite_config["judge_model_path"] = str(judge_model_path)

    # Validate generator ≠ judge before loading anything.
    if judge_model_path:
        gp = Path(generator_path).resolve()
        jp = Path(str(judge_model_path)).expanduser().resolve()
        if gp == jp:
            raise ValueError(
                "multiturn: judge_model_path must differ from the generator "
                "model — the judge cannot grade itself."
            )

    print(
        f"[multiturn] loading generator {Path(generator_path).name} "
        f"(n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers})",
        flush=True,
    )
    generator = Llama(
        model_path=str(generator_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )
    try:
        return run_multiturn(
            generator,
            model_config=model_config,
            suite_config=suite_config,
            judge_llm=reuse_judge_llm,
        )
    finally:
        del generator


__all__ = [
    "run_multiturn",
    "run_multiturn_for_model",
    "VALID_MODES",
]
