"""
PPB Phase 4 — Context-Rot evaluation (semantic Needle-in-a-Haystack).

This module measures how a model's answer accuracy degrades as context length
increases.  It is inspired by NVIDIA's RULER benchmark and uses real ShareGPT
conversations as the haystack distractor text, with a synthetic factual needle
injected at five depth positions per haystack length.

Method
------
1. Build haystacks of target token lengths
   ``[4096, 8192, 16384, 32768, 65536, 131072]`` by concatenating ShareGPT
   turns.  Token counts are measured with the *model's own* tokenizer
   (``llama_cpp.Llama.tokenize``).
2. Insert a synthetic needle at depth positions 10/30/50/70/90 % of the
   total token count → 6 lengths × 5 depths = 30 cases per model.
3. Query the model with temperature=0, max_tokens=20 and exact-match score
   each response.
4. Aggregate to per-length, per-depth, and overall accuracy.

Public entry point
------------------
``run_context_rot(llm, model_config, suite_config) -> dict``

Returns a dict with three keys (suitable for merging into the PPB result
envelope):

* ``context_rot_accuracy_by_length``  — ``{token_count: float | None}``
* ``context_rot_accuracy_by_depth``   — ``{depth_pct: float}``
* ``context_rot_score``               — ``float`` (mean over all 30 cases)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterable

from datasets.sharegpt import (
    SHAREGPT_FILENAME,
    SHAREGPT_REPO,
    download_dataset,
)

log = logging.getLogger("ppb")

# ---------------------------------------------------------------------------
# Defaults (overridable via suite_config)
# ---------------------------------------------------------------------------

DEFAULT_HAYSTACK_LENGTHS: list[int] = [4096, 8192, 16384, 32768, 65536, 131072]
DEFAULT_DEPTHS_PCT: list[int] = [10, 30, 50, 70, 90]

DEFAULT_NEEDLE_TEXT = (
    "The secret launch code for Project Nightingale is ALPHA-7-DELTA."
)
DEFAULT_NEEDLE_QUERY = (
    "Based only on the text provided, what is the secret launch code for "
    "Project Nightingale? Answer with just the code."
)
DEFAULT_NEEDLE_ANSWER = "ALPHA-7-DELTA"


# ---------------------------------------------------------------------------
# Haystack construction
# ---------------------------------------------------------------------------


def _iter_sharegpt_text(json_path: Path) -> Iterable[str]:
    """Yield concatenated conversation turns from a ShareGPT JSON file."""
    import json

    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    for conv in data:
        turns = conv.get("conversations") or []
        for turn in turns:
            value = (turn.get("value") or "").strip()
            if value:
                yield value


def _build_token_pool(llm: Any, target_max: int) -> list[int]:
    """Build a pool of haystack tokens at least ``target_max`` long.

    Iterates ShareGPT turns, encoding each one with the model's tokenizer
    until the pool is large enough.  The pool is reused across all
    haystack lengths so each is a contiguous prefix of the same text.
    """
    json_path = download_dataset(SHAREGPT_REPO, SHAREGPT_FILENAME)
    pool: list[int] = []
    # Add a generous safety margin so we have room for slicing + needle insertion.
    needed = int(target_max * 1.10) + 1024

    for chunk in _iter_sharegpt_text(json_path):
        if len(pool) >= needed:
            break
        # add_bos=False so we don't pepper BOS tokens through the haystack.
        try:
            ids = llm.tokenize(chunk.encode("utf-8"), add_bos=False, special=False)
        except TypeError:
            # Older llama-cpp-python signatures differ; fall back gracefully.
            ids = llm.tokenize(chunk.encode("utf-8"), add_bos=False)
        pool.extend(ids)
        # Add a newline separator between turns.
        try:
            pool.extend(llm.tokenize(b"\n\n", add_bos=False, special=False))
        except TypeError:
            pool.extend(llm.tokenize(b"\n\n", add_bos=False))

    if len(pool) < needed:
        log.warning(
            "ShareGPT pool only produced %d tokens (wanted %d); "
            "longer haystacks may be truncated.",
            len(pool),
            needed,
        )
    return pool


def _detokenize(llm: Any, ids: list[int]) -> bytes:
    """Decode a token list back to bytes via the model's tokenizer."""
    out = llm.detokenize(ids)
    if isinstance(out, str):
        return out.encode("utf-8")
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_response(response: str, expected: str) -> int:
    """Exact substring match (case-insensitive) → 1 pass / 0 fail."""
    if not response:
        return 0
    return 1 if expected.strip().lower() in response.strip().lower() else 0


# ---------------------------------------------------------------------------
# Single test case
# ---------------------------------------------------------------------------


def _run_single_case(
    llm: Any,
    *,
    token_pool: list[int],
    needle_tokens: list[int],
    needle_query: str,
    needle_answer: str,
    target_ctx: int,
    depth_pct: int,
    case_idx: int,
    total_cases: int,
) -> tuple[int, float, str]:
    """Run one (length × depth) case and return (pass/fail, elapsed_s, response)."""
    # Reserve room for the trailing query + small answer budget.
    # Encode the query so we know its size precisely.
    try:
        query_tokens = llm.tokenize(
            needle_query.encode("utf-8"), add_bos=False, special=False
        )
    except TypeError:
        query_tokens = llm.tokenize(needle_query.encode("utf-8"), add_bos=False)

    answer_budget = 20
    haystack_budget = target_ctx - len(needle_tokens) - len(query_tokens) - answer_budget
    if haystack_budget <= 0:
        raise ValueError(
            f"target_ctx={target_ctx} too small for needle+query+answer"
        )

    if len(token_pool) < haystack_budget:
        raise ValueError(
            f"token pool ({len(token_pool)}) smaller than haystack budget "
            f"({haystack_budget})"
        )

    haystack = token_pool[:haystack_budget]

    # Insert the needle at the requested depth.
    insert_at = (len(haystack) * depth_pct) // 100
    composed = haystack[:insert_at] + needle_tokens + haystack[insert_at:] + query_tokens

    # Decode back to text, then call the model.  We pass text rather than
    # token ids because llama-cpp-python's high-level call expects a prompt
    # string; this also lets the chat-template-free path be portable across
    # base/instruct models.
    prompt_bytes = _detokenize(llm, composed)
    prompt = prompt_bytes.decode("utf-8", errors="replace")

    t0 = time.time()
    out = llm(
        prompt,
        max_tokens=answer_budget,
        temperature=0.0,
        top_p=1.0,
        echo=False,
    )
    elapsed = time.time() - t0

    # Extract text from llama-cpp-python's OpenAI-style completion response.
    try:
        response = out["choices"][0]["text"]
    except (KeyError, IndexError, TypeError):
        response = str(out)

    passed = _score_response(response, needle_answer)
    status = "pass" if passed else "fail"
    print(
        f"  ✓ [{case_idx}/{total_cases}] ctx={target_ctx} "
        f"depth={depth_pct}% {status} ({elapsed:.1f}s)",
        flush=True,
    )
    return passed, elapsed, response


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_context_rot(
    llm: Any,
    model_config: dict[str, Any] | None = None,
    suite_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the context-rot evaluation suite against ``llm``.

    Parameters
    ----------
    llm
        A ``llama_cpp.Llama`` instance, already loaded with a context
        window large enough for the requested haystack lengths.
    model_config
        Optional dict.  Recognised key:

        * ``max_ctx`` (int) — the model's measured VRAM-cliff cap.
          Haystack lengths exceeding this value are skipped (recorded
          as ``None``).
    suite_config
        Optional dict (typically the ``[qualitative]`` TOML section).
        Recognised keys:

        * ``haystack_lengths`` (list[int])
        * ``depths_pct``       (list[int])
        * ``needle_text``      (str)
        * ``needle_query``     (str)
        * ``needle_answer``    (str)

    Returns
    -------
    dict
        ``{
            "context_rot_accuracy_by_length": {token_count: float|None, …},
            "context_rot_accuracy_by_depth":  {depth_pct: float, …},
            "context_rot_score":              float,
        }``
    """
    model_config = model_config or {}
    suite_config = suite_config or {}

    haystack_lengths: list[int] = list(
        suite_config.get("haystack_lengths") or DEFAULT_HAYSTACK_LENGTHS
    )
    depths_pct: list[int] = list(suite_config.get("depths_pct") or DEFAULT_DEPTHS_PCT)

    needle_text: str = suite_config.get("needle_text") or DEFAULT_NEEDLE_TEXT
    needle_query: str = suite_config.get("needle_query") or DEFAULT_NEEDLE_QUERY
    needle_answer: str = suite_config.get("needle_answer") or DEFAULT_NEEDLE_ANSWER

    max_ctx_cap: int | None = model_config.get("max_ctx")

    # Filter haystack lengths against the VRAM cliff.
    runnable_lengths: list[int] = []
    skipped_lengths: list[int] = []
    for L in haystack_lengths:
        if max_ctx_cap is not None and L > max_ctx_cap:
            skipped_lengths.append(L)
        else:
            runnable_lengths.append(L)

    total_cases = len(runnable_lengths) * len(depths_pct)
    print(
        f"\n[context-rot] {len(runnable_lengths)} length(s) × {len(depths_pct)} depth(s) "
        f"= {total_cases} case(s)"
        + (f"  [skipped {len(skipped_lengths)} > VRAM cap {max_ctx_cap}]" if skipped_lengths else ""),
        flush=True,
    )

    if total_cases == 0:
        return {
            "context_rot_accuracy_by_length": {L: None for L in haystack_lengths},
            "context_rot_accuracy_by_depth": {d: 0.0 for d in depths_pct},
            "context_rot_score": 0.0,
        }

    # Tokenise the needle once.
    try:
        needle_tokens = llm.tokenize(
            needle_text.encode("utf-8"), add_bos=False, special=False
        )
    except TypeError:
        needle_tokens = llm.tokenize(needle_text.encode("utf-8"), add_bos=False)

    # Build a token pool large enough for the longest runnable haystack.
    target_max = max(runnable_lengths)
    pool = _build_token_pool(llm, target_max)

    # Run all cases.
    passes_by_length: dict[int, list[int]] = {L: [] for L in runnable_lengths}
    passes_by_depth: dict[int, list[int]] = {d: [] for d in depths_pct}

    case_idx = 0
    for L in runnable_lengths:
        for depth in depths_pct:
            case_idx += 1
            try:
                passed, _elapsed, _resp = _run_single_case(
                    llm,
                    token_pool=pool,
                    needle_tokens=needle_tokens,
                    needle_query=needle_query,
                    needle_answer=needle_answer,
                    target_ctx=L,
                    depth_pct=depth,
                    case_idx=case_idx,
                    total_cases=total_cases,
                )
            except Exception as exc:  # pragma: no cover — defensive
                log.warning(
                    "[context-rot] case %d (ctx=%d depth=%d%%) failed: %s",
                    case_idx,
                    L,
                    depth,
                    exc,
                )
                passed = 0
            passes_by_length[L].append(passed)
            passes_by_depth[depth].append(passed)

    # Aggregate.
    accuracy_by_length: dict[int, float | None] = {}
    for L in haystack_lengths:
        if L in passes_by_length and passes_by_length[L]:
            accuracy_by_length[L] = sum(passes_by_length[L]) / len(passes_by_length[L])
        else:
            accuracy_by_length[L] = None

    accuracy_by_depth: dict[int, float] = {
        d: (sum(passes_by_depth[d]) / len(passes_by_depth[d]))
        if passes_by_depth[d]
        else 0.0
        for d in depths_pct
    }

    all_passes: list[int] = [p for ps in passes_by_length.values() for p in ps]
    overall = sum(all_passes) / len(all_passes) if all_passes else 0.0

    print(
        f"[context-rot] overall score: {overall:.3f}  "
        f"(by length: "
        + ", ".join(
            f"{L}={'-' if v is None else f'{v:.2f}'}"
            for L, v in accuracy_by_length.items()
        )
        + ")",
        flush=True,
    )

    return {
        "context_rot_accuracy_by_length": accuracy_by_length,
        "context_rot_accuracy_by_depth": accuracy_by_depth,
        "context_rot_score": overall,
    }


# ---------------------------------------------------------------------------
# Convenience helper for ppb.py integration
# ---------------------------------------------------------------------------


def run_context_rot_for_model(
    model_path: Path,
    *,
    suite_config: dict[str, Any] | None = None,
    max_ctx_cap: int | None = None,
    n_gpu_layers: int = -1,
    verbose: bool = False,
) -> dict[str, Any]:
    """Load *model_path* with llama-cpp-python and run context-rot.

    This is the entry point ``ppb.py`` uses; it isolates the
    ``llama_cpp.Llama`` lifecycle from the orchestrator so that the
    model is loaded, evaluated, and freed within a single call.
    """
    try:
        from llama_cpp import Llama
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ppb_context_rot requires llama-cpp-python. "
            "Install with: pip install llama-cpp-python"
        ) from exc

    suite_config = suite_config or {}
    haystack_lengths: list[int] = list(
        suite_config.get("haystack_lengths") or DEFAULT_HAYSTACK_LENGTHS
    )

    # Pick a context window large enough for the longest runnable haystack
    # (clamped by the VRAM cliff cap if known).
    runnable = [L for L in haystack_lengths if max_ctx_cap is None or L <= max_ctx_cap]
    n_ctx = max(runnable) if runnable else (max_ctx_cap or 4096)

    print(
        f"[context-rot] loading {model_path.name} (n_ctx={n_ctx}, "
        f"n_gpu_layers={n_gpu_layers})",
        flush=True,
    )
    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )
    try:
        return run_context_rot(
            llm,
            model_config={"max_ctx": max_ctx_cap},
            suite_config=suite_config,
        )
    finally:
        # Free the model before returning so the orchestrator can reload
        # it for the next phase without VRAM contention.
        del llm
