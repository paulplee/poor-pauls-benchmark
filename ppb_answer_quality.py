"""
PPB — Answer Faithfulness & Quality (Phase 6).

Scores model answers on three orthogonal dimensions using a *two-model*
pipeline so the model under test never grades itself:

* **Faithfulness**     — fraction of factual claims in the response that
  the judge model considers correct under common knowledge.
* **Answer Relevancy** — how directly the response addresses the prompt
  (judge-rated 1–5, normalised to 0–1).
* **Coherence**        — logical consistency of the response
  (judge-rated 1–10, normalised to 0–1).

Inputs
------
* ``generator_llm`` — the model under test (already-loaded
  ``llama_cpp.Llama`` instance from ``ppb.py``).
* ``judge_llm``     — a small, fixed local judge model (separate
  ``llama_cpp.Llama``).  The judge MUST NOT be the same model as the
  generator.

Public entry points
-------------------
* ``run_answer_quality(generator_llm, judge_llm, model_config, suite_config) -> dict``
  Pure scorer: assumes both ``Llama`` instances are already constructed.

* ``run_answer_quality_for_model(generator_path, judge_model_path, ...)``
  Convenience wrapper used by ``ppb.py`` — loads the generator, then
  loads the judge *only* for the duration of this phase, scores the
  responses, and disposes of the judge before returning.  Keeping the
  judge's lifetime scoped to this call avoids double-VRAM occupancy
  while other qualitative phases run.

The 50-prompt evaluation set is sampled once from
``anon8231489123/ShareGPT_Vicuna_unfiltered`` and cached in
``ppb_quality_prompts_cache.json`` so every model in the leaderboard is
graded on identical prompts.  The cache file's SHA-256 is published in
the result row's ``meta`` block as ``quality_prompts_cache_hash`` so
downstream consumers can detect cache drift.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

from ppb_datasets.sharegpt import (
    SHAREGPT_FILENAME,
    SHAREGPT_REPO,
    download_dataset,
    load_sharegpt_prompts,
)

log = logging.getLogger("ppb")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_SIZE = 50
DEFAULT_GEN_MAX_TOKENS = 512
DEFAULT_GEN_TEMPERATURE = 0.7
DEFAULT_JUDGE_MAX_TOKENS = 512
DEFAULT_JUDGE_TEMPERATURE = 0.0
_MIN_PROMPT_LEN = 20
_PROMPT_CACHE_SEED = 20260426  # frozen so re-sampling is deterministic
_PROMPT_CACHE_VERSION = 1

PROMPT_CACHE_PATH = Path(__file__).resolve().parent / "ppb_quality_prompts_cache.json"


# ---------------------------------------------------------------------------
# Prompt cache management
# ---------------------------------------------------------------------------


def _build_prompt_cache(sample_size: int) -> list[str]:
    """Sample *sample_size* single-turn factual ShareGPT prompts.

    Filter: the first human turn of each conversation, length > 20 chars.
    A frozen RNG seed makes the sample deterministic for a given dataset
    snapshot.
    """
    json_path = download_dataset(SHAREGPT_REPO, SHAREGPT_FILENAME)
    # Pull a generous over-sample from a shuffled copy of the dataset so
    # we can re-filter on length without exhausting the pool.
    candidates = load_sharegpt_prompts(
        json_path,
        max_prompts=sample_size * 20,
        shuffle=True,
        seed=_PROMPT_CACHE_SEED,
    )
    selected = [p for p in candidates if len(p) > _MIN_PROMPT_LEN][:sample_size]
    if len(selected) < sample_size:
        raise RuntimeError(
            f"answer_quality: only found {len(selected)} eligible prompts in "
            f"{SHAREGPT_REPO} (need {sample_size}). Lower "
            f"answer_quality_sample_size or check the dataset."
        )
    return selected


def _load_or_build_prompt_cache(sample_size: int) -> tuple[list[str], str]:
    """Return ``(prompts, sha256_hex)``.

    Loads ``ppb_quality_prompts_cache.json`` if present and the cached
    sample size matches the requested one; otherwise rebuilds and writes
    the cache atomically.
    """
    if PROMPT_CACHE_PATH.exists():
        try:
            payload = json.loads(PROMPT_CACHE_PATH.read_text(encoding="utf-8"))
            cached = payload.get("prompts") or []
            if (
                isinstance(cached, list)
                and len(cached) == sample_size
                and all(isinstance(p, str) for p in cached)
            ):
                digest = hashlib.sha256(PROMPT_CACHE_PATH.read_bytes()).hexdigest()
                log.info(
                    "[answer-quality] loaded %d cached prompts (sha256=%s…)",
                    len(cached),
                    digest[:8],
                )
                return cached, digest
            log.info(
                "[answer-quality] cache present but sample_size mismatch "
                "(cached=%d, requested=%d) — rebuilding",
                len(cached) if isinstance(cached, list) else -1,
                sample_size,
            )
        except Exception as exc:
            log.warning(
                "[answer-quality] failed to read cache %s: %s — rebuilding",
                PROMPT_CACHE_PATH.name,
                exc,
            )

    prompts = _build_prompt_cache(sample_size)
    payload = {
        "version": _PROMPT_CACHE_VERSION,
        "source": f"{SHAREGPT_REPO}/{SHAREGPT_FILENAME}",
        "seed": _PROMPT_CACHE_SEED,
        "sample_size": sample_size,
        "min_prompt_len": _MIN_PROMPT_LEN,
        "prompts": prompts,
    }
    tmp = PROMPT_CACHE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(PROMPT_CACHE_PATH)
    digest = hashlib.sha256(PROMPT_CACHE_PATH.read_bytes()).hexdigest()
    log.info(
        "[answer-quality] wrote %s (%d prompts, sha256=%s…)",
        PROMPT_CACHE_PATH.name,
        len(prompts),
        digest[:8],
    )
    return prompts, digest


# ---------------------------------------------------------------------------
# Llama call helpers (defensive against output shape variation)
# ---------------------------------------------------------------------------


def _llm_complete(llm: Any, prompt: str, *, max_tokens: int, temperature: float) -> str:
    """Call ``llm(prompt, ...)`` and return the response text (or "")."""
    try:
        out = llm(prompt, max_tokens=max_tokens, temperature=temperature)
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("[answer-quality] llm call failed: %s", exc)
        return ""
    try:
        return out["choices"][0]["text"]
    except Exception:
        return ""


_INT_RE = re.compile(r"-?\d+")


def _first_int(text: str) -> int | None:
    """Return the first integer found in *text*, or ``None``."""
    if not text:
        return None
    m = _INT_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_claims_json(text: str) -> list[str]:
    """Best-effort ``{"claims": [...]}`` parse from *text*."""
    if not text:
        return []
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    obj: Any = None
    try:
        obj = json.loads(text)
    except Exception:
        m = _JSON_OBJECT_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if isinstance(obj, dict):
        claims = obj.get("claims")
        if isinstance(claims, list):
            return [str(c).strip() for c in claims if str(c).strip()]
    return []


# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------


_CLAIMS_PROMPT = (
    "Extract every distinct factual claim from the response below as a JSON "
    'object of the form {"claims": ["claim 1", "claim 2", ...]}. '
    "Each claim should be a single, self-contained, verifiable statement. "
    "If the response contains no factual claims (e.g. it is a refusal or "
    'pure opinion), return {"claims": []}. Respond ONLY with the JSON.\n\n'
    "Response:\n{response}\n\nJSON:"
)

_VERIFY_CLAIM_PROMPT = (
    "Is the following claim factually correct and not contradicted by common "
    "knowledge? Answer YES or NO only.\n\nClaim: {claim}\n\nAnswer:"
)

_RELEVANCY_PROMPT = (
    "On a scale of 1 to 5, how directly does the response answer the "
    "question? Reply with only a single integer.\n\n"
    "Question: {question}\n\nResponse: {response}\n\nScore:"
)

_COHERENCE_PROMPT = (
    "Rate the following response on coherence and logical consistency on a "
    "scale of 1 to 10. Reply with only a single integer.\n\n"
    "Response: {response}\n\nScore:"
)


# ---------------------------------------------------------------------------
# Per-prompt scoring
# ---------------------------------------------------------------------------


def _score_faithfulness(judge: Any, response: str) -> tuple[float | None, int, int]:
    """Return ``(faithfulness, n_yes, n_total)``.

    ``faithfulness`` is ``None`` when zero claims were extracted.
    """
    raw = _llm_complete(
        judge,
        _CLAIMS_PROMPT.format(response=response),
        max_tokens=DEFAULT_JUDGE_MAX_TOKENS,
        temperature=DEFAULT_JUDGE_TEMPERATURE,
    )
    claims = _parse_claims_json(raw)
    if not claims:
        return None, 0, 0
    n_yes = 0
    for claim in claims:
        verdict = _llm_complete(
            judge,
            _VERIFY_CLAIM_PROMPT.format(claim=claim),
            max_tokens=8,
            temperature=DEFAULT_JUDGE_TEMPERATURE,
        )
        if verdict and "yes" in verdict.strip().lower()[:5]:
            n_yes += 1
    return n_yes / len(claims), n_yes, len(claims)


def _score_relevancy(judge: Any, prompt: str, response: str) -> float | None:
    raw = _llm_complete(
        judge,
        _RELEVANCY_PROMPT.format(question=prompt, response=response),
        max_tokens=8,
        temperature=DEFAULT_JUDGE_TEMPERATURE,
    )
    score = _first_int(raw)
    if score is None:
        return None
    score = max(1, min(5, score))
    # Normalise 1–5 → 0–1 (linear, anchored at 1=0.0, 5=1.0).
    return (score - 1) / 4.0


def _score_coherence(judge: Any, response: str) -> float | None:
    raw = _llm_complete(
        judge,
        _COHERENCE_PROMPT.format(response=response),
        max_tokens=8,
        temperature=DEFAULT_JUDGE_TEMPERATURE,
    )
    score = _first_int(raw)
    if score is None:
        return None
    score = max(1, min(10, score))
    return (score - 1) / 9.0


# ---------------------------------------------------------------------------
# Public scorer
# ---------------------------------------------------------------------------


def run_answer_quality(
    generator_llm: Any,
    judge_llm: Any,
    model_config: dict[str, Any] | None = None,
    suite_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score the generator on faithfulness, relevancy, and coherence.

    Both ``generator_llm`` and ``judge_llm`` must be *separate*
    ``llama_cpp.Llama`` instances — the spec forbids self-grading.

    Returns a dict whose top-level keys slot directly into the
    ``qualitative`` block of the composable schema, plus a ``meta``
    block carrying the prompt-cache hash for reproducibility.
    """
    _ = model_config  # reserved for future use
    suite_config = suite_config or {}
    sample_size = int(
        suite_config.get("answer_quality_sample_size", DEFAULT_SAMPLE_SIZE)
    )

    if generator_llm is judge_llm:
        raise ValueError(
            "answer_quality: generator_llm and judge_llm must be different "
            "Llama instances (the judge must not grade itself)."
        )

    prompts, cache_hash = _load_or_build_prompt_cache(sample_size)
    total = len(prompts)
    print(
        f"\n[answer-quality] {total} prompt(s) (cache sha256={cache_hash[:12]}…)",
        flush=True,
    )

    faith_scores: list[float] = []
    relevancy_scores: list[float] = []
    coherence_scores: list[float] = []
    n_no_claims = 0

    for i, prompt in enumerate(prompts, start=1):
        t0 = time.time()
        response = _llm_complete(
            generator_llm,
            prompt,
            max_tokens=DEFAULT_GEN_MAX_TOKENS,
            temperature=DEFAULT_GEN_TEMPERATURE,
        )
        if not response.strip():
            # Empty generation — record a 0 across the board, but no
            # claims to verify, so faithfulness is undefined for this
            # prompt and excluded from the mean (consistent with the
            # spec's "null when no claims extracted" rule).
            n_no_claims += 1
            relevancy_scores.append(0.0)
            coherence_scores.append(0.0)
            print(
                f"  ✓ [{i}/{total}] empty generation — "
                f"faithfulness=none relevancy=0.00 coherence=0.00 "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )
            continue

        faith, _n_yes, n_total = _score_faithfulness(judge_llm, response)
        if faith is None:
            n_no_claims += 1
        else:
            faith_scores.append(faith)
        relevancy = _score_relevancy(judge_llm, prompt, response)
        coherence = _score_coherence(judge_llm, response)
        if relevancy is not None:
            relevancy_scores.append(relevancy)
        if coherence is not None:
            coherence_scores.append(coherence)

        composite_components = [
            x for x in (faith, relevancy, coherence) if x is not None
        ]
        composite = (
            sum(composite_components) / len(composite_components)
            if composite_components
            else float("nan")
        )
        elapsed = time.time() - t0
        print(
            f"  ✓ [{i}/{total}] "
            f"faithfulness={('%.2f' % faith) if faith is not None else 'none'} "
            f"relevancy={('%.2f' % relevancy) if relevancy is not None else 'none'} "
            f"coherence={('%.2f' % coherence) if coherence is not None else 'none'} "
            f"composite={('%.2f' % composite) if not math.isnan(composite) else 'none'} "
            f"(claims={n_total}, {elapsed:.1f}s)",
            flush=True,
        )

    faithfulness_mean = statistics.fmean(faith_scores) if faith_scores else None
    faithfulness_std = statistics.stdev(faith_scores) if len(faith_scores) > 1 else None
    answer_relevancy_mean = (
        statistics.fmean(relevancy_scores) if relevancy_scores else None
    )
    coherence_mean = statistics.fmean(coherence_scores) if coherence_scores else None
    composite_components = [
        m
        for m in (faithfulness_mean, answer_relevancy_mean, coherence_mean)
        if m is not None
    ]
    quality_composite_score = (
        sum(composite_components) / len(composite_components)
        if composite_components
        else None
    )

    print(
        f"[answer-quality] "
        f"faithfulness_mean={faithfulness_mean if faithfulness_mean is None else f'{faithfulness_mean:.3f}'} "
        f"(std={faithfulness_std if faithfulness_std is None else f'{faithfulness_std:.3f}'}) "
        f"relevancy_mean={answer_relevancy_mean if answer_relevancy_mean is None else f'{answer_relevancy_mean:.3f}'} "
        f"coherence_mean={coherence_mean if coherence_mean is None else f'{coherence_mean:.3f}'} "
        f"composite={quality_composite_score if quality_composite_score is None else f'{quality_composite_score:.3f}'}",
        flush=True,
    )

    return {
        "faithfulness_mean": faithfulness_mean,
        "faithfulness_std": faithfulness_std,
        "answer_relevancy_mean": answer_relevancy_mean,
        "coherence_mean": coherence_mean,
        "quality_composite_score": quality_composite_score,
        # Diagnostic / meta — passed through to the result envelope.
        "n_prompts": total,
        "n_no_claims": n_no_claims,
        "quality_prompts_cache_hash": cache_hash,
    }


# ---------------------------------------------------------------------------
# Convenience wrapper: handles judge-Llama lifecycle for ppb.py
# ---------------------------------------------------------------------------


def run_answer_quality_for_model(
    generator_path: Path,
    judge_model_path: Path | str,
    *,
    suite_config: dict[str, Any] | None = None,
    n_ctx: int = 4096,
    judge_n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    judge_n_gpu_layers: int = -1,
    verbose: bool = False,
) -> dict[str, Any]:
    """Load generator + judge, score answer quality, dispose of judge.

    The judge ``Llama`` is instantiated only for the duration of this
    call so the next qualitative phase doesn't pay double VRAM.
    """
    if not judge_model_path:
        raise ValueError(
            "answer_quality phase requires judge_model_path in suite TOML."
        )
    judge_path = Path(judge_model_path).expanduser()
    if not judge_path.exists():
        raise FileNotFoundError(
            f"answer_quality: judge_model_path does not exist: {judge_path}"
        )
    if Path(generator_path).resolve() == judge_path.resolve():
        raise ValueError(
            "answer_quality: judge_model_path must differ from the generator "
            "model — the judge cannot grade itself."
        )

    try:
        from llama_cpp import Llama
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ppb_answer_quality requires llama-cpp-python. "
            "Install with: pip install llama-cpp-python"
        ) from exc

    print(
        f"[answer-quality] loading generator {Path(generator_path).name} "
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
        print(
            f"[answer-quality] loading judge {judge_path.name} "
            f"(n_ctx={judge_n_ctx}, n_gpu_layers={judge_n_gpu_layers})",
            flush=True,
        )
        judge = Llama(
            model_path=str(judge_path),
            n_ctx=judge_n_ctx,
            n_gpu_layers=judge_n_gpu_layers,
            verbose=verbose,
        )
        try:
            return run_answer_quality(
                generator,
                judge,
                model_config=None,
                suite_config=suite_config or {},
            )
        finally:
            del judge
    finally:
        del generator


__all__ = [
    "run_answer_quality",
    "run_answer_quality_for_model",
    "PROMPT_CACHE_PATH",
]
