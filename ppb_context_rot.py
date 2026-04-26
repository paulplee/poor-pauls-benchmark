"""
PPB — Context-Rot evaluation (semantic Needle-in-a-Haystack).

This module measures how a model's answer accuracy degrades as context length
increases.  It is inspired by NVIDIA's RULER benchmark and uses real ShareGPT
conversations as the haystack distractor text, with synthetic factual needles
injected at five depth positions per haystack length.

Multi-needle design
-------------------
A single hard-coded needle creates training-contamination risk and yields
zero variance over needle content.  PPB therefore ships a pool of 15
diverse needles (``DEFAULT_NEEDLES``) covering different answer shapes
(alphanumeric codes, names, dates, integers, currency, places, colours,
measurements, slugs, species, formulae, titles, jobs, constellations and
short phrases).  For every (haystack_length × depth_pct) cell, one needle
is selected deterministically with ``random.Random(seed + cell_index)``,
so the 30 default cells use 30 different selections (cycling if needed)
and a given ``seed`` reproduces exactly the same plan across runs.

Multi-needle inference (optional)
---------------------------------
The "context confusion" / "lost-in-the-middle" phenomenon — where large
contexts cause incoherent reasoning even when all relevant facts are
present — is not exercised by single-fact retrieval.  When
``suite_config['multi_needle_enabled']`` is true, PPB additionally runs
one scenario from ``DEFAULT_MULTI_NEEDLES`` per context-length
checkpoint.  Each scenario plants 2–3 facts and asks a question whose
correct answer requires arithmetic, comparison or conditional inference
across every planted fact.

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
4. Aggregate to per-length, per-depth, per-needle, and overall accuracy.
5. (Optional) Run multi-needle inference scenarios.

Public entry point
------------------
``run_context_rot(llm, model_config, suite_config) -> dict``

Returns a dict with the following keys:

* ``context_rot_accuracy_by_length``   — ``{token_count: float | None}``
* ``context_rot_accuracy_by_depth``    — ``{depth_pct: float}``
* ``context_rot_accuracy_by_needle``   — ``{needle_label: float}``
* ``context_rot_score``                — ``float`` (mean over all cells)
* ``multi_needle_score``               — ``float | None``
* ``multi_needle_accuracy_by_length``  — ``{token_count: float} | None``
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Iterable

from ppb_datasets.sharegpt import (
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

# Default seed used to deterministically choose a needle per cell.  Override
# via ``suite_config['needle_seed']``.
DEFAULT_NEEDLE_SEED: int = 20260426

# 15 diverse needles with deliberately varied answer shapes.  Each answer is
# short, unambiguous, and unlikely to occur verbatim in ShareGPT haystack
# text by coincidence.  ``label`` is a stable short identifier used to key
# per-needle accuracy aggregation.
DEFAULT_NEEDLES: list[dict[str, str]] = [
    {
        "label": "code",
        "text": "The internal access code for the Marigold archive is QX9-MERIDIAN-42.",
        "query": "Based only on the text provided, what is the internal access code for the Marigold archive? Answer with just the code.",
        "answer": "QX9-MERIDIAN-42",
    },
    {
        "label": "name",
        "text": "The lead curator of the Aldworth Botanical Garden is Dr. Elspeth Quinones.",
        "query": "Based only on the text provided, who is the lead curator of the Aldworth Botanical Garden? Answer with just the full name.",
        "answer": "Elspeth Quinones",
    },
    {
        "label": "date",
        "text": "The Halberd-IV satellite was decommissioned on 14 March 2031.",
        "query": "Based only on the text provided, on what date was the Halberd-IV satellite decommissioned? Answer with just the date.",
        "answer": "14 March 2031",
    },
    {
        "label": "integer",
        "text": "The Cobalt-Spire research vessel houses exactly 387 climate sensors.",
        "query": "Based only on the text provided, how many climate sensors does the Cobalt-Spire research vessel house? Answer with just the integer.",
        "answer": "387",
    },
    {
        "label": "currency",
        "text": "The Lazuli Foundation pledged a grant of \u00a32,419,500 to the project.",
        "query": "Based only on the text provided, what amount did the Lazuli Foundation pledge to the project? Answer with just the amount including the currency symbol.",
        "answer": "\u00a32,419,500",
    },
    {
        "label": "city",
        "text": "The 2034 Tessellation Symposium will be held in Reykjav\u00edk.",
        "query": "Based only on the text provided, in which city will the 2034 Tessellation Symposium be held? Answer with just the city name.",
        "answer": "Reykjav\u00edk",
    },
    {
        "label": "colour",
        "text": "The exterior of the Hexalith pavilion is painted vermilion.",
        "query": "Based only on the text provided, what colour is the exterior of the Hexalith pavilion painted? Answer with just the colour.",
        "answer": "vermilion",
    },
    {
        "label": "measurement",
        "text": "The Greythorne suspension bridge spans 1,742 metres across the gorge.",
        "query": "Based only on the text provided, how long is the Greythorne suspension bridge in metres? Answer with just the value and unit.",
        "answer": "1,742 metres",
    },
    {
        "label": "slug",
        "text": "The internal documentation portal lives at /docs/quasar-orbit-handbook.",
        "query": "Based only on the text provided, at which path does the internal documentation portal live? Answer with just the URL slug.",
        "answer": "/docs/quasar-orbit-handbook",
    },
    {
        "label": "species",
        "text": "The mascot of the Findhorn observatory is the Atlantic puffin (Fratercula arctica).",
        "query": "Based only on the text provided, which species is the mascot of the Findhorn observatory? Answer with just the common species name.",
        "answer": "Atlantic puffin",
    },
    {
        "label": "formula",
        "text": "The new catalyst synthesised by the Trentham team has the chemical formula C8H10N4O2.",
        "query": "Based only on the text provided, what is the chemical formula of the catalyst synthesised by the Trentham team? Answer with just the formula.",
        "answer": "C8H10N4O2",
    },
    {
        "label": "film",
        "text": "The retrospective will close with a screening of the 1974 film 'Lanterns of Carmine Bay'.",
        "query": "Based only on the text provided, which film will close the retrospective? Answer with just the film title.",
        "answer": "Lanterns of Carmine Bay",
    },
    {
        "label": "job_title",
        "text": "At the Vasari Institute, Priya Khatri serves as Principal Cryogenics Engineer.",
        "query": "Based only on the text provided, what job title does Priya Khatri hold at the Vasari Institute? Answer with just the job title.",
        "answer": "Principal Cryogenics Engineer",
    },
    {
        "label": "constellation",
        "text": "The Brindley star catalogue lists HD-204941 as the brightest member of Lacerta.",
        "query": "Based only on the text provided, in which constellation does the Brindley catalogue place HD-204941? Answer with just the constellation name.",
        "answer": "Lacerta",
    },
    {
        "label": "phrase",
        "text": "The Whitechapel weather station's official motto is 'still skies, steady hands'.",
        "query": "Based only on the text provided, what is the official motto of the Whitechapel weather station? Answer with just the motto.",
        "answer": "still skies, steady hands",
    },
]

# Multi-needle inference scenarios.  Each scenario plants several facts and
# asks a question whose answer requires reasoning over ALL the facts.
DEFAULT_MULTI_NEEDLES: list[dict[str, Any]] = [
    {
        "label": "arithmetic",
        "needles": [
            {"label": "crew", "text": "The Helix-9 polar expedition has a crew of 23 researchers."},
            {"label": "sleds", "text": "Each Helix-9 expedition member is assigned 4 supply sleds."},
        ],
        "query": "Based only on the text provided, how many supply sleds in total are assigned to the Helix-9 polar expedition? Answer with just the integer.",
        "answer": "92",
    },
    {
        "label": "superlative",
        "needles": [
            {"label": "alpha", "text": "The Alphard reservoir holds 14,200 megalitres of water."},
            {"label": "betel", "text": "The Betelgeuse reservoir holds 9,750 megalitres of water."},
            {"label": "casto", "text": "The Castor reservoir holds 18,640 megalitres of water."},
        ],
        "query": "Based only on the text provided, which reservoir holds the most water? Answer with just the reservoir name.",
        "answer": "Castor",
    },
    {
        "label": "conditional",
        "needles": [
            {"label": "rule", "text": "The Nimbus library opens at 09:00 on weekdays and at 11:00 at weekends."},
            {"label": "fact", "text": "Today's date in this scenario is Saturday 12 October 2030."},
        ],
        "query": "Based only on the text provided, at what time does the Nimbus library open today? Answer with just the time in HH:MM.",
        "answer": "11:00",
    },
    {
        "label": "composition",
        "needles": [
            {"label": "founded", "text": "The Glasshouse cooperative was founded in 1998."},
            {"label": "members", "text": "The Glasshouse cooperative currently has 53 active members."},
        ],
        "query": "Based only on the text provided, how many years has the Glasshouse cooperative been active in 2026, multiplied by its current membership? Answer with just the integer.",
        "answer": "1484",
    },
    {
        "label": "chain",
        "needles": [
            {"label": "plane", "text": "The Skylark-7 aircraft cruises at 480 knots."},
            {"label": "distance", "text": "The route from Brackenford to Lyssan Point is 2,160 nautical miles."},
        ],
        "query": "Based only on the text provided, how many hours does it take a Skylark-7 to fly from Brackenford to Lyssan Point at cruise speed? Answer with just the integer number of hours.",
        "answer": "4 hours",
    },
]


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

    log.info(
        "[context-rot] building ShareGPT token pool (target: %d tokens) — "
        "this may take ~10–30 s on the first run …",
        needed,
    )

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
    label: str | None = None,
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
    haystack_budget = (
        target_ctx - len(needle_tokens) - len(query_tokens) - answer_budget
    )
    # Note: callers (run_context_rot) pre-validate haystack_budget > 0
    # before invoking this helper, so the only remaining failure mode is
    # a token pool smaller than the requested haystack — still a hard error.
    if len(token_pool) < haystack_budget:
        raise ValueError(
            f"token pool ({len(token_pool)}) smaller than haystack budget "
            f"({haystack_budget})"
        )

    haystack = token_pool[:haystack_budget]

    # Insert the needle at the requested depth.
    insert_at = (len(haystack) * depth_pct) // 100
    composed = (
        haystack[:insert_at] + needle_tokens + haystack[insert_at:] + query_tokens
    )

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
    label_part = f" needle={label}" if label else ""
    print(
        f"  ✓ [{case_idx}/{total_cases}] ctx={target_ctx} "
        f"depth={depth_pct}%{label_part} {status} ({elapsed:.1f}s)",
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
        * ``needles``          (list[dict[text, query, answer, label?]])
        * ``needle_seed``      (int)
        * ``multi_needle_enabled`` (bool, default False)
        * ``multi_needles``    (list of multi-needle scenarios)

    Returns
    -------
    dict
        See module docstring for the full schema.
    """
    model_config = model_config or {}
    suite_config = suite_config or {}

    haystack_lengths: list[int] = list(
        suite_config.get("haystack_lengths") or DEFAULT_HAYSTACK_LENGTHS
    )
    depths_pct: list[int] = list(suite_config.get("depths_pct") or DEFAULT_DEPTHS_PCT)

    needles: list[dict[str, str]] = list(
        suite_config.get("needles") or DEFAULT_NEEDLES
    )
    if not needles:
        raise ValueError(
            "context-rot: at least one needle must be provided in suite_config['needles']."
        )
    needle_seed: int = int(suite_config.get("needle_seed") or DEFAULT_NEEDLE_SEED)

    multi_needle_enabled: bool = bool(suite_config.get("multi_needle_enabled", False))
    multi_needles: list[dict[str, Any]] = list(
        suite_config.get("multi_needles") or DEFAULT_MULTI_NEEDLES
    )

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
        + (
            f"  [skipped {len(skipped_lengths)} > VRAM cap {max_ctx_cap}]"
            if skipped_lengths
            else ""
        ),
        flush=True,
    )

    if total_cases == 0:
        return {
            "context_rot_accuracy_by_length": {L: None for L in haystack_lengths},
            "context_rot_accuracy_by_depth": {d: 0.0 for d in depths_pct},
            "context_rot_accuracy_by_needle": {},
            "context_rot_score": 0.0,
            "multi_needle_score": None,
            "multi_needle_accuracy_by_length": None,
        }

    # Pre-tokenise every needle in the pool so we can reuse the encoded
    # form when a needle is selected for a cell (and validate budgets).
    def _tokenize(text: str) -> list[int]:
        try:
            return llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)
        except TypeError:
            return llm.tokenize(text.encode("utf-8"), add_bos=False)

    needle_specs: list[dict[str, Any]] = []
    for i, n in enumerate(needles):
        text = str(n.get("text") or "").strip()
        query = str(n.get("query") or "").strip()
        answer = str(n.get("answer") or "").strip()
        if not text or not query or not answer:
            raise ValueError(
                f"context-rot: needle #{i} is missing text/query/answer"
            )
        needle_specs.append(
            {
                "label": str(n.get("label") or f"needle_{i}"),
                "text": text,
                "query": query,
                "answer": answer,
                "needle_tokens": _tokenize(text),
                "query_tokens": _tokenize(query),
            }
        )

    # Build a token pool large enough for the longest runnable haystack.
    target_max = max(runnable_lengths)
    pool = _build_token_pool(llm, target_max)

    # Run all cases.
    passes_by_length: dict[int, list[int]] = {L: [] for L in runnable_lengths}
    passes_by_depth: dict[int, list[int]] = {d: [] for d in depths_pct}
    passes_by_needle: dict[str, list[int]] = {n["label"]: [] for n in needle_specs}

    _answer_budget = 20

    case_idx = 0
    cell_index = 0
    for L in runnable_lengths:
        for depth in depths_pct:
            case_idx += 1
            # Deterministic per-cell needle selection: stable across runs
            # for a given seed, but rotates across the 30 default cells.
            rng = random.Random(needle_seed + cell_index)
            spec = rng.choice(needle_specs)
            cell_index += 1

            haystack_budget = (
                L
                - len(spec["needle_tokens"])
                - len(spec["query_tokens"])
                - _answer_budget
            )
            if haystack_budget <= 0:
                log.warning(
                    "[context-rot] skipping ctx=%d depth=%d%% needle=%s: "
                    "needle+query+answer leaves no room for a haystack",
                    L,
                    depth,
                    spec["label"],
                )
                passes_by_length[L].append(0)
                passes_by_depth[depth].append(0)
                passes_by_needle[spec["label"]].append(0)
                continue
            try:
                passed, _elapsed, _resp = _run_single_case(
                    llm,
                    token_pool=pool,
                    needle_tokens=spec["needle_tokens"],
                    needle_query=spec["query"],
                    needle_answer=spec["answer"],
                    target_ctx=L,
                    depth_pct=depth,
                    case_idx=case_idx,
                    total_cases=total_cases,
                    label=spec["label"],
                )
            except Exception as exc:  # pragma: no cover — defensive
                log.warning(
                    "[context-rot] case %d (ctx=%d depth=%d%% needle=%s) failed: %s",
                    case_idx,
                    L,
                    depth,
                    spec["label"],
                    exc,
                )
                passed = 0
            passes_by_length[L].append(passed)
            passes_by_depth[depth].append(passed)
            passes_by_needle[spec["label"]].append(passed)

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

    accuracy_by_needle: dict[str, float] = {
        label: (sum(arr) / len(arr)) for label, arr in passes_by_needle.items() if arr
    }

    all_passes: list[int] = [p for ps in passes_by_length.values() for p in ps]
    overall = sum(all_passes) / len(all_passes) if all_passes else 0.0

    # ---- Multi-needle inference (optional) -----------------------------
    multi_needle_score: float | None = None
    multi_needle_accuracy_by_length: dict[int, float] | None = None

    if multi_needle_enabled and runnable_lengths and multi_needles:
        print(
            f"\n[context-rot] multi-needle: {len(runnable_lengths)} length(s) \u00d7 "
            f"1 scenario each = {len(runnable_lengths)} case(s)",
            flush=True,
        )
        multi_needle_accuracy_by_length = {}
        mn_passes: list[int] = []
        for i, L in enumerate(runnable_lengths):
            scenario = multi_needles[i % len(multi_needles)]
            sc_label = str(scenario.get("label") or f"scenario_{i}")
            sc_query = str(scenario.get("query") or "").strip()
            sc_answer = str(scenario.get("answer") or "").strip()
            sc_facts = scenario.get("needles") or []
            if not sc_query or not sc_answer or not sc_facts:
                log.warning(
                    "[context-rot] multi-needle scenario %s is malformed; skipping",
                    sc_label,
                )
                continue

            # Tokenise the planted facts and the query.
            fact_token_blocks: list[list[int]] = []
            for f in sc_facts:
                ftext = str(f.get("text") or "").strip()
                if ftext:
                    fact_token_blocks.append(_tokenize(ftext))
            query_tokens = _tokenize(sc_query)

            facts_total = sum(len(b) for b in fact_token_blocks)
            haystack_budget = L - facts_total - len(query_tokens) - _answer_budget
            if haystack_budget <= 0:
                log.warning(
                    "[context-rot] multi-needle skip ctx=%d %s: facts+query "
                    "exceed budget",
                    L,
                    sc_label,
                )
                multi_needle_accuracy_by_length[L] = 0.0
                mn_passes.append(0)
                continue
            if len(pool) < haystack_budget:
                log.warning(
                    "[context-rot] multi-needle skip ctx=%d %s: token pool too small",
                    L,
                    sc_label,
                )
                multi_needle_accuracy_by_length[L] = 0.0
                mn_passes.append(0)
                continue

            haystack = pool[:haystack_budget]
            # Insert the facts at evenly-spaced depths within the haystack.
            n_facts = len(fact_token_blocks)
            insertion_points = [
                ((j + 1) * len(haystack)) // (n_facts + 1) for j in range(n_facts)
            ]
            # Insert from the end so earlier indices remain valid.
            composed = list(haystack)
            for idx, block in zip(reversed(insertion_points), reversed(fact_token_blocks)):
                composed = composed[:idx] + block + composed[idx:]
            composed = composed + query_tokens

            prompt_bytes = _detokenize(llm, composed)
            prompt = prompt_bytes.decode("utf-8", errors="replace")

            t0 = time.time()
            try:
                out = llm(
                    prompt,
                    max_tokens=_answer_budget,
                    temperature=0.0,
                    top_p=1.0,
                    echo=False,
                )
                response = out["choices"][0]["text"]
            except Exception as exc:  # pragma: no cover — defensive
                log.warning(
                    "[context-rot] multi-needle ctx=%d %s failed: %s", L, sc_label, exc
                )
                response = ""
            elapsed = time.time() - t0

            passed = _score_response(response, sc_answer)
            multi_needle_accuracy_by_length[L] = float(passed)
            mn_passes.append(passed)
            print(
                f"  \u2713 [multi-needle ctx={L}] {sc_label} "
                f"{'pass' if passed else 'fail'} ({elapsed:.1f}s)",
                flush=True,
            )

        multi_needle_score = (
            sum(mn_passes) / len(mn_passes) if mn_passes else None
        )

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
        "context_rot_accuracy_by_needle": accuracy_by_needle,
        "context_rot_score": overall,
        "multi_needle_score": multi_needle_score,
        "multi_needle_accuracy_by_length": multi_needle_accuracy_by_length,
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
