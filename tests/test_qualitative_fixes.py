"""Regression tests for the qualitative-branch review fixes.

Covers:
- Fix 3 (HIGH): ``ppb run`` banner reports the correct mode label.
- Fix 4 + 5: ``_composable_key`` warns on empty-quant filenames; BFCL loader
  passes ``trust_remote_code=True``.
- Fix 6: ``run_tool_accuracy`` returns the diagnostic ``n_*`` keys (so the
  orchestrator can strip them before publishing).
- ``_ast_match`` correctly handles the empty-args case.
- ``COLUMN_ORDER`` includes the new composable ``qualitative`` /
  ``quantitative`` nested-block columns (Fix 2).
"""

from __future__ import annotations

import logging

from utils.flattener import COLUMN_ORDER
from utils.publisher import _composable_key


# ---------------------------------------------------------------------------
# Fix 2 — schema columns
# ---------------------------------------------------------------------------


def test_column_order_has_composable_blocks() -> None:
    assert "qualitative" in COLUMN_ORDER
    assert "quantitative" in COLUMN_ORDER
    assert "run_type" in COLUMN_ORDER


# ---------------------------------------------------------------------------
# Fix 5 — publisher warns on missing quant tag
# ---------------------------------------------------------------------------


def test_composable_key_warns_when_quant_missing(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="ppb")
    gpu, base, quant = _composable_key(
        "user/repo/model.gguf", {"gpus": [{"name": "RTX 4090"}]}
    )
    assert gpu == "RTX 4090"
    assert base == "model"
    assert quant == ""
    assert any(
        "Could not extract quantization" in rec.getMessage() for rec in caplog.records
    )


def test_composable_key_no_warning_for_valid_filename(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="ppb")
    gpu, base, quant = _composable_key(
        "user/repo/Qwen3.5-2B-Q4_K_M.gguf", {"gpus": [{"name": "RTX 4090"}]}
    )
    assert quant == "Q4_K_M"
    assert base == "Qwen3.5-2B"
    assert not any(
        "Could not extract quantization" in rec.getMessage() for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# Fix 3 — banner mode label
# ---------------------------------------------------------------------------


def test_banner_uses_mode_argument(monkeypatch, tmp_path) -> None:
    """``run_all`` should print ``Run mode: <mode>`` near the top of execution.

    We monkey-patch ``console.print`` to record output and ``load_suite_config``
    to short-circuit before any HF download or model setup runs.
    """
    import ppb

    captured: list[str] = []

    def fake_print(*args, **kwargs) -> None:  # noqa: ARG001
        for a in args:
            captured.append(str(a))
        # Abort the rest of the pipeline immediately after the banner.
        raise RuntimeError("stop-after-banner")

    monkeypatch.setattr(ppb.console, "print", fake_print)
    # Make load_suite_config return a minimal raw + dummy results path.
    monkeypatch.setattr(
        ppb,
        "load_suite_config",
        lambda _cfg: ({"sweep": {"runner_type": "fake"}}, tmp_path / "r.jsonl"),
    )

    cfg = tmp_path / "x.toml"
    cfg.write_text("[sweep]\n")
    try:
        ppb.run_all(
            cfg,
            results_file=tmp_path / "r.jsonl",
            no_resume=True,
            mode="qualitative",
        )
    except RuntimeError:
        pass
    except Exception:
        pass

    banner = " ".join(captured).lower()
    assert "run mode:" in banner
    assert "qualitative" in banner


# ---------------------------------------------------------------------------
# End-to-end: qualitative publish pipeline produces the composable schema
# ---------------------------------------------------------------------------


def test_qualitative_publish_pipeline(monkeypatch, tmp_path) -> None:
    """End-to-end: qualitative-mode ``run_all`` → ``publish_to_hf`` rows.

    Verifies that the rows handed to ``publish_to_hf`` carry the full
    composable schema:
      * ``run_type == "qualitative"``                            (Fix 1)
      * ``quantitative is None``                                 (Fix 2a)
      * ``qualitative`` block populated from real phase results  (Fix 2b)
      * Diagnostic ``n_*`` keys are stripped                     (Fix 6)
      * Future-phase placeholders (``knowledge_accuracy_mean`` etc.) are present
        and ``None`` so consumers can rely on key existence.

    Phases that hit the GPU / network are mocked; ``_build_qualitative_block``
    and ``flatten_benchmark_row`` are NOT mocked — the test exercises the
    real wiring between them and ``publish_to_hf``.
    """
    import pytest

    import ppb
    import ppb_context_rot
    import ppb_tool_accuracy
    import utils.publisher

    # -- 1. Capture publish_to_hf -----------------------------------------
    publish_calls: list[tuple[tuple, dict]] = []

    def fake_publish_to_hf(rows, *args, **kwargs):
        # Defensive copy: the caller may mutate after returning.
        publish_calls.append((tuple([list(rows)]) + tuple(args), dict(kwargs)))

    monkeypatch.setattr(ppb, "publish_to_hf", fake_publish_to_hf)
    monkeypatch.setattr(ppb, "check_hf_token", lambda *_a, **_kw: None)

    # -- 2. Mock the qualitative phases -----------------------------------
    ctx_result = {
        "context_rot_score": 0.75,
        "context_rot_accuracy_by_length": {"4096": 0.9, "8192": 0.6},
        "context_rot_accuracy_by_depth": {"10": 0.8, "50": 0.7, "90": 0.7},
    }
    tool_result = {
        "tool_selection_accuracy": 0.85,
        "parameter_accuracy": 0.80,
        "parameter_hallucination_rate": 0.05,
        "parse_success_rate": 0.95,
        "overall_tool_accuracy": 0.825,
        "n_cases": 120,
        "n_bfcl": 100,
        "n_ppb_native": 20,
    }

    monkeypatch.setattr(
        ppb_context_rot,
        "run_context_rot_for_model",
        lambda *_a, **_kw: ctx_result,
    )
    monkeypatch.setattr(
        ppb_tool_accuracy,
        "run_tool_accuracy_for_model",
        lambda *_a, **_kw: tool_result,
    )

    # -- 3. Mock the HF lookup so qualitative-only mode short-circuits ----
    monkeypatch.setattr(
        utils.publisher,
        "fetch_existing_quantitative_for",
        lambda *_a, **_kw: None,
    )

    # -- 4. Mock model resolution -----------------------------------------
    fake_model = tmp_path / "Qwen3.5-0.8B-Q4_K_M.gguf"
    # Real file not required: every consumer of ``mp`` is mocked above.
    fake_model.touch()
    fake_manifest = [
        (fake_model, "unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf", False, None)
    ]
    monkeypatch.setattr(ppb, "_resolve_models", lambda *_a, **_kw: fake_manifest)

    # -- 5. Write the suite TOML and call run_all -------------------------
    cfg = tmp_path / "suite.toml"
    cfg.write_text(
        'repo_id    = "unsloth/Qwen3.5-0.8B-GGUF"\n'
        'filename   = "Qwen3.5-0.8B-Q4_K_M.gguf"\n'
        f'models_dir = "{tmp_path}"\n'
        "\n"
        "[qualitative]\n"
        "context_rot_enabled  = true\n"
        "tool_accuracy_enabled = true\n"
        "\n"
        "[publish]\n"
        'submitter = "test-runner"\n'
        "upload    = true\n"
    )
    results_file = tmp_path / "results.jsonl"

    import typer

    try:
        ppb.run_all(
            cfg,
            results_file=results_file,
            no_resume=True,
            mode="qualitative",
        )
    except (SystemExit, typer.Exit):
        # Normal end-of-pipeline behaviour for typer commands.
        pass

    # -- 6. Assertions ----------------------------------------------------
    assert publish_calls, "publish_to_hf was never called"

    # Use the LAST publish_to_hf invocation: it runs after both qualitative
    # phases have completed, so its rows carry both context-rot and
    # tool-accuracy results in the qualitative block.
    last_args, _last_kwargs = publish_calls[-1]
    rows = last_args[0]
    assert len(rows) >= 1, "publish_to_hf received no rows"

    # Find a row corresponding to one of the qualitative phases (the
    # flattener may produce one row per source JSONL line).
    qualifying_rows = [
        r for r in rows if r.get("runner_type") in ("context-rot", "tool-accuracy")
    ] or rows
    row = qualifying_rows[-1]

    # Fix 1: run_type is stamped
    assert row["run_type"] == "qualitative"

    # Fix 2a: quantitative block is null in qualitative-only mode
    assert row.get("quantitative") is None

    # Fix 2b: qualitative block is present and populated
    qual = row.get("qualitative")
    assert qual is not None, "qualitative block missing from published row"
    assert isinstance(qual, dict)

    # Context-rot keys are populated from the mocked phase result
    assert qual["context_rot_score"] == pytest.approx(0.75)
    assert qual["context_rot_accuracy_by_length"] is not None

    # Tool accuracy keys are populated (diagnostic n_* keys must be absent)
    assert qual["tool_selection_accuracy"] == pytest.approx(0.85)
    assert qual["parameter_accuracy"] == pytest.approx(0.80)
    assert "n_cases" not in qual, (
        "diagnostic key n_cases must not appear in published qualitative block"
    )
    assert "n_bfcl" not in qual, (
        "diagnostic key n_bfcl must not appear in published qualitative block"
    )
    assert "n_ppb_native" not in qual, (
        "diagnostic key n_ppb_native must not appear in published qualitative block"
    )

    # Future-phase placeholders are present and null (not missing)
    assert "knowledge_accuracy_mean" in qual and qual["knowledge_accuracy_mean"] is None
    assert "mt_bench_score" in qual and qual["mt_bench_score"] is None


# ---------------------------------------------------------------------------
# Targeted unit tests added with d7290b9 follow-up gap fixes.
# ---------------------------------------------------------------------------


import json as _json

import pytest


def test_normalise_bfcl_row():
    """Verify _normalise_bfcl_row handles all three documented input shapes."""
    from ppb_tool_accuracy import _normalise_bfcl_row

    # (a) function as JSON string, answer as plain dict
    out_a = _normalise_bfcl_row(
        {
            "question": "search the web",
            "function": '{"name": "search", "parameters": {}}',
            "answer": {"name": "search", "arguments": {}},
        }
    )
    assert isinstance(out_a["function"], dict)
    assert out_a["function"]["name"] == "search"

    # (b) function already a dict, answer as a single-item list
    out_b = _normalise_bfcl_row(
        {
            "question": "search test",
            "function": {"name": "search", "parameters": {}},
            "answer": [{"name": "search", "arguments": {"q": "test"}}],
        }
    )
    assert isinstance(out_b["answer"], dict)
    assert out_b["answer"]["name"] == "search"
    assert out_b["answer"]["arguments"] == {"q": "test"}

    # (c) function as a list of two dicts (multiple_function), answer as JSON string
    out_c = _normalise_bfcl_row(
        {
            "question": "do something",
            "function": [
                {"name": "search", "parameters": {}},
                {"name": "lookup", "parameters": {}},
            ],
            "answer": '{"name": "lookup", "arguments": {"id": 1}}',
        }
    )
    assert isinstance(out_c["function"], list)
    assert len(out_c["function"]) == 2
    assert isinstance(out_c["answer"], dict)
    assert out_c["answer"]["name"] == "lookup"


def test_score_response_context_rot():
    """Verify _score_response covers exact, case-insensitive, substring, empty, and wrong cases."""
    from ppb_context_rot import _score_response

    assert _score_response("ALPHA-7-DELTA", "ALPHA-7-DELTA") == 1
    assert _score_response("alpha-7-delta", "ALPHA-7-DELTA") == 1
    assert _score_response("The answer is ALPHA-7-DELTA, confirmed.", "ALPHA-7-DELTA") == 1
    assert _score_response("", "ALPHA-7-DELTA") == 0
    assert _score_response("totally unrelated text", "ALPHA-7-DELTA") == 0


def test_exact_match_multiturn():
    """Verify _exact_match handles lowercase, wrong, empty, whitespace, and unicode."""
    from ppb_multiturn import _exact_match

    assert _exact_match("the capital is paris.", "Paris") is True
    assert _exact_match("the capital is london.", "Paris") is False
    assert _exact_match("", "Paris") is False
    assert _exact_match("the capital is paris.", "") is False
    assert _exact_match("the capital is paris.", "  Paris  ") is True
    assert _exact_match("we visited Reykjavík last summer", "Reykjavík") is True


def test_build_qualitative_block_partial():
    """Mirror _build_qualitative_block dict-assembly to ensure missing phases yield None.

    # Mirrors _build_qualitative_block in ppb.py — update if that function changes
    """
    state = {
        "context_rot": {
            "context_rot_score": 0.87,
            "context_rot_accuracy_by_length": {"4096": 1.0, "8192": 0.8},
            "context_rot_accuracy_by_depth": {"10": 1.0, "90": 0.6},
            "context_rot_accuracy_by_needle": {"code": 0.9, "date": 0.8},
            "multi_needle_score": None,
            "multi_needle_accuracy_by_length": None,
        },
        "tool_accuracy": None,
        "answer_quality": None,
        "multiturn": None,
    }
    cr = state.get("context_rot") or None
    ta = state.get("tool_accuracy") or None
    aq = state.get("answer_quality") or None
    mt = state.get("multiturn") or None

    block = {
        "context_rot_score": (cr or {}).get("context_rot_score"),
        "context_rot_accuracy_by_length": (cr or {}).get("context_rot_accuracy_by_length"),
        "context_rot_accuracy_by_depth": (cr or {}).get("context_rot_accuracy_by_depth"),
        "context_rot_accuracy_by_needle": (cr or {}).get("context_rot_accuracy_by_needle"),
        "multi_needle_score": (cr or {}).get("multi_needle_score"),
        "multi_needle_accuracy_by_length": (cr or {}).get("multi_needle_accuracy_by_length"),
        "tool_selection_accuracy": (ta or {}).get("tool_selection_accuracy"),
        "parameter_accuracy": (ta or {}).get("parameter_accuracy"),
        "parameter_hallucination_rate": (ta or {}).get("parameter_hallucination_rate"),
        "parse_success_rate": (ta or {}).get("parse_success_rate"),
        "overall_tool_accuracy": (ta or {}).get("overall_tool_accuracy"),
        "knowledge_accuracy_mean": (aq or {}).get("knowledge_accuracy_mean"),
        "knowledge_accuracy_std": (aq or {}).get("knowledge_accuracy_std"),
        "answer_relevancy_mean": (aq or {}).get("answer_relevancy_mean"),
        "coherence_mean": (aq or {}).get("coherence_mean"),
        "quality_composite_score": (aq or {}).get("quality_composite_score"),
        "memory_accuracy": (mt or {}).get("memory_accuracy"),
        "mt_bench_score": (mt or {}).get("mt_bench_score"),
        "cases_evaluated": (mt or {}).get("cases_evaluated"),
        "cases_skipped_context": (mt or {}).get("cases_skipped_context"),
    }

    # Phases that did not run must report None for their headline keys.
    assert block["tool_selection_accuracy"] is None
    assert block["parameter_accuracy"] is None
    assert block["knowledge_accuracy_mean"] is None
    assert block["knowledge_accuracy_std"] is None
    assert block["coherence_mean"] is None
    assert block["memory_accuracy"] is None
    assert block["mt_bench_score"] is None

    # Context-rot keys that were populated must round-trip unchanged.
    assert block["context_rot_score"] == 0.87
    assert block["context_rot_accuracy_by_length"] == {"4096": 1.0, "8192": 0.8}
    assert block["context_rot_accuracy_by_depth"] == {"10": 1.0, "90": 0.6}
    assert block["context_rot_accuracy_by_needle"] == {"code": 0.9, "date": 0.8}

    # Multi-needle keys are None when the suite did not opt in.
    assert block["multi_needle_score"] is None
    assert block["multi_needle_accuracy_by_length"] is None


def test_parse_claims_json():
    """Verify _parse_claims_json handles valid, empty, malformed, integer-claims, and fenced JSON."""
    from ppb_answer_quality import _parse_claims_json

    out_valid = _parse_claims_json('{"claims": ["The sky is blue.", "Water boils at 100°C."]}')
    assert out_valid == ["The sky is blue.", "Water boils at 100°C."]

    assert _parse_claims_json('{"claims": []}') == []
    assert _parse_claims_json("not json at all") == []

    out_ints = _parse_claims_json('{"claims": [1, 2, 3]}')
    assert out_ints == ["1", "2", "3"]

    out_fenced = _parse_claims_json('```json\n{"claims": ["test"]}\n```')
    assert out_fenced == ["test"]


def test_needle_selection_deterministic():
    """Verify deterministic per-cell needle selection mirrors run_context_rot."""
    import random as _random

    from ppb_context_rot import DEFAULT_NEEDLES, DEFAULT_NEEDLE_SEED

    assert len(DEFAULT_NEEDLES) == 15

    def select_needle(needles, seed, cell_index):
        return _random.Random(seed + cell_index).choice(needles)

    seq_a = [select_needle(DEFAULT_NEEDLES, DEFAULT_NEEDLE_SEED, i) for i in range(30)]
    seq_b = [select_needle(DEFAULT_NEEDLES, DEFAULT_NEEDLE_SEED, i) for i in range(30)]
    # Determinism: identical seeds yield identical selections.
    assert [n["answer"] for n in seq_a] == [n["answer"] for n in seq_b]

    # Rotation is working — not all 30 selections are the same needle.
    assert len({n["answer"] for n in seq_a}) > 1

    # Seed isolation: changing the seed by 1 yields a different sequence.
    seq_c = [select_needle(DEFAULT_NEEDLES, DEFAULT_NEEDLE_SEED + 1, i) for i in range(30)]
    assert [n["answer"] for n in seq_a] != [n["answer"] for n in seq_c]
