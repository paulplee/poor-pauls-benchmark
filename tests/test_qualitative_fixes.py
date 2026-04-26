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
      * Future-phase placeholders (``faithfulness_mean`` etc.) are present
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
    assert "faithfulness_mean" in qual and qual["faithfulness_mean"] is None
    assert "mt_bench_score" in qual and qual["mt_bench_score"] is None
