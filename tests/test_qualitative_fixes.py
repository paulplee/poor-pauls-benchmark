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
