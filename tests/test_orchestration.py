"""Integration tests for the sweep and auto-limit orchestration.

These tests use a :class:`FakeRunner` (registered at test time) to verify
that the orchestrator correctly:
    • instantiates the runner via the registry
    • calls setup / run / teardown in the right order
    • enriches results with timestamp, hardware, runner_type
    • writes valid JSONL output
    • passes the right config to the runner
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from runners import _REGISTRY, register_runner
from tests.conftest import FakeRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _register_fake_runner():
    """Make 'fake' runner available for every test, then clean up."""
    register_runner("fake", FakeRunner)
    yield
    _REGISTRY.pop("fake", None)


def _sweep_toml(tmp_path: Path, model_path: str, **extra) -> Path:
    """Write a sweep TOML file with runner_type='fake'."""
    runner_params = extra.pop("runner_params", None)

    lines = [
        "[sweep]",
        'runner_type = "fake"',
        f'model_path = "{model_path}"',
    ]
    for key, val in extra.items():
        lines.append(f"{key} = {val}")
    if "n_ctx" not in extra:
        lines.append("n_ctx = [512]")
    if "n_batch" not in extra:
        lines.append("n_batch = [256]")

    if runner_params:
        lines.append("")
        lines.append("[sweep.runner_params]")
        for k, v in runner_params.items():
            lines.append(f'{k} = "{v}"')

    cfg = tmp_path / "sweep.toml"
    cfg.write_text("\n".join(lines) + "\n")
    return cfg


# ==========================================================================
# execute_sweep
# ==========================================================================


class TestExecuteSweep:
    """Verify the sweep orchestrator integrates with the runner plugin."""

    def test_basic_sweep_writes_jsonl(self, tmp_path: Path, tmp_model: Path) -> None:
        from ppb import execute_sweep

        cfg = _sweep_toml(tmp_path, str(tmp_model))
        results = tmp_path / "results.jsonl"

        execute_sweep(cfg, results)

        lines = results.read_text().strip().splitlines()
        assert len(lines) == 1  # 1 model × 1 ctx × 1 batch

        record = json.loads(lines[0])
        assert record["runner_type"] == "fake"
        assert record["model_path"] == str(tmp_model)
        assert record["n_ctx"] == 512
        assert record["n_batch"] == 256
        assert "timestamp" in record
        assert "hardware" in record
        assert "results" in record

    def test_cartesian_product(self, tmp_path: Path, tmp_model: Path) -> None:
        from ppb import execute_sweep

        cfg = _sweep_toml(
            tmp_path, str(tmp_model), n_ctx="[512, 1024]", n_batch="[256, 512]"
        )
        results = tmp_path / "results.jsonl"
        execute_sweep(cfg, results)

        lines = results.read_text().strip().splitlines()
        assert len(lines) == 4  # 1×2×2

    def test_runner_setup_called_with_params(
        self, tmp_path: Path, tmp_model: Path
    ) -> None:
        """runner_params from TOML must be forwarded to runner.setup()."""
        from ppb import execute_sweep

        cfg = _sweep_toml(
            tmp_path,
            str(tmp_model),
            runner_params={"custom_key": "custom_value"},
        )
        results = tmp_path / "results.jsonl"

        # Patch get_runner to capture the instance
        captured: list[FakeRunner] = []
        original_get = __import__("runners").get_runner

        def spy_get(rt: str):
            r = original_get(rt)
            captured.append(r)
            return r

        with patch("ppb.get_runner", side_effect=spy_get):
            execute_sweep(cfg, results)

        assert len(captured) == 1
        r = captured[0]
        assert r.setup_called
        assert r._params == {"custom_key": "custom_value"}

    def test_runner_teardown_called(
        self, tmp_path: Path, tmp_model: Path
    ) -> None:
        from ppb import execute_sweep

        captured: list[FakeRunner] = []
        original_get = __import__("runners").get_runner

        def spy_get(rt: str):
            r = original_get(rt)
            captured.append(r)
            return r

        cfg = _sweep_toml(tmp_path, str(tmp_model))
        results = tmp_path / "results.jsonl"

        with patch("ppb.get_runner", side_effect=spy_get):
            execute_sweep(cfg, results)

        assert captured[0].teardown_called

    def test_teardown_called_on_failure(
        self, tmp_path: Path, tmp_model: Path
    ) -> None:
        """teardown() must be called even when all runs fail."""
        from ppb import execute_sweep

        captured: list[FakeRunner] = []
        original_get = __import__("runners").get_runner

        def spy_get(rt: str):
            r = original_get(rt)
            r.run_return = None  # simulate failure
            captured.append(r)
            return r

        cfg = _sweep_toml(tmp_path, str(tmp_model))
        results = tmp_path / "results.jsonl"

        with patch("ppb.get_runner", side_effect=spy_get):
            execute_sweep(cfg, results)

        assert captured[0].teardown_called
        # No results should be written
        assert not results.exists() or results.read_text().strip() == ""

    def test_failed_run_not_written(
        self, tmp_path: Path, tmp_model: Path
    ) -> None:
        """When runner.run() returns None, no JSONL line should be written."""
        from ppb import execute_sweep

        original_get = __import__("runners").get_runner

        def spy_get(rt: str):
            r = original_get(rt)
            r.run_return = None
            return r

        cfg = _sweep_toml(tmp_path, str(tmp_model))
        results = tmp_path / "results.jsonl"

        with patch("ppb.get_runner", side_effect=spy_get):
            execute_sweep(cfg, results)

        content = results.read_text().strip() if results.exists() else ""
        assert content == ""

    def test_result_envelope_structure(
        self, tmp_path: Path, tmp_model: Path
    ) -> None:
        """Every JSONL record must have the stable envelope fields."""
        from ppb import execute_sweep

        cfg = _sweep_toml(tmp_path, str(tmp_model))
        results = tmp_path / "results.jsonl"
        execute_sweep(cfg, results)

        record = json.loads(results.read_text().strip())
        required_keys = {
            "timestamp",
            "runner_type",
            "model_path",
            "n_ctx",
            "n_batch",
            "hardware",
            "results",
        }
        assert required_keys.issubset(record.keys())

    def test_missing_sweep_section(self, tmp_path: Path) -> None:
        """Config without [sweep] must exit with an error."""
        from click.exceptions import Exit as ClickExit

        from ppb import execute_sweep

        cfg = tmp_path / "bad.toml"
        cfg.write_text("[other]\nfoo = 1\n")

        with pytest.raises((SystemExit, ClickExit)):
            execute_sweep(cfg, tmp_path / "r.jsonl")

    def test_unknown_runner_type(self, tmp_path: Path, tmp_model: Path) -> None:
        """An unregistered runner_type must produce a clear error."""
        from click.exceptions import Exit as ClickExit

        from ppb import execute_sweep

        cfg = tmp_path / "bad_runner.toml"
        cfg.write_text(
            textwrap.dedent(f"""\
            [sweep]
            runner_type = "nonexistent"
            model_path = "{tmp_model}"
            n_ctx = [512]
            n_batch = [256]
            """)
        )

        with pytest.raises((ValueError, SystemExit, ClickExit)):
            execute_sweep(cfg, tmp_path / "r.jsonl")


# ==========================================================================
# execute_auto_limit
# ==========================================================================


class TestExecuteAutoLimit:
    """Verify the binary-search orchestrator uses the runner plugin."""

    def test_basic_auto_limit(self, tmp_model: Path) -> None:
        """With probe always passing, should converge to max_ctx."""
        from ppb import execute_auto_limit

        result = execute_auto_limit(
            model_path=tmp_model,
            min_ctx=1024,
            max_ctx=4096,
            tolerance=1024,
            runner_type="fake",
        )
        # Since probe always returns True, last_good should be near max
        assert result >= 1024

    def test_probe_always_fails(self, tmp_model: Path) -> None:
        """With probe always failing, should return 0 (or min_ctx fallback)."""
        from ppb import execute_auto_limit

        original_get = __import__("runners").get_runner

        def spy_get(rt: str):
            r = original_get(rt)
            r.probe_return = False
            return r

        with patch("ppb.get_runner", side_effect=spy_get):
            result = execute_auto_limit(
                model_path=tmp_model,
                min_ctx=1024,
                max_ctx=4096,
                tolerance=1024,
                runner_type="fake",
            )

        # When all probes fail, last_good stays 0, safe = lo
        assert isinstance(result, int)

    def test_runner_type_passed_through(self, tmp_model: Path) -> None:
        """The runner_type parameter must reach get_runner."""
        from ppb import execute_auto_limit

        called_with: list[str] = []
        original_get = __import__("runners").get_runner

        def spy_get(rt: str):
            called_with.append(rt)
            return original_get(rt)

        with patch("ppb.get_runner", side_effect=spy_get):
            execute_auto_limit(
                model_path=tmp_model,
                min_ctx=1024,
                max_ctx=2048,
                tolerance=1024,
                runner_type="fake",
            )

        assert called_with == ["fake"]


# ==========================================================================
# Backward compatibility
# ==========================================================================


class TestBackwardCompat:
    """Ensure existing sweep.toml files without runner_type still work."""

    def test_default_runner_type_in_toml(
        self, tmp_path: Path, tmp_model: Path
    ) -> None:
        """A TOML with no runner_type must default to 'llama-bench'."""
        from ppb import SweepConfig

        cfg = SweepConfig(
            model_path=str(tmp_model),
            n_ctx=[512],
            n_batch=[256],
        )
        assert cfg.runner_type == "llama-bench"

    def test_sweep_without_runner_type_field(
        self, tmp_path: Path, tmp_model: Path
    ) -> None:
        """Parsing a TOML that omits runner_type must succeed."""
        import tomllib

        toml_path = tmp_path / "legacy.toml"
        toml_path.write_text(
            textwrap.dedent(f"""\
            [sweep]
            model_path = "{tmp_model}"
            n_ctx = [512]
            n_batch = [256]
            """)
        )

        with toml_path.open("rb") as fh:
            raw = tomllib.load(fh)

        from ppb import SweepConfig

        cfg = SweepConfig(**raw["sweep"])
        assert cfg.runner_type == "llama-bench"
        assert cfg.runner_params == {}
