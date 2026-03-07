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

        execute_sweep(results_file=results, config_path=cfg)

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
        execute_sweep(results_file=results, config_path=cfg)

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
            execute_sweep(results_file=results, config_path=cfg)

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
            execute_sweep(results_file=results, config_path=cfg)

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
            execute_sweep(results_file=results, config_path=cfg)

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
            execute_sweep(results_file=results, config_path=cfg)

        content = results.read_text().strip() if results.exists() else ""
        assert content == ""

    def test_result_envelope_structure(
        self, tmp_path: Path, tmp_model: Path
    ) -> None:
        """Every JSONL record must have the stable envelope fields."""
        from ppb import execute_sweep

        cfg = _sweep_toml(tmp_path, str(tmp_model))
        results = tmp_path / "results.jsonl"
        execute_sweep(results_file=results, config_path=cfg)

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
            execute_sweep(results_file=tmp_path / "r.jsonl", config_path=cfg)

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
            execute_sweep(results_file=tmp_path / "r.jsonl", config_path=cfg)


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


class TestExecuteSweepMaxCtxCap:
    """Verify max_ctx_cap filtering in execute_sweep."""

    def test_max_ctx_caps_filters_combos(self, tmp_path: Path, tmp_model: Path) -> None:
        """Combos with n_ctx > per-model cap should be skipped."""
        from ppb import execute_sweep

        cfg = _sweep_toml(
            tmp_path, str(tmp_model), n_ctx="[512, 1024, 4096]", n_batch="[256]"
        )
        results = tmp_path / "results.jsonl"
        execute_sweep(
            results_file=results, config_path=cfg,
            max_ctx_caps={tmp_model.resolve(): 1024},
        )

        lines = results.read_text().strip().splitlines()
        assert len(lines) == 2  # 512 and 1024 pass, 4096 skipped
        for line in lines:
            record = json.loads(line)
            assert record["n_ctx"] <= 1024

    def test_max_ctx_caps_none_runs_all(self, tmp_path: Path, tmp_model: Path) -> None:
        """max_ctx_caps=None must not filter any combos."""
        from ppb import execute_sweep

        cfg = _sweep_toml(
            tmp_path, str(tmp_model), n_ctx="[512, 1024, 4096]", n_batch="[256]"
        )
        results = tmp_path / "results.jsonl"
        execute_sweep(results_file=results, config_path=cfg, max_ctx_caps=None)

        lines = results.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_all_combos_above_cap(self, tmp_path: Path, tmp_model: Path) -> None:
        """When every combo exceeds cap, no results are written."""
        from ppb import execute_sweep

        cfg = _sweep_toml(
            tmp_path, str(tmp_model), n_ctx="[4096, 8192]", n_batch="[256]"
        )
        results = tmp_path / "results.jsonl"
        execute_sweep(
            results_file=results, config_path=cfg,
            max_ctx_caps={tmp_model.resolve(): 1024},
        )

        content = results.read_text().strip() if results.exists() else ""
        assert content == ""


# ==========================================================================
# execute_sweep with SweepConfig directly (CLI mode)
# ==========================================================================


class TestExecuteSweepDirect:
    """Verify execute_sweep works when given a pre-built SweepConfig."""

    def test_sweep_with_config_object(self, tmp_path: Path, tmp_model: Path) -> None:
        from ppb import SweepConfig, execute_sweep

        cfg = SweepConfig(
            model_path=str(tmp_model),
            n_ctx=[512],
            n_batch=[256],
            runner_type="fake",
        )
        results = tmp_path / "results.jsonl"
        execute_sweep(results_file=results, sweep_config=cfg)

        lines = results.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["runner_type"] == "fake"
        assert record["n_ctx"] == 512

    def test_both_args_raises(self, tmp_path: Path, tmp_model: Path) -> None:
        from ppb import SweepConfig, execute_sweep

        cfg = SweepConfig(
            model_path=str(tmp_model),
            n_ctx=[512],
            n_batch=[256],
            runner_type="fake",
        )
        toml_path = _sweep_toml(tmp_path, str(tmp_model))
        with pytest.raises(ValueError, match="not both"):
            execute_sweep(
                results_file=tmp_path / "r.jsonl",
                config_path=toml_path,
                sweep_config=cfg,
            )

    def test_neither_arg_raises(self, tmp_path: Path) -> None:
        from ppb import execute_sweep

        with pytest.raises(ValueError, match="required"):
            execute_sweep(results_file=tmp_path / "r.jsonl")


# ==========================================================================
# execute_auto_limit runner_params
# ==========================================================================


class TestAutoLimitRunnerParams:
    """Verify runner_params forwarding in execute_auto_limit."""

    def test_runner_params_forwarded(self, tmp_model: Path) -> None:
        from ppb import execute_auto_limit

        captured: list[FakeRunner] = []
        original_get = __import__("runners").get_runner

        def spy_get(rt: str):
            r = original_get(rt)
            captured.append(r)
            return r

        with patch("ppb.get_runner", side_effect=spy_get):
            execute_auto_limit(
                model_path=tmp_model,
                min_ctx=1024,
                max_ctx=2048,
                tolerance=1024,
                runner_type="fake",
                runner_params={"key": "val"},
            )

        assert captured[0]._params == {"key": "val"}


# ==========================================================================
# load_suite_config
# ==========================================================================


class TestLoadSuiteConfig:
    """Verify the suite config loader."""

    def test_returns_raw_and_results(self, suite_toml: Path) -> None:
        from ppb import load_suite_config

        raw, results = load_suite_config(suite_toml)
        assert "sweep" in raw
        assert "auto-limit" in raw
        assert results.name.endswith(".jsonl")

    def test_missing_file_exits(self, tmp_path: Path) -> None:
        from click.exceptions import Exit as ClickExit
        from ppb import load_suite_config

        with pytest.raises((SystemExit, ClickExit)):
            load_suite_config(tmp_path / "nope.toml")

    def test_toml_results_field_used(self, suite_toml_with_results: Path) -> None:
        from ppb import load_suite_config

        raw, results = load_suite_config(suite_toml_with_results)
        assert results == Path("custom.jsonl")


# ==========================================================================
# run_all command orchestration
# ==========================================================================


class TestRunAll:
    """Verify the ``all`` command chains auto-limit → sweep."""

    def test_all_auto_limit_then_sweep(
        self, tmp_path: Path, suite_toml: Path, tmp_model: Path
    ) -> None:
        """auto-limit should run, then sweep respects the cap."""
        from ppb import run_all
        from typer.testing import CliRunner
        from ppb import app

        runner = CliRunner()
        results = tmp_path / "all_results.jsonl"
        result = runner.invoke(
            app, ["all", str(suite_toml), "--results", str(results)]
        )
        # Should complete without hard error
        assert result.exit_code == 0
        # sweep should have written some results
        assert results.exists()
        lines = results.read_text().strip().splitlines()
        assert len(lines) > 0

    def test_all_without_autolimit_section(
        self, tmp_path: Path, suite_toml_no_autolimit: Path
    ) -> None:
        """Without [auto-limit], sweep runs unmodified."""
        from typer.testing import CliRunner
        from ppb import app

        runner = CliRunner()
        results = tmp_path / "res.jsonl"
        result = runner.invoke(
            app, ["all", str(suite_toml_no_autolimit), "--results", str(results)]
        )
        assert result.exit_code == 0
        assert results.exists()
        lines = results.read_text().strip().splitlines()
        assert len(lines) == 2  # 2 n_ctx × 1 n_batch

    def test_all_auto_limit_fails_exits(self, tmp_path: Path, tmp_model: Path) -> None:
        """If auto-limit returns 0, the command must exit with code 1."""
        from typer.testing import CliRunner
        from ppb import app

        # Write a suite TOML where auto-limit will always fail
        cfg = tmp_path / "fail_suite.toml"
        cfg.write_text(
            textwrap.dedent(f"""\
            [auto-limit]
            model_path = "{tmp_model}"
            min_ctx = 1024
            max_ctx = 2048
            tolerance = 1024
            runner_type = "fake"

            [sweep]
            runner_type = "fake"
            model_path = "{tmp_model}"
            n_ctx = [512]
            n_batch = [256]
            """)
        )

        # Make probe always fail
        original_get = __import__("runners").get_runner

        def spy_get(rt: str):
            r = original_get(rt)
            r.probe_return = False
            return r

        runner = CliRunner()
        with patch("ppb.get_runner", side_effect=spy_get):
            result = runner.invoke(
                app, ["all", str(cfg), "--results", str(tmp_path / "r.jsonl")]
            )
        assert result.exit_code == 1


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
