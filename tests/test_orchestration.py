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

        execute_sweep(config_path=cfg, results_file=results)

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
        execute_sweep(config_path=cfg, results_file=results)

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
            execute_sweep(config_path=cfg, results_file=results)

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
            execute_sweep(config_path=cfg, results_file=results)

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
            execute_sweep(config_path=cfg, results_file=results)

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
            execute_sweep(config_path=cfg, results_file=results)

        content = results.read_text().strip() if results.exists() else ""
        assert content == ""

    def test_result_envelope_structure(
        self, tmp_path: Path, tmp_model: Path
    ) -> None:
        """Every JSONL record must have the stable envelope fields."""
        from ppb import execute_sweep

        cfg = _sweep_toml(tmp_path, str(tmp_model))
        results = tmp_path / "results.jsonl"
        execute_sweep(config_path=cfg, results_file=results)

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
            execute_sweep(config_path=cfg, results_file=tmp_path / "r.jsonl")

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
            execute_sweep(config_path=cfg, results_file=tmp_path / "r.jsonl")


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


# ==========================================================================
# execute_sweep — new features
# ==========================================================================


class TestExecuteSweepMaxCtxCap:
    """Test per-model max_ctx_caps filtering in execute_sweep."""

    def test_cap_filters_combos(self, tmp_path: Path, tmp_model: Path) -> None:
        from ppb import execute_sweep

        cfg = _sweep_toml(
            tmp_path, str(tmp_model), n_ctx="[512, 1024, 2048]", n_batch="[256]"
        )
        results = tmp_path / "results.jsonl"
        execute_sweep(
            config_path=cfg, results_file=results,
            max_ctx_caps={tmp_model.resolve(): 1024},
        )

        lines = results.read_text().strip().splitlines()
        assert len(lines) == 2  # Only 512 and 1024
        for line in lines:
            record = json.loads(line)
            assert record["n_ctx"] <= 1024

    def test_cap_none_runs_all(self, tmp_path: Path, tmp_model: Path) -> None:
        from ppb import execute_sweep

        cfg = _sweep_toml(
            tmp_path, str(tmp_model), n_ctx="[512, 1024, 2048]", n_batch="[256]"
        )
        results = tmp_path / "results.jsonl"
        execute_sweep(config_path=cfg, results_file=results, max_ctx_caps=None)

        lines = results.read_text().strip().splitlines()
        assert len(lines) == 3


class TestExecuteSweepDirect:
    """Test passing a SweepConfig object directly (no TOML)."""

    def test_sweep_config_direct(self, tmp_path: Path, tmp_model: Path) -> None:
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
        assert record["n_ctx"] == 512

    def test_both_config_path_and_sweep_config_raises(
        self, tmp_path: Path, tmp_model: Path
    ) -> None:
        from ppb import SweepConfig, execute_sweep

        cfg_obj = SweepConfig(
            model_path=str(tmp_model),
            n_ctx=[512],
            n_batch=[256],
            runner_type="fake",
        )
        cfg_file = _sweep_toml(tmp_path, str(tmp_model))

        with pytest.raises(ValueError, match="not both"):
            execute_sweep(
                config_path=cfg_file,
                results_file=tmp_path / "r.jsonl",
                sweep_config=cfg_obj,
            )

    def test_neither_raises(self, tmp_path: Path) -> None:
        from ppb import execute_sweep

        with pytest.raises(ValueError, match="required"):
            execute_sweep(results_file=tmp_path / "r.jsonl")


class TestAutoLimitRunnerParams:
    """Test that runner_params flows through execute_auto_limit."""

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
                runner_params={"my_param": "val"},
            )

        assert captured[0]._params == {"my_param": "val"}

    def test_runner_params_default_empty(self, tmp_model: Path) -> None:
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
            )

        assert captured[0]._params == {}


# ==========================================================================
# run_all CLI command
# ==========================================================================


class TestRunAll:
    """Test the ``ppb all`` command end-to-end via CliRunner."""

    def test_all_with_autolimit_and_sweep(
        self, suite_toml: Path, tmp_path: Path
    ) -> None:
        from typer.testing import CliRunner

        from ppb import app

        runner = CliRunner()
        results_file = tmp_path / "all_results.jsonl"
        result = runner.invoke(
            app, ["all", str(suite_toml), "--results", str(results_file)]
        )
        assert result.exit_code == 0, result.output

        # The sweep should have written results
        assert results_file.exists()
        lines = results_file.read_text().strip().splitlines()
        # 1 model × 2 ctx × 1 batch = 2, but some may be filtered by max_ctx_cap
        assert len(lines) >= 1

    def test_all_without_autolimit(
        self, suite_toml_no_autolimit: Path, tmp_path: Path
    ) -> None:
        from typer.testing import CliRunner

        from ppb import app

        runner = CliRunner()
        results_file = tmp_path / "no_al_results.jsonl"
        result = runner.invoke(
            app,
            ["all", str(suite_toml_no_autolimit), "--results", str(results_file)],
        )
        assert result.exit_code == 0, result.output
        assert results_file.exists()
        lines = results_file.read_text().strip().splitlines()
        assert len(lines) == 2  # 2 ctx × 1 batch

    def test_all_auto_results_name(self, suite_toml_no_autolimit: Path) -> None:
        """When no --results flag, the output file should be auto-named."""
        from typer.testing import CliRunner

        from ppb import app

        runner = CliRunner()
        result = runner.invoke(app, ["all", str(suite_toml_no_autolimit)])
        assert result.exit_code == 0, result.output
        # Output should mention the auto-generated filename
        assert "suite_no_al_" in result.output


# ==========================================================================
# Shared root-level params integration
# ==========================================================================


class TestSharedRootParams:
    """Verify that root-level shared params are inherited by sections."""

    def _root_params_toml(
        self, tmp_path: Path, model_path: str, **section_extras
    ) -> Path:
        """Write a TOML with model_path + runner_type at root level."""
        lines = [
            f'model_path = "{model_path}"',
            'runner_type = "fake"',
            "",
            "[sweep]",
            "n_ctx = [512]",
            "n_batch = [256]",
        ]
        for k, v in section_extras.items():
            lines.append(f"{k} = {v}")
        cfg = tmp_path / "root_params.toml"
        cfg.write_text("\n".join(lines) + "\n")
        return cfg

    def test_sweep_inherits_root_model(
        self, tmp_path: Path, tmp_model: Path
    ) -> None:
        from ppb import execute_sweep

        cfg = self._root_params_toml(tmp_path, str(tmp_model))
        results = tmp_path / "results.jsonl"
        execute_sweep(config_path=cfg, results_file=results)

        lines = results.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["runner_type"] == "fake"
        assert record["model_path"] == str(tmp_model)

    def test_section_overrides_root_model(
        self, tmp_path: Path, tmp_model: Path, tmp_model_dir: Path
    ) -> None:
        """If [sweep] has its own model_path, it takes priority."""
        # root = tmp_model, but section overrides to tmp_model_dir
        lines = [
            f'model_path = "{tmp_model}"',
            'runner_type = "fake"',
            "",
            "[sweep]",
            f'model_path = "{tmp_model_dir}"',
            "n_ctx = [512]",
            "n_batch = [256]",
        ]
        cfg = tmp_path / "override.toml"
        cfg.write_text("\n".join(lines) + "\n")

        results = tmp_path / "results.jsonl"
        from ppb import execute_sweep

        execute_sweep(config_path=cfg, results_file=results)

        lines_out = results.read_text().strip().splitlines()
        # tmp_model_dir has 2 models (alpha.gguf, beta.gguf)
        assert len(lines_out) == 2

    def test_backward_compat_section_only(
        self, tmp_path: Path, tmp_model: Path
    ) -> None:
        """Old-style TOML with everything in [sweep] still works."""
        from ppb import execute_sweep

        cfg = _sweep_toml(tmp_path, str(tmp_model))
        results = tmp_path / "results.jsonl"
        execute_sweep(config_path=cfg, results_file=results)

        lines = results.read_text().strip().splitlines()
        assert len(lines) == 1
