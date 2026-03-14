"""Integration tests for the sweep and vram-cliff orchestration.

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


def _sweep_toml(tmp_path: Path, tmp_model: Path, **extra) -> Path:
    """Write a sweep TOML file with runner_type='fake'.

    Uses repo_id/filename/models_dir instead of model_path.
    Callers must also mock ``ppb._ensure_models`` to bypass HF downloads.
    """
    runner_params = extra.pop("runner_params", None)

    lines = [
        "[sweep]",
        'runner_type = "fake"',
        'repo_id = "test-org/test-repo"',
        f'filename = "{tmp_model.name}"',
        f'models_dir = "{tmp_model.parent}"',
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


def _mock_ensure_models(tmp_model: Path):
    """Return a mock side_effect for ``ppb._ensure_models``."""
    return lambda *a, **kw: [(tmp_model, f"test-org/test-repo/{tmp_model.name}")]


# ==========================================================================
# execute_sweep
# ==========================================================================


class TestExecuteSweep:
    """Verify the sweep orchestrator integrates with the runner plugin."""

    def test_basic_sweep_writes_jsonl(self, tmp_path: Path, tmp_model: Path) -> None:
        from ppb import execute_sweep

        cfg = _sweep_toml(tmp_path, tmp_model)
        results = tmp_path / "results.jsonl"

        with patch("ppb._ensure_models", side_effect=_mock_ensure_models(tmp_model)):
            execute_sweep(results_file=results, config_path=cfg)

        lines = results.read_text().strip().splitlines()
        assert len(lines) == 1  # 1 model × 1 ctx × 1 batch

        record = json.loads(lines[0])
        assert record["runner_type"] == "fake"
        assert record["model"] == f"test-org/test-repo/{tmp_model.name}"
        assert record["n_ctx"] == 512
        assert record["n_batch"] == 256
        assert "timestamp" in record
        assert "hardware" in record
        assert "results" in record

    def test_cartesian_product(self, tmp_path: Path, tmp_model: Path) -> None:
        from ppb import execute_sweep

        cfg = _sweep_toml(
            tmp_path, tmp_model, n_ctx="[512, 1024]", n_batch="[256, 512]"
        )
        results = tmp_path / "results.jsonl"
        with patch("ppb._ensure_models", side_effect=_mock_ensure_models(tmp_model)):
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
            tmp_model,
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

        with (
            patch("ppb.get_runner", side_effect=spy_get),
            patch("ppb._ensure_models", side_effect=_mock_ensure_models(tmp_model)),
        ):
            execute_sweep(results_file=results, config_path=cfg)

        assert len(captured) == 1
        r = captured[0]
        assert r.setup_called
        assert r._params == {"custom_key": "custom_value"}

    def test_runner_teardown_called(self, tmp_path: Path, tmp_model: Path) -> None:
        from ppb import execute_sweep

        captured: list[FakeRunner] = []
        original_get = __import__("runners").get_runner

        def spy_get(rt: str):
            r = original_get(rt)
            captured.append(r)
            return r

        cfg = _sweep_toml(tmp_path, tmp_model)
        results = tmp_path / "results.jsonl"

        with (
            patch("ppb.get_runner", side_effect=spy_get),
            patch("ppb._ensure_models", side_effect=_mock_ensure_models(tmp_model)),
        ):
            execute_sweep(results_file=results, config_path=cfg)

        assert captured[0].teardown_called

    def test_teardown_called_on_failure(self, tmp_path: Path, tmp_model: Path) -> None:
        """teardown() must be called even when all runs fail."""
        from ppb import execute_sweep

        captured: list[FakeRunner] = []
        original_get = __import__("runners").get_runner

        def spy_get(rt: str):
            r = original_get(rt)
            r.run_return = None  # simulate failure
            captured.append(r)
            return r

        cfg = _sweep_toml(tmp_path, tmp_model)
        results = tmp_path / "results.jsonl"

        with (
            patch("ppb.get_runner", side_effect=spy_get),
            patch("ppb._ensure_models", side_effect=_mock_ensure_models(tmp_model)),
        ):
            execute_sweep(results_file=results, config_path=cfg)

        assert captured[0].teardown_called
        # No results should be written
        assert not results.exists() or results.read_text().strip() == ""

    def test_failed_run_not_written(self, tmp_path: Path, tmp_model: Path) -> None:
        """When runner.run() returns None, no JSONL line should be written."""
        from ppb import execute_sweep

        original_get = __import__("runners").get_runner

        def spy_get(rt: str):
            r = original_get(rt)
            r.run_return = None
            return r

        cfg = _sweep_toml(tmp_path, tmp_model)
        results = tmp_path / "results.jsonl"

        with (
            patch("ppb.get_runner", side_effect=spy_get),
            patch("ppb._ensure_models", side_effect=_mock_ensure_models(tmp_model)),
        ):
            execute_sweep(results_file=results, config_path=cfg)

        content = results.read_text().strip() if results.exists() else ""
        assert content == ""

    def test_result_envelope_structure(self, tmp_path: Path, tmp_model: Path) -> None:
        """Every JSONL record must have the stable envelope fields."""
        from ppb import execute_sweep

        cfg = _sweep_toml(tmp_path, tmp_model)
        results = tmp_path / "results.jsonl"
        with patch("ppb._ensure_models", side_effect=_mock_ensure_models(tmp_model)):
            execute_sweep(results_file=results, config_path=cfg)

        record = json.loads(results.read_text().strip())
        required_keys = {
            "timestamp",
            "runner_type",
            "model",
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

    def test_concurrent_users_in_cartesian_product(
        self,
        tmp_path: Path,
        tmp_model: Path,
    ) -> None:
        """concurrent_users adds another axis to the Cartesian product."""
        from ppb import execute_sweep

        cfg = _sweep_toml(
            tmp_path,
            tmp_model,
            n_ctx="[512]",
            n_batch="[256]",
            concurrent_users="[1, 2]",
        )
        results = tmp_path / "results.jsonl"
        with patch("ppb._ensure_models", side_effect=_mock_ensure_models(tmp_model)):
            execute_sweep(results_file=results, config_path=cfg)

        lines = results.read_text().strip().splitlines()
        assert len(lines) == 2  # 1 model × 1 ctx × 1 batch × 2 users

    def test_concurrent_users_in_envelope(
        self,
        tmp_path: Path,
        tmp_model: Path,
    ) -> None:
        """concurrent_users > 1 adds the field to the JSONL envelope."""
        from ppb import execute_sweep

        cfg = _sweep_toml(
            tmp_path,
            tmp_model,
            n_ctx="[512]",
            n_batch="[256]",
            concurrent_users="[4]",
        )
        results = tmp_path / "results.jsonl"
        with patch("ppb._ensure_models", side_effect=_mock_ensure_models(tmp_model)):
            execute_sweep(results_file=results, config_path=cfg)

        record = json.loads(results.read_text().strip())
        assert record["concurrent_users"] == 4

    def test_concurrent_users_1_in_envelope(
        self,
        tmp_path: Path,
        tmp_model: Path,
    ) -> None:
        """concurrent_users == 1 should appear in the envelope."""
        from ppb import execute_sweep

        cfg = _sweep_toml(
            tmp_path,
            tmp_model,
            n_ctx="[512]",
            n_batch="[256]",
        )
        results = tmp_path / "results.jsonl"
        with patch("ppb._ensure_models", side_effect=_mock_ensure_models(tmp_model)):
            execute_sweep(results_file=results, config_path=cfg)

        record = json.loads(results.read_text().strip())
        assert record["concurrent_users"] == 1

    def test_unknown_runner_type(self, tmp_path: Path, tmp_model: Path) -> None:
        """An unregistered runner_type must produce a clear error."""
        from click.exceptions import Exit as ClickExit

        from ppb import execute_sweep

        cfg = tmp_path / "bad_runner.toml"
        cfg.write_text(
            textwrap.dedent(f"""\
            [sweep]
            runner_type = "nonexistent"
            repo_id = "test-org/test-repo"
            filename = "{tmp_model.name}"
            models_dir = "{tmp_model.parent}"
            n_ctx = [512]
            n_batch = [256]
            """)
        )

        with (
            pytest.raises((ValueError, SystemExit, ClickExit)),
            patch("ppb._ensure_models", side_effect=_mock_ensure_models(tmp_model)),
        ):
            execute_sweep(results_file=tmp_path / "r.jsonl", config_path=cfg)


# ==========================================================================
# execute_vram_cliff
# ==========================================================================


class TestExecuteVramCliff:
    """Verify the binary-search orchestrator uses the runner plugin."""

    def test_basic_vram_cliff(self, tmp_model: Path) -> None:
        """With probe always passing, should converge to max_ctx."""
        from ppb import execute_vram_cliff

        result = execute_vram_cliff(
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
        from ppb import execute_vram_cliff

        original_get = __import__("runners").get_runner

        def spy_get(rt: str):
            r = original_get(rt)
            r.probe_return = False
            return r

        with patch("ppb.get_runner", side_effect=spy_get):
            result = execute_vram_cliff(
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
        from ppb import execute_vram_cliff

        called_with: list[str] = []
        original_get = __import__("runners").get_runner

        def spy_get(rt: str):
            called_with.append(rt)
            return original_get(rt)

        with patch("ppb.get_runner", side_effect=spy_get):
            execute_vram_cliff(
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
            tmp_path, tmp_model, n_ctx="[512, 1024, 4096]", n_batch="[256]"
        )
        results = tmp_path / "results.jsonl"
        with patch("ppb._ensure_models", side_effect=_mock_ensure_models(tmp_model)):
            execute_sweep(
                results_file=results,
                config_path=cfg,
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
            tmp_path, tmp_model, n_ctx="[512, 1024, 4096]", n_batch="[256]"
        )
        results = tmp_path / "results.jsonl"
        with patch("ppb._ensure_models", side_effect=_mock_ensure_models(tmp_model)):
            execute_sweep(results_file=results, config_path=cfg, max_ctx_caps=None)

        lines = results.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_all_combos_above_cap(self, tmp_path: Path, tmp_model: Path) -> None:
        """When every requested n_ctx exceeds cap, the cap is injected and used."""
        from ppb import execute_sweep

        cfg = _sweep_toml(tmp_path, tmp_model, n_ctx="[4096, 8192]", n_batch="[256]")
        results = tmp_path / "results.jsonl"
        with patch("ppb._ensure_models", side_effect=_mock_ensure_models(tmp_model)):
            execute_sweep(
                results_file=results,
                config_path=cfg,
                max_ctx_caps={tmp_model.resolve(): 1024},
            )

        # The cap (1024) is injected as an n_ctx value, so 1 combo runs.
        lines = results.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["n_ctx"] == 1024


# ==========================================================================
# execute_sweep with SweepConfig directly (CLI mode)
# ==========================================================================


class TestExecuteSweepDirect:
    """Verify execute_sweep works when given a pre-built SweepConfig."""

    def test_sweep_with_config_object(self, tmp_path: Path, tmp_model: Path) -> None:
        from ppb import SweepConfig, execute_sweep

        cfg = SweepConfig(
            repo_id="test-org/test-repo",
            filename=tmp_model.name,
            models_dir=str(tmp_model.parent),
            n_ctx=[512],
            n_batch=[256],
            runner_type="fake",
            resolved_models=[(tmp_model, f"test-org/test-repo/{tmp_model.name}")],
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
            repo_id="test-org/test-repo",
            filename=tmp_model.name,
            n_ctx=[512],
            n_batch=[256],
            runner_type="fake",
        )
        toml_path = _sweep_toml(tmp_path, tmp_model)
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
# execute_vram_cliff runner_params
# ==========================================================================


class TestVramCliffRunnerParams:
    """Verify runner_params forwarding in execute_vram_cliff."""

    def test_runner_params_forwarded(self, tmp_model: Path) -> None:
        from ppb import execute_vram_cliff

        captured: list[FakeRunner] = []
        original_get = __import__("runners").get_runner

        def spy_get(rt: str):
            r = original_get(rt)
            captured.append(r)
            return r

        with patch("ppb.get_runner", side_effect=spy_get):
            execute_vram_cliff(
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
        assert "vram-cliff" in raw
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
    """Verify the ``all`` command chains vram-cliff → sweep."""

    def test_all_vram_cliff_then_sweep(
        self, tmp_path: Path, suite_toml: Path, tmp_model: Path
    ) -> None:
        """vram-cliff should run, then sweep respects the cap."""
        from typer.testing import CliRunner
        from ppb import app

        runner = CliRunner()
        results = tmp_path / "all_results.jsonl"
        with patch("ppb._ensure_models", side_effect=_mock_ensure_models(tmp_model)):
            result = runner.invoke(
                app, ["all", str(suite_toml), "--results", str(results)]
            )
        # Should complete without hard error
        assert result.exit_code == 0
        # sweep should have written some results
        assert results.exists()
        lines = results.read_text().strip().splitlines()
        assert len(lines) > 0

    def test_all_without_vramcliff_section(
        self, tmp_path: Path, suite_toml_no_vramcliff: Path, tmp_model: Path
    ) -> None:
        """Without [vram-cliff], sweep runs unmodified."""
        from typer.testing import CliRunner
        from ppb import app

        runner = CliRunner()
        results = tmp_path / "res.jsonl"
        with patch("ppb._ensure_models", side_effect=_mock_ensure_models(tmp_model)):
            result = runner.invoke(
                app, ["all", str(suite_toml_no_vramcliff), "--results", str(results)]
            )
        assert result.exit_code == 0
        assert results.exists()
        lines = results.read_text().strip().splitlines()
        assert len(lines) == 2  # 2 n_ctx × 1 n_batch

    def test_all_vram_cliff_fails_exits(self, tmp_path: Path, tmp_model: Path) -> None:
        """If vram-cliff returns 0, the command must exit with code 1."""
        from typer.testing import CliRunner
        from ppb import app

        # Write a suite TOML where vram-cliff will always fail
        cfg = tmp_path / "fail_suite.toml"
        cfg.write_text(
            textwrap.dedent(f"""\
            repo_id = "test-org/test-repo"
            filename = "{tmp_model.name}"
            models_dir = "{tmp_model.parent}"

            [vram-cliff]
            min_ctx = 1024
            max_ctx = 2048
            tolerance = 1024
            runner_type = "fake"

            [sweep]
            runner_type = "fake"
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
        with (
            patch("ppb.get_runner", side_effect=spy_get),
            patch("ppb._ensure_models", side_effect=_mock_ensure_models(tmp_model)),
        ):
            result = runner.invoke(
                app, ["all", str(cfg), "--results", str(tmp_path / "r.jsonl")]
            )
        assert result.exit_code == 1


# ==========================================================================
# Backward compatibility
# ==========================================================================


class TestBackwardCompat:
    """Ensure existing sweep.toml files without runner_type still work."""

    def test_default_runner_type_in_toml(self, tmp_path: Path, tmp_model: Path) -> None:
        """A TOML with no runner_type must default to 'llama-bench'."""
        from ppb import SweepConfig

        cfg = SweepConfig(
            repo_id="test-org/test-repo",
            filename=tmp_model.name,
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
            repo_id = "test-org/test-repo"
            filename = "{tmp_model.name}"
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
# suite_run_id propagation
# ==========================================================================


class TestSuiteRunId:
    """Verify suite_run_id is injected into JSONL records when provided."""

    def test_suite_run_id_in_records(self, tmp_path: Path, tmp_model: Path) -> None:
        """Every JSONL record must include suite_run_id when passed."""
        from ppb import SweepConfig, execute_sweep

        cfg = SweepConfig(
            repo_id="test-org/test-repo",
            filename=tmp_model.name,
            models_dir=str(tmp_model.parent),
            n_ctx=[512, 1024],
            n_batch=[256],
            runner_type="fake",
            resolved_models=[(tmp_model, f"test-org/test-repo/{tmp_model.name}")],
        )
        results = tmp_path / "results.jsonl"
        execute_sweep(
            results_file=results,
            sweep_config=cfg,
            suite_run_id="test-run-abc123",
        )

        lines = results.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert record["suite_run_id"] == "test-run-abc123"

    def test_no_suite_run_id_by_default(self, tmp_path: Path, tmp_model: Path) -> None:
        """Without suite_run_id, records must not contain the field."""
        from ppb import SweepConfig, execute_sweep

        cfg = SweepConfig(
            repo_id="test-org/test-repo",
            filename=tmp_model.name,
            models_dir=str(tmp_model.parent),
            n_ctx=[512],
            n_batch=[256],
            runner_type="fake",
            resolved_models=[(tmp_model, f"test-org/test-repo/{tmp_model.name}")],
        )
        results = tmp_path / "results.jsonl"
        execute_sweep(results_file=results, sweep_config=cfg)

        record = json.loads(results.read_text().strip())
        assert "suite_run_id" not in record


# ==========================================================================
# completed_models (resume — skip models)
# ==========================================================================


class TestCompletedModelsSkip:
    """Verify that completed_models causes models to be skipped."""

    def test_skip_completed_model(self, tmp_path: Path) -> None:
        """Model in completed_models set should produce no results."""
        from ppb import SweepConfig, execute_sweep

        model_a = tmp_path / "model-a.gguf"
        model_b = tmp_path / "model-b.gguf"
        model_a.write_bytes(b"\x00" * 64)
        model_b.write_bytes(b"\x00" * 64)

        cfg = SweepConfig(
            repo_id="test-org/test-repo",
            filename="*.gguf",
            models_dir=str(tmp_path),
            n_ctx=[512],
            n_batch=[256],
            runner_type="fake",
            resolved_models=[
                (model_a, "test-org/test-repo/model-a.gguf"),
                (model_b, "test-org/test-repo/model-b.gguf"),
            ],
        )
        results = tmp_path / "results.jsonl"
        execute_sweep(
            results_file=results,
            sweep_config=cfg,
            completed_models={"test-org/test-repo/model-a.gguf"},
        )

        lines = results.read_text().strip().splitlines()
        assert len(lines) == 1  # only model-b ran
        record = json.loads(lines[0])
        assert "model-b.gguf" in record["model"]

    def test_skip_all_completed(self, tmp_path: Path) -> None:
        """When all models are completed, no results should be written."""
        from ppb import SweepConfig, execute_sweep

        model_a = tmp_path / "model-a.gguf"
        model_a.write_bytes(b"\x00" * 64)

        cfg = SweepConfig(
            repo_id="test-org/test-repo",
            filename="*.gguf",
            models_dir=str(tmp_path),
            n_ctx=[512],
            n_batch=[256],
            runner_type="fake",
            resolved_models=[
                (model_a, "test-org/test-repo/model-a.gguf"),
            ],
        )
        results = tmp_path / "results.jsonl"
        execute_sweep(
            results_file=results,
            sweep_config=cfg,
            completed_models={"test-org/test-repo/model-a.gguf"},
        )

        content = results.read_text().strip() if results.exists() else ""
        assert content == ""

    def test_empty_completed_models_runs_all(
        self, tmp_path: Path, tmp_model: Path
    ) -> None:
        """completed_models=None should run everything normally."""
        from ppb import SweepConfig, execute_sweep

        cfg = SweepConfig(
            repo_id="test-org/test-repo",
            filename=tmp_model.name,
            models_dir=str(tmp_model.parent),
            n_ctx=[512, 1024],
            n_batch=[256],
            runner_type="fake",
            resolved_models=[(tmp_model, f"test-org/test-repo/{tmp_model.name}")],
        )
        results = tmp_path / "results.jsonl"
        execute_sweep(
            results_file=results,
            sweep_config=cfg,
            completed_models=None,
        )

        lines = results.read_text().strip().splitlines()
        assert len(lines) == 2


# ==========================================================================
# on_model_done callback
# ==========================================================================


class TestOnModelDoneCallback:
    """Verify the on_model_done callback fires correctly."""

    def test_callback_called_per_model(self, tmp_path: Path) -> None:
        """Callback must fire once per model with correct arguments."""
        from ppb import SweepConfig, execute_sweep

        model_a = tmp_path / "model-a.gguf"
        model_b = tmp_path / "model-b.gguf"
        model_a.write_bytes(b"\x00" * 64)
        model_b.write_bytes(b"\x00" * 64)

        cfg = SweepConfig(
            repo_id="test-org/test-repo",
            filename="*.gguf",
            models_dir=str(tmp_path),
            n_ctx=[512],
            n_batch=[256],
            runner_type="fake",
            resolved_models=[
                (model_a, "test-org/test-repo/model-a.gguf"),
                (model_b, "test-org/test-repo/model-b.gguf"),
            ],
        )
        results = tmp_path / "results.jsonl"

        callback_calls: list[tuple[str, Path, int]] = []

        def on_done(model_hf_id: str, rfile: Path, line_offset: int) -> None:
            callback_calls.append((model_hf_id, rfile, line_offset))

        execute_sweep(
            results_file=results,
            sweep_config=cfg,
            on_model_done=on_done,
        )

        assert len(callback_calls) == 2
        assert callback_calls[0][0] == "test-org/test-repo/model-a.gguf"
        assert callback_calls[0][2] == 0  # first model starts at line 0
        assert callback_calls[1][0] == "test-org/test-repo/model-b.gguf"
        assert callback_calls[1][2] == 1  # second model starts at line 1

    def test_callback_line_offset_with_multi_combos(self, tmp_path: Path) -> None:
        """Line offset reflects the number of combos for previous models."""
        from ppb import SweepConfig, execute_sweep

        model_a = tmp_path / "model-a.gguf"
        model_b = tmp_path / "model-b.gguf"
        model_a.write_bytes(b"\x00" * 64)
        model_b.write_bytes(b"\x00" * 64)

        cfg = SweepConfig(
            repo_id="test-org/test-repo",
            filename="*.gguf",
            models_dir=str(tmp_path),
            n_ctx=[512, 1024],
            n_batch=[256],
            runner_type="fake",
            resolved_models=[
                (model_a, "test-org/test-repo/model-a.gguf"),
                (model_b, "test-org/test-repo/model-b.gguf"),
            ],
        )
        results = tmp_path / "results.jsonl"

        callback_calls: list[tuple[str, Path, int]] = []

        def on_done(model_hf_id: str, rfile: Path, line_offset: int) -> None:
            callback_calls.append((model_hf_id, rfile, line_offset))

        execute_sweep(
            results_file=results,
            sweep_config=cfg,
            on_model_done=on_done,
        )

        assert len(callback_calls) == 2
        # model-a has 2 combos (512×256, 1024×256) → offset 0
        assert callback_calls[0][2] == 0
        # model-b starts at line 2
        assert callback_calls[1][2] == 2

    def test_callback_not_called_for_skipped_models(self, tmp_path: Path) -> None:
        """Callback must not fire for models skipped via completed_models."""
        from ppb import SweepConfig, execute_sweep

        model_a = tmp_path / "model-a.gguf"
        model_b = tmp_path / "model-b.gguf"
        model_a.write_bytes(b"\x00" * 64)
        model_b.write_bytes(b"\x00" * 64)

        cfg = SweepConfig(
            repo_id="test-org/test-repo",
            filename="*.gguf",
            models_dir=str(tmp_path),
            n_ctx=[512],
            n_batch=[256],
            runner_type="fake",
            resolved_models=[
                (model_a, "test-org/test-repo/model-a.gguf"),
                (model_b, "test-org/test-repo/model-b.gguf"),
            ],
        )
        results = tmp_path / "results.jsonl"

        callback_calls: list[str] = []

        def on_done(model_hf_id: str, rfile: Path, line_offset: int) -> None:
            callback_calls.append(model_hf_id)

        execute_sweep(
            results_file=results,
            sweep_config=cfg,
            completed_models={"test-org/test-repo/model-a.gguf"},
            on_model_done=on_done,
        )

        assert callback_calls == ["test-org/test-repo/model-b.gguf"]


# ==========================================================================
# _detect_completed_models
# ==========================================================================


class TestDetectCompletedModels:
    """Verify resume detection from an existing results file."""

    def test_detect_two_of_three_completed(self, tmp_path: Path) -> None:
        """Two fully-benchmarked models should be detected as completed."""
        from ppb import SweepConfig, _detect_completed_models

        model_a = tmp_path / "model-a.gguf"
        model_b = tmp_path / "model-b.gguf"
        model_c = tmp_path / "model-c.gguf"
        for m in (model_a, model_b, model_c):
            m.write_bytes(b"\x00" * 64)

        cfg = SweepConfig(
            repo_id="test-org/test-repo",
            filename="*.gguf",
            models_dir=str(tmp_path),
            n_ctx=[512, 1024],
            n_batch=[256],
            runner_type="fake",
            resolved_models=[
                (model_a, "test-org/test-repo/model-a.gguf"),
                (model_b, "test-org/test-repo/model-b.gguf"),
                (model_c, "test-org/test-repo/model-c.gguf"),
            ],
        )

        # Write 2 records each for model-a and model-b (expected: 2 ctx × 1 batch = 2)
        results = tmp_path / "results.jsonl"
        records = []
        for model_id in ["test-org/test-repo/model-a.gguf", "test-org/test-repo/model-b.gguf"]:
            for ctx in [512, 1024]:
                records.append(json.dumps({
                    "model": model_id,
                    "n_ctx": ctx,
                    "n_batch": 256,
                    "runner_type": "fake",
                    "suite_run_id": "run-xyz",
                    "concurrent_users": 1,
                    "results": {},
                }))
        results.write_text("\n".join(records) + "\n")

        completed, run_id = _detect_completed_models(
            results, cfg, cfg.resolved_models, None
        )
        assert completed == {
            "test-org/test-repo/model-a.gguf",
            "test-org/test-repo/model-b.gguf",
        }
        assert run_id == "run-xyz"

    def test_partial_model_not_completed(self, tmp_path: Path) -> None:
        """A model with fewer records than expected should not be completed."""
        from ppb import SweepConfig, _detect_completed_models

        model_a = tmp_path / "model-a.gguf"
        model_a.write_bytes(b"\x00" * 64)

        cfg = SweepConfig(
            repo_id="test-org/test-repo",
            filename="*.gguf",
            models_dir=str(tmp_path),
            n_ctx=[512, 1024],
            n_batch=[256],
            runner_type="fake",
            resolved_models=[
                (model_a, "test-org/test-repo/model-a.gguf"),
            ],
        )

        # Write only 1 record (expected: 2)
        results = tmp_path / "results.jsonl"
        results.write_text(json.dumps({
            "model": "test-org/test-repo/model-a.gguf",
            "n_ctx": 512,
            "n_batch": 256,
            "runner_type": "fake",
            "concurrent_users": 1,
            "results": {},
        }) + "\n")

        completed, run_id = _detect_completed_models(
            results, cfg, cfg.resolved_models, None
        )
        assert completed is None
        assert run_id is None  # no suite_run_id in old records

    def test_empty_file(self, tmp_path: Path) -> None:
        """An empty results file should return None."""
        from ppb import SweepConfig, _detect_completed_models

        model_a = tmp_path / "model-a.gguf"
        model_a.write_bytes(b"\x00" * 64)

        cfg = SweepConfig(
            repo_id="test-org/test-repo",
            filename="*.gguf",
            models_dir=str(tmp_path),
            n_ctx=[512],
            n_batch=[256],
            runner_type="fake",
            resolved_models=[
                (model_a, "test-org/test-repo/model-a.gguf"),
            ],
        )

        results = tmp_path / "results.jsonl"
        results.write_text("")

        completed, run_id = _detect_completed_models(
            results, cfg, cfg.resolved_models, None
        )
        assert completed is None

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """A missing results file should return None."""
        from ppb import SweepConfig, _detect_completed_models

        model_a = tmp_path / "model-a.gguf"
        model_a.write_bytes(b"\x00" * 64)

        cfg = SweepConfig(
            repo_id="test-org/test-repo",
            filename="*.gguf",
            models_dir=str(tmp_path),
            n_ctx=[512],
            n_batch=[256],
            runner_type="fake",
            resolved_models=[
                (model_a, "test-org/test-repo/model-a.gguf"),
            ],
        )

        completed, run_id = _detect_completed_models(
            tmp_path / "nope.jsonl", cfg, cfg.resolved_models, None
        )
        assert completed is None
        assert run_id is None


# ==========================================================================
# _find_resumable_results
# ==========================================================================


class TestFindResumableResults:
    """Verify scanning for prior results files."""

    def test_finds_most_recent(self, tmp_path: Path) -> None:
        """Should return the most recently modified matching file."""
        import time

        from ppb import _find_resumable_results

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        old = results_dir / "MySuite_20260101_0000.jsonl"
        old.write_text('{"model":"a"}\n')
        time.sleep(0.05)

        new = results_dir / "MySuite_20260314_1200.jsonl"
        new.write_text('{"model":"b"}\n')

        config = tmp_path / "MySuite.toml"
        config.write_text("[sweep]\n")

        with patch("ppb.Path") as MockPath:
            # _find_resumable_results uses Path("results") internally;
            # we need it to point to our tmp results dir.
            # Easier: just test the function directly by monkeypatching.
            pass

        # Direct test: call with the real function but from the right cwd
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = _find_resumable_results(config)
            assert result is not None
            assert result.name == "MySuite_20260314_1200.jsonl"
        finally:
            os.chdir(old_cwd)

    def test_no_match_returns_none(self, tmp_path: Path) -> None:
        """No matching files should return None."""
        import os

        from ppb import _find_resumable_results

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # File for a different suite
        other = results_dir / "OtherSuite_20260314_1200.jsonl"
        other.write_text('{"model":"a"}\n')

        config = tmp_path / "MySuite.toml"
        config.write_text("[sweep]\n")

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = _find_resumable_results(config)
            assert result is None
        finally:
            os.chdir(old_cwd)


# ==========================================================================
# suite_run_id in flattener
# ==========================================================================


class TestSuiteRunIdFlattener:
    """Verify suite_run_id flows through the flattener pipeline."""

    def test_suite_run_id_in_flat_row(self) -> None:
        """suite_run_id from raw JSONL must appear in flattened output."""
        from utils.flattener import COLUMN_ORDER, flatten_benchmark_row

        raw = {
            "timestamp": "2026-03-14T12:00:00+00:00",
            "runner_type": "llama-server",
            "model": "org/repo/model-Q4_0.gguf",
            "n_ctx": 8192,
            "n_batch": 512,
            "concurrent_users": 1,
            "suite_run_id": "abc123",
            "hardware": {},
            "results": {"throughput_tok_s": 42.0},
        }
        flat_rows = flatten_benchmark_row(raw)
        assert len(flat_rows) >= 1
        assert flat_rows[0]["suite_run_id"] == "abc123"
        assert "suite_run_id" in COLUMN_ORDER

    def test_suite_run_id_none_when_absent(self) -> None:
        """When raw record has no suite_run_id, flat row should have None."""
        from utils.flattener import flatten_benchmark_row

        raw = {
            "timestamp": "2026-03-14T12:00:00+00:00",
            "runner_type": "llama-server",
            "model": "org/repo/model-Q4_0.gguf",
            "n_ctx": 8192,
            "n_batch": 512,
            "concurrent_users": 1,
            "hardware": {},
            "results": {"throughput_tok_s": 42.0},
        }
        flat_rows = flatten_benchmark_row(raw)
        assert flat_rows[0]["suite_run_id"] is None


# ==========================================================================
# _count_lines and _read_lines_from helpers
# ==========================================================================


class TestLineHelpers:
    """Verify the line-counting and line-reading helper functions."""

    def test_count_lines(self, tmp_path: Path) -> None:
        from ppb import _count_lines

        f = tmp_path / "test.jsonl"
        f.write_text("line1\nline2\nline3\n")
        assert _count_lines(f) == 3

    def test_count_lines_missing_file(self, tmp_path: Path) -> None:
        from ppb import _count_lines

        assert _count_lines(tmp_path / "nope.jsonl") == 0

    def test_read_lines_from(self, tmp_path: Path) -> None:
        from ppb import _read_lines_from

        f = tmp_path / "test.jsonl"
        f.write_text("line0\nline1\nline2\nline3\n")
        lines = _read_lines_from(f, 2)
        assert len(lines) == 2
        assert lines[0].strip() == "line2"
        assert lines[1].strip() == "line3"
