"""Tests for SweepConfig, BenchCombo, and the _write_result helper."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest


# ==========================================================================
# SweepConfig
# ==========================================================================


class TestSweepConfig:
    """Validate the Pydantic model that drives sweep configuration."""

    def _make_config(self, **overrides):
        """Build a SweepConfig from defaults + overrides.

        Lazily imported to keep test collection fast when ppb has
        heavy transitive imports (pynvml, etc.).
        """
        from ppb import SweepConfig

        defaults = {
            "model_path": str(overrides.pop("_model_file", "/dev/null")),
            "n_ctx": [512],
            "n_batch": [256],
        }
        defaults.update(overrides)
        return SweepConfig(**defaults)

    # -- runner_type defaults -------------------------------------------------

    def test_default_runner_type(self, tmp_model: Path) -> None:
        cfg = self._make_config(_model_file=tmp_model)
        assert cfg.runner_type == "llama-bench"

    def test_custom_runner_type(self, tmp_model: Path) -> None:
        cfg = self._make_config(_model_file=tmp_model, runner_type="llama-server")
        assert cfg.runner_type == "llama-server"

    # -- runner_params --------------------------------------------------------

    def test_default_runner_params_empty(self, tmp_model: Path) -> None:
        cfg = self._make_config(_model_file=tmp_model)
        assert cfg.runner_params == {}

    def test_custom_runner_params(self, tmp_model: Path) -> None:
        cfg = self._make_config(
            _model_file=tmp_model, runner_params={"key": "val"}
        )
        assert cfg.runner_params == {"key": "val"}

    # -- model_path resolution ------------------------------------------------

    def test_single_file(self, tmp_model: Path) -> None:
        cfg = self._make_config(_model_file=tmp_model)
        assert cfg.model_paths == [tmp_model]

    def test_directory(self, tmp_model_dir: Path) -> None:
        cfg = self._make_config(model_path=str(tmp_model_dir))
        names = sorted(p.name for p in cfg.model_paths)
        assert names == ["alpha.gguf", "beta.gguf"]

    def test_glob_pattern(self, tmp_model_dir: Path) -> None:
        pattern = str(tmp_model_dir / "alpha*")
        cfg = self._make_config(model_path=pattern)
        assert len(cfg.model_paths) == 1
        assert cfg.model_paths[0].name == "alpha.gguf"

    def test_no_match_raises(self, tmp_path: Path) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            self._make_config(model_path=str(tmp_path / "nonexistent*.gguf"))

    # -- combos ---------------------------------------------------------------

    def test_combos_cartesian_product(self, tmp_model_dir: Path) -> None:
        cfg = self._make_config(
            model_path=str(tmp_model_dir),
            n_ctx=[512, 1024],
            n_batch=[256, 512],
        )
        combos = cfg.combos()
        # 2 models × 2 ctx × 2 batch = 8
        assert len(combos) == 8

    def test_combos_with_concurrent_users(self, tmp_model: Path) -> None:
        cfg = self._make_config(
            _model_file=tmp_model,
            n_ctx=[512],
            n_batch=[256],
            concurrent_users=[1, 2, 4],
        )
        combos = cfg.combos()
        # 1 model × 1 ctx × 1 batch × 3 users = 3
        assert len(combos) == 3
        user_counts = sorted(c.concurrent_users for c in combos)
        assert user_counts == [1, 2, 4]

    def test_combos_concurrent_users_default(self, tmp_model: Path) -> None:
        """Default concurrent_users=[1] doesn't inflate combo count."""
        cfg = self._make_config(
            _model_file=tmp_model, n_ctx=[512, 1024], n_batch=[256],
        )
        combos = cfg.combos()
        assert len(combos) == 2
        assert all(c.concurrent_users == 1 for c in combos)

    def test_combos_fields(self, tmp_model: Path) -> None:
        cfg = self._make_config(
            _model_file=tmp_model, n_ctx=[1024], n_batch=[256]
        )
        combos = cfg.combos()
        assert len(combos) == 1
        c = combos[0]
        assert c.model_path == tmp_model
        assert c.n_ctx == 1024
        assert c.n_batch == 256


# ==========================================================================
# BenchCombo
# ==========================================================================


class TestBenchCombo:
    def test_fields(self) -> None:
        from ppb import BenchCombo

        c = BenchCombo(model_path=Path("/m.gguf"), n_ctx=4096, n_batch=512)
        assert c.model_path == Path("/m.gguf")
        assert c.n_ctx == 4096
        assert c.n_batch == 512

    def test_concurrent_users_default(self) -> None:
        from ppb import BenchCombo

        c = BenchCombo(model_path=Path("/m.gguf"), n_ctx=4096, n_batch=512)
        assert c.concurrent_users == 1

    def test_concurrent_users_set(self) -> None:
        from ppb import BenchCombo

        c = BenchCombo(
            model_path=Path("/m.gguf"), n_ctx=4096, n_batch=512,
            concurrent_users=8,
        )
        assert c.concurrent_users == 8


# ==========================================================================
# _write_result
# ==========================================================================


class TestWriteResult:
    """Verify the JSONL writer creates valid output."""

    def test_appends_jsonl(self, tmp_path: Path) -> None:
        from ppb import _write_result

        outfile = tmp_path / "out.jsonl"
        record1 = {"a": 1}
        record2 = {"b": 2}

        _write_result(record1, outfile)
        _write_result(record2, outfile)

        lines = outfile.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == record1
        assert json.loads(lines[1]) == record2

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        from ppb import _write_result

        outfile = tmp_path / "deep" / "nested" / "results.jsonl"
        _write_result({"x": 1}, outfile)
        assert outfile.exists()

    def test_output_is_valid_json(self, tmp_path: Path) -> None:
        from ppb import _write_result

        outfile = tmp_path / "valid.jsonl"
        record = {
            "timestamp": "2026-03-07T00:00:00+00:00",
            "runner_type": "llama-bench",
            "model_path": "/m.gguf",
            "n_ctx": 8192,
            "n_batch": 512,
            "hardware": {"os": {"system": "Linux"}},
            "results": [{"avg_ts": 42.0}],
        }
        _write_result(record, outfile)

        parsed = json.loads(outfile.read_text().strip())
        assert parsed["runner_type"] == "llama-bench"
        assert parsed["results"][0]["avg_ts"] == 42.0


# ==========================================================================
# AutoLimitConfig
# ==========================================================================


class TestAutoLimitConfig:
    """Validate the Pydantic model for [auto-limit] config."""

    def test_defaults(self, tmp_model: Path) -> None:
        from ppb import AutoLimitConfig

        cfg = AutoLimitConfig(model_path=str(tmp_model))
        assert cfg.min_ctx == 2048
        assert cfg.max_ctx == 131072
        assert cfg.tolerance == 1024
        assert cfg.runner_type == "llama-bench"
        assert cfg.runner_params == {}

    def test_model_paths_single_file(self, tmp_model: Path) -> None:
        from ppb import AutoLimitConfig

        cfg = AutoLimitConfig(model_path=str(tmp_model))
        assert cfg.model_paths == [tmp_model.resolve()]

    def test_model_paths_directory(self, tmp_model_dir: Path) -> None:
        from ppb import AutoLimitConfig

        cfg = AutoLimitConfig(model_path=str(tmp_model_dir))
        names = sorted(p.name for p in cfg.model_paths)
        assert names == ["alpha.gguf", "beta.gguf"]

    def test_model_paths_glob(self, tmp_model_dir: Path) -> None:
        from ppb import AutoLimitConfig

        pattern = str(tmp_model_dir / "alpha*")
        cfg = AutoLimitConfig(model_path=pattern)
        assert len(cfg.model_paths) == 1
        assert cfg.model_paths[0].name == "alpha.gguf"

    def test_missing_model_file(self, tmp_path: Path) -> None:
        from pydantic import ValidationError
        from ppb import AutoLimitConfig

        with pytest.raises(ValidationError, match="No files match pattern"):
            AutoLimitConfig(model_path=str(tmp_path / "nonexistent.gguf"))

    def test_all_fields_override(self, tmp_model: Path) -> None:
        from ppb import AutoLimitConfig

        cfg = AutoLimitConfig(
            model_path=str(tmp_model),
            min_ctx=512,
            max_ctx=65536,
            tolerance=512,
            runner_type="custom",
            runner_params={"key": "val"},
        )
        assert cfg.min_ctx == 512
        assert cfg.max_ctx == 65536
        assert cfg.tolerance == 512
        assert cfg.runner_type == "custom"
        assert cfg.runner_params == {"key": "val"}


# ==========================================================================
# _resolve_results_file
# ==========================================================================


class TestResolveResultsFile:
    """Verify results file resolution precedence."""

    def test_cli_override_wins(self, tmp_path: Path) -> None:
        from ppb import _resolve_results_file

        result = _resolve_results_file(
            config_path=tmp_path / "suite.toml",
            cli_override=Path("explicit.jsonl"),
            toml_results="toml_out.jsonl",
        )
        assert result == Path("explicit.jsonl")

    def test_toml_results_field(self, tmp_path: Path) -> None:
        from ppb import _resolve_results_file

        result = _resolve_results_file(
            config_path=tmp_path / "suite.toml",
            cli_override=None,
            toml_results="toml_out.jsonl",
        )
        assert result == Path("toml_out.jsonl")

    def test_auto_generated_name(self, tmp_path: Path) -> None:
        from ppb import _resolve_results_file

        result = _resolve_results_file(
            config_path=tmp_path / "my_suite.toml",
            cli_override=None,
            toml_results=None,
        )
        name = result.name
        assert name.startswith("my_suite_")
        assert name.endswith(".jsonl")

    def test_auto_generated_no_seconds(self, tmp_path: Path) -> None:
        """Timestamp should be YYYYMMDD_HHMM — no seconds."""
        import re
        from ppb import _resolve_results_file

        result = _resolve_results_file(
            config_path=tmp_path / "x.toml",
            cli_override=None,
            toml_results=None,
        )
        # e.g. x_20260307_1430.jsonl — exactly 13 chars for the timestamp part
        assert re.match(r"^x_\d{8}_\d{4}\.jsonl$", result.name)

    def test_no_config_path_uses_results_stem(self) -> None:
        from ppb import _resolve_results_file

        result = _resolve_results_file(
            config_path=None,
            cli_override=None,
            toml_results=None,
        )
        assert result.name.startswith("results_")
