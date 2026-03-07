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
    """Validate the Pydantic model for [auto-limit] configuration."""

    def test_defaults(self, tmp_model: Path) -> None:
        from ppb import AutoLimitConfig

        cfg = AutoLimitConfig(model_path=str(tmp_model))
        assert cfg.min_ctx == 2048
        assert cfg.max_ctx == 131072
        assert cfg.tolerance == 1024
        assert cfg.runner_type == "llama-bench"
        assert cfg.runner_params == {}
        assert cfg.model_paths == [tmp_model.resolve()]

    def test_custom_values(self, tmp_model: Path) -> None:
        from ppb import AutoLimitConfig

        cfg = AutoLimitConfig(
            model_path=str(tmp_model),
            min_ctx=512,
            max_ctx=8192,
            tolerance=256,
            runner_type="fake",
            runner_params={"key": "val"},
        )
        assert cfg.min_ctx == 512
        assert cfg.max_ctx == 8192
        assert cfg.tolerance == 256
        assert cfg.runner_type == "fake"
        assert cfg.runner_params == {"key": "val"}

    def test_nonexistent_model_raises(self, tmp_path: Path) -> None:
        from pydantic import ValidationError

        from ppb import AutoLimitConfig

        with pytest.raises(ValidationError, match="No files match pattern"):
            AutoLimitConfig(model_path=str(tmp_path / "no_such.gguf"))

    def test_directory_model(self, tmp_model_dir: Path) -> None:
        from ppb import AutoLimitConfig

        cfg = AutoLimitConfig(model_path=str(tmp_model_dir))
        names = sorted(p.name for p in cfg.model_paths)
        assert names == ["alpha.gguf", "beta.gguf"]

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        from pydantic import ValidationError

        from ppb import AutoLimitConfig

        d = tmp_path / "emptydir"
        d.mkdir()
        with pytest.raises(ValidationError, match="No .gguf files found"):
            AutoLimitConfig(model_path=str(d))

    def test_glob_model(self, tmp_model_dir: Path) -> None:
        from ppb import AutoLimitConfig

        pattern = str(tmp_model_dir / "alpha*")
        cfg = AutoLimitConfig(model_path=pattern)
        assert len(cfg.model_paths) == 1
        assert cfg.model_paths[0].name == "alpha.gguf"


# ==========================================================================
# _resolve_results_file
# ==========================================================================


class TestResolveResultsFile:
    """Test the results-file resolution helper."""

    def test_cli_override_wins(self, tmp_path: Path) -> None:
        from ppb import _resolve_results_file

        result = _resolve_results_file(
            config_path=tmp_path / "suite.toml",
            cli_override=Path("override.jsonl"),
            toml_results="from_toml.jsonl",
        )
        assert result == Path("override.jsonl")

    def test_toml_results_used(self, tmp_path: Path) -> None:
        from ppb import _resolve_results_file

        result = _resolve_results_file(
            config_path=tmp_path / "suite.toml",
            cli_override=None,
            toml_results="from_toml.jsonl",
        )
        assert result == Path("from_toml.jsonl")

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
        # Format: my_suite_YYYYMMDD_HHMM.jsonl
        assert len(name) == len("my_suite_") + len("YYYYMMDD_HHMM") + len(".jsonl")

    def test_auto_generated_no_config(self) -> None:
        from ppb import _resolve_results_file

        result = _resolve_results_file(
            config_path=None,
            cli_override=None,
            toml_results=None,
        )
        name = result.name
        assert name.startswith("results_")
        assert name.endswith(".jsonl")


# ==========================================================================
# load_suite_config
# ==========================================================================


class TestLoadSuiteConfig:
    """Test the suite TOML loader."""

    def test_load_with_results(self, suite_toml_with_results: Path) -> None:
        from ppb import load_suite_config

        raw, results_path = load_suite_config(suite_toml_with_results)
        assert "sweep" in raw
        assert results_path == Path("my_output.jsonl")

    def test_load_auto_generated(self, suite_toml_no_autolimit: Path) -> None:
        from ppb import load_suite_config

        raw, results_path = load_suite_config(suite_toml_no_autolimit)
        assert "sweep" in raw
        # Should be auto-generated from stem
        assert results_path.name.startswith("suite_no_al_")

    def test_missing_file_exits(self, tmp_path: Path) -> None:
        from click.exceptions import Exit as ClickExit

        from ppb import load_suite_config

        with pytest.raises((SystemExit, ClickExit)):
            load_suite_config(tmp_path / "nonexistent.toml")


# ==========================================================================
# _merge_shared_params
# ==========================================================================


class TestMergeSharedParams:
    """Test root-level → section merging for shared TOML keys."""

    def test_root_model_path_inherited(self) -> None:
        from ppb import _merge_shared_params

        raw = {
            "model_path": "/root/model.gguf",
            "results": "out.jsonl",
            "sweep": {"n_ctx": [512], "n_batch": [256]},
        }
        merged = _merge_shared_params(raw, "sweep")
        assert merged["model_path"] == "/root/model.gguf"
        assert merged["n_ctx"] == [512]
        assert merged["n_batch"] == [256]
        # 'results' is NOT a shared key — must not leak in
        assert "results" not in merged

    def test_section_overrides_root(self) -> None:
        from ppb import _merge_shared_params

        raw = {
            "model_path": "/root/model.gguf",
            "runner_type": "llama-bench",
            "sweep": {
                "model_path": "/section/other.gguf",
                "n_ctx": [512],
                "n_batch": [256],
            },
        }
        merged = _merge_shared_params(raw, "sweep")
        assert merged["model_path"] == "/section/other.gguf"  # section wins
        assert merged["runner_type"] == "llama-bench"  # inherited

    def test_root_runner_type_inherited(self) -> None:
        from ppb import _merge_shared_params

        raw = {
            "runner_type": "fake",
            "auto-limit": {"model_path": "/m.gguf"},
        }
        merged = _merge_shared_params(raw, "auto-limit")
        assert merged["runner_type"] == "fake"
        assert merged["model_path"] == "/m.gguf"

    def test_empty_section_gets_root_only(self) -> None:
        from ppb import _merge_shared_params

        raw = {
            "model_path": "/m.gguf",
            "runner_type": "fake",
            "auto-limit": {},
        }
        merged = _merge_shared_params(raw, "auto-limit")
        assert merged["model_path"] == "/m.gguf"
        assert merged["runner_type"] == "fake"

    def test_missing_section_returns_root_shared_only(self) -> None:
        from ppb import _merge_shared_params

        raw = {"model_path": "/m.gguf", "results": "out.jsonl"}
        merged = _merge_shared_params(raw, "nonexistent")
        assert merged == {"model_path": "/m.gguf"}

    def test_backward_compat_section_level_only(self) -> None:
        """Old-style TOML with params only in sections still works."""
        from ppb import _merge_shared_params

        raw = {
            "sweep": {
                "model_path": "/m.gguf",
                "runner_type": "fake",
                "n_ctx": [512],
                "n_batch": [256],
            }
        }
        merged = _merge_shared_params(raw, "sweep")
        assert merged["model_path"] == "/m.gguf"
        assert merged["runner_type"] == "fake"

    def test_runner_params_inherited(self) -> None:
        from ppb import _merge_shared_params

        raw = {
            "runner_params": {"llama_bench_cmd": "/usr/bin/llama-bench"},
            "sweep": {"model_path": "/m.gguf", "n_ctx": [512], "n_batch": [256]},
        }
        merged = _merge_shared_params(raw, "sweep")
        assert merged["runner_params"] == {"llama_bench_cmd": "/usr/bin/llama-bench"}
