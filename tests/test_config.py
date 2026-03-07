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
