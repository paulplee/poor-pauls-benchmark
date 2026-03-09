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
        """Build a SweepConfig from defaults + overrides."""
        from ppb import SweepConfig

        defaults = {
            "repo_id": "test-org/test-repo",
            "filename": "*.gguf",
            "n_ctx": [512],
            "n_batch": [256],
        }
        defaults.update(overrides)
        return SweepConfig(**defaults)

    # -- runner_type defaults -------------------------------------------------

    def test_default_runner_type(self) -> None:
        cfg = self._make_config()
        assert cfg.runner_type == "llama-bench"

    def test_custom_runner_type(self) -> None:
        cfg = self._make_config(runner_type="llama-server")
        assert cfg.runner_type == "llama-server"

    # -- runner_params --------------------------------------------------------

    def test_default_runner_params_empty(self) -> None:
        cfg = self._make_config()
        assert cfg.runner_params == {}

    def test_custom_runner_params(self) -> None:
        cfg = self._make_config(runner_params={"key": "val"})
        assert cfg.runner_params == {"key": "val"}

    # -- models_dir default ---------------------------------------------------

    def test_default_models_dir(self) -> None:
        cfg = self._make_config()
        assert cfg.models_dir == "./models"

    def test_custom_models_dir(self) -> None:
        cfg = self._make_config(models_dir="/data/models")
        assert cfg.models_dir == "/data/models"

    # -- combos ---------------------------------------------------------------

    def test_combos_cartesian_product(self, tmp_path: Path) -> None:
        cfg = self._make_config(n_ctx=[512, 1024], n_batch=[256, 512])
        m1 = tmp_path / "alpha.gguf"
        m2 = tmp_path / "beta.gguf"
        cfg.resolved_models = [
            (m1, "org/repo/alpha.gguf"),
            (m2, "org/repo/beta.gguf"),
        ]
        combos = cfg.combos()
        # 2 models × 2 ctx × 2 batch = 8
        assert len(combos) == 8

    def test_combos_with_concurrent_users(self, tmp_path: Path) -> None:
        cfg = self._make_config(
            n_ctx=[512], n_batch=[256], concurrent_users=[1, 2, 4],
        )
        m = tmp_path / "m.gguf"
        cfg.resolved_models = [(m, "org/repo/m.gguf")]
        combos = cfg.combos()
        # 1 model × 1 ctx × 1 batch × 3 users = 3
        assert len(combos) == 3
        user_counts = sorted(c.concurrent_users for c in combos)
        assert user_counts == [1, 2, 4]

    def test_combos_concurrent_users_default(self, tmp_path: Path) -> None:
        """Default concurrent_users=[1] doesn't inflate combo count."""
        cfg = self._make_config(n_ctx=[512, 1024], n_batch=[256])
        m = tmp_path / "m.gguf"
        cfg.resolved_models = [(m, "org/repo/m.gguf")]
        combos = cfg.combos()
        assert len(combos) == 2
        assert all(c.concurrent_users == 1 for c in combos)

    def test_combos_fields(self, tmp_path: Path) -> None:
        cfg = self._make_config(n_ctx=[1024], n_batch=[256])
        m = tmp_path / "m.gguf"
        cfg.resolved_models = [(m, "org/repo/m.gguf")]
        combos = cfg.combos()
        assert len(combos) == 1
        c = combos[0]
        assert c.model_path == m
        assert c.model == "org/repo/m.gguf"
        assert c.n_ctx == 1024
        assert c.n_batch == 256


# ==========================================================================
# BenchCombo
# ==========================================================================


class TestBenchCombo:
    def test_fields(self) -> None:
        from ppb import BenchCombo

        c = BenchCombo(
            model_path=Path("/m.gguf"), model="org/repo/m.gguf",
            n_ctx=4096, n_batch=512,
        )
        assert c.model_path == Path("/m.gguf")
        assert c.model == "org/repo/m.gguf"
        assert c.n_ctx == 4096
        assert c.n_batch == 512

    def test_concurrent_users_default(self) -> None:
        from ppb import BenchCombo

        c = BenchCombo(
            model_path=Path("/m.gguf"), model="org/repo/m.gguf",
            n_ctx=4096, n_batch=512,
        )
        assert c.concurrent_users == 1

    def test_concurrent_users_set(self) -> None:
        from ppb import BenchCombo

        c = BenchCombo(
            model_path=Path("/m.gguf"), model="org/repo/m.gguf",
            n_ctx=4096, n_batch=512, concurrent_users=8,
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
            "model": "org/repo/m.gguf",
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
# VramCliffConfig
# ==========================================================================


class TestVramCliffConfig:
    """Validate the Pydantic model for [vram-cliff] config."""

    def test_defaults(self) -> None:
        from ppb import VramCliffConfig

        cfg = VramCliffConfig(repo_id="org/repo", filename="*.gguf")
        assert cfg.min_ctx == 2048
        assert cfg.max_ctx == 131072
        assert cfg.tolerance == 1024
        assert cfg.runner_type == "llama-bench"
        assert cfg.runner_params == {}
        assert cfg.models_dir == "./models"

    def test_all_fields_override(self) -> None:
        from ppb import VramCliffConfig

        cfg = VramCliffConfig(
            repo_id="org/repo",
            filename="model.gguf",
            models_dir="/data/models",
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
        assert cfg.models_dir == "/data/models"


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
