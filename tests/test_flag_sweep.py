"""Tests for the llama.cpp flag sweep feature (utils/flag_utils.py + ppb.py integration)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.flag_utils import (
    build_extra_cli_args,
    expand_llama_cpp_args,
    parse_flag_entry,
)


# ---------------------------------------------------------------------------
# parse_flag_entry
# ---------------------------------------------------------------------------


class TestParseFlagEntry:
    def test_empty_dict(self):
        flags, label, efr = parse_flag_entry({})
        assert flags == {}
        assert label is None
        assert efr is None

    def test_structured_flags_only(self):
        flags, label, efr = parse_flag_entry({"ncmoe": 20, "cmoe": True})
        assert flags == {"ncmoe": 20, "cmoe": True}
        assert label is None
        assert efr is None

    def test_label_extracted(self):
        flags, label, efr = parse_flag_entry({"ncmoe": 20, "_label": "ncmoe_20"})
        assert flags == {"ncmoe": 20}
        assert label == "ncmoe_20"
        assert efr is None

    def test_extra_flags_extracted(self):
        flags, label, efr = parse_flag_entry({"fa": True, "extra_flags": "-rtr"})
        assert flags == {"fa": True}
        assert label is None
        assert efr == "-rtr"

    def test_all_special_keys(self):
        entry = {"ncmoe": 40, "_label": "test", "extra_flags": "--some-flag 1"}
        flags, label, efr = parse_flag_entry(entry)
        assert flags == {"ncmoe": 40}
        assert label == "test"
        assert efr == "--some-flag 1"

    def test_original_dict_not_mutated(self):
        entry = {"ncmoe": 20, "_label": "x"}
        parse_flag_entry(entry)
        assert "_label" in entry  # original untouched


# ---------------------------------------------------------------------------
# build_extra_cli_args
# ---------------------------------------------------------------------------


class TestBuildExtraCliArgs:
    def test_empty_produces_nothing(self):
        assert build_extra_cli_args({}) == []

    def test_short_flag_integer(self):
        assert build_extra_cli_args({"ncmoe": 20}) == ["-ncmoe", "20"]

    def test_short_flag_boolean_true(self):
        assert build_extra_cli_args({"cmoe": True}) == ["-cmoe"]

    def test_short_flag_boolean_false_omitted(self):
        assert build_extra_cli_args({"cmoe": False}) == []

    def test_long_flag_unknown_key(self):
        # underscore → dash, double-dash prefix
        assert build_extra_cli_args({"flash_attn": True}) == ["--flash-attn"]

    def test_long_flag_unknown_key_with_value(self):
        assert build_extra_cli_args({"cache_type": "q8_0"}) == ["--cache-type", "q8_0"]

    def test_multiple_flags_order_preserved(self):
        # Python dicts preserve insertion order
        result = build_extra_cli_args({"ncmoe": 20, "cmoe": True, "fa": False})
        assert result == ["-ncmoe", "20", "-cmoe"]

    def test_extra_flags_raw_appended(self):
        result = build_extra_cli_args({}, extra_flags_raw="-rtr")
        assert result == ["-rtr"]

    def test_extra_flags_raw_split_by_whitespace(self):
        result = build_extra_cli_args({}, extra_flags_raw="-rtr --some-flag 42")
        assert result == ["-rtr", "--some-flag", "42"]

    def test_combined_structured_and_raw(self):
        result = build_extra_cli_args({"ncmoe": 20, "cmoe": True}, extra_flags_raw="-rtr")
        assert result == ["-ncmoe", "20", "-cmoe", "-rtr"]

    def test_ngl_short_flag(self):
        assert build_extra_cli_args({"ngl": 35}) == ["-ngl", "35"]

    def test_fa_short_flag(self):
        assert build_extra_cli_args({"fa": True}) == ["-fa"]

    def test_moe_combined(self):
        result = build_extra_cli_args({"cmoe": True, "ncmoe": 60})
        assert result == ["-cmoe", "-ncmoe", "60"]


# ---------------------------------------------------------------------------
# expand_llama_cpp_args
# ---------------------------------------------------------------------------


class TestExpandLlamaCppArgs:
    def test_no_range_returns_explicit(self):
        explicit = [{"ncmoe": 20}, {"ncmoe": 40}]
        result = expand_llama_cpp_args(explicit, {})
        assert result == explicit

    def test_empty_explicit_empty_range(self):
        result = expand_llama_cpp_args([], {})
        assert result == []

    def test_single_range_flag(self):
        result = expand_llama_cpp_args([], {"ncmoe": {"from": 20, "to": 40, "step": 20}})
        assert result == [{"ncmoe": 20}, {"ncmoe": 40}]

    def test_range_step_1_default(self):
        result = expand_llama_cpp_args([], {"n": {"from": 1, "to": 3}})
        assert result == [{"n": 1}, {"n": 2}, {"n": 3}]

    def test_range_does_not_exceed_to(self):
        # from=20 to=99 step=10 → values stop before 99
        result = expand_llama_cpp_args([], {"ncmoe": {"from": 20, "to": 99, "step": 10}})
        values = [r["ncmoe"] for r in result]
        assert all(v <= 99 for v in values)
        assert 20 in values
        assert 90 in values

    def test_range_appended_after_explicit(self):
        explicit = [{}]
        result = expand_llama_cpp_args(explicit, {"ncmoe": {"from": 20, "to": 20, "step": 10}})
        assert result[0] == {}    # baseline first
        assert result[1] == {"ncmoe": 20}  # range after

    def test_multi_flag_cartesian_product(self):
        result = expand_llama_cpp_args(
            [],
            {
                "ncmoe": {"from": 20, "to": 40, "step": 20},
                "fa": {"from": 0, "to": 1, "step": 1},
            },
        )
        # 2 ncmoe values × 2 fa values = 4 combos
        assert len(result) == 4
        ncmoe_values = {r["ncmoe"] for r in result}
        fa_values = {r["fa"] for r in result}
        assert ncmoe_values == {20, 40}
        assert fa_values == {0, 1}

    def test_invalid_range_spec_raises(self):
        with pytest.raises((ValueError, KeyError)):
            expand_llama_cpp_args([], {"ncmoe": "not-a-dict"})

    def test_empty_range_produces_no_values_raises(self):
        with pytest.raises(ValueError, match="produced no values"):
            expand_llama_cpp_args([], {"ncmoe": {"from": 100, "to": 20, "step": 10}})


# ---------------------------------------------------------------------------
# SweepConfig.combos() integration
# ---------------------------------------------------------------------------


class TestSweepConfigCombos:
    """Verify that the flag dimension is correctly added to the combo product."""

    def _make_config(self, **kwargs):
        """Import SweepConfig inline to avoid module-level import issues."""
        import importlib
        ppb = importlib.import_module("ppb")
        SweepConfig = ppb.SweepConfig
        return SweepConfig(**kwargs)

    def _resolved_models(self, tmp_path: Path):
        m = tmp_path / "model.gguf"
        m.touch()
        return [(m, "org/repo/model.gguf")]

    def test_no_flags_single_baseline(self, tmp_path):
        """Default llama_cpp_args=[{}] produces one flag point per combo."""
        cfg = self._make_config(
            repo_id="org/repo",
            filename="*.gguf",
            n_ctx=[8192],
            n_batch=[512],
            concurrent_users=[1],
            resolved_models=self._resolved_models(tmp_path),
        )
        combos = cfg.combos()
        assert len(combos) == 1
        assert combos[0].llm_flags == {}
        assert combos[0].llm_flags_label is None

    def test_two_flag_sets_doubles_combos(self, tmp_path):
        cfg = self._make_config(
            repo_id="org/repo",
            filename="*.gguf",
            n_ctx=[8192],
            n_batch=[512],
            concurrent_users=[1],
            resolved_models=self._resolved_models(tmp_path),
            llama_cpp_args=[{}, {"ncmoe": 20, "_label": "ncmoe_20"}],
        )
        combos = cfg.combos()
        assert len(combos) == 2
        flags_seen = {c.llm_flags.get("ncmoe") for c in combos}
        assert flags_seen == {None, 20}
        labels = {c.llm_flags_label for c in combos}
        assert labels == {None, "ncmoe_20"}

    def test_range_expansion_in_combos(self, tmp_path):
        cfg = self._make_config(
            repo_id="org/repo",
            filename="*.gguf",
            n_ctx=[8192],
            n_batch=[512],
            concurrent_users=[1],
            resolved_models=self._resolved_models(tmp_path),
            llama_cpp_args=[{}],
            llama_cpp_args_range={"ncmoe": {"from": 20, "to": 40, "step": 20}},
        )
        combos = cfg.combos()
        # [{}] baseline + [{ncmoe:20}, {ncmoe:40}] range = 3 flag sets
        assert len(combos) == 3

    def test_extra_flags_raw_preserved(self, tmp_path):
        cfg = self._make_config(
            repo_id="org/repo",
            filename="*.gguf",
            n_ctx=[8192],
            n_batch=[512],
            concurrent_users=[1],
            resolved_models=self._resolved_models(tmp_path),
            llama_cpp_args=[{"extra_flags": "-rtr", "_label": "rtr"}],
        )
        combos = cfg.combos()
        assert len(combos) == 1
        assert combos[0].extra_flags_raw == "-rtr"
        assert combos[0].llm_flags_label == "rtr"
        assert combos[0].llm_flags == {}  # extra_flags not in flags dict


# ---------------------------------------------------------------------------
# Backfill script unit tests
# ---------------------------------------------------------------------------


class TestBackfillRecord:
    def test_adds_missing_fields(self):
        from scripts.backfill_flags import _backfill_record

        rec = {"model": "test/model.gguf", "n_ctx": 8192}
        out = _backfill_record(rec, "b5063 (58ab80c3)")

        assert out["llm_flags"] == "{}"
        assert out["llm_flags_label"] == "default"
        assert out["extra_flags_raw"] is None
        assert out["llm_engine_version"] == "b5063 (58ab80c3)"
        assert "backfilled_flags" in json.loads(out["meta"])

    def test_does_not_overwrite_existing_flags(self):
        from scripts.backfill_flags import _backfill_record

        rec = {
            "llm_flags": '{"ncmoe": 20}',
            "llm_flags_label": "ncmoe_20",
            "extra_flags_raw": None,
        }
        out = _backfill_record(rec, None)
        # Existing values preserved
        assert out["llm_flags"] == '{"ncmoe": 20}'
        assert out["llm_flags_label"] == "ncmoe_20"

    def test_does_not_touch_fingerprints(self):
        from scripts.backfill_flags import _backfill_record

        rec = {
            "row_id": "abc",
            "result_fingerprint": "fp1",
            "run_fingerprint": "fp2",
            "machine_fingerprint": "fp3",
        }
        out = _backfill_record(rec, None)
        assert out["row_id"] == "abc"
        assert out["result_fingerprint"] == "fp1"
        assert out["run_fingerprint"] == "fp2"
        assert out["machine_fingerprint"] == "fp3"

    def test_needs_backfill_detection(self):
        from scripts.backfill_flags import _needs_backfill

        assert _needs_backfill({}) is True
        assert _needs_backfill({"llm_flags": "{}", "llm_flags_label": "default"}) is False
        assert _needs_backfill({"llm_flags": "{}"}) is True  # label still missing

    def test_backfill_file_roundtrip(self, tmp_path):
        from scripts.backfill_flags import backfill_file

        jsonl = tmp_path / "test.jsonl"
        records = [
            {"model": "m.gguf", "n_ctx": 8192, "results": {}},
            {"model": "m.gguf", "n_ctx": 16384, "llm_flags": "{}", "llm_flags_label": "default", "extra_flags_raw": None},
        ]
        jsonl.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        n = backfill_file(jsonl, "b5063", dry_run=False)
        assert n == 1  # only first record needed backfill

        out_records = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
        assert out_records[0]["llm_flags"] == "{}"
        assert out_records[1]["llm_flags"] == "{}"  # unchanged

    def test_dry_run_does_not_write(self, tmp_path):
        from scripts.backfill_flags import backfill_file

        jsonl = tmp_path / "test.jsonl"
        original = '{"model": "m.gguf"}\n'
        jsonl.write_text(original)

        backfill_file(jsonl, None, dry_run=True)
        assert jsonl.read_text() == original  # file unchanged


# ---------------------------------------------------------------------------
# Flattener: new columns appear
# ---------------------------------------------------------------------------


class TestFlattenerNewColumns:
    def _make_bench_row(self, **extra):
        return {
            "runner_type": "llama-bench",
            "model": "unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
            "n_ctx": 8192,
            "n_batch": 512,
            "concurrent_users": 1,
            "llm_flags": '{"ncmoe": 20}',
            "llm_flags_label": "ncmoe_20",
            "extra_flags_raw": None,
            "llm_engine_name": "llama.cpp",
            "llm_engine_version": "b5063 (58ab80c3)",
            "hardware": {},
            "results": [{"n_prompt": 8192, "n_gen": 0, "t_pp": 1.0, "t_tg": 0.0, "avg_pp": 1000.0, "avg_tg": 0.0}],
            **extra,
        }

    def test_llm_flags_in_flat_row(self):
        from utils.flattener import flatten_benchmark_row

        row = self._make_bench_row()
        flat_rows = flatten_benchmark_row(row)
        assert len(flat_rows) >= 1
        flat = flat_rows[0]
        assert flat["llm_flags"] == '{"ncmoe": 20}'
        assert flat["llm_flags_label"] == "ncmoe_20"
        assert flat["extra_flags_raw"] is None
        assert flat["llm_engine_version"] == "b5063 (58ab80c3)"

    def test_missing_flags_become_none(self):
        from utils.flattener import flatten_benchmark_row

        row = self._make_bench_row()
        del row["llm_flags"]
        del row["llm_flags_label"]
        flat_rows = flatten_benchmark_row(row)
        flat = flat_rows[0]
        assert flat["llm_flags"] is None
        assert flat["llm_flags_label"] is None

    def test_schema_version_bumped(self):
        from utils.flattener import _SCHEMA_VERSION

        # Schema was bumped to 0.10.0 when llm_flags fields were added
        assert _SCHEMA_VERSION == "0.10.0"
