"""Tests for utils.flattener — row normalisation logic."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from utils.flattener import (
    COLUMN_ORDER,
    MASTER_SCHEMA,
    compute_file_sha256,
    flatten_benchmark_row,
    _parse_model_provenance,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal but structurally complete benchmark rows
# ---------------------------------------------------------------------------


def _make_hardware():
    return {
        "os": {
            "system": "Linux",
            "release": "6.17.0",
            "machine": "x86_64",
            "distro": "Ubuntu",
            "distro_version": "24.04",
        },
        "cpu": {"model": "AMD Ryzen 7 7800X3D", "cores": "16"},
        "ram": {"total_gb": 124.9},
        "gpus": [
            {
                "name": "NVIDIA RTX 5090",
                "vram_total_gb": 32.0,
                "driver": "590.40",
                "cuda_version": "13.0",
            }
        ],
        "runtime": {"python_version": "3.13.12"},
    }


def _make_dual_gpu_hardware():
    """Hardware snapshot for a dual NVIDIA GeForce RTX 4060 Ti system."""
    return {
        "os": {
            "system": "Linux",
            "release": "6.17.0",
            "machine": "x86_64",
            "distro": "Ubuntu",
            "distro_version": "24.04",
        },
        "cpu": {"model": "Intel Core i9-14900K", "cores": "24"},
        "ram": {"total_gb": 64.0},
        "gpus": [
            {
                "index": 0,
                "name": "NVIDIA GeForce RTX 4060 Ti",
                "vram_total_gb": 16.0,
                "driver": "570.86",
                "cuda_version": "12.8",
            },
            {
                "index": 1,
                "name": "NVIDIA GeForce RTX 4060 Ti",
                "vram_total_gb": 16.0,
                "driver": "570.86",
                "cuda_version": "12.8",
            },
        ],
        "runtime": {"python_version": "3.13.12"},
    }


LLAMA_BENCH_ROW = {
    "timestamp": "2026-03-08T12:00:00+00:00",
    "runner_type": "llama-bench",
    "model": "test-org/test-repo/Qwen3.5-9B-Q8_0.gguf",
    "n_ctx": 8192,
    "n_batch": 512,
    "llm_engine_name": "llama.cpp",
    "llm_engine_version": "b5063 (58ab80c3)",
    "task_type": "text-generation",
    "hardware": _make_hardware(),
    "results": [
        {
            "gpu_info": "NVIDIA RTX 5090",
            "backends": "CUDA",
            "n_prompt": 512,
            "n_gen": 0,
            "avg_ts": 1200.5,
        },
        {
            "gpu_info": "NVIDIA RTX 5090",
            "backends": "CUDA",
            "n_prompt": 0,
            "n_gen": 128,
            "avg_ts": 131.0,
        },
    ],
}

LLAMA_SERVER_ROW = {
    "timestamp": "2026-03-08T12:10:00+00:00",
    "runner_type": "llama-server",
    "model": "test-org/test-repo/Qwen3.5-9B-Q8_0.gguf",
    "n_ctx": 16384,
    "n_batch": 512,
    "llm_engine_name": "llama.cpp",
    "llm_engine_version": "b5063 (58ab80c3)",
    "task_type": "text-generation",
    "prompt_dataset": "sharegpt-v3",
    "hardware": _make_hardware(),
    "results": {
        "num_prompts_attempted": 10,
        "num_prompts_succeeded": 10,
        "n_predict": 256,
        "throughput_tok_s": 59.26,
        "avg_ttft_ms": 142.5,
        "p50_ttft_ms": 138.2,
        "p99_ttft_ms": 210.7,
        "avg_itl_ms": 12.3,
        "p50_itl_ms": 11.8,
        "p99_itl_ms": 18.4,
    },
}

LOADTEST_ROW = {
    "timestamp": "2026-03-08T12:20:00+00:00",
    "runner_type": "llama-server-loadtest",
    "model": "test-org/test-repo/Qwen3.5-9B-Q8_0.gguf",
    "n_ctx": 8192,
    "n_batch": 512,
    "llm_engine_name": "llama.cpp",
    "task_type": "text-generation",
    "prompt_dataset": "sharegpt-v3",
    "hardware": _make_hardware(),
    "results": {
        "error_threshold": 0.1,
        "concurrency_curve": [
            {"concurrent_users": 1, "aggregate_throughput_tok_s": 55.0},
            {"concurrent_users": 2, "aggregate_throughput_tok_s": 100.0},
            {"concurrent_users": 4, "aggregate_throughput_tok_s": 180.0},
            {"concurrent_users": 8, "aggregate_throughput_tok_s": 200.0},
        ],
    },
}


# ---------------------------------------------------------------------------
# Model provenance (model_org / model_repo)
# ---------------------------------------------------------------------------


class TestModelProvenance:
    def test_hf_path_extracts_org_and_repo(self):
        assert _parse_model_provenance("unsloth/Qwen3.5-2B-GGUF/model.gguf") == (
            "unsloth",
            "unsloth/Qwen3.5-2B-GGUF",
        )

    def test_local_absolute_path_returns_none(self):
        assert _parse_model_provenance("/Volumes/DATA/models/model.gguf") == (
            None,
            None,
        )

    def test_local_relative_path_returns_none(self):
        assert _parse_model_provenance("./models/model.gguf") == (None, None)

    def test_none_returns_none(self):
        assert _parse_model_provenance(None) == (None, None)

    def test_flat_row_has_provenance_fields(self):
        row = {
            **LLAMA_BENCH_ROW,
            "model": "unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q4_K_M.gguf",
        }
        flat = flatten_benchmark_row(row)[0]
        assert flat["model_org"] == "unsloth"
        assert flat["model_repo"] == "unsloth/Qwen3.5-2B-GGUF"

    def test_local_model_provenance_is_null(self):
        row = {**LLAMA_BENCH_ROW, "model": "/local/path/to/model.gguf"}
        flat = flatten_benchmark_row(row)[0]
        assert flat["model_org"] is None
        assert flat["model_repo"] is None


# ---------------------------------------------------------------------------
# Multi-GPU hardware fields
# ---------------------------------------------------------------------------


class TestMultiGpuHardware:
    def test_dual_gpu_count(self):
        row = {**LLAMA_SERVER_ROW, "hardware": _make_dual_gpu_hardware()}
        flat = flatten_benchmark_row(row)[0]
        assert flat["gpu_count"] == 2

    def test_dual_gpu_names_joined(self):
        row = {**LLAMA_SERVER_ROW, "hardware": _make_dual_gpu_hardware()}
        flat = flatten_benchmark_row(row)[0]
        assert (
            flat["gpu_names"]
            == "NVIDIA GeForce RTX 4060 Ti, NVIDIA GeForce RTX 4060 Ti"
        )

    def test_dual_gpu_total_vram(self):
        row = {**LLAMA_SERVER_ROW, "hardware": _make_dual_gpu_hardware()}
        flat = flatten_benchmark_row(row)[0]
        assert flat["gpu_total_vram_gb"] == 32.0

    def test_dual_gpu_name_is_primary_gpu(self):
        """gpu_name stays as GPU 0 for backward compat with viz site filters."""
        row = {**LLAMA_SERVER_ROW, "hardware": _make_dual_gpu_hardware()}
        flat = flatten_benchmark_row(row)[0]
        assert flat["gpu_name"] == "NVIDIA GeForce RTX 4060 Ti"

    def test_no_gpu_count_is_zero(self):
        row = {**LLAMA_SERVER_ROW, "hardware": {**_make_hardware(), "gpus": []}}
        flat = flatten_benchmark_row(row)[0]
        assert flat["gpu_count"] == 0
        assert flat["gpu_names"] is None
        assert flat["gpu_total_vram_gb"] is None

    def test_split_mode_and_tensor_split_from_record(self):
        row = {
            **LLAMA_SERVER_ROW,
            "split_mode": "layer",
            "tensor_split": "1,1",
        }
        flat = flatten_benchmark_row(row)[0]
        assert flat["split_mode"] == "layer"
        assert flat["tensor_split"] == "1,1"

    def test_split_mode_null_when_absent(self):
        flat = flatten_benchmark_row(LLAMA_SERVER_ROW)[0]
        assert flat["split_mode"] is None
        assert flat["tensor_split"] is None


# ---------------------------------------------------------------------------
# Model parsing (model_base / quant extraction)
# ---------------------------------------------------------------------------


class TestModelParsing:
    def test_extracts_base_and_quant(self):
        row = {**LLAMA_BENCH_ROW, "model": "org/repo/Qwen3.5-9B-Q8_0.gguf"}
        flat = flatten_benchmark_row(row)[0]
        assert flat["model"] == "org/repo/Qwen3.5-9B-Q8_0.gguf"
        assert flat["model_base"] == "Qwen3.5-9B"
        assert flat["quant"] == "Q8_0"

    def test_various_quant_formats(self):
        cases = [
            ("org/repo/Llama-3.1-8B-Q4_K_M.gguf", "Llama-3.1-8B", "Q4_K_M"),
            ("org/repo/Phi-4-F16.gguf", "Phi-4", "F16"),
            ("org/repo/model-IQ2_XXS.gguf", "model", "IQ2_XXS"),
            ("org/repo/Falcon-7B-BF16.gguf", "Falcon-7B", "BF16"),
        ]
        for model, expected_base, expected_quant in cases:
            row = {**LLAMA_BENCH_ROW, "model": model}
            flat = flatten_benchmark_row(row)[0]
            assert flat["model_base"] == expected_base, f"Failed base for {model}"
            assert flat["quant"] == expected_quant, f"Failed quant for {model}"

    def test_plain_gguf_no_quant(self):
        row = {**LLAMA_BENCH_ROW, "model": "org/repo/my-model.gguf"}
        flat = flatten_benchmark_row(row)[0]
        assert flat["model_base"] == "my-model"
        assert flat["quant"] is None

    def test_none_when_no_model(self):
        row = {**LLAMA_BENCH_ROW, "model": None}
        flat = flatten_benchmark_row(row)[0]
        assert flat["model"] is None
        assert flat["model_base"] is None
        assert flat["quant"] is None


# ---------------------------------------------------------------------------
# backends enrichment
# ---------------------------------------------------------------------------


class TestBackendsEnrichment:
    def test_cuda_version_appended(self):
        """When hardware has cuda_version, backends should read 'CUDA 13.0'."""
        flat = flatten_benchmark_row(LLAMA_BENCH_ROW)[0]
        assert flat["backends"] == "CUDA 13.0"

    def test_no_cuda_version_keeps_plain(self):
        """When gpus list is empty, backends stays as-is."""
        row = {**LLAMA_BENCH_ROW, "hardware": {**_make_hardware(), "gpus": []}}
        flat = flatten_benchmark_row(row)[0]
        assert flat["backends"] == "CUDA"

    def test_non_cuda_backend_untouched(self):
        row = {**LLAMA_BENCH_ROW}
        row["results"] = [{**LLAMA_BENCH_ROW["results"][0], "backends": "Metal"}]
        flat = flatten_benchmark_row(row)[0]
        assert flat["backends"] == "Metal"


# ---------------------------------------------------------------------------
# llama-bench
# ---------------------------------------------------------------------------


class TestLlamaBench:
    def test_explodes_to_n_rows(self):
        rows = flatten_benchmark_row(LLAMA_BENCH_ROW)
        assert len(rows) == 2

    def test_envelope_fields_present(self):
        row = flatten_benchmark_row(LLAMA_BENCH_ROW)[0]
        assert row["timestamp"] == "2026-03-08T12:00:00+00:00"
        assert row["runner_type"] == "llama-bench"
        assert row["model_base"] == "Qwen3.5-9B"
        assert row["quant"] == "Q8_0"
        assert row["n_ctx"] == 8192
        assert row["n_batch"] == 512

    def test_hardware_fields_flattened(self):
        row = flatten_benchmark_row(LLAMA_BENCH_ROW)[0]
        assert row["os_system"] == "Linux"
        assert row["cpu_model"] == "AMD Ryzen 7 7800X3D"
        assert row["ram_total_gb"] == 124.9
        assert row["gpu_name"] == "NVIDIA RTX 5090"
        assert row["gpu_vram_gb"] == 32.0
        assert row["gpu_driver"] == "590.40"
        # Single-GPU multi-GPU fields
        assert row["gpu_count"] == 1
        assert row["gpu_names"] == "NVIDIA RTX 5090"
        assert row["gpu_total_vram_gb"] == 32.0

    def test_per_item_fields(self):
        rows = flatten_benchmark_row(LLAMA_BENCH_ROW)
        # First item: prompt processing
        assert rows[0]["throughput_tok_s"] == 1200.5
        assert rows[0]["backends"] == "CUDA 13.0"
        # Second item: generation
        assert rows[1]["throughput_tok_s"] == 131.0

    def test_raw_payload_roundtrips(self):
        row = flatten_benchmark_row(LLAMA_BENCH_ROW)[0]
        restored = json.loads(row["raw_payload"])
        assert restored["runner_type"] == "llama-bench"
        assert len(restored["results"]) == 2

    def test_gpu_name_fallback_when_gpus_empty(self):
        """When hardware.gpus is empty, gpu_name falls back to result-level gpu_info."""
        row = {**LLAMA_BENCH_ROW, "hardware": {**_make_hardware(), "gpus": []}}
        flat = flatten_benchmark_row(row)[0]
        assert flat["gpu_name"] == "NVIDIA RTX 5090"


# ---------------------------------------------------------------------------
# llama-server
# ---------------------------------------------------------------------------


class TestLlamaServer:
    def test_single_row(self):
        rows = flatten_benchmark_row(LLAMA_SERVER_ROW)
        assert len(rows) == 1

    def test_throughput_and_latency(self):
        row = flatten_benchmark_row(LLAMA_SERVER_ROW)[0]
        assert row["throughput_tok_s"] == 59.26
        assert row["avg_ttft_ms"] == 142.5
        assert row["p50_ttft_ms"] == 138.2
        assert row["p99_ttft_ms"] == 210.7
        assert row["avg_itl_ms"] == 12.3
        assert row["p50_itl_ms"] == 11.8
        assert row["p99_itl_ms"] == 18.4


# ---------------------------------------------------------------------------
# llama-server-loadtest
# ---------------------------------------------------------------------------


class TestLoadtest:
    def test_single_row(self):
        rows = flatten_benchmark_row(LOADTEST_ROW)
        assert len(rows) == 1

    def test_max_sustainable_users_absent(self):
        """max_sustainable_users was removed from the schema in v2."""
        row = flatten_benchmark_row(LOADTEST_ROW)[0]
        assert "max_sustainable_users" not in row
        assert "max_concurrent_users" not in row

    def test_best_throughput_from_curve(self):
        row = flatten_benchmark_row(LOADTEST_ROW)[0]
        assert row["throughput_tok_s"] == 200.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unknown_runner_type(self):
        row = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "runner_type": "future-runner",
            "model": "org/repo/model.gguf",
            "n_ctx": 4096,
            "n_batch": 256,
            "hardware": {},
            "results": {"something": 42},
        }
        rows = flatten_benchmark_row(row)
        assert len(rows) == 1
        assert rows[0]["throughput_tok_s"] is None

    def test_missing_hardware(self):
        row = {**LLAMA_SERVER_ROW, "hardware": None}
        rows = flatten_benchmark_row(row)
        assert len(rows) == 1
        assert rows[0]["gpu_name"] is None

    def test_empty_concurrency_curve(self):
        row = {
            **LOADTEST_ROW,
            "results": {**LOADTEST_ROW["results"], "concurrency_curve": []},
        }
        flat = flatten_benchmark_row(row)[0]
        assert flat["throughput_tok_s"] is None


# ---------------------------------------------------------------------------
# Unified schema — every row must have exactly the MASTER_SCHEMA keys
# ---------------------------------------------------------------------------


class TestUnifiedSchema:
    """Ensure every runner produces rows with the exact MASTER_SCHEMA keys."""

    _expected_keys = set(MASTER_SCHEMA)

    @pytest.mark.parametrize(
        "fixture",
        [LLAMA_BENCH_ROW, LLAMA_SERVER_ROW, LOADTEST_ROW],
        ids=["llama-bench", "llama-server", "llama-server-loadtest"],
    )
    def test_all_runners_have_exact_schema(self, fixture):
        for flat in flatten_benchmark_row(fixture):
            assert set(flat) == self._expected_keys

    def test_unknown_runner_has_exact_schema(self):
        row = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "runner_type": "future-runner",
            "model": "org/repo/model.gguf",
            "n_ctx": 4096,
            "n_batch": 256,
            "hardware": {},
            "results": {"something": 42},
        }
        for flat in flatten_benchmark_row(row):
            assert set(flat) == self._expected_keys

    def test_missing_hardware_has_exact_schema(self):
        row = {**LLAMA_SERVER_ROW, "hardware": None}
        for flat in flatten_benchmark_row(row):
            assert set(flat) == self._expected_keys


# ---------------------------------------------------------------------------
# Provenance / fingerprint fields
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_schema_version(self):
        flat = flatten_benchmark_row(LLAMA_SERVER_ROW)[0]
        assert flat["schema_version"] == "0.1.0"

    def test_benchmark_version_present(self):
        flat = flatten_benchmark_row(LLAMA_SERVER_ROW)[0]
        assert isinstance(flat["benchmark_version"], str)
        assert len(flat["benchmark_version"]) > 0

    def test_row_id_unique_across_exploded_rows(self):
        rows = flatten_benchmark_row(LLAMA_BENCH_ROW)
        assert len(rows) == 2
        assert rows[0]["row_id"] != rows[1]["row_id"]

    def test_machine_fingerprint_deterministic(self):
        a = flatten_benchmark_row(LLAMA_SERVER_ROW)[0]
        b = flatten_benchmark_row(LLAMA_SERVER_ROW)[0]
        assert a["machine_fingerprint"] == b["machine_fingerprint"]
        assert len(a["machine_fingerprint"]) == 64  # SHA-256 hex

    def test_machine_fingerprint_changes_with_hardware(self):
        row_a = LLAMA_SERVER_ROW
        hw_b = {**_make_hardware()}
        hw_b["gpus"] = [{**hw_b["gpus"][0], "name": "NVIDIA RTX 4090"}]
        row_b = {**LLAMA_SERVER_ROW, "hardware": hw_b}
        fp_a = flatten_benchmark_row(row_a)[0]["machine_fingerprint"]
        fp_b = flatten_benchmark_row(row_b)[0]["machine_fingerprint"]
        assert fp_a != fp_b


# ---------------------------------------------------------------------------
# Column order
# ---------------------------------------------------------------------------


class TestColumnOrder:
    """COLUMN_ORDER drives CSV headers and HF viewer column order."""

    def test_column_order_does_not_include_raw_payload(self):
        assert "raw_payload" not in COLUMN_ORDER

    def test_column_order_does_not_include_removed_fields(self):
        assert "max_sustainable_users" not in COLUMN_ORDER
        assert "max_concurrent_users" not in COLUMN_ORDER
        assert "avg_gpu_power_w" not in COLUMN_ORDER
        assert "max_gpu_power_w" not in COLUMN_ORDER

    @pytest.mark.parametrize(
        "fixture",
        [LLAMA_BENCH_ROW, LLAMA_SERVER_ROW, LOADTEST_ROW],
        ids=["llama-bench", "llama-server", "llama-server-loadtest"],
    )
    def test_flat_row_keys_start_with_column_order(self, fixture):
        """All public columns appear in canonical order at the start of the row."""
        flat = flatten_benchmark_row(fixture)[0]
        public_keys = [k for k in flat if k != "raw_payload"]
        assert public_keys == COLUMN_ORDER

    def test_master_schema_keys_are_column_order_plus_raw_payload(self):
        assert list(MASTER_SCHEMA) == COLUMN_ORDER + ["raw_payload"]

    def test_run_fingerprint_deterministic(self):
        a = flatten_benchmark_row(LLAMA_SERVER_ROW)[0]
        b = flatten_benchmark_row(LLAMA_SERVER_ROW)[0]
        assert a["run_fingerprint"] == b["run_fingerprint"]
        assert len(a["run_fingerprint"]) == 64

    def test_run_fingerprint_changes_with_n_ctx(self):
        row_a = LLAMA_SERVER_ROW
        row_b = {**LLAMA_SERVER_ROW, "n_ctx": 4096}
        fp_a = flatten_benchmark_row(row_a)[0]["run_fingerprint"]
        fp_b = flatten_benchmark_row(row_b)[0]["run_fingerprint"]
        assert fp_a != fp_b

    def test_result_fingerprint_changes_with_throughput(self):
        row_a = LLAMA_BENCH_ROW
        altered_results = [{**LLAMA_BENCH_ROW["results"][0], "avg_ts": 9999.9}]
        row_b = {**LLAMA_BENCH_ROW, "results": altered_results}
        fp_a = flatten_benchmark_row(row_a)[0]["result_fingerprint"]
        fp_b = flatten_benchmark_row(row_b)[0]["result_fingerprint"]
        assert fp_a != fp_b

    def test_result_fingerprint_stable_for_same_metrics(self):
        a = flatten_benchmark_row(LLAMA_SERVER_ROW)[0]
        b = flatten_benchmark_row(LLAMA_SERVER_ROW)[0]
        assert a["result_fingerprint"] == b["result_fingerprint"]

    def test_caller_fields_default_to_none(self):
        flat = flatten_benchmark_row(LLAMA_SERVER_ROW)[0]
        assert flat["submission_id"] is None
        assert flat["submitted_at"] is None
        assert flat["source_file_sha256"] is None


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------


class TestFileHash:
    def test_compute_file_sha256_matches_hashlib(self):
        content = b"hello world\nsome benchmark data\n"
        expected = hashlib.sha256(content).hexdigest()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            assert compute_file_sha256(tmp_path) == expected
        finally:
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# New schema columns — LLM engine, workload, quality, OS distro, tags
# ---------------------------------------------------------------------------


class TestLlmEngine:
    def test_engine_name_propagated(self):
        flat = flatten_benchmark_row(LLAMA_BENCH_ROW)[0]
        assert flat["llm_engine_name"] == "llama.cpp"

    def test_engine_version_propagated(self):
        flat = flatten_benchmark_row(LLAMA_SERVER_ROW)[0]
        assert flat["llm_engine_version"] == "b5063 (58ab80c3)"

    def test_engine_name_defaults_to_none(self):
        row = {**LLAMA_BENCH_ROW}
        del row["llm_engine_name"]
        flat = flatten_benchmark_row(row)[0]
        assert flat["llm_engine_name"] is None

    def test_engine_version_defaults_to_none(self):
        row = {**LLAMA_BENCH_ROW}
        del row["llm_engine_version"]
        flat = flatten_benchmark_row(row)[0]
        assert flat["llm_engine_version"] is None


class TestOsDistro:
    def test_distro_extracted(self):
        flat = flatten_benchmark_row(LLAMA_SERVER_ROW)[0]
        assert flat["os_distro"] == "Ubuntu"

    def test_distro_version_extracted(self):
        flat = flatten_benchmark_row(LLAMA_SERVER_ROW)[0]
        assert flat["os_distro_version"] == "24.04"

    def test_distro_none_when_missing(self):
        hw = {**_make_hardware(), "os": {"system": "Linux", "release": "6.17.0", "machine": "x86_64"}}
        row = {**LLAMA_SERVER_ROW, "hardware": hw}
        flat = flatten_benchmark_row(row)[0]
        assert flat["os_distro"] is None
        assert flat["os_distro_version"] is None

    def test_distro_none_when_no_hardware(self):
        row = {**LLAMA_SERVER_ROW, "hardware": None}
        flat = flatten_benchmark_row(row)[0]
        assert flat["os_distro"] is None
        assert flat["os_distro_version"] is None


class TestWorkloadColumns:
    def test_task_type_propagated(self):
        flat = flatten_benchmark_row(LLAMA_SERVER_ROW)[0]
        assert flat["task_type"] == "text-generation"

    def test_task_type_defaults_to_none(self):
        row = {**LLAMA_BENCH_ROW}
        del row["task_type"]
        flat = flatten_benchmark_row(row)[0]
        assert flat["task_type"] is None

    def test_prompt_dataset_propagated(self):
        flat = flatten_benchmark_row(LLAMA_SERVER_ROW)[0]
        assert flat["prompt_dataset"] == "sharegpt-v3"

    def test_prompt_dataset_none_for_bench(self):
        flat = flatten_benchmark_row(LLAMA_BENCH_ROW)[0]
        assert flat["prompt_dataset"] is None

    def test_num_prompts_from_results(self):
        flat = flatten_benchmark_row(LLAMA_SERVER_ROW)[0]
        assert flat["num_prompts"] == 10

    def test_num_prompts_none_for_bench(self):
        flat = flatten_benchmark_row(LLAMA_BENCH_ROW)[0]
        assert flat["num_prompts"] is None

    def test_n_predict_from_results(self):
        flat = flatten_benchmark_row(LLAMA_SERVER_ROW)[0]
        assert flat["n_predict"] == 256

    def test_n_predict_none_for_bench(self):
        flat = flatten_benchmark_row(LLAMA_BENCH_ROW)[0]
        assert flat["n_predict"] is None


class TestQualityScore:
    def test_defaults_to_none(self):
        flat = flatten_benchmark_row(LLAMA_BENCH_ROW)[0]
        assert flat["quality_score"] is None

    def test_propagated_when_present(self):
        row = {**LLAMA_SERVER_ROW, "quality_score": 0.85}
        flat = flatten_benchmark_row(row)[0]
        assert flat["quality_score"] == 0.85


class TestTags:
    def test_defaults_to_none(self):
        flat = flatten_benchmark_row(LLAMA_BENCH_ROW)[0]
        assert flat["tags"] is None

    def test_string_passed_through(self):
        row = {**LLAMA_SERVER_ROW, "tags": '{"env": "ci"}'}
        flat = flatten_benchmark_row(row)[0]
        assert flat["tags"] == '{"env": "ci"}'

    def test_dict_serialized_to_json(self):
        row = {**LLAMA_SERVER_ROW, "tags": json.dumps({"env": "ci", "run": 42})}
        flat = flatten_benchmark_row(row)[0]
        parsed = json.loads(flat["tags"])
        assert parsed["env"] == "ci"
        assert parsed["run"] == 42
