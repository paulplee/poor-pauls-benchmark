"""Tests for utils.flattener — row normalisation logic."""

from __future__ import annotations

import json

import pytest

from utils.flattener import MASTER_SCHEMA, flatten_benchmark_row


# ---------------------------------------------------------------------------
# Fixtures — minimal but structurally complete benchmark rows
# ---------------------------------------------------------------------------

def _make_hardware():
    return {
        "os": {"system": "Linux", "release": "6.17.0", "machine": "x86_64"},
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


LLAMA_BENCH_ROW = {
    "timestamp": "2026-03-08T12:00:00+00:00",
    "runner_type": "llama-bench",
    "model_path": "/data/models/Qwen3.5-9B-Q8_0.gguf",
    "n_ctx": 8192,
    "n_batch": 512,
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
    "model_path": "/data/models/Qwen3.5-9B-Q8_0.gguf",
    "n_ctx": 16384,
    "n_batch": 512,
    "hardware": _make_hardware(),
    "results": {
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
    "model_path": "/data/models/Qwen3.5-9B-Q8_0.gguf",
    "n_ctx": 8192,
    "n_batch": 512,
    "hardware": _make_hardware(),
    "results": {
        "max_sustainable_users": 8,
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
# model_id derivation
# ---------------------------------------------------------------------------

class TestModelId:
    def test_strips_quant_suffix(self):
        row = {**LLAMA_BENCH_ROW, "model_path": "/data/models/Qwen3.5-9B-Q8_0.gguf"}
        flat = flatten_benchmark_row(row)[0]
        assert flat["model_id"] == "Qwen3.5-9B"

    def test_strips_other_quant_formats(self):
        cases = [
            ("/m/Llama-3.1-8B-Q4_K_M.gguf", "Llama-3.1-8B"),
            ("/m/Phi-4-F16.gguf", "Phi-4"),
            ("/m/model-IQ2_XXS.gguf", "model"),
            ("/m/Falcon-7B-BF16.gguf", "Falcon-7B"),
        ]
        for path, expected in cases:
            row = {**LLAMA_BENCH_ROW, "model_path": path}
            flat = flatten_benchmark_row(row)[0]
            assert flat["model_id"] == expected, f"Failed for {path}"

    def test_plain_gguf_no_quant(self):
        row = {**LLAMA_BENCH_ROW, "model_path": "/data/models/my-model.gguf"}
        flat = flatten_benchmark_row(row)[0]
        assert flat["model_id"] == "my-model"

    def test_none_when_no_model_path(self):
        row = {**LLAMA_BENCH_ROW, "model_path": None}
        flat = flatten_benchmark_row(row)[0]
        assert flat["model_id"] is None


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
        assert row["model_id"] == "Qwen3.5-9B"
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

    def test_max_users(self):
        row = flatten_benchmark_row(LOADTEST_ROW)[0]
        assert row["max_sustainable_users"] == 8

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
            "model_path": "/data/model.gguf",
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
            "model_path": "/data/model.gguf",
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
