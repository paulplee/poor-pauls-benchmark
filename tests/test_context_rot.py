"""Unit tests for the context-rot module and the flattener's context-rot path.

These tests do not require ``llama-cpp-python`` to be installed; they only
exercise the pure-Python helpers and the flattener's row-shaping logic.
"""

from __future__ import annotations

import json

import pytest

from ppb_context_rot import _score_response
from utils.flattener import flatten_benchmark_row


# ---------------------------------------------------------------------------
# _score_response — pure function, no mocks needed
# ---------------------------------------------------------------------------


def test_score_response_exact_match() -> None:
    assert _score_response("ALPHA-7-DELTA", "ALPHA-7-DELTA") == 1


def test_score_response_case_insensitive() -> None:
    assert _score_response("The code is alpha-7-delta.", "ALPHA-7-DELTA") == 1


def test_score_response_no_match() -> None:
    assert _score_response("I don't know.", "ALPHA-7-DELTA") == 0


def test_score_response_empty_response() -> None:
    assert _score_response("", "ALPHA-7-DELTA") == 0


def test_score_response_partial_substring() -> None:
    assert _score_response("ALPHA-7-DELTA is the answer", "ALPHA-7-DELTA") == 1


# ---------------------------------------------------------------------------
# flatten_benchmark_row — context-rot row shaping
# ---------------------------------------------------------------------------


def test_flatten_context_rot_row() -> None:
    """flatten_benchmark_row correctly handles a context-rot runner row."""
    row = {
        "timestamp": "2026-04-26T00:00:00+00:00",
        "runner_type": "context-rot",
        "run_type": "qualitative",
        "model": "unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
        "n_ctx": 32768,
        "n_batch": None,
        "concurrent_users": 1,
        "hardware": {
            "gpus": [{"name": "RTX 4090", "vram_gb": 24.0}],
            "os": {},
            "cpu": {},
            "ram": {},
            "runtime": {},
        },
        "suite_run_id": "abc123",
        "task_type": "context-rot-niah",
        "prompt_dataset": "sharegpt-v3",
        "llm_engine_name": "llama-cpp-python",
        "llm_engine_version": None,
        "results": {
            "context_rot_score": 0.867,
            "context_rot_accuracy_by_length": {4096: 1.0, 8192: 0.8, 16384: 0.6},
            "context_rot_accuracy_by_depth": {10: 0.9, 50: 0.8, 90: 0.7},
        },
    }

    rows = flatten_benchmark_row(row)
    assert len(rows) == 1
    flat = rows[0]

    assert flat["runner_type"] == "context-rot"
    assert flat["run_type"] == "qualitative"
    assert flat["context_rot_score"] == pytest.approx(0.867)

    # Dicts are serialised as JSON strings for Arrow compatibility.
    by_len = json.loads(flat["context_rot_accuracy_by_length"])
    assert by_len["4096"] == pytest.approx(1.0)  # JSON keys are strings

    assert flat["model_base"] == "Qwen3.5-0.8B"
    assert flat["quant"] == "Q4_K_M"
    assert flat["gpu_name"] == "RTX 4090"
