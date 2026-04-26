"""Unit tests for ppb_multiturn (Phase 7).

These tests exercise the pure helpers and the flattener path; they do
not require ``llama-cpp-python`` or the ``datasets`` library to be
installed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from ppb_multiturn import (
    VALID_MODES,
    _exact_match,
    _first_int,
    _render_session_history,
    run_multiturn,
)
from utils.flattener import flatten_benchmark_row


# ---------------------------------------------------------------------------
# _exact_match
# ---------------------------------------------------------------------------


def test_exact_match_true() -> None:
    assert _exact_match("The answer is BLUE.", "blue") is True


def test_exact_match_case_insensitive() -> None:
    assert _exact_match("answer: BLUE", "Blue") is True


def test_exact_match_false() -> None:
    assert _exact_match("I don't know.", "blue") is False


def test_exact_match_empty() -> None:
    assert _exact_match("", "blue") is False
    assert _exact_match("foo", "") is False


# ---------------------------------------------------------------------------
# _first_int
# ---------------------------------------------------------------------------


def test_first_int_basic() -> None:
    assert _first_int("Score: 7") == 7


def test_first_int_negative() -> None:
    assert _first_int("rating -3 is bad") == -3


def test_first_int_none() -> None:
    assert _first_int("") is None
    assert _first_int("no digits here") is None


# ---------------------------------------------------------------------------
# _render_session_history
# ---------------------------------------------------------------------------


def test_render_session_history_string_passthrough() -> None:
    assert _render_session_history("hello there") == "hello there"


def test_render_session_history_flat_list() -> None:
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    rendered = _render_session_history(history)
    assert "User: Hi" in rendered
    assert "Assistant: Hello" in rendered


def test_render_session_history_nested_sessions() -> None:
    history = [
        [
            {"role": "user", "content": "First turn"},
            {"role": "assistant", "content": "Reply"},
        ],
        [
            {"role": "user", "content": "Second turn"},
        ],
    ]
    rendered = _render_session_history(history)
    assert "First turn" in rendered
    assert "Second turn" in rendered


def test_render_session_history_non_list_returns_empty() -> None:
    assert _render_session_history(None) == ""
    assert _render_session_history(42) == ""


# ---------------------------------------------------------------------------
# run_multiturn — config validation
# ---------------------------------------------------------------------------


def test_run_multiturn_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown multiturn_mode"):
        run_multiturn(
            llm=object(),
            suite_config={"multiturn_mode": "bogus"},
        )


def test_run_multiturn_quick_without_judge_raises() -> None:
    with pytest.raises(ValueError, match="MT-Bench quick mode requires"):
        run_multiturn(
            llm=object(),
            suite_config={"multiturn_mode": "quick"},
            judge_llm=None,
        )


def test_valid_modes_constant() -> None:
    assert "longmemeval_s" in VALID_MODES
    assert "quick" in VALID_MODES


# ---------------------------------------------------------------------------
# run_multiturn — LongMemEval flow with mocked dataset + LLM
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Stand-in for ``llama_cpp.Llama`` returning a fixed response."""

    def __init__(self, response: str = "BLUE") -> None:
        self.response = response
        self.calls: list[str] = []

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(prompt)
        return {"choices": [{"text": self.response}]}

    def tokenize(self, text: bytes) -> list[int]:
        # 1 "token" per 4 bytes for predictable VRAM-cliff math.
        return [0] * max(1, len(text) // 4)


def _fake_longmemeval(_sample_size: int) -> list[dict[str, Any]]:
    return [
        {
            "session_history": [
                {"role": "user", "content": "What's the colour of the sky?"},
                {"role": "assistant", "content": "Blue."},
            ],
            "question": "What colour did I just say the sky was?",
            "answer": "blue",
        },
        {
            "session_history": [{"role": "user", "content": "Tell me a story."}],
            "question": "What did the king say?",
            "answer": "rosebud",
        },
    ]


def test_run_multiturn_longmemeval_exact_match() -> None:
    llm = _FakeLLM(response="The sky is blue, definitely.")
    with patch("ppb_multiturn._load_longmemeval", side_effect=_fake_longmemeval):
        result = run_multiturn(
            llm=llm,
            suite_config={
                "multiturn_mode": "longmemeval_s",
                "multiturn_sample_size": 2,
            },
        )

    assert result["mt_bench_score"] is None
    assert result["cases_evaluated"] == 2
    assert result["cases_skipped_context"] == 0
    # First case: response contains "blue" -> match. Second: doesn't contain
    # "rosebud" -> no match. Accuracy = 0.5.
    assert result["memory_accuracy"] == pytest.approx(0.5)


def test_run_multiturn_longmemeval_skips_when_over_vram_cliff() -> None:
    llm = _FakeLLM(response="blue")
    with patch("ppb_multiturn._load_longmemeval", side_effect=_fake_longmemeval):
        result = run_multiturn(
            llm=llm,
            model_config={"vram_cliff_tokens": 1},  # forces every case to be skipped
            suite_config={"multiturn_mode": "longmemeval_s"},
        )

    assert result["cases_evaluated"] == 0
    assert result["cases_skipped_context"] == 2
    assert result["memory_accuracy"] is None


def test_run_multiturn_longmemeval_with_judge() -> None:
    llm = _FakeLLM(response="The sky is blue.")
    judge = _FakeLLM(response="YES")
    with patch("ppb_multiturn._load_longmemeval", side_effect=_fake_longmemeval):
        result = run_multiturn(
            llm=llm,
            judge_llm=judge,
            suite_config={
                "multiturn_mode": "longmemeval_s",
                "multiturn_sample_size": 2,
            },
        )

    # Judge says YES on every case -> 100% accuracy.
    assert result["memory_accuracy"] == pytest.approx(1.0)
    assert result["cases_evaluated"] == 2


# ---------------------------------------------------------------------------
# flattener — multiturn runner row shaping
# ---------------------------------------------------------------------------


def test_flatten_multiturn_longmemeval_row() -> None:
    row = {
        "timestamp": "2026-04-26T00:00:00+00:00",
        "runner_type": "multiturn",
        "run_type": "qualitative",
        "model": "unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
        "n_ctx": 32768,
        "n_batch": None,
        "concurrent_users": None,
        "hardware": {
            "gpus": [{"name": "RTX 4090", "vram_gb": 24.0}],
            "os": {},
            "cpu": {},
            "ram": {},
            "runtime": {},
        },
        "task_type": "multiturn-longmemeval_s",
        "prompt_dataset": "longmemeval-cleaned",
        "llm_engine_name": "llama-cpp-python",
        "llm_engine_version": None,
        "results": {
            "memory_accuracy": 0.72,
            "mt_bench_score": None,
            "cases_evaluated": 47,
            "cases_skipped_context": 3,
        },
    }

    rows = flatten_benchmark_row(row)
    assert len(rows) == 1
    flat = rows[0]

    assert flat["runner_type"] == "multiturn"
    assert flat["run_type"] == "qualitative"
    assert flat["memory_accuracy"] == pytest.approx(0.72)
    assert flat["mt_bench_score"] is None
    assert flat["model_base"] == "Qwen3.5-0.8B"
    assert flat["quant"] == "Q4_K_M"
    assert flat["gpu_name"] == "RTX 4090"


def test_flatten_multiturn_quick_row() -> None:
    row = {
        "timestamp": "2026-04-26T00:00:00+00:00",
        "runner_type": "multiturn",
        "run_type": "qualitative",
        "model": "unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
        "n_ctx": 8192,
        "hardware": {
            "gpus": [{"name": "RTX 4090", "vram_gb": 24.0}],
            "os": {},
            "cpu": {},
            "ram": {},
            "runtime": {},
        },
        "results": {
            "memory_accuracy": None,
            "mt_bench_score": 7.4,
            "cases_evaluated": 80,
            "cases_skipped_context": 0,
        },
    }

    rows = flatten_benchmark_row(row)
    flat = rows[0]
    assert flat["mt_bench_score"] == pytest.approx(7.4)
    assert flat["memory_accuracy"] is None


def test_flatten_multiturn_default_run_type_is_qualitative() -> None:
    """Legacy rows without explicit run_type still classify correctly."""
    row = {
        "timestamp": "2026-04-26T00:00:00+00:00",
        "runner_type": "multiturn",
        "model": "x/y/Foo-Q4_K_M.gguf",
        "hardware": {"gpus": [], "os": {}, "cpu": {}, "ram": {}, "runtime": {}},
        "results": {"memory_accuracy": 0.5},
    }
    flat = flatten_benchmark_row(row)[0]
    assert flat["run_type"] == "qualitative"


def test_flatten_multiturn_row_includes_case_counts() -> None:
    """cases_evaluated and cases_skipped_context must survive flattening."""
    row = {
        "timestamp": "2026-04-26T00:00:00+00:00",
        "runner_type": "multiturn",
        "run_type": "qualitative",
        "model": "unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
        "n_ctx": 32768,
        "hardware": {
            "gpus": [{"name": "RTX 4090", "vram_gb": 24.0}],
            "os": {},
            "cpu": {},
            "ram": {},
            "runtime": {},
        },
        "results": {
            "memory_accuracy": 0.72,
            "mt_bench_score": None,
            "cases_evaluated": 47,
            "cases_skipped_context": 3,
        },
    }
    flat = flatten_benchmark_row(row)[0]
    assert flat["cases_evaluated"] == 47
    assert flat["cases_skipped_context"] == 3


def test_flatten_multiturn_case_counts_default_to_none() -> None:
    """Rows without case counts flatten to None, not KeyError."""
    row = {
        "timestamp": "2026-04-26T00:00:00+00:00",
        "runner_type": "multiturn",
        "model": "x/y/Foo-Q4_K_M.gguf",
        "hardware": {"gpus": [], "os": {}, "cpu": {}, "ram": {}, "runtime": {}},
        "results": {"memory_accuracy": 0.5},
    }
    flat = flatten_benchmark_row(row)[0]
    assert flat["cases_evaluated"] is None
    assert flat["cases_skipped_context"] is None
