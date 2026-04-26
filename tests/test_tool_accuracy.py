"""Unit tests for the tool-accuracy module.

These tests exercise the pure-Python helpers (parsing, AST match, scoring)
and do NOT require ``llama-cpp-python`` to be installed.
"""

from __future__ import annotations

import pytest

from ppb_tool_accuracy import (
    _ast_match,
    _expected_schema,
    _parse_response,
    _type_compatible,
    run_tool_accuracy,
)
from utils.flattener import flatten_benchmark_row


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


def test_parse_response_clean_json() -> None:
    assert _parse_response('{"name": "foo", "arguments": {"x": 1}}') == {
        "name": "foo",
        "arguments": {"x": 1},
    }


def test_parse_response_code_fence() -> None:
    text = '```json\n{"name": "foo", "arguments": {}}\n```'
    parsed = _parse_response(text)
    assert parsed is not None
    assert parsed["name"] == "foo"


def test_parse_response_with_prose() -> None:
    text = 'Sure! Here is the call: {"name": "foo", "arguments": {"x": 1}}'
    parsed = _parse_response(text)
    assert parsed is not None
    assert parsed["name"] == "foo"


def test_parse_response_invalid() -> None:
    assert _parse_response("not json at all") is None
    assert _parse_response("") is None


# ---------------------------------------------------------------------------
# _type_compatible
# ---------------------------------------------------------------------------


def test_type_compatible_basic() -> None:
    assert _type_compatible(1, "integer")
    assert _type_compatible(1.5, "number")
    assert _type_compatible("x", "string")
    assert _type_compatible(True, "boolean")
    assert _type_compatible([1], "array")
    assert _type_compatible({"a": 1}, "object")


def test_type_compatible_bool_is_not_number() -> None:
    # Guard against Python's bool-is-int subtlety.
    assert not _type_compatible(True, "integer")
    assert not _type_compatible(False, "number")


def test_type_compatible_unknown_type_passes() -> None:
    assert _type_compatible("anything", None)
    assert _type_compatible("anything", "fancy_custom_type")


# ---------------------------------------------------------------------------
# _ast_match
# ---------------------------------------------------------------------------


_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {"type": "string"},
        "gpu_vram_gb": {"type": "number"},
        "priority": {"type": "string"},
    },
    "required": ["model", "gpu_vram_gb", "priority"],
}

_TRUTH = {
    "name": "recommend_quantization",
    "arguments": {"model": "Qwen3-30B", "gpu_vram_gb": 24, "priority": "speed"},
}


def test_ast_match_perfect() -> None:
    name, params, hall = _ast_match(_TRUTH, _TRUTH, _SCHEMA)
    assert name and params and not hall


def test_ast_match_wrong_name() -> None:
    resp = {"name": "wrong_tool", "arguments": _TRUTH["arguments"]}
    name, params, _ = _ast_match(resp, _TRUTH, _SCHEMA)
    assert not name
    assert params  # required params still all present and correct


def test_ast_match_missing_required_param() -> None:
    resp = {
        "name": "recommend_quantization",
        "arguments": {"model": "Qwen3-30B", "priority": "speed"},
    }
    name, params, _ = _ast_match(resp, _TRUTH, _SCHEMA)
    assert name and not params


def test_ast_match_wrong_value() -> None:
    resp = {
        "name": "recommend_quantization",
        "arguments": {"model": "Qwen3-30B", "gpu_vram_gb": 24, "priority": "quality"},
    }
    name, params, _ = _ast_match(resp, _TRUTH, _SCHEMA)
    assert name and not params


def test_ast_match_hallucinated_param() -> None:
    resp = {
        "name": "recommend_quantization",
        "arguments": {
            "model": "Qwen3-30B",
            "gpu_vram_gb": 24,
            "priority": "speed",
            "secret_extra_flag": True,
        },
    }
    name, params, hall = _ast_match(resp, _TRUTH, _SCHEMA)
    assert name and params and hall


def test_ast_match_wrong_type() -> None:
    resp = {
        "name": "recommend_quantization",
        "arguments": {"model": 12345, "gpu_vram_gb": 24, "priority": "speed"},
    }
    name, params, _ = _ast_match(resp, _TRUTH, _SCHEMA)
    assert name and not params


def test_ast_match_empty_arguments() -> None:
    """Zero-arg tools (e.g. ``list_tested_configs()``) score full credit.

    Regression for Fix 8: ground-truth cases with empty argument dicts and
    empty schemas must yield ``(name=True, params=True, hall=False)`` rather
    than crash or undercount parameter accuracy.
    """
    truth = {"name": "list_tested_configs", "arguments": {}}
    schema = {"type": "object", "properties": {}, "required": []}
    name, params, hall = _ast_match(truth, truth, schema)
    assert name is True
    assert params is True
    assert hall is False


# ---------------------------------------------------------------------------
# _expected_schema
# ---------------------------------------------------------------------------


def test_expected_schema_single() -> None:
    fn = {"name": "foo", "parameters": {"properties": {"x": {"type": "integer"}}}}
    schema = _expected_schema(fn, "foo")
    assert schema["properties"]["x"]["type"] == "integer"


def test_expected_schema_list_pick_by_name() -> None:
    fns = [
        {"name": "foo", "parameters": {"properties": {"a": {"type": "string"}}}},
        {"name": "bar", "parameters": {"properties": {"b": {"type": "integer"}}}},
    ]
    schema = _expected_schema(fns, "bar")
    assert "b" in schema["properties"]


def test_expected_schema_missing_returns_empty() -> None:
    assert _expected_schema({"name": "foo"}, "nonexistent") == {}


# ---------------------------------------------------------------------------
# run_tool_accuracy — end-to-end with a stub LLM (no llama-cpp-python)
# ---------------------------------------------------------------------------


class _StubLLM:
    """Returns a fixed response for every prompt — emulates a perfect model."""

    def __init__(self, response_text: str) -> None:
        self._text = response_text

    def __call__(self, prompt: str, **_: object) -> dict:
        return {"choices": [{"text": self._text}]}


def test_run_tool_accuracy_zero_cases_returns_nulls(monkeypatch) -> None:
    # Force both data sources to return empty so we hit the empty-set branch.
    import ppb_tool_accuracy as mod

    monkeypatch.setattr(mod, "_load_bfcl", lambda n: [])
    monkeypatch.setattr(mod, "_load_ppb_native", lambda: [])

    out = run_tool_accuracy(
        _StubLLM("{}"), suite_config={"tool_accuracy_sample_size": 0}
    )
    assert out["n_cases"] == 0
    assert out["overall_tool_accuracy"] is None


def test_run_tool_accuracy_perfect_run(monkeypatch) -> None:
    import ppb_tool_accuracy as mod

    case = {
        "question": "Recommend a quant.",
        "function": {
            "name": "recommend_quantization",
            "parameters": _SCHEMA,
        },
        "answer": _TRUTH,
    }
    monkeypatch.setattr(mod, "_load_bfcl", lambda n: [])
    monkeypatch.setattr(mod, "_load_ppb_native", lambda: [case, case])

    perfect_response = (
        '{"name": "recommend_quantization", "arguments": '
        '{"model": "Qwen3-30B", "gpu_vram_gb": 24, "priority": "speed"}}'
    )
    out = run_tool_accuracy(_StubLLM(perfect_response))
    assert out["n_cases"] == 2
    assert out["tool_selection_accuracy"] == pytest.approx(1.0)
    assert out["parameter_accuracy"] == pytest.approx(1.0)
    assert out["parse_success_rate"] == pytest.approx(1.0)
    assert out["parameter_hallucination_rate"] == pytest.approx(0.0)
    assert out["overall_tool_accuracy"] == pytest.approx(1.0)


def test_run_tool_accuracy_parse_failure(monkeypatch) -> None:
    import ppb_tool_accuracy as mod

    case = {
        "question": "Recommend a quant.",
        "function": {"name": "foo", "parameters": _SCHEMA},
        "answer": _TRUTH,
    }
    monkeypatch.setattr(mod, "_load_bfcl", lambda n: [])
    monkeypatch.setattr(mod, "_load_ppb_native", lambda: [case])

    out = run_tool_accuracy(_StubLLM("complete garbage, not json"))
    assert out["parse_success_rate"] == pytest.approx(0.0)
    assert out["tool_selection_accuracy"] == pytest.approx(0.0)
    assert out["parameter_accuracy"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# flatten_benchmark_row — tool-accuracy runner path
# ---------------------------------------------------------------------------


def test_flatten_tool_accuracy_row() -> None:
    row = {
        "timestamp": "2026-04-26T00:00:00+00:00",
        "runner_type": "tool-accuracy",
        "run_type": "qualitative",
        "model": "unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
        "n_ctx": 4096,
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
        "task_type": "tool-call-accuracy",
        "prompt_dataset": "bfcl+ppb-mcp",
        "llm_engine_name": "llama-cpp-python",
        "llm_engine_version": None,
        "results": {
            "tool_selection_accuracy": 0.9,
            "parameter_accuracy": 0.8,
            "parameter_hallucination_rate": 0.1,
            "parse_success_rate": 0.95,
            "overall_tool_accuracy": 0.848,
            "n_cases": 120,
        },
    }
    rows = flatten_benchmark_row(row)
    assert len(rows) == 1
    flat = rows[0]
    assert flat["runner_type"] == "tool-accuracy"
    assert flat["run_type"] == "qualitative"
    assert flat["tool_selection_accuracy"] == pytest.approx(0.9)
    assert flat["parameter_accuracy"] == pytest.approx(0.8)
    assert flat["parameter_hallucination_rate"] == pytest.approx(0.1)
    assert flat["parse_success_rate"] == pytest.approx(0.95)
    assert flat["overall_tool_accuracy"] == pytest.approx(0.848)
    assert flat["model_base"] == "Qwen3.5-0.8B"
    assert flat["quant"] == "Q4_K_M"
    assert flat["gpu_name"] == "RTX 4090"


def test_run_type_default_for_tool_accuracy_runner() -> None:
    """When run_type is not explicit, runner_type=tool-accuracy → qualitative."""
    row = {
        "timestamp": "2026-04-26T00:00:00+00:00",
        "runner_type": "tool-accuracy",
        "model": "unsloth/Foo/Foo-Q4_K_M.gguf",
        "results": {
            "tool_selection_accuracy": 0.5,
            "parameter_accuracy": 0.5,
            "parameter_hallucination_rate": 0.0,
            "parse_success_rate": 1.0,
            "overall_tool_accuracy": 0.5,
        },
    }
    rows = flatten_benchmark_row(row)
    assert rows[0]["run_type"] == "qualitative"


# ---------------------------------------------------------------------------
# BFCL v4 — _normalise_bfcl_row, _score_no_call, irrelevance routing
# ---------------------------------------------------------------------------


def test_normalise_bfcl_row() -> None:
    """BFCL v4 row normalisation handles well-formed, malformed, and irrelevance."""
    from ppb_tool_accuracy import _normalise_bfcl_row

    # (a) well-formed v4 positive case with merged ground truth.
    raw_ok = {
        "id": "simple_python_0",
        "question": [[{"role": "user", "content": "Recommend a quant."}]],
        "function": [
            {
                "name": "recommend_quantization",
                "description": "...",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    }
    gt_map = {
        "simple_python_0": [
            "recommend_quantization(model='Qwen3-30B', gpu_vram_gb=24, priority='speed')"
        ]
    }
    out = _normalise_bfcl_row(raw_ok, ground_truth_map=gt_map)
    assert out is not None
    assert out["question"] == "Recommend a quant."
    assert (
        isinstance(out["function"], list)
        and out["function"][0]["name"] == "recommend_quantization"
    )
    assert out["answer"]["name"] == "recommend_quantization"
    assert out["answer"]["arguments"] == {
        "model": "Qwen3-30B",
        "gpu_vram_gb": 24,
        "priority": "speed",
    }
    assert out["expected_no_call"] is False

    # (b) malformed nesting (single-list shape, not double-nested) → row skipped.
    raw_bad = {
        "id": "bad_0",
        "question": [{"role": "user", "content": "missing outer list"}],
        "function": [],
    }
    assert _normalise_bfcl_row(raw_bad, ground_truth_map={}) is None

    # (c) irrelevance row → expected_no_call=True, answer empty.
    raw_irr = {
        "id": "irrelevance_42",
        "question": [[{"role": "user", "content": "What is the weather like?"}]],
        "function": [{"name": "calculator", "parameters": {}}],
    }
    out_irr = _normalise_bfcl_row(raw_irr, ground_truth_map={}, expected_no_call=True)
    assert out_irr is not None
    assert out_irr["expected_no_call"] is True
    assert out_irr["answer"] == {}
    assert out_irr["question"] == "What is the weather like?"


def test_score_no_call() -> None:
    """`_score_no_call` correctly identifies abstention vs tool-call attempts."""
    from ppb_tool_accuracy import _score_no_call

    # Plain prose abstentions → True.
    assert _score_no_call("I cannot help with that without more context.") is True
    assert _score_no_call("Sorry, I don't know.") is True
    assert _score_no_call("") is True
    assert _score_no_call("   \n  ") is True

    # Tool-call attempts → False.
    assert _score_no_call("get_weather(city='London')") is False
    assert _score_no_call('{"function_call": {"name": "x"}}') is False
    assert _score_no_call('{"tool_call": {}}') is False
    assert _score_no_call("<tool_call>foo</tool_call>") is False


def test_irrelevance_routing(monkeypatch) -> None:
    """Irrelevance cases populate `no_call_accuracy` and skip positive scoring."""
    import ppb_tool_accuracy as mod

    irr_case = {
        "question": "What is the meaning of life?",
        "function": [{"name": "calculator", "parameters": {}}],
        "answer": {},
        "expected_no_call": True,
    }
    pos_case = {
        "question": "Recommend a quant.",
        "function": {"name": "recommend_quantization", "parameters": _SCHEMA},
        "answer": _TRUTH,
    }
    monkeypatch.setattr(mod, "_load_bfcl", lambda n: [irr_case])
    monkeypatch.setattr(mod, "_load_ppb_native", lambda: [pos_case])

    # Stub returns a Python-style call (contains ``(``) so both prompts
    # produce the same string.  Note ``_parse_response`` extracts the
    # embedded JSON for the positive case (still parseable).
    response = (
        "recommend_quantization(model=...) → "
        '{"name": "recommend_quantization", "arguments": '
        '{"model": "Qwen3-30B", "gpu_vram_gb": 24, "priority": "speed"}}'
    )
    out = run_tool_accuracy(_StubLLM(response))

    assert out["n_cases"] == 2
    # Positive denominator excludes irrelevance: only 1 positive case → 1.0.
    assert out["tool_selection_accuracy"] == pytest.approx(1.0)
    assert out["parameter_accuracy"] == pytest.approx(1.0)
    assert out["parse_success_rate"] == pytest.approx(1.0)
    # Irrelevance: response contains "(" → scored as a call → 0.0.
    assert out["no_call_accuracy"] == pytest.approx(0.0)
    # Geometric mean over positive cases only — does NOT fold in no_call.
    assert out["overall_tool_accuracy"] == pytest.approx(1.0)


def test_irrelevance_routing_correct_abstention(monkeypatch) -> None:
    """When the model abstains in prose, no_call_accuracy is 1.0."""
    import ppb_tool_accuracy as mod

    irr_case = {
        "question": "What is the weather like in Paris?",
        "function": [{"name": "calculator", "parameters": {}}],
        "answer": {},
        "expected_no_call": True,
    }
    monkeypatch.setattr(mod, "_load_bfcl", lambda n: [irr_case])
    monkeypatch.setattr(mod, "_load_ppb_native", lambda: [])

    out = run_tool_accuracy(_StubLLM("I don't have access to weather data."))
    assert out["n_cases"] == 1
    assert out["no_call_accuracy"] == pytest.approx(1.0)
    # No positive cases → all positive metrics are None.
    assert out["tool_selection_accuracy"] is None
    assert out["parameter_accuracy"] is None
    assert out["overall_tool_accuracy"] is None
