"""
PPB — Tool-Call Accuracy (BFCL v4 + PPB-native).

Measures whether a model produces structurally valid, schema-correct tool
calls.  Uses the Berkeley Function Calling Leaderboard (BFCL) v4 single-turn
splits (``simple_python``, ``multiple``, ``parallel``, ``irrelevance``) as
the primary evaluation set, supplemented by a PPB-native MCP ground-truth
set covering the four ``ppb-mcp`` tools.

Method
------
For each test case:

1. Build a system prompt that lists the available tool schema(s) and
   instructs the model to respond with a single JSON object
   ``{"name": ..., "arguments": ...}``.
2. Prompt the model with the user question (temperature=0, max_tokens=256).
3. Parse the response as JSON.  Failures are recorded as ``parse_fail``.
4. AST-match against the ground-truth call:
   - ``name`` must match exactly,
   - all required parameters must be present and correctly typed,
   - extra parameter keys not in the schema are flagged as hallucinations.

Aggregated metrics (returned in the result dict, slot directly into the
``qualitative`` block of the composable schema):

* ``tool_selection_accuracy``       — fraction with matching ``name``
* ``parameter_accuracy``            — fraction with all required params correct
* ``parameter_hallucination_rate``  — fraction with extra/unknown params
* ``parse_success_rate``            — fraction of valid-JSON responses
* ``overall_tool_accuracy``         — geometric mean of selection × parameter

Public entry point
------------------
``run_tool_accuracy(llm, model_config, suite_config) -> dict``
"""

from __future__ import annotations

import ast
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("ppb")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

# Single-turn splits used by PPB.
# Excludes: live/*, multi_turn/*, memory, web_search, java, javascript.
BFCL_SPLITS = (
    "BFCL_v4_simple_python.json",  # 399 cases — one tool, one call
    "BFCL_v4_multiple.json",  # 199 cases — tool selection from candidates
    "BFCL_v4_parallel.json",  # 199 cases — parallel / batched calls
    "BFCL_v4_irrelevance.json",  # 239 cases — model must decline to call
)
BFCL_ANSWER_PREFIX = "possible_answer/"  # ground-truth subfolder in the same repo
# Split filename suffix that signals "model must NOT emit a tool call".
BFCL_IRRELEVANCE_SPLIT = "BFCL_v4_irrelevance.json"

DEFAULT_BFCL_SAMPLE_SIZE = 100
DEFAULT_MAX_TOKENS = 256

PPB_GROUND_TRUTH_PATH = (
    Path(__file__).resolve().parent
    / "ppb_datasets"
    / "data"
    / "ppb_mcp_ground_truth.json"
)

SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful assistant. You have access to the following tools. "
    "When you want to call a tool, respond ONLY with a JSON object with "
    "keys `name` (string) and `arguments` (object). Do not add any prose.\n\n"
    "Available tools:\n{tool_schemas}"
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _parse_bfcl_ground_truth(call_str: str) -> dict[str, Any]:
    """Parse a BFCL v4 ground-truth call string into a ``{name, arguments}`` dict.

    BFCL v4 expresses ground truth as Python function-call syntax, e.g.
    ``"recommend_quantization(model='Qwen3-30B', gpu_vram_gb=24)"``.
    Returns ``{}`` on any parse failure.
    """
    if not call_str or not isinstance(call_str, str):
        return {}
    try:
        tree = ast.parse(call_str.strip(), mode="eval")
    except SyntaxError:
        return {}
    call = tree.body
    if not isinstance(call, ast.Call):
        return {}
    if isinstance(call.func, ast.Name):
        name = call.func.id
    elif isinstance(call.func, ast.Attribute):
        name = call.func.attr
    else:
        return {}
    arguments: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue
        try:
            arguments[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            return {}
    return {"name": name, "arguments": arguments}


def _load_bfcl_answers(split_filename: str) -> dict[str, list[str]]:
    """Download and parse the ``possible_answer/<split>`` ground-truth file.

    Returns a mapping from BFCL ``id`` to its ``ground_truth`` list (a list
    of Python function-call strings).  Returns an empty dict on failure so
    the question rows can still be loaded (and skipped per-row when no
    ground truth is found).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:  # pragma: no cover
        return {}
    try:
        local_path = hf_hub_download(
            repo_id=BFCL_REPO,
            repo_type="dataset",
            filename=f"{BFCL_ANSWER_PREFIX}{split_filename}",
        )
    except Exception as exc:  # pragma: no cover — network/dataset issues
        log.warning(
            "[tool-accuracy] Failed to download BFCL ground-truth for %r: %s",
            split_filename,
            exc,
        )
        return {}
    answers: dict[str, list[str]] = {}
    try:
        with open(local_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = obj.get("id")
                gt = obj.get("ground_truth") or []
                if qid is not None and isinstance(gt, list):
                    answers[qid] = gt
    except Exception as exc:  # pragma: no cover
        log.warning(
            "[tool-accuracy] Failed to read BFCL ground-truth %r: %s",
            split_filename,
            exc,
        )
    return answers


def _load_bfcl(sample_size: int) -> list[dict[str, Any]]:
    """Load BFCL v4 single-turn rows from each split, capped at *sample_size*.

    Each split contributes ``sample_size // len(BFCL_SPLITS)`` rows.  Ground
    truth is fetched separately from ``possible_answer/<split>`` and merged
    into each row by ``id``.  Falls back to an empty list (with a warning)
    if the dataset cannot be fetched.
    """
    if sample_size <= 0:
        return []

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:  # pragma: no cover
        log.warning(
            "[tool-accuracy] `huggingface_hub` not installed; skipping BFCL evaluation."
        )
        return []

    rows: list[dict[str, Any]] = []
    per_split = max(1, sample_size // len(BFCL_SPLITS))

    for filename in BFCL_SPLITS:
        is_irrelevance = filename == BFCL_IRRELEVANCE_SPLIT
        try:
            local_path = hf_hub_download(
                repo_id=BFCL_REPO,
                repo_type="dataset",
                filename=filename,
            )
        except Exception as exc:  # pragma: no cover — network/dataset issues
            log.warning(
                "[tool-accuracy] Failed to download BFCL split %r: %s. "
                "Skipping this split \u2014 other splits and PPB-native cases will still run.",
                filename,
                exc,
            )
            continue

        # Irrelevance split has no positive ground truth; skip the answer fetch.
        ground_truth_map: dict[str, list[str]] = (
            {} if is_irrelevance else _load_bfcl_answers(filename)
        )

        count = 0
        try:
            with open(local_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if count >= per_split:
                        break
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    normalised = _normalise_bfcl_row(
                        raw,
                        ground_truth_map=ground_truth_map,
                        expected_no_call=is_irrelevance,
                    )
                    if normalised is None:
                        continue
                    rows.append(normalised)
                    count += 1
        except Exception as exc:  # pragma: no cover
            log.warning(
                "[tool-accuracy] Failed to read BFCL split %r: %s.",
                filename,
                exc,
            )
            continue

        if len(rows) >= sample_size:
            break

    return rows[:sample_size]


def _normalise_bfcl_row(
    row: dict[str, Any],
    *,
    ground_truth_map: dict[str, list[str]] | None = None,
    expected_no_call: bool = False,
) -> dict[str, Any] | None:
    """Normalise a raw BFCL v4 row into the same shape as PPB-native cases.

    The v4 schema is fixed:

    * ``question`` is ``[[{"role": "user", "content": "..."}]]`` (the user
      message lives at ``question[0][0]["content"]``).
    * ``function`` is always a list of tool-schema dicts.
    * Ground truth lives in a separate ``possible_answer/<split>`` file and
      is supplied via *ground_truth_map* keyed by ``id``.

    Returns ``None`` (and logs a warning) if the question nesting is shallower
    than expected.  Irrelevance rows return ``answer={}`` and
    ``expected_no_call=True``; the scorer routes them through ``_score_no_call``.
    """
    qid = row.get("id", "")
    question_field = row.get("question")
    try:
        if question_field is None:
            raise TypeError
        content = question_field[0][0]["content"]
    except (TypeError, KeyError, IndexError):
        log.warning(
            "[tool-accuracy] malformed question shape for %r; skipping row.", qid
        )
        return None
    if not isinstance(content, str):
        log.warning(
            "[tool-accuracy] non-string question content for %r; skipping row.", qid
        )
        return None

    function = row.get("function") or []
    if isinstance(function, dict):
        function = [function]
    elif not isinstance(function, list):
        function = []

    if expected_no_call:
        answer: dict[str, Any] = {}
    else:
        gt_list = (ground_truth_map or {}).get(qid) or []
        # BFCL parallel cases have multiple ground-truth calls; PPB scores
        # against the first call (single-call response model).
        answer = _parse_bfcl_ground_truth(gt_list[0]) if gt_list else {}

    return {
        "id": qid,
        "question": content,
        "function": function,
        "answer": answer,
        "expected_no_call": expected_no_call,
    }


def _load_ppb_native() -> list[dict[str, Any]]:
    """Load the PPB-native MCP ground-truth cases."""
    if not PPB_GROUND_TRUTH_PATH.exists():
        log.warning(
            "[tool-accuracy] %s not found; skipping PPB-native cases.",
            PPB_GROUND_TRUTH_PATH.name,
        )
        return []
    with PPB_GROUND_TRUTH_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _render_tool_schemas(function: Any) -> str:
    """Render the ``function`` field as a JSON-pretty list of schemas."""
    if isinstance(function, list):
        schemas = function
    elif isinstance(function, dict):
        schemas = [function]
    else:
        schemas = []
    return "\n".join(json.dumps(s, indent=2) for s in schemas)


def _build_prompt(case: dict[str, Any]) -> str:
    schemas = _render_tool_schemas(case["function"])
    system = SYSTEM_PROMPT_TEMPLATE.format(tool_schemas=schemas)
    return f"{system}\n\nUser: {case['question']}\nAssistant:"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from a model response.

    Tries direct ``json.loads`` first; if that fails, extracts the first
    ``{...}`` block.  Returns ``None`` on hard failure.
    """
    if not text:
        return None
    text = text.strip()
    # Strip common code-fence wrappers.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        obj = json.loads(text)
    except Exception:
        m = _JSON_OBJECT_RE.search(text)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
    return obj if isinstance(obj, dict) else None


# ---------------------------------------------------------------------------
# Schema-aware AST match
# ---------------------------------------------------------------------------


def _expected_schema(function: Any, name: str) -> dict[str, Any]:
    """Return the parameter schema for the tool *name* (or empty dict)."""
    schemas = function if isinstance(function, list) else [function]
    for schema in schemas:
        if isinstance(schema, dict) and schema.get("name") == name:
            return (schema.get("parameters") or {}) if isinstance(schema, dict) else {}
    return {}


def _type_compatible(value: Any, json_type: str | None) -> bool:
    if json_type is None:
        return True
    mapping = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected = mapping.get(json_type)
    if expected is None:
        return True
    # Bool is a subclass of int — guard against a stray True being accepted
    # as a number.
    if json_type in ("number", "integer") and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def _ast_match(
    response: dict[str, Any],
    truth: dict[str, Any],
    schema_params: dict[str, Any],
) -> tuple[bool, bool, bool]:
    """Compare *response* against *truth* under *schema_params*.

    Returns ``(name_match, param_match, hallucinated)``.
    """
    name_match = response.get("name") == truth.get("name")

    resp_args = response.get("arguments") or {}
    truth_args = truth.get("arguments") or {}
    if not isinstance(resp_args, dict):
        return name_match, False, False

    properties = (
        schema_params.get("properties") if isinstance(schema_params, dict) else None
    ) or {}
    required = (
        schema_params.get("required") if isinstance(schema_params, dict) else None
    ) or list(truth_args.keys())

    # All required params (per ground truth + schema) must be present and
    # match the ground-truth value where one is given.
    param_match = True
    for key in required:
        if key not in resp_args:
            param_match = False
            break
        if key in truth_args and resp_args[key] != truth_args[key]:
            param_match = False
            break
        json_type = (properties.get(key) or {}).get("type") if properties else None
        if not _type_compatible(resp_args[key], json_type):
            param_match = False
            break

    # Hallucination: keys present in response but absent from the schema.
    hallucinated = False
    if properties:
        hallucinated = any(k not in properties for k in resp_args.keys())

    return name_match, param_match, hallucinated


# ---------------------------------------------------------------------------
# Single-case evaluation
# ---------------------------------------------------------------------------


# Substrings that strongly suggest the model attempted a tool / function call.
# Used by ``_score_no_call`` to score irrelevance cases (where the correct
# behaviour is to abstain from calling any tool).
_TOOL_CALL_MARKERS: tuple[str, ...] = ("(", "function_call", "tool_call", "<tool")


def _score_no_call(model_response: str) -> bool:
    """Return ``True`` when *model_response* does NOT look like a tool call.

    Used for the BFCL ``irrelevance`` split where the correct behaviour is to
    refuse and answer in plain prose.  This is a deliberately simple heuristic
    that flags the response as a tool-call attempt if it contains any of:

    * ``"("``           — bare Python-style call ``foo(...)``
    * ``"function_call"`` / ``"tool_call"`` — common JSON-wrapper keys
    * ``"<tool"``       — XML-style ``<tool_call>`` openers used by some models

    False-positive risk: prose containing an open paren (e.g. *"(per the user)"*)
    is incorrectly flagged as a call.  Treat ``no_call_accuracy`` as a coarse
    floor on abstention skill, not a rigorous metric.  Empty / whitespace-only
    responses are treated as a successful abstention.
    """
    if not model_response:
        return True
    text = model_response.strip()
    if not text:
        return True
    return not any(marker in text for marker in _TOOL_CALL_MARKERS)


def _evaluate_case(
    llm: Any,
    case: dict[str, Any],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Run *case* through the model and score it.

    Irrelevance cases (``case["expected_no_call"] is True``) are routed
    through :func:`_score_no_call` and their result populates ``no_call_match``;
    they intentionally do not contribute to ``tool_match`` / ``param_match``
    / ``parse_ok`` so positive-case metrics remain interpretable.
    """
    prompt = _build_prompt(case)
    t0 = time.time()
    out = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    elapsed = time.time() - t0

    try:
        text = out["choices"][0]["text"]
    except Exception:
        text = ""

    if case.get("expected_no_call"):
        return {
            "expected_no_call": True,
            "no_call_match": _score_no_call(text),
            "parse_ok": False,
            "tool_match": False,
            "param_match": False,
            "hallucinated": False,
            "elapsed_s": elapsed,
            "raw_response": text,
        }

    parsed = _parse_response(text)
    truth = case.get("answer") or {}
    schema_params = _expected_schema(case.get("function"), truth.get("name", ""))

    if parsed is None:
        return {
            "expected_no_call": False,
            "no_call_match": False,
            "parse_ok": False,
            "tool_match": False,
            "param_match": False,
            "hallucinated": False,
            "elapsed_s": elapsed,
            "raw_response": text,
        }

    name_match, param_match, hallucinated = _ast_match(parsed, truth, schema_params)
    return {
        "expected_no_call": False,
        "no_call_match": False,
        "parse_ok": True,
        "tool_match": bool(name_match),
        "param_match": bool(param_match),
        "hallucinated": bool(hallucinated),
        "elapsed_s": elapsed,
        "raw_response": text,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_tool_accuracy(
    llm: Any,
    model_config: dict[str, Any] | None = None,
    suite_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full tool-accuracy evaluation against *llm*.

    *model_config* is currently unused but accepted for symmetry with the
    other phase entry points.  *suite_config* may set
    ``tool_accuracy_sample_size`` (BFCL only) and ``max_tokens``.
    """
    _ = model_config  # reserved for future use
    suite_config = suite_config or {}
    sample_size = int(
        suite_config.get("tool_accuracy_sample_size", DEFAULT_BFCL_SAMPLE_SIZE)
    )
    max_tokens = int(suite_config.get("tool_accuracy_max_tokens", DEFAULT_MAX_TOKENS))

    bfcl_cases = _load_bfcl(sample_size)
    ppb_cases = _load_ppb_native()
    cases = bfcl_cases + ppb_cases
    total = len(cases)

    print(
        f"\n[tool-accuracy] {len(bfcl_cases)} BFCL + {len(ppb_cases)} PPB-native "
        f"= {total} case(s)",
        flush=True,
    )

    if total == 0:
        return {
            "tool_selection_accuracy": None,
            "parameter_accuracy": None,
            "parameter_hallucination_rate": None,
            "parse_success_rate": None,
            "no_call_accuracy": None,
            "overall_tool_accuracy": None,
            "n_cases": 0,
        }

    n_parse_ok = 0
    n_tool_match = 0
    n_param_match = 0
    n_hallucinated = 0
    no_call_scores: list[bool] = []
    n_positive = 0  # cases that contribute to tool/param/parse metrics

    for i, case in enumerate(cases, start=1):
        try:
            result = _evaluate_case(llm, case, max_tokens=max_tokens)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("[tool-accuracy] case %d failed: %s", i, exc)
            result = {
                "expected_no_call": bool(case.get("expected_no_call")),
                "no_call_match": False,
                "parse_ok": False,
                "tool_match": False,
                "param_match": False,
                "hallucinated": False,
                "elapsed_s": 0.0,
            }
        if result.get("expected_no_call"):
            no_call_scores.append(bool(result["no_call_match"]))
            print(
                f"  ✓ [{i}/{total}] no_call_match={result['no_call_match']} "
                f"(irrelevance, {result['elapsed_s']:.1f}s)",
                flush=True,
            )
            continue

        n_positive += 1
        n_parse_ok += int(result["parse_ok"])
        n_tool_match += int(result["tool_match"])
        n_param_match += int(result["param_match"])
        n_hallucinated += int(result["hallucinated"])
        print(
            f"  ✓ [{i}/{total}] tool_match={result['tool_match']} "
            f"param_match={result['param_match']} "
            f"hallucination={result['hallucinated']} "
            f"({result['elapsed_s']:.1f}s)",
            flush=True,
        )

    if n_positive > 0:
        tool_selection_accuracy = n_tool_match / n_positive
        parameter_accuracy = n_param_match / n_positive
        parameter_hallucination_rate = n_hallucinated / n_positive
        parse_success_rate = n_parse_ok / n_positive
        # Geometric mean: collapses to 0 if either selection OR parameter accuracy
        # is 0. This is intentional — a model that can't reliably select the right
        # tool OR correctly parameterise it is not usable for tool calling, regardless
        # of the other dimension. Use arithmetic mean only if you want partial credit.
        # Note: ``no_call_accuracy`` is NOT folded into this geometric mean — it
        # measures a different (negative) capability and is reported alongside.
        overall_tool_accuracy = math.sqrt(tool_selection_accuracy * parameter_accuracy)
    else:
        tool_selection_accuracy = None
        parameter_accuracy = None
        parameter_hallucination_rate = None
        parse_success_rate = None
        overall_tool_accuracy = None

    no_call_accuracy = (
        float(sum(no_call_scores) / len(no_call_scores)) if no_call_scores else None
    )

    print(
        "[tool-accuracy] "
        f"tool_selection={tool_selection_accuracy if tool_selection_accuracy is None else f'{tool_selection_accuracy:.3f}'} "
        f"param_accuracy={parameter_accuracy if parameter_accuracy is None else f'{parameter_accuracy:.3f}'} "
        f"hallucination={parameter_hallucination_rate if parameter_hallucination_rate is None else f'{parameter_hallucination_rate:.3f}'} "
        f"parse_success={parse_success_rate if parse_success_rate is None else f'{parse_success_rate:.3f}'} "
        f"no_call={no_call_accuracy if no_call_accuracy is None else f'{no_call_accuracy:.3f}'} "
        f"overall={overall_tool_accuracy if overall_tool_accuracy is None else f'{overall_tool_accuracy:.3f}'}",
        flush=True,
    )

    return {
        "tool_selection_accuracy": tool_selection_accuracy,
        "parameter_accuracy": parameter_accuracy,
        "parameter_hallucination_rate": parameter_hallucination_rate,
        "parse_success_rate": parse_success_rate,
        "no_call_accuracy": no_call_accuracy,
        "overall_tool_accuracy": overall_tool_accuracy,
        "n_cases": total,
        "n_bfcl": len(bfcl_cases),
        "n_ppb_native": len(ppb_cases),
    }


# ---------------------------------------------------------------------------
# Convenience helper for ppb.py integration
# ---------------------------------------------------------------------------


def run_tool_accuracy_for_model(
    model_path: Path,
    *,
    suite_config: dict[str, Any] | None = None,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    verbose: bool = False,
) -> dict[str, Any]:
    """Load *model_path* with llama-cpp-python and run tool-accuracy.

    Mirrors ``ppb_context_rot.run_context_rot_for_model`` so ``ppb.py`` can
    invoke the phase without managing the ``llama_cpp.Llama`` lifecycle.
    """
    try:
        from llama_cpp import Llama  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ppb_tool_accuracy requires llama-cpp-python. "
            "Install with: pip install llama-cpp-python"
        ) from exc

    print(
        f"[tool-accuracy] loading {model_path.name} (n_ctx={n_ctx}, "
        f"n_gpu_layers={n_gpu_layers})",
        flush=True,
    )
    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )
    try:
        return run_tool_accuracy(
            llm,
            model_config=None,
            suite_config=suite_config or {},
        )
    finally:
        del llm
