"""
PPB — Tool-Call Accuracy (BFCL + PPB-native).

Measures whether a model produces structurally valid, schema-correct tool
calls.  Uses the Berkeley Function Calling Leaderboard (BFCL) ``simple`` and
``multiple_function`` splits as the primary evaluation set, supplemented by
a PPB-native MCP ground-truth set covering the four ``ppb-mcp`` tools.

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
BFCL_SPLITS = ("simple", "multiple_function")

DEFAULT_BFCL_SAMPLE_SIZE = 100
DEFAULT_MAX_TOKENS = 256

PPB_GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "ppb_mcp_ground_truth.json"

SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful assistant. You have access to the following tools. "
    "When you want to call a tool, respond ONLY with a JSON object with "
    "keys `name` (string) and `arguments` (object). Do not add any prose.\n\n"
    "Available tools:\n{tool_schemas}"
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_bfcl(sample_size: int) -> list[dict[str, Any]]:
    """Load BFCL simple + multiple_function rows, capped at *sample_size*.

    Falls back to an empty list (with a warning) if the dataset cannot be
    fetched — the PPB-native ground truth still runs.
    """
    if sample_size <= 0:
        return []
    try:
        from datasets import load_dataset
    except ImportError:  # pragma: no cover
        log.warning(
            "[tool-accuracy] `datasets` not installed; skipping BFCL evaluation."
        )
        return []

    rows: list[dict[str, Any]] = []
    per_split = max(1, sample_size // len(BFCL_SPLITS))
    for split in BFCL_SPLITS:
        try:
            ds = load_dataset(BFCL_REPO, split=split, trust_remote_code=True)
        except Exception as exc:  # pragma: no cover — network/dataset issues
            log.warning(
                "[tool-accuracy] WARNING: Failed to load BFCL split %r: %s. "
                "Falling back to PPB-native cases only (20 cases). "
                "Set BFCL_TRUST_REMOTE_CODE=1 to enable trust_remote_code if needed.",
                split,
                exc,
            )
            continue
        for i, row in enumerate(ds):
            if i >= per_split:
                break
            rows.append(_normalise_bfcl_row(row))
        if len(rows) >= sample_size:
            break
    return rows[:sample_size]


def _normalise_bfcl_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalise a raw BFCL row into the same shape as PPB-native cases.

    BFCL stores ``function`` as either a JSON string or an already-parsed
    dict, and ``answer`` similarly.  We normalise both to dicts here.
    """
    func = row.get("function")
    if isinstance(func, str):
        try:
            func = json.loads(func)
        except Exception:
            func = {}
    # BFCL ``function`` can be a list of schemas (multiple_function); we keep
    # the list shape if so and let the prompt builder render all of them.
    answer = row.get("answer")
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except Exception:
            answer = {}
    # BFCL answer schema: typically ``{"name": ..., "arguments": {...}}`` or
    # a list with one such object — normalise to a single dict.
    if isinstance(answer, list) and answer:
        answer = answer[0]
    return {
        "question": row.get("question", ""),
        "function": func,
        "answer": answer if isinstance(answer, dict) else {},
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


def _evaluate_case(
    llm: Any,
    case: dict[str, Any],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Run *case* through the model and score it."""
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

    parsed = _parse_response(text)
    truth = case.get("answer") or {}
    schema_params = _expected_schema(case.get("function"), truth.get("name", ""))

    if parsed is None:
        return {
            "parse_ok": False,
            "tool_match": False,
            "param_match": False,
            "hallucinated": False,
            "elapsed_s": elapsed,
            "raw_response": text,
        }

    name_match, param_match, hallucinated = _ast_match(parsed, truth, schema_params)
    return {
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
            "overall_tool_accuracy": None,
            "n_cases": 0,
        }

    n_parse_ok = 0
    n_tool_match = 0
    n_param_match = 0
    n_hallucinated = 0

    for i, case in enumerate(cases, start=1):
        try:
            result = _evaluate_case(llm, case, max_tokens=max_tokens)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("[tool-accuracy] case %d failed: %s", i, exc)
            result = {
                "parse_ok": False,
                "tool_match": False,
                "param_match": False,
                "hallucinated": False,
                "elapsed_s": 0.0,
            }
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

    tool_selection_accuracy = n_tool_match / total
    parameter_accuracy = n_param_match / total
    parameter_hallucination_rate = n_hallucinated / total
    parse_success_rate = n_parse_ok / total
    # Geometric mean: collapses to 0 if either selection OR parameter accuracy
    # is 0. This is intentional — a model that can't reliably select the right
    # tool OR correctly parameterise it is not usable for tool calling, regardless
    # of the other dimension. Use arithmetic mean only if you want partial credit.
    overall_tool_accuracy = math.sqrt(tool_selection_accuracy * parameter_accuracy)

    print(
        f"[tool-accuracy] tool_selection={tool_selection_accuracy:.3f} "
        f"param_accuracy={parameter_accuracy:.3f} "
        f"hallucination={parameter_hallucination_rate:.3f} "
        f"parse_success={parse_success_rate:.3f} "
        f"overall={overall_tool_accuracy:.3f}",
        flush=True,
    )

    return {
        "tool_selection_accuracy": tool_selection_accuracy,
        "parameter_accuracy": parameter_accuracy,
        "parameter_hallucination_rate": parameter_hallucination_rate,
        "parse_success_rate": parse_success_rate,
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
        from llama_cpp import Llama
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
