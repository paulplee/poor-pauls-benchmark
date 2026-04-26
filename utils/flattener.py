"""Normalize deeply nested benchmark rows into flat, Arrow-friendly dicts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

# ---------------------------------------------------------------------------
# Strict unified schema — every flat row will contain exactly these keys.
# Keys not applicable to a given runner are set to None, which serialises
# as ``null`` in JSON and as an empty cell in CSV.  This guarantees that
# Apache Arrow / Hugging Face datasets see identical columns across files.
#
# COLUMN_ORDER defines the canonical output order, intentionally optimised
# for human readability in the Hugging Face dataset viewer:
#   - benchmark identity and configuration first
#   - performance metrics next
#   - OS / hardware context after
#   - provenance / fingerprint fields last
# ---------------------------------------------------------------------------

COLUMN_ORDER: list[str] = [
    # Composable-schema join key (added 2026-04 — nullable on legacy rows).
    # Together (gpu_name, model_name, quantization, run_type) are the
    # primary key used by downstream consumers (ppb-mcp, poorpaul.dev) to
    # JOIN quantitative- and qualitative-only rows into a single profile.
    "run_type",  # "all" | "quantitative" | "qualitative"
    "model_name",  # alias of model_base — duplicated for the composable key
    "quantization",  # alias of quant     — duplicated for the composable key
    # Benchmark identity
    "model",
    "model_base",
    "quant",
    "model_org",
    "model_repo",
    "runner_type",
    # LLM engine (inference backend)
    "llm_engine_name",
    "llm_engine_version",
    # Hardware
    "gpu_name",
    "gpu_vram_gb",
    "gpu_driver",
    "gpu_count",
    "gpu_names",
    "gpu_total_vram_gb",
    "backends",
    "cpu_model",
    # Configuration
    "n_ctx",
    "n_batch",
    "split_mode",
    "tensor_split",
    "concurrent_users",
    # Workload
    "task_type",
    "prompt_dataset",
    "num_prompts",
    "n_predict",
    # Performance — raw speed
    "throughput_tok_s",
    # Quantitative composable block (nullable on qualitative-only rows).
    "vram_used_gb",
    "vram_cliff_tokens",
    "tokens_per_sec_prompt",
    "tokens_per_sec_generation",
    # Performance — power efficiency
    "avg_power_w",
    "max_power_w",
    # Performance — thermal
    "avg_gpu_temp_c",
    "max_gpu_temp_c",
    "avg_cpu_temp_c",
    "max_cpu_temp_c",
    "avg_fan_speed_rpm",
    "max_fan_speed_rpm",
    # Performance — user experience
    "avg_ttft_ms",
    "p50_ttft_ms",
    "p99_ttft_ms",
    "avg_itl_ms",
    "p50_itl_ms",
    "p99_itl_ms",
    # Performance — quality
    "quality_score",
    # Context-Rot (Semantic NIAH)
    "context_rot_score",
    "context_rot_accuracy_by_length",
    "context_rot_accuracy_by_depth",
    # Future qualitative blocks (nullable placeholders — populated by
    # forthcoming phases: tool_accuracy, answer_quality, multiturn).
    "tool_selection_accuracy",
    "parameter_accuracy",
    "parameter_hallucination_rate",
    "parse_success_rate",
    "overall_tool_accuracy",
    "faithfulness_mean",
    "faithfulness_std",
    "answer_relevancy_mean",
    "coherence_mean",
    "quality_composite_score",
    "memory_accuracy",
    "mt_bench_score",
    # Composable nested blocks (opaque JSON dicts written verbatim to
    # JSONL; CSV writer JSON-stringifies them).  Downstream consumers
    # (ppb-mcp, poorpaul.dev) read these to assemble model profiles by
    # JOINing on (gpu_name, model_name, quantization, run_type).
    "qualitative",
    "quantitative",
    # Per-row meta block — opaque JSON dict carrying reproducibility
    # hints (e.g. ``quality_prompts_cache_hash``).  Nullable.
    "meta",
    # OS / system context
    "os_system",
    "os_release",
    "os_machine",
    "os_distro",
    "os_distro_version",
    "cpu_cores",
    "ram_total_gb",
    # Submission metadata
    "submitter",
    "timestamp",
    "submitted_at",
    # Provenance / dedup
    "schema_version",
    "benchmark_version",
    "suite_run_id",
    "submission_id",
    "row_id",
    "machine_fingerprint",
    "run_fingerprint",
    "result_fingerprint",
    "source_file_sha256",
    # Extensibility
    "tags",
]

# raw_payload is included in the row for internal use / HF uploads,
# but excluded from CSV output (not in COLUMN_ORDER).
MASTER_SCHEMA: dict[str, None] = dict.fromkeys(COLUMN_ORDER + ["raw_payload"])

# ---------------------------------------------------------------------------
# Provenance / fingerprint helpers
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = "0.1.0"


def _get_benchmark_version() -> str:
    """Return the installed package version, or ``'unknown'``."""
    try:
        return importlib.metadata.version("poor-pauls-benchmark")
    except importlib.metadata.PackageNotFoundError:
        pass
    # Fallback: read version from pyproject.toml when not pip-installed
    try:
        import tomllib

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if pyproject.is_file():
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            return data.get("project", {}).get("version", "unknown")
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _norm(value: Any) -> Any:
    """Normalize a value for deterministic hashing.

    Strings are trimmed and lowercased.  Numerics and ``None`` pass through
    unchanged so that ``json.dumps`` produces a stable canonical form.
    """
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _sha256_dict(d: dict[str, Any]) -> str:
    """Canonical SHA-256 hex digest of a dict (sorted keys, deterministic)."""
    payload = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _compute_machine_fingerprint(flat: dict[str, Any]) -> str:
    """SHA-256 of stable hardware-identity fields."""
    return _sha256_dict(
        {
            "os_machine": _norm(flat.get("os_machine")),
            "cpu_model": _norm(flat.get("cpu_model")),
            "ram_total_gb": flat.get("ram_total_gb"),
            "gpu_name": _norm(flat.get("gpu_name")),
            "gpu_vram_gb": flat.get("gpu_vram_gb"),
            "os_release": _norm(flat.get("os_release")),
        }
    )


def _compute_run_fingerprint(
    flat: dict[str, Any],
    machine_fp: str,
    benchmark_version: str,
) -> str:
    """SHA-256 of benchmark-identity fields (same config on same machine)."""
    return _sha256_dict(
        {
            "runner_type": _norm(flat.get("runner_type")),
            "llm_engine_name": _norm(flat.get("llm_engine_name")),
            "model": _norm(flat.get("model")),
            "n_ctx": flat.get("n_ctx"),
            "n_batch": flat.get("n_batch"),
            "concurrent_users": flat.get("concurrent_users"),
            "machine_fingerprint": machine_fp,
            "benchmark_version": _norm(benchmark_version),
        }
    )


def _compute_result_fingerprint(
    flat: dict[str, Any],
    run_fp: str,
) -> str:
    """SHA-256 of run identity + measured metrics — uniquely identifies one result."""
    return _sha256_dict(
        {
            "run_fingerprint": run_fp,
            "throughput_tok_s": flat.get("throughput_tok_s"),
            "avg_ttft_ms": flat.get("avg_ttft_ms"),
            "avg_itl_ms": flat.get("avg_itl_ms"),
            "p99_ttft_ms": flat.get("p99_ttft_ms"),
            "p99_itl_ms": flat.get("p99_itl_ms"),
        }
    )


def compute_file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file (streamed, memory-safe)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(8192):
            h.update(chunk)
    return h.hexdigest()


def _stamp_provenance(flat: dict[str, Any]) -> None:
    """Set provenance fields on a flat row (in-place).

    ``submission_id``, ``submitted_at``, and ``source_file_sha256`` are left
    as ``None`` — they are batch-level values set by the caller.
    """
    bv = _get_benchmark_version()
    machine_fp = _compute_machine_fingerprint(flat)
    run_fp = _compute_run_fingerprint(flat, machine_fp, bv)
    result_fp = _compute_result_fingerprint(flat, run_fp)

    flat["schema_version"] = _SCHEMA_VERSION
    flat["benchmark_version"] = bv
    flat["row_id"] = uuid.uuid4().hex
    flat["machine_fingerprint"] = machine_fp
    flat["run_fingerprint"] = run_fp
    flat["result_fingerprint"] = result_fp


# ---------------------------------------------------------------------------
# Model filename parsing
# ---------------------------------------------------------------------------

# Regex to capture a quantisation suffix before .gguf.
# e.g. "Qwen3.5-9B-Q8_0.gguf" → separator="-", quant="Q8_0"
_QUANT_SUFFIX_RE = re.compile(r"[-_]([QqFfIiBb][A-Za-z0-9_]*)\.gguf$")


def _parse_model_provenance(model: str | None) -> tuple[str | None, str | None]:
    """Extract (model_org, model_repo) from a Hugging Face model path.

    HF model paths follow the pattern ``org/repo/filename.gguf``.
    Local paths (starting with ``/``, ``.``, or ``~``) return ``(None, None)``.

    Examples::

        >>> _parse_model_provenance("unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q4_K_M.gguf")
        ('unsloth', 'unsloth/Qwen3.5-2B-GGUF')
        >>> _parse_model_provenance("/Volumes/DATA/models/model.gguf")
        (None, None)
        >>> _parse_model_provenance(None)
        (None, None)
    """
    if not model:
        return None, None
    if model.startswith("/") or model.startswith(".") or model.startswith("~"):
        return None, None
    parts = model.split("/")
    if len(parts) >= 3:
        return parts[0], f"{parts[0]}/{parts[1]}"
    return None, None


def _parse_model_filename(filename: str | None) -> tuple[str | None, str | None]:
    """Extract (model_base, quant) from a GGUF filename.

    Examples::

        >>> _parse_model_filename("Qwen3.5-9B-Q8_0.gguf")
        ('Qwen3.5-9B', 'Q8_0')
        >>> _parse_model_filename("my-model.gguf")
        ('my-model', None)
        >>> _parse_model_filename(None)
        (None, None)
    """
    if not filename:
        return None, None
    m = _QUANT_SUFFIX_RE.search(filename)
    if m:
        base = filename[: m.start()]
        quant = m.group(1)
        return (base or None), quant
    # No quant suffix — just drop .gguf
    base = filename.removesuffix(".gguf")
    return (base or None), None


def _enrich_backends(backends: str | None, cuda_version: str | None) -> str | None:
    """Append CUDA version to backends string when available.

    ``"CUDA"`` + ``"13.0"`` → ``"CUDA 13.0"``
    """
    if not backends:
        return backends
    if cuda_version and "CUDA" in backends and cuda_version not in backends:
        return backends.replace("CUDA", f"CUDA {cuda_version}")
    return backends


def _new_row() -> dict[str, Any]:
    """Return a fresh copy of the master schema (all values ``None``)."""
    return dict(MASTER_SCHEMA)


def flatten_benchmark_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one raw JSONL row into one or more flat dictionaries.

    For ``llama-bench`` rows the ``results`` list is *exploded* — one flat
    row per result item (typically prompt-processing and generation).  All
    other runner types produce exactly one flat row.

    The returned dicts have a strict, predictable set of keys suitable for
    ``csv.DictWriter`` or Apache Arrow schema inference.  The original
    nested row is preserved verbatim in the ``raw_payload`` column (as a
    JSON string).
    """
    runner_type = row.get("runner_type", "")
    results = row.get("results")

    envelope = _extract_envelope(row)
    hw_fields, cuda_version = _extract_hardware(row.get("hardware") or {})
    raw_payload = json.dumps(row, default=str)

    if runner_type == "llama-bench" and isinstance(results, list):
        rows = _flatten_llama_bench(
            envelope, hw_fields, cuda_version, results, raw_payload
        )
    elif runner_type == "llama-server" and isinstance(results, dict):
        rows = [
            _flatten_llama_server(
                envelope, hw_fields, cuda_version, results, raw_payload
            )
        ]
    elif runner_type == "llama-server-loadtest" and isinstance(results, dict):
        rows = [
            _flatten_llama_server_loadtest(envelope, hw_fields, results, raw_payload)
        ]
    elif runner_type == "context-rot" and isinstance(results, dict):
        flat = _new_row()
        flat.update(envelope)
        flat.update(hw_fields)
        # Serialize the per-length and per-depth dicts as JSON strings so
        # they fit cleanly into a single CSV cell / Arrow string column.
        by_len = results.get("context_rot_accuracy_by_length")
        by_depth = results.get("context_rot_accuracy_by_depth")
        flat["context_rot_score"] = results.get("context_rot_score")
        flat["context_rot_accuracy_by_length"] = (
            json.dumps(by_len) if by_len is not None else None
        )
        flat["context_rot_accuracy_by_depth"] = (
            json.dumps(by_depth) if by_depth is not None else None
        )
        flat["raw_payload"] = raw_payload
        _stamp_provenance(flat)
        rows = [flat]
    elif runner_type == "tool-accuracy" and isinstance(results, dict):
        flat = _new_row()
        flat.update(envelope)
        flat.update(hw_fields)
        flat["tool_selection_accuracy"] = results.get("tool_selection_accuracy")
        flat["parameter_accuracy"] = results.get("parameter_accuracy")
        flat["parameter_hallucination_rate"] = results.get(
            "parameter_hallucination_rate"
        )
        flat["parse_success_rate"] = results.get("parse_success_rate")
        flat["overall_tool_accuracy"] = results.get("overall_tool_accuracy")
        flat["raw_payload"] = raw_payload
        _stamp_provenance(flat)
        rows = [flat]
    elif runner_type == "answer-quality" and isinstance(results, dict):
        flat = _new_row()
        flat.update(envelope)
        flat.update(hw_fields)
        flat["faithfulness_mean"] = results.get("faithfulness_mean")
        flat["faithfulness_std"] = results.get("faithfulness_std")
        flat["answer_relevancy_mean"] = results.get("answer_relevancy_mean")
        flat["coherence_mean"] = results.get("coherence_mean")
        flat["quality_composite_score"] = results.get("quality_composite_score")
        flat["raw_payload"] = raw_payload
        _stamp_provenance(flat)
        rows = [flat]
    else:
        # Unknown / unsupported runner — emit one row with Nones
        flat = _new_row()
        flat.update(envelope)
        flat.update(hw_fields)
        flat["raw_payload"] = raw_payload
        _stamp_provenance(flat)
        rows = [flat]

    # Return rows with all MASTER_SCHEMA keys in canonical order.
    # raw_payload is included; COLUMN_ORDER (used by _write_csv) excludes it.
    schema_keys = list(MASTER_SCHEMA)
    return [{k: r.get(k) for k in schema_keys} for r in rows]


# -- helpers ---------------------------------------------------------------


def _extract_envelope(row: dict[str, Any]) -> dict[str, Any]:
    """Pull the common top-level fields shared by every runner."""
    model = row.get("model")
    fname = PurePosixPath(model).name if model else None
    base, quant = _parse_model_filename(fname)
    model_org, model_repo = _parse_model_provenance(model)
    # Promote workload fields from results when present at top level
    results = row.get("results") or {}
    results_dict = results if isinstance(results, dict) else {}

    return {
        "timestamp": row.get("timestamp"),
        "runner_type": row.get("runner_type"),
        # Composable-schema join key.  ``run_type`` distinguishes
        # quantitative-only, qualitative-only, and combined runs so that
        # downstream consumers can JOIN them by (gpu_name, model_name,
        # quantization).  Defaults to "quantitative" for legacy rows that
        # don't set it explicitly (anything driven by `ppb sweep` or the
        # original `ppb all`).
        "run_type": row.get("run_type")
        or (
            "qualitative"
            if row.get("runner_type")
            in ("context-rot", "tool-accuracy", "answer-quality")
            else "quantitative"
        ),
        "model": model,
        "model_base": base,
        "quant": quant,
        "model_name": base,  # alias for the composable key
        "quantization": quant,  # alias for the composable key
        "model_org": model_org,
        "model_repo": model_repo,
        # LLM engine (inference backend)
        "llm_engine_name": row.get("llm_engine_name"),
        "llm_engine_version": row.get("llm_engine_version"),
        "n_ctx": row.get("n_ctx"),
        "n_batch": row.get("n_batch"),
        "split_mode": row.get("split_mode"),
        "tensor_split": row.get("tensor_split"),
        "concurrent_users": row.get("concurrent_users"),
        # Workload
        "task_type": row.get("task_type"),
        "prompt_dataset": row.get("prompt_dataset"),
        "num_prompts": row.get("num_prompts")
        or results_dict.get("num_prompts_attempted"),
        "n_predict": row.get("n_predict") or results_dict.get("n_predict"),
        # Quality
        "quality_score": row.get("quality_score"),
        # Power draw measured during the run (Watts).
        # Source: NVIDIA NVML (GPU board), Linux RAPL (CPU package),
        # or macOS powermetrics (SoC). None when unavailable.
        "avg_power_w": row.get("avg_power_w"),
        "max_power_w": row.get("max_power_w"),
        # Thermal readings sampled during the run.
        "avg_gpu_temp_c": row.get("avg_gpu_temp_c"),
        "max_gpu_temp_c": row.get("max_gpu_temp_c"),
        "avg_cpu_temp_c": row.get("avg_cpu_temp_c"),
        "max_cpu_temp_c": row.get("max_cpu_temp_c"),
        "avg_fan_speed_rpm": row.get("avg_fan_speed_rpm"),
        "max_fan_speed_rpm": row.get("max_fan_speed_rpm"),
        "suite_run_id": row.get("suite_run_id"),
        # Extensibility
        "tags": row.get("tags"),
    }


def _extract_hardware(hw: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Flatten the nested ``hardware`` dictionary.

    Returns a tuple of (schema-conformant fields, cuda_version).
    ``cuda_version`` is kept separate because it is used to enrich
    the ``backends`` string but is not a published column itself.
    """
    os_info = hw.get("os") or {}
    cpu = hw.get("cpu") or {}
    ram = hw.get("ram") or {}
    gpus = hw.get("gpus") or []
    gpu0: dict[str, Any] = gpus[0] if gpus else {}

    fields = {
        "os_system": os_info.get("system"),
        "os_release": os_info.get("release"),
        "os_machine": os_info.get("machine"),
        "os_distro": os_info.get("distro"),
        "os_distro_version": os_info.get("distro_version"),
        "cpu_model": cpu.get("model"),
        "cpu_cores": cpu.get("cores"),
        "ram_total_gb": ram.get("total_gb"),
        "gpu_name": gpu0.get("name"),
        "gpu_vram_gb": gpu0.get("vram_total_gb"),
        "gpu_driver": gpu0.get("driver"),
        # Multi-GPU fields (backward-compat: single-GPU runs still populate these)
        "gpu_count": len(gpus),
        "gpu_names": ", ".join(g["name"] for g in gpus if g.get("name")) or None,
        "gpu_total_vram_gb": (
            round(
                sum(
                    g["vram_total_gb"]
                    for g in gpus
                    if g.get("vram_total_gb") is not None
                ),
                1,
            )
            if gpus and any(g.get("vram_total_gb") is not None for g in gpus)
            else None
        ),
    }
    return fields, gpu0.get("cuda_version")


def _flatten_llama_bench(
    envelope: dict[str, Any],
    hw_fields: dict[str, Any],
    cuda_version: str | None,
    results: list[dict[str, Any]],
    raw_payload: str,
) -> list[dict[str, Any]]:
    """Explode llama-bench results — one flat row per result item."""
    rows: list[dict[str, Any]] = []
    for item in results:
        flat = _new_row()
        flat.update(envelope)
        flat.update(hw_fields)
        # llama-bench embeds gpu_info inside each result item (hardware.gpus
        # can be empty), so prefer the result-level value as a fallback.
        if flat.get("gpu_name") is None and item.get("gpu_info"):
            flat["gpu_name"] = item.get("gpu_info")
        # Enrich backends with CUDA version when available
        flat["backends"] = _enrich_backends(item.get("backends"), cuda_version)
        flat["throughput_tok_s"] = item.get("avg_ts")
        flat["raw_payload"] = raw_payload
        _stamp_provenance(flat)
        rows.append(flat)
    return rows


def _flatten_llama_server(
    envelope: dict[str, Any],
    hw_fields: dict[str, Any],
    cuda_version: str | None,
    results: dict[str, Any],
    raw_payload: str,
) -> dict[str, Any]:
    """Flatten a single llama-server results dict."""
    flat = _new_row()
    flat.update(envelope)
    flat.update(hw_fields)
    # Infer compute backend from hardware (llama-bench emits this directly;
    # llama-server does not, so we derive it from the detected platform).
    if flat.get("os_system") == "Darwin":
        flat["backends"] = flat.get("gpu_driver") or "Metal"
    elif flat.get("gpu_name"):
        flat["backends"] = f"CUDA {cuda_version}" if cuda_version else "CUDA"
    flat["throughput_tok_s"] = results.get("throughput_tok_s")
    flat["avg_ttft_ms"] = results.get("avg_ttft_ms")
    flat["p50_ttft_ms"] = results.get("p50_ttft_ms")
    flat["p99_ttft_ms"] = results.get("p99_ttft_ms")
    flat["avg_itl_ms"] = results.get("avg_itl_ms")
    flat["p50_itl_ms"] = results.get("p50_itl_ms")
    flat["p99_itl_ms"] = results.get("p99_itl_ms")
    # concurrent_users may be nested inside the results dict for server runs
    if flat.get("concurrent_users") is None:
        flat["concurrent_users"] = results.get("concurrent_users")
    flat["raw_payload"] = raw_payload
    _stamp_provenance(flat)
    return flat


def _flatten_llama_server_loadtest(
    envelope: dict[str, Any],
    hw_fields: dict[str, Any],
    results: dict[str, Any],
    raw_payload: str,
) -> dict[str, Any]:
    """Flatten a single llama-server-loadtest results dict."""
    flat = _new_row()
    flat.update(envelope)
    flat.update(hw_fields)
    # max_sustainable_users is not in the public schema; derive concurrent_users
    # from the highest tested level in the concurrency curve instead.
    curve = results.get("concurrency_curve") or []
    best_throughput = None
    for level in curve:
        val = level.get("aggregate_throughput_tok_s")
        if val is not None:
            best_throughput = val
    flat["throughput_tok_s"] = best_throughput
    flat["raw_payload"] = raw_payload
    _stamp_provenance(flat)
    return flat
