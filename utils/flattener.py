"""Normalize deeply nested benchmark rows into flat, Arrow-friendly dicts."""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from typing import Any

# ---------------------------------------------------------------------------
# Strict unified schema — every flat row will contain exactly these keys.
# Keys not applicable to a given runner are set to None, which serialises
# as ``null`` in JSON and as an empty cell in CSV.  This guarantees that
# Apache Arrow / Hugging Face datasets see identical columns across files.
#
# Column order matters — it defines the CSV column order and the HF display.
# Priority columns are listed first.
# ---------------------------------------------------------------------------

MASTER_SCHEMA: dict[str, None] = {
    # Identity
    "model_id": None,
    # Hardware
    "gpu_name": None,
    "gpu_vram_gb": None,
    "gpu_driver": None,
    "backends": None,
    "cpu_model": None,
    # Configuration
    "n_ctx": None,
    # Performance — raw speed
    "throughput_tok_s": None,
    # Performance — user experience
    "avg_ttft_ms": None,
    "p99_ttft_ms": None,
    # Metadata
    "timestamp": None,
    "runner_type": None,
    "n_batch": None,
    "submitter": None,
    "os_system": None,
    "os_release": None,
    "os_machine": None,
    "cpu_cores": None,
    "ram_total_gb": None,
    "concurrent_users": None,
    "p50_ttft_ms": None,
    "avg_itl_ms": None,
    "p50_itl_ms": None,
    "p99_itl_ms": None,
    "max_sustainable_users": None,
    # Raw
    "raw_payload": None,
}

# Regex to strip quantisation suffix + .gguf extension from a model filename.
# e.g. "Qwen3.5-9B-Q8_0.gguf" → "Qwen3.5-9B"
_QUANT_SUFFIX_RE = re.compile(r"[-_][QqFfIiBb][A-Za-z0-9_]*\.gguf$")


def _derive_model_id(model_path: str | None) -> str | None:
    """Best-effort model ID from the file path (strip quant suffix + .gguf)."""
    if not model_path:
        return None
    basename = PurePosixPath(model_path).name
    result = _QUANT_SUFFIX_RE.sub("", basename)
    # If regex didn't match, just drop the extension
    if result == basename:
        result = basename.removesuffix(".gguf")
    return result or None


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
        return _flatten_llama_bench(envelope, hw_fields, cuda_version, results, raw_payload)
    elif runner_type == "llama-server" and isinstance(results, dict):
        return [_flatten_llama_server(envelope, hw_fields, results, raw_payload)]
    elif runner_type == "llama-server-loadtest" and isinstance(results, dict):
        return [_flatten_llama_server_loadtest(envelope, hw_fields, results, raw_payload)]
    else:
        # Unknown / unsupported runner — emit one row with Nones
        flat = _new_row()
        flat.update(envelope)
        flat.update(hw_fields)
        flat["raw_payload"] = raw_payload
        return [flat]


# -- helpers ---------------------------------------------------------------


def _extract_envelope(row: dict[str, Any]) -> dict[str, Any]:
    """Pull the common top-level fields shared by every runner."""
    return {
        "timestamp": row.get("timestamp"),
        "runner_type": row.get("runner_type"),
        "model_id": _derive_model_id(row.get("model_path")),
        "n_ctx": row.get("n_ctx"),
        "n_batch": row.get("n_batch"),
        "concurrent_users": row.get("concurrent_users"),
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
        "cpu_model": cpu.get("model"),
        "cpu_cores": cpu.get("cores"),
        "ram_total_gb": ram.get("total_gb"),
        "gpu_name": gpu0.get("name"),
        "gpu_vram_gb": gpu0.get("vram_total_gb"),
        "gpu_driver": gpu0.get("driver"),
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
        flat["backends"] = _enrich_backends(
            item.get("backends"), cuda_version
        )
        flat["throughput_tok_s"] = item.get("avg_ts")
        flat["raw_payload"] = raw_payload
        rows.append(flat)
    return rows


def _flatten_llama_server(
    envelope: dict[str, Any],
    hw_fields: dict[str, Any],
    results: dict[str, Any],
    raw_payload: str,
) -> dict[str, Any]:
    """Flatten a single llama-server results dict."""
    flat = _new_row()
    flat.update(envelope)
    flat.update(hw_fields)
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
    flat["max_sustainable_users"] = results.get("max_sustainable_users")

    # Best aggregate throughput comes from the last successful entry
    # in the concurrency curve.
    curve = results.get("concurrency_curve") or []
    best_throughput = None
    for level in curve:
        val = level.get("aggregate_throughput_tok_s")
        if val is not None:
            best_throughput = val
    flat["throughput_tok_s"] = best_throughput
    flat["raw_payload"] = raw_payload
    return flat
