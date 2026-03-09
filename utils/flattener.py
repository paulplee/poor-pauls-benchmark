"""Normalize deeply nested benchmark rows into flat, Arrow-friendly dicts."""

from __future__ import annotations

import json
from typing import Any


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

    base = _extract_envelope(row)
    base.update(_extract_hardware(row.get("hardware") or {}))

    raw_payload = json.dumps(row, default=str)

    if runner_type == "llama-bench" and isinstance(results, list):
        return _flatten_llama_bench(base, results, raw_payload)
    elif runner_type == "llama-server" and isinstance(results, dict):
        return [_flatten_llama_server(base, results, raw_payload)]
    elif runner_type == "llama-server-loadtest" and isinstance(results, dict):
        return [_flatten_llama_server_loadtest(base, results, raw_payload)]
    else:
        # Unknown / unsupported runner — emit one row with Nones
        flat = {**base, "throughput_tok_s": None, "raw_payload": raw_payload}
        return [flat]


# -- helpers ---------------------------------------------------------------


def _extract_envelope(row: dict[str, Any]) -> dict[str, Any]:
    """Pull the common top-level fields shared by every runner."""
    return {
        "timestamp": row.get("timestamp"),
        "runner_type": row.get("runner_type"),
        "model_path": row.get("model_path"),
        "n_ctx": row.get("n_ctx"),
        "n_batch": row.get("n_batch"),
        "concurrent_users": row.get("concurrent_users"),
    }


def _extract_hardware(hw: dict[str, Any]) -> dict[str, Any]:
    """Flatten the nested ``hardware`` dictionary."""
    os_info = hw.get("os") or {}
    cpu = hw.get("cpu") or {}
    ram = hw.get("ram") or {}
    gpus = hw.get("gpus") or []
    gpu0: dict[str, Any] = gpus[0] if gpus else {}
    runtime = hw.get("runtime") or {}

    return {
        "os_system": os_info.get("system"),
        "os_release": os_info.get("release"),
        "os_machine": os_info.get("machine"),
        "cpu_model": cpu.get("model"),
        "cpu_cores": cpu.get("cores"),
        "ram_total_gb": ram.get("total_gb"),
        "gpu_name": gpu0.get("name"),
        "gpu_vram_gb": gpu0.get("vram_total_gb"),
        "gpu_driver": gpu0.get("driver"),
        "python_version": runtime.get("python_version"),
    }


def _flatten_llama_bench(
    base: dict[str, Any],
    results: list[dict[str, Any]],
    raw_payload: str,
) -> list[dict[str, Any]]:
    """Explode llama-bench results — one flat row per result item."""
    rows: list[dict[str, Any]] = []
    for item in results:
        flat = {**base}
        # llama-bench embeds gpu_info inside each result item (hardware.gpus
        # can be empty), so prefer the result-level value as a fallback.
        if flat.get("gpu_name") is None and item.get("gpu_info"):
            flat["gpu_name"] = item.get("gpu_info")
        flat["gpu_info"] = item.get("gpu_info")
        flat["backends"] = item.get("backends")
        flat["n_prompt"] = item.get("n_prompt")
        flat["n_gen"] = item.get("n_gen")
        flat["throughput_tok_s"] = item.get("avg_ts")
        flat["raw_payload"] = raw_payload
        rows.append(flat)
    return rows


def _flatten_llama_server(
    base: dict[str, Any],
    results: dict[str, Any],
    raw_payload: str,
) -> dict[str, Any]:
    """Flatten a single llama-server results dict."""
    flat = {**base}
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
    base: dict[str, Any],
    results: dict[str, Any],
    raw_payload: str,
) -> dict[str, Any]:
    """Flatten a single llama-server-loadtest results dict."""
    flat = {**base}
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
