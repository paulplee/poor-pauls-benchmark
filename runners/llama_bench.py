"""
llama-bench runner — the default benchmark backend for PPB.

Wraps ``llama-bench`` from `llama.cpp <https://github.com/ggerganov/llama.cpp>`_
via :mod:`subprocess`.  Supports both regular benchmark runs **and** OOM probing
for the ``auto-limit`` command.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from .base import BaseRunner

log = logging.getLogger("ppb")


class LlamaBenchRunner(BaseRunner):
    """Benchmark runner that delegates to ``llama-bench``."""

    runner_type: str = "llama-bench"

    # Substrings in stderr/stdout that signal an out-of-memory condition.
    OOM_MARKERS: tuple[str, ...] = (
        "out of memory",
        "bad alloc",
        "bad_alloc",
        "cudaerroroutofmemory",
        "rocm out of memory",
        "failed to allocate",
    )

    def __init__(self) -> None:
        self._cmd: str = ""

    # ---- lifecycle ----------------------------------------------------------

    def setup(self, runner_params: dict[str, Any]) -> None:
        """Resolve the ``llama-bench`` binary.

        Precedence (highest → lowest):
            1. ``runner_params["llama_bench_cmd"]``  (sweep.toml)
            2. ``PPB_LLAMA_BENCH`` env-var
            3. ``"llama-bench"`` on ``$PATH``
        """
        self._cmd = runner_params.get(
            "llama_bench_cmd",
            os.getenv("PPB_LLAMA_BENCH", "llama-bench"),
        )

    def run(self, config: dict[str, Any]) -> dict | None:
        """Run ``llama-bench`` for one (model, n_ctx, n_batch) combo.

        Parameters
        ----------
        config:
            Must contain ``"model_path"`` (str), ``"n_ctx"`` (int),
            ``"n_batch"`` (int).

        Returns
        -------
        dict | None
            ``{"results": <parsed JSON array>}`` on success, *None* on
            failure.
        """
        cmd: list[str] = [
            self._cmd,
            "-m", str(config["model_path"]),
            "-p", str(config["n_ctx"]),
            "-b", str(config["n_batch"]),
            "-o", "json",
        ]

        log.debug("Running: %s", " ".join(cmd))

        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            log.error(
                "llama-bench exited with code %d\n%s",
                proc.returncode,
                proc.stderr.strip(),
            )
            return None

        try:
            bench_data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            log.error(
                "Failed to parse llama-bench output: %s\nRaw stdout:\n%s",
                exc,
                proc.stdout[:500],
            )
            return None

        return {"results": bench_data}

    def teardown(self) -> None:
        """No-op — each ``llama-bench`` call is a fresh subprocess."""

    # ---- OOM probing --------------------------------------------------------

    def probe_ctx(self, model_path: Path, n_ctx: int) -> bool:
        """Return *True* when llama-bench can allocate a KV cache at *n_ctx*.

        Runs ``llama-bench -n 0`` (skip generation) with the given context
        size and inspects the exit code + output for OOM markers.
        """
        cmd: list[str] = [
            self._cmd,
            "-m", str(model_path),
            "-p", str(n_ctx),
            "-n", "0",          # allocation-only — skip token generation
            "-o", "json",
        ]
        log.debug("probe_ctx n_ctx=%d — running: %s", n_ctx, " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            log.debug("probe_ctx n_ctx=%d — exit %d", n_ctx, proc.returncode)
            return False

        combined = (proc.stdout + proc.stderr).lower()
        if any(marker in combined for marker in self.OOM_MARKERS):
            log.debug("probe_ctx n_ctx=%d — OOM marker found in output", n_ctx)
            return False

        return True
