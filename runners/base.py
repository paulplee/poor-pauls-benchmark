"""
Abstract base class for benchmark runners.

Every benchmark backend (llama-bench, llama-server, vLLM, …) inherits from
:class:`BaseRunner` and implements the three lifecycle methods:

    setup  → configure / connect
    run    → execute a single benchmark combo → return raw results
    teardown → release resources

The orchestrator in ``ppb.py`` owns the JSONL result envelope (timestamp,
hardware fingerprint, runner_type) — runners MUST NOT write files or attach
metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseRunner(ABC):
    """Contract that every benchmark runner must satisfy."""

    # Subclasses MUST set this to a unique identifier, e.g. "llama-bench".
    runner_type: str = ""

    # ---- lifecycle ----------------------------------------------------------

    @abstractmethod
    def setup(self, runner_params: dict[str, Any]) -> None:
        """Prepare the runner (resolve binary paths, open connections, …).

        Parameters
        ----------
        runner_params:
            Runner-specific configuration read from the optional
            ``[sweep.runner_params]`` table in the sweep TOML file.
        """

    @abstractmethod
    def run(self, config: dict[str, Any]) -> dict | None:
        """Execute **one** benchmark for the given parameter combination.

        Parameters
        ----------
        config:
            Always contains ``"model_path"`` (str).  Other keys are
            runner-specific (e.g. ``"n_ctx"``, ``"n_batch"`` for
            llama-bench).

        Returns
        -------
        dict | None
            ``{"results": <raw benchmark payload>}`` on success, or
            *None* when the run failed.  The caller enriches this dict
            with timestamp, hardware, runner_type, etc. before persisting.
        """

    @abstractmethod
    def teardown(self) -> None:
        """Release any resources acquired during :meth:`setup`."""

    # ---- optional capabilities ----------------------------------------------

    def metadata(self) -> dict[str, Any]:
        """Return engine metadata for the result envelope.

        Override in subclasses to provide the inference engine name and
        version.  The orchestrator merges this into every JSONL record.

        Returns
        -------
        dict
            ``{"llm_engine_name": str, "llm_engine_version": str | None}``
        """
        return {"llm_engine_name": "unknown", "llm_engine_version": None}

    # Populated by probe_ctx when it returns False — contains the
    # server stderr or error message from the failed probe, if available.
    last_probe_error: str = ""

    def probe_ctx(self, model_path: Path, n_ctx: int) -> bool:
        """Return *True* if *model_path* can allocate a KV cache of *n_ctx*.

        Override this in runners that support OOM probing.  The default
        implementation raises :exc:`NotImplementedError` so callers get a
        clear message when a runner does not support ``vram-cliff``.

        When returning *False*, implementations should set
        :attr:`last_probe_error` to the stderr / error message from the
        failed attempt so callers (e.g. ``execute_vram_cliff``) can
        report the real cause.
        """
        raise NotImplementedError(
            f"Runner {self.runner_type!r} does not support context-size probing."
        )

    # ---- server reuse (optional) --------------------------------------------
    # Runners that launch a long-lived server process (e.g. llama-server)
    # can override these to let the orchestrator keep the server alive
    # across multiple combos that share the same (model, n_ctx).

    @property
    def supports_server_reuse(self) -> bool:
        """Return *True* if this runner supports managed server lifecycle."""
        return False

    def ensure_server(self, model_path: Path, n_ctx: int, parallel: int = 1) -> None:
        """Start (or keep running) a server for *model_path* at *n_ctx*.

        If a compatible server is already running, this is a no-op.
        If the server is running with different parameters, it is
        stopped and restarted.
        """
        raise NotImplementedError

    def run_on_server(self, config: dict[str, Any]) -> dict | None:
        """Run a benchmark combo against the already-running server.

        The server must have been started via :meth:`ensure_server`.
        Unlike :meth:`run`, this does NOT start/stop the server.
        """
        raise NotImplementedError

    def stop_managed_server(self) -> None:
        """Stop the managed server if one is running."""
        raise NotImplementedError
