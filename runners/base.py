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

    def probe_ctx(self, model_path: Path, n_ctx: int) -> bool:
        """Return *True* if *model_path* can allocate a KV cache of *n_ctx*.

        Override this in runners that support OOM probing.  The default
        implementation raises :exc:`NotImplementedError` so callers get a
        clear message when a runner does not support ``vram-cliff``.
        """
        raise NotImplementedError(
            f"Runner {self.runner_type!r} does not support context-size probing."
        )
