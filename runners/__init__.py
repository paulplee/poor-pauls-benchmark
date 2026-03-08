"""
Runner registry — maps ``runner_type`` strings to :class:`BaseRunner` classes.

Usage::

    from runners import get_runner

    runner = get_runner("llama-bench")
    runner.setup(runner_params={})
    result = runner.run({"model_path": ..., "n_ctx": 8192, "n_batch": 512})
    runner.teardown()

Third-party runners call :func:`register_runner` at import time to add
themselves to the registry without touching core PPB code.
"""

from __future__ import annotations

from .base import BaseRunner

__all__ = ["BaseRunner", "get_runner", "register_runner"]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseRunner]] = {}


def register_runner(runner_type: str, cls: type[BaseRunner]) -> None:
    """Add a runner class to the global registry.

    Parameters
    ----------
    runner_type:
        Short identifier such as ``"llama-bench"`` or ``"llama-server"``.
    cls:
        A concrete :class:`BaseRunner` subclass.
    """
    _REGISTRY[runner_type] = cls


def get_runner(runner_type: str) -> BaseRunner:
    """Instantiate and return a runner for *runner_type*.

    Raises
    ------
    ValueError
        If *runner_type* is not registered — the error message lists all
        available runners so the user knows what to put in ``sweep.toml``.
    """
    cls = _REGISTRY.get(runner_type)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(
            f"Unknown runner_type {runner_type!r}.  "
            f"Available runners: {available}"
        )
    return cls()


# ---------------------------------------------------------------------------
# Built-in runners (auto-registered at import time)
# ---------------------------------------------------------------------------

from .llama_bench import LlamaBenchRunner  # noqa: E402
from .llama_server import LlamaServerRunner  # noqa: E402
from .llama_server_loadtest import LlamaServerLoadTestRunner  # noqa: E402

register_runner("llama-bench", LlamaBenchRunner)
register_runner("llama-server", LlamaServerRunner)
register_runner("llama-server-loadtest", LlamaServerLoadTestRunner)
