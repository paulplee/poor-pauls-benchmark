"""
Shared server-management utilities for llama-server based runners.

Extracted from :mod:`runners.llama_server` so that both
:class:`~runners.llama_server.LlamaServerRunner` and
:class:`~runners.llama_server_loadtest.LlamaServerLoadTestRunner`
can reuse the same start/stop/health-check logic.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
from pathlib import Path
import subprocess
import time

import httpx

log = logging.getLogger("ppb")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HEALTH_POLL_INTERVAL_S = 0.5  # seconds between /health polls
_HEALTH_TIMEOUT_S = 120  # max seconds to wait for server readiness
_SERVER_STOP_TIMEOUT_S = 8  # seconds to wait after SIGINT/SIGTERM before SIGKILL
_POST_KILL_DELAY_S = 3  # seconds to wait after SIGKILL for GPU resource reclamation
_DEFAULT_N_PREDICT = 256  # max tokens to generate per prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_free_port() -> int:
    """Return a free TCP port on localhost.

    Binds to port 0 (OS-assigned), reads the allocated port, then
    closes the socket immediately.  A small race window exists between
    close and the server binding, but it's negligible in practice.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def percentile(data: list[float], pct: int) -> float:
    """Return the *pct*-th percentile of *data* (nearest-rank method).

    *data* must be non-empty and pre-sorted.
    """
    if not data:
        return 0.0
    k = max(0, min(len(data) - 1, int(len(data) * pct / 100)))
    return data[k]


class ServerMixin:
    """Mixin providing llama-server start/stop/health-check helpers.

    Subclasses must set:
    * ``_cmd``  — the llama-server binary path/name
    * ``_health_timeout``  — seconds to wait for health-check
    * ``_port``  — populated by :meth:`start_server`
    * ``_process``  — populated by :meth:`start_server`
    """

    _cmd: str
    _health_timeout: float
    _stop_timeout: float
    _port: int
    _process: subprocess.Popen[str] | None

    def resolve_server_cmd(self, runner_params: dict) -> str:
        """Determine the ``llama-server`` binary from params / env / PATH."""
        return runner_params.get(
            "llama_server_cmd",
            os.getenv("PPB_LLAMA_SERVER", "llama-server"),
        )

    def start_server(
        self,
        model_path: str | Path,
        n_ctx: int,
        parallel: int = 1,
    ) -> subprocess.Popen[str]:
        """Launch ``llama-server`` and wait for ``/health`` to become OK."""
        self._port = find_free_port()
        cmd: list[str] = [
            self._cmd,
            "-m",
            str(model_path),
            "-c",
            str(n_ctx),
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
        ]
        if parallel > 1:
            cmd += ["--parallel", str(parallel)]

        log.debug("Starting llama-server: %s", " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._process = proc

        # Poll /health until the server is ready.
        url = f"http://127.0.0.1:{self._port}/health"
        deadline = time.monotonic() + self._health_timeout

        while time.monotonic() < deadline:
            # Check server hasn't crashed.
            if proc.poll() is not None:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise OSError(
                    f"llama-server exited with code {proc.returncode} "
                    f"before becoming healthy.\nstderr: {stderr[:500]}"
                )
            try:
                resp = httpx.get(url, timeout=2.0)
                if resp.status_code == 200:
                    log.debug("llama-server healthy on port %d", self._port)
                    return proc
            except httpx.ConnectError:
                pass  # server not listening yet
            except httpx.TimeoutException:
                pass  # health endpoint slow

            time.sleep(_HEALTH_POLL_INTERVAL_S)

        # Timed out — capture stderr for diagnostics, then kill.
        stderr = ""
        if proc.stderr:
            # Non-blocking read: grab whatever the server has written so far.
            import selectors

            sel = selectors.DefaultSelector()
            sel.register(proc.stderr, selectors.EVENT_READ)
            ready = sel.select(timeout=0)
            if ready:
                stderr = proc.stderr.read(4096)
            sel.close()

        self.stop_server(proc)
        msg = (
            f"llama-server did not become healthy within "
            f"{self._health_timeout}s on port {self._port}"
        )
        if stderr:
            msg += f"\nServer stderr (last 500 chars): {stderr[-500:]}"
        raise TimeoutError(msg)

    def stop_server(self, proc: subprocess.Popen[str]) -> None:
        """Gracefully stop the server: SIGINT → SIGTERM → SIGKILL."""
        if proc.poll() is not None:
            return  # already exited

        timeout = getattr(self, "_stop_timeout", _SERVER_STOP_TIMEOUT_S)
        log.debug("Stopping llama-server (pid %d), timeout=%ds", proc.pid, timeout)
        needed_sigkill = False
        try:
            # SIGINT triggers llama-server's graceful shutdown handler;
            # SIGTERM is often ignored when the server is under heavy load.
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log.debug("SIGINT timed out — escalating to SIGTERM (pid %d)", proc.pid)
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                log.warning("SIGTERM timed out — sending SIGKILL to pid %d", proc.pid)
                needed_sigkill = True
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    log.warning(
                        "Process %d did not exit after SIGKILL — "
                        "may be in uninterruptible sleep",
                        proc.pid,
                    )
            except OSError:
                pass  # process already gone
        except OSError:
            pass  # process already gone

        # After a forced kill the GPU driver may need a moment to reclaim
        # VRAM.  Only pause when SIGKILL was required — clean shutdowns
        # release resources before the process exits.
        if needed_sigkill:
            time.sleep(_POST_KILL_DELAY_S)

        if proc is self._process:
            self._process = None
