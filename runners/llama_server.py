"""
llama-server runner — real-world inference benchmark for PPB.

Starts ``llama-server`` from `llama.cpp <https://github.com/ggerganov/llama.cpp>`_
as a subprocess, sends realistic ShareGPT prompts to the ``/completion``
endpoint with **server-sent events (SSE)** streaming enabled, and records
UX-relevant latency metrics:

    **TTFT** — Time-To-First-Token  (how long the user waits for output to begin)
    **ITL**  — Inter-Token Latency  (perceived smoothness of the streaming output)

These complement the raw throughput numbers from ``llama-bench`` with
user-experience data that matters for interactive applications.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from datasets import download_dataset, load_sharegpt_prompts
from datasets.sharegpt import SHAREGPT_FILENAME, SHAREGPT_REPO

from .base import BaseRunner

log = logging.getLogger("ppb")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HEALTH_POLL_INTERVAL_S = 0.5  # seconds between /health polls
_HEALTH_TIMEOUT_S = 120  # max seconds to wait for server readiness
_SERVER_STOP_TIMEOUT_S = 5  # seconds to wait after SIGTERM before SIGKILL
_DEFAULT_N_PREDICT = 256  # max tokens to generate per prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Return a free TCP port on localhost.

    Binds to port 0 (OS-assigned), reads the allocated port, then
    closes the socket immediately.  A small race window exists between
    close and the server binding, but it's negligible in practice.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _percentile(data: list[float], pct: int) -> float:
    """Return the *pct*-th percentile of *data* (nearest-rank method).

    *data* must be non-empty and pre-sorted.
    """
    if not data:
        return 0.0
    k = max(0, min(len(data) - 1, int(len(data) * pct / 100)))
    return data[k]


def _parse_sse_lines(raw_lines: list[str]) -> list[dict[str, Any]]:
    """Parse Server-Sent Event lines into a list of data payloads.

    Each SSE event looks like::

        data: {"content": "Hello", "stop": false}

    Lines that are empty, comments (``:``) , or ``data: [DONE]`` are
    skipped.  Returns a list of parsed JSON dicts.
    """
    import json

    payloads: list[dict[str, Any]] = []
    for line in raw_lines:
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            body = line[len("data:") :].strip()
            if not body or body == "[DONE]":
                continue
            try:
                payloads.append(json.loads(body))
            except json.JSONDecodeError:
                log.debug("Skipping unparseable SSE body: %s", body[:120])
    return payloads


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class LlamaServerRunner(BaseRunner):
    """Benchmark runner that starts ``llama-server`` and streams completions.

    Collects real-world UX metrics — **TTFT** and **ITL** — by sending
    conversational prompts to the ``/completion`` endpoint with SSE
    streaming enabled.

    runner_params
    -------------
    llama_server_cmd : str
        Path or name of the ``llama-server`` binary.
        Falls back to ``PPB_LLAMA_SERVER`` env-var → ``"llama-server"`` on
        ``$PATH``.
    num_prompts : int
        Number of prompts to send per benchmark run (default: 10).
    n_predict : int
        Maximum tokens to generate per prompt (default: 256).
    health_timeout : int | float
        Seconds to wait for the server to become healthy (default: 120).
    dataset_dir : str
        Directory for cached dataset files.
    dataset_repo : str
        Hugging Face Hub repository ID for the dataset
        (default: ``anon8231489123/ShareGPT_Vicuna_unfiltered``).
    dataset_filename : str
        Filename to download from the repository
        (default: ``ShareGPT_V3_unfiltered_cleaned_split.json``).
    shuffle : bool
        Randomise prompt order so repeated runs use different workloads
        (default: ``false``).
    seed : int
        RNG seed for reproducible shuffling (optional).
    """

    runner_type: str = "llama-server"

    def __init__(self) -> None:
        self._cmd: str = ""
        self._prompts: list[str] = []
        self._n_predict: int = _DEFAULT_N_PREDICT
        self._health_timeout: float = _HEALTH_TIMEOUT_S
        self._process: subprocess.Popen[str] | None = None
        self._port: int = 0

    # ---- lifecycle ----------------------------------------------------------

    def setup(self, runner_params: dict[str, Any]) -> None:
        """Resolve binary, download dataset, load prompts."""
        self._cmd = runner_params.get(
            "llama_server_cmd",
            os.getenv("PPB_LLAMA_SERVER", "llama-server"),
        )
        self._n_predict = int(runner_params.get("n_predict", _DEFAULT_N_PREDICT))
        self._health_timeout = float(
            runner_params.get("health_timeout", _HEALTH_TIMEOUT_S)
        )

        num_prompts = int(runner_params.get("num_prompts", 10))
        dataset_dir_str = runner_params.get("dataset_dir")
        dataset_dir = Path(dataset_dir_str) if dataset_dir_str else None

        repo_id = str(runner_params.get("dataset_repo", SHAREGPT_REPO))
        filename = str(runner_params.get("dataset_filename", SHAREGPT_FILENAME))
        shuffle = bool(runner_params.get("shuffle", False))
        seed_val = runner_params.get("seed")
        seed = int(seed_val) if seed_val is not None else None

        # Download + load prompts.
        dataset_path = download_dataset(
            repo_id=repo_id, filename=filename, dataset_dir=dataset_dir,
        )
        self._prompts = load_sharegpt_prompts(
            dataset_path,
            max_prompts=num_prompts,
            shuffle=shuffle,
            seed=seed,
        )

        if not self._prompts:
            log.warning("No usable prompts loaded from dataset")

    def run(self, config: dict[str, Any]) -> dict | None:
        """Start the server, stream prompts, collect latency metrics.

        Parameters
        ----------
        config:
            Must contain ``"model_path"`` (str) and ``"n_ctx"`` (int).
            ``"n_batch"`` is accepted but not used by this runner
            (llama-server manages its own batching).

        Returns
        -------
        dict | None
            ``{"results": <metrics dict>}`` on success, *None* on failure.
        """
        model_path = config["model_path"]
        n_ctx = config["n_ctx"]

        # --- start server ---------------------------------------------------
        try:
            proc = self._start_server(Path(model_path), n_ctx)
        except TimeoutError as exc:
            log.error("Server health-check timed out: %s", exc)
            return None
        except OSError as exc:
            log.error("Failed to start llama-server: %s", exc)
            return None

        # --- run prompts and collect metrics ---------------------------------
        base_url = f"http://127.0.0.1:{self._port}"
        all_ttft: list[float] = []
        all_itl: list[float] = []
        total_tokens = 0
        successful_prompts = 0
        t_start = time.monotonic()

        try:
            with httpx.Client(base_url=base_url, timeout=300.0) as client:
                for i, prompt in enumerate(self._prompts, 1):
                    log.debug("Prompt %d/%d (%d chars)", i, len(self._prompts), len(prompt))
                    result = self._stream_completion(client, prompt)
                    if result is not None:
                        ttft, itl_list, n_tokens = result
                        all_ttft.append(ttft)
                        all_itl.extend(itl_list)
                        total_tokens += n_tokens
                        successful_prompts += 1
        finally:
            self._stop_server(proc)

        total_duration = time.monotonic() - t_start

        if successful_prompts == 0:
            log.error("All prompts failed — no metrics to report")
            return None

        # --- aggregate metrics -----------------------------------------------
        sorted_ttft = sorted(all_ttft)
        sorted_itl = sorted(all_itl) if all_itl else [0.0]

        metrics: dict[str, Any] = {
            "num_prompts_attempted": len(self._prompts),
            "num_prompts_succeeded": successful_prompts,
            "n_predict": self._n_predict,
            "total_tokens": total_tokens,
            "total_duration_s": round(total_duration, 3),
            "throughput_tok_s": round(total_tokens / total_duration, 2)
            if total_duration > 0
            else 0.0,
            # TTFT — Time-To-First-Token (milliseconds)
            "avg_ttft_ms": round(statistics.mean(sorted_ttft) * 1000, 2),
            "p50_ttft_ms": round(_percentile(sorted_ttft, 50) * 1000, 2),
            "p99_ttft_ms": round(_percentile(sorted_ttft, 99) * 1000, 2),
            # ITL — Inter-Token Latency (milliseconds)
            "avg_itl_ms": round(statistics.mean(sorted_itl) * 1000, 2)
            if sorted_itl
            else 0.0,
            "p50_itl_ms": round(_percentile(sorted_itl, 50) * 1000, 2),
            "p99_itl_ms": round(_percentile(sorted_itl, 99) * 1000, 2),
        }

        log.info(
            "llama-server benchmark: %d/%d prompts, %d tokens, "
            "TTFT p50=%.1fms p99=%.1fms, ITL p50=%.1fms p99=%.1fms",
            successful_prompts,
            len(self._prompts),
            total_tokens,
            metrics["p50_ttft_ms"],
            metrics["p99_ttft_ms"],
            metrics["p50_itl_ms"],
            metrics["p99_itl_ms"],
        )

        return {"results": metrics}

    def teardown(self) -> None:
        """Safety-net: kill any lingering server process."""
        if self._process is not None and self._process.poll() is None:
            log.warning("teardown: killing lingering llama-server (pid %d)", self._process.pid)
            self._stop_server(self._process)
            self._process = None

    # ---- optional: probe_ctx ------------------------------------------------

    def probe_ctx(self, model_path: Path, n_ctx: int) -> bool:
        """Return *True* if llama-server can start with *n_ctx*.

        Attempts to launch the server, waits for ``/health`` to return
        a success response, then shuts the server down.  A health-check
        timeout or launch failure returns *False*.
        """
        try:
            proc = self._start_server(model_path, n_ctx)
        except (TimeoutError, OSError):
            return False
        else:
            self._stop_server(proc)
            return True

    # ---- internal -----------------------------------------------------------

    def _start_server(self, model_path: Path, n_ctx: int) -> subprocess.Popen[str]:
        """Launch ``llama-server`` and wait for ``/health`` to become OK."""
        self._port = _find_free_port()
        cmd: list[str] = [
            self._cmd,
            "-m", str(model_path),
            "-c", str(n_ctx),
            "--host", "127.0.0.1",
            "--port", str(self._port),
        ]

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

        # Timed out — kill and raise.
        self._stop_server(proc)
        raise TimeoutError(
            f"llama-server did not become healthy within "
            f"{self._health_timeout}s on port {self._port}"
        )

    def _stop_server(self, proc: subprocess.Popen[str]) -> None:
        """Gracefully stop the server: SIGTERM → wait → SIGKILL."""
        if proc.poll() is not None:
            return  # already exited

        log.debug("Stopping llama-server (pid %d)", proc.pid)
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=_SERVER_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            log.warning("SIGTERM timed out — sending SIGKILL to pid %d", proc.pid)
            proc.kill()
            proc.wait(timeout=5)
        except OSError:
            pass  # process already gone

        if proc is self._process:
            self._process = None

    def _stream_completion(
        self,
        client: httpx.Client,
        prompt: str,
    ) -> tuple[float, list[float], int] | None:
        """Send a streaming ``/completion`` request and measure latency.

        Returns
        -------
        tuple[float, list[float], int] | None
            ``(ttft_seconds, itl_list_seconds, token_count)`` on success,
            or *None* if the request failed.
        """
        payload = {
            "prompt": prompt,
            "n_predict": self._n_predict,
            "stream": True,
        }

        t_request = time.monotonic()
        t_first_token: float | None = None
        token_timestamps: list[float] = []
        n_tokens = 0

        try:
            with client.stream("POST", "/completion", json=payload) as response:
                if response.status_code != 200:
                    log.warning(
                        "/completion returned %d: %s",
                        response.status_code,
                        response.read().decode()[:200],
                    )
                    return None

                for line in response.iter_lines():
                    line = line.strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue

                    body = line[len("data:"):].strip()
                    if not body or body == "[DONE]":
                        continue

                    try:
                        import json

                        event = json.loads(body)
                    except Exception:
                        continue

                    content = event.get("content", "")
                    if not content:
                        continue

                    now = time.monotonic()
                    n_tokens += 1

                    if t_first_token is None:
                        t_first_token = now
                    else:
                        token_timestamps.append(now)

        except httpx.HTTPError as exc:
            log.warning("Completion request failed: %s", exc)
            return None

        if t_first_token is None or n_tokens == 0:
            log.debug("No tokens received for prompt (len=%d)", len(prompt))
            return None

        ttft = t_first_token - t_request

        # ITL: gaps between consecutive token arrivals.
        itl_list: list[float] = []
        prev = t_first_token
        for ts in token_timestamps:
            itl_list.append(ts - prev)
            prev = ts

        return ttft, itl_list, n_tokens
