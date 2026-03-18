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

Supports **concurrent user simulation**: set ``concurrent_users`` in the
sweep config to send multiple prompts simultaneously and measure how
latency degrades under load.
"""

from __future__ import annotations

import asyncio
import json as _json
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
from ._server_mixin import (
    ServerMixin,
    find_free_port,
    percentile,
    _HEALTH_POLL_INTERVAL_S,
    _HEALTH_TIMEOUT_S,
    _SERVER_STOP_TIMEOUT_S,
    _DEFAULT_N_PREDICT,
)

log = logging.getLogger("ppb")

# ---------------------------------------------------------------------------
# Backward-compat aliases (used by tests and external code)
# ---------------------------------------------------------------------------

_find_free_port = find_free_port
_percentile = percentile


def _parse_sse_lines(raw_lines: list[str]) -> list[dict[str, Any]]:
    """Parse Server-Sent Event lines into a list of data payloads.

    Each SSE event looks like::

        data: {"content": "Hello", "stop": false}

    Lines that are empty, comments (``:``) , or ``data: [DONE]`` are
    skipped.  Returns a list of parsed JSON dicts.
    """
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
                payloads.append(_json.loads(body))
            except _json.JSONDecodeError:
                log.debug("Skipping unparseable SSE body: %s", body[:120])
    return payloads


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class LlamaServerRunner(ServerMixin, BaseRunner):
    """Benchmark runner that starts ``llama-server`` and streams completions.

    Collects real-world UX metrics — **TTFT** and **ITL** — by sending
    conversational prompts to the ``/completion`` endpoint with SSE
    streaming enabled.

    When ``concurrent_users`` > 1 in the run config, prompts are sent
    by multiple simulated users in parallel via ``asyncio``, measuring
    how latency degrades under load.

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
    prompt_distribution : str
        How prompts are divided among concurrent users:
        ``"shared"`` (default) — each user gets all prompts;
        ``"split"`` — prompts are round-robined across users.
    """

    runner_type: str = "llama-server"

    def __init__(self) -> None:
        self._cmd: str = ""
        self._prompts: list[str] = []
        self._n_predict: int = _DEFAULT_N_PREDICT
        self._health_timeout: float = _HEALTH_TIMEOUT_S
        self._stop_timeout: float = _SERVER_STOP_TIMEOUT_S
        self._process: subprocess.Popen[str] | None = None
        self._port: int = 0
        self._prompt_distribution: str = "shared"

    # ---- lifecycle ----------------------------------------------------------

    def setup(self, runner_params: dict[str, Any]) -> None:
        """Resolve binary, download dataset, load prompts."""
        self._cmd = self.resolve_server_cmd(runner_params)
        self._n_predict = int(runner_params.get("n_predict", _DEFAULT_N_PREDICT))
        self._health_timeout = float(
            runner_params.get("health_timeout", _HEALTH_TIMEOUT_S)
        )
        self._prompt_distribution = str(
            runner_params.get("prompt_distribution", "shared")
        )
        self._stop_timeout = float(
            runner_params.get("stop_timeout", _SERVER_STOP_TIMEOUT_S)
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
            ``"concurrent_users"`` (int, default 1) — number of
            simulated users sending prompts simultaneously.

        Returns
        -------
        dict | None
            ``{"results": <metrics dict>}`` on success, *None* on failure.
        """
        model_path = config["model_path"]
        n_ctx = config["n_ctx"]
        concurrent_users = int(config.get("concurrent_users", 1))

        # --- start server ---------------------------------------------------
        try:
            proc = self._start_server(Path(model_path), n_ctx)
        except TimeoutError as exc:
            log.error("Server health-check timed out: %s", exc)
            return None
        except OSError as exc:
            log.error("Failed to start llama-server: %s", exc)
            return None

        # --- choose execution path -------------------------------------------
        try:
            if concurrent_users <= 1:
                return self._run_serial(proc)
            else:
                return self._run_concurrent(proc, concurrent_users)
        finally:
            self._stop_server(proc)

    def _run_serial(self, proc: subprocess.Popen[str]) -> dict | None:
        """Original serial prompt execution — one prompt at a time."""
        base_url = f"http://127.0.0.1:{self._port}"
        all_ttft: list[float] = []
        all_itl: list[float] = []
        total_tokens = 0
        successful_prompts = 0
        t_start = time.monotonic()

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

        total_duration = time.monotonic() - t_start
        return self._aggregate_metrics(
            all_ttft, all_itl, total_tokens, successful_prompts,
            total_duration, concurrent_users=1,
        )

    def _run_concurrent(
        self, proc: subprocess.Popen[str], concurrent_users: int,
    ) -> dict | None:
        """Send prompts from *concurrent_users* simulated users in parallel."""
        user_prompts = self._distribute_prompts(concurrent_users)
        return asyncio.run(
            self._async_run(concurrent_users, user_prompts)
        )

    def _distribute_prompts(self, concurrent_users: int) -> list[list[str]]:
        """Split ``self._prompts`` among users based on distribution mode."""
        if self._prompt_distribution == "split":
            buckets: list[list[str]] = [[] for _ in range(concurrent_users)]
            for i, p in enumerate(self._prompts):
                buckets[i % concurrent_users].append(p)
            return buckets
        # "shared" — every user gets the full prompt list
        return [list(self._prompts) for _ in range(concurrent_users)]

    async def _async_run(
        self,
        concurrent_users: int,
        user_prompts: list[list[str]],
    ) -> dict | None:
        """Async entry point: launch *concurrent_users* tasks, aggregate."""
        base_url = f"http://127.0.0.1:{self._port}"
        t_start = time.monotonic()

        # Scale timeout with concurrency: each user's TTFT grows
        # linearly with queue depth (prefill is sequential).
        request_timeout = max(300.0, concurrent_users * 60.0)
        async with httpx.AsyncClient(
            base_url=base_url, timeout=request_timeout,
        ) as client:
            tasks = [
                self._async_user_session(client, prompts, user_id)
                for user_id, prompts in enumerate(user_prompts)
            ]
            user_results = await asyncio.gather(*tasks)

        total_duration = time.monotonic() - t_start

        # Flatten results from all users
        all_ttft: list[float] = []
        all_itl: list[float] = []
        all_queue: list[float] = []
        total_tokens = 0
        successful_prompts = 0
        per_user_stats: list[dict[str, Any]] = []

        for uid, uresult in enumerate(user_results):
            u_ttft = uresult["ttft"]
            u_itl = uresult["itl"]
            u_queue = uresult["queue"]
            u_tokens = uresult["tokens"]
            u_succeeded = uresult["succeeded"]
            u_attempted = uresult["attempted"]

            all_ttft.extend(u_ttft)
            all_itl.extend(u_itl)
            all_queue.extend(u_queue)
            total_tokens += u_tokens
            successful_prompts += u_succeeded

            # Per-user breakdown
            sorted_u_ttft = sorted(u_ttft)
            sorted_u_itl = sorted(u_itl) if u_itl else [0.0]
            per_user_stats.append({
                "user_id": uid,
                "prompts_attempted": u_attempted,
                "prompts_succeeded": u_succeeded,
                "tokens": u_tokens,
                "avg_ttft_ms": round(statistics.mean(sorted_u_ttft) * 1000, 2) if sorted_u_ttft else 0.0,
                "p50_ttft_ms": round(percentile(sorted_u_ttft, 50) * 1000, 2),
                "avg_itl_ms": round(statistics.mean(sorted_u_itl) * 1000, 2) if sorted_u_itl else 0.0,
                "p50_itl_ms": round(percentile(sorted_u_itl, 50) * 1000, 2),
            })

        total_attempted = sum(r["attempted"] for r in user_results)
        return self._aggregate_metrics(
            all_ttft, all_itl, total_tokens, successful_prompts,
            total_duration, concurrent_users,
            queue_times=all_queue,
            per_user_stats=per_user_stats,
            total_attempted=total_attempted,
        )

    async def _async_user_session(
        self,
        client: httpx.AsyncClient,
        prompts: list[str],
        user_id: int,
    ) -> dict[str, Any]:
        """One simulated user: send prompts sequentially, collect metrics."""
        ttft_list: list[float] = []
        itl_list: list[float] = []
        queue_list: list[float] = []
        tokens = 0
        succeeded = 0

        for i, prompt in enumerate(prompts):
            log.debug("User %d prompt %d/%d (%d chars)", user_id, i + 1, len(prompts), len(prompt))
            result = await self._astream_completion(client, prompt)
            if result is not None:
                ttft, itl, n_tokens, queue_time = result
                ttft_list.append(ttft)
                itl_list.extend(itl)
                queue_list.append(queue_time)
                tokens += n_tokens
                succeeded += 1

        return {
            "ttft": ttft_list,
            "itl": itl_list,
            "queue": queue_list,
            "tokens": tokens,
            "succeeded": succeeded,
            "attempted": len(prompts),
        }

    async def _astream_completion(
        self,
        client: httpx.AsyncClient,
        prompt: str,
    ) -> tuple[float, list[float], int, float] | None:
        """Async version of :meth:`_stream_completion` with queue-time tracking.

        Returns
        -------
        tuple[float, list[float], int, float] | None
            ``(ttft_seconds, itl_list_seconds, token_count, queue_seconds)``
            or *None* on failure.
        """
        payload = {
            "prompt": prompt,
            "n_predict": self._n_predict,
            "stream": True,
        }

        t_request = time.monotonic()
        t_first_byte: float | None = None
        t_first_token: float | None = None
        token_timestamps: list[float] = []
        n_tokens = 0

        try:
            async with client.stream("POST", "/completion", json=payload) as response:
                if t_first_byte is None:
                    t_first_byte = time.monotonic()

                if response.status_code != 200:
                    body = await response.aread()
                    log.warning(
                        "/completion returned %d: %s",
                        response.status_code,
                        body.decode()[:200],
                    )
                    return None

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue

                    body_str = line[len("data:"):].strip()
                    if not body_str or body_str == "[DONE]":
                        continue

                    try:
                        event = _json.loads(body_str)
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
        queue_time = (t_first_byte - t_request) if t_first_byte else ttft

        # ITL: gaps between consecutive token arrivals.
        itl_list: list[float] = []
        prev = t_first_token
        for ts in token_timestamps:
            itl_list.append(ts - prev)
            prev = ts

        return ttft, itl_list, n_tokens, queue_time

    def _aggregate_metrics(
        self,
        all_ttft: list[float],
        all_itl: list[float],
        total_tokens: int,
        successful_prompts: int,
        total_duration: float,
        concurrent_users: int,
        *,
        queue_times: list[float] | None = None,
        per_user_stats: list[dict[str, Any]] | None = None,
        total_attempted: int | None = None,
    ) -> dict | None:
        """Build the metrics dict from collected timing data."""
        if successful_prompts == 0:
            log.error("All prompts failed — no metrics to report")
            return None

        attempted = total_attempted if total_attempted is not None else len(self._prompts)
        sorted_ttft = sorted(all_ttft)
        sorted_itl = sorted(all_itl) if all_itl else [0.0]

        metrics: dict[str, Any] = {
            "num_prompts_attempted": attempted,
            "num_prompts_succeeded": successful_prompts,
            "n_predict": self._n_predict,
            "total_tokens": total_tokens,
            "total_duration_s": round(total_duration, 3),
            "throughput_tok_s": round(total_tokens / total_duration, 2)
            if total_duration > 0
            else 0.0,
            # TTFT — Time-To-First-Token (milliseconds)
            "avg_ttft_ms": round(statistics.mean(sorted_ttft) * 1000, 2),
            "p50_ttft_ms": round(percentile(sorted_ttft, 50) * 1000, 2),
            "p99_ttft_ms": round(percentile(sorted_ttft, 99) * 1000, 2),
            # ITL — Inter-Token Latency (milliseconds)
            "avg_itl_ms": round(statistics.mean(sorted_itl) * 1000, 2)
            if sorted_itl
            else 0.0,
            "p50_itl_ms": round(percentile(sorted_itl, 50) * 1000, 2),
            "p99_itl_ms": round(percentile(sorted_itl, 99) * 1000, 2),
        }

        # Concurrent-specific metrics (only when concurrent_users > 1)
        if concurrent_users > 1:
            metrics["concurrent_users"] = concurrent_users
            metrics["aggregate_throughput_tok_s"] = metrics["throughput_tok_s"]
            metrics["per_user_throughput_tok_s"] = round(
                metrics["throughput_tok_s"] / concurrent_users, 2
            )
            if queue_times:
                sorted_q = sorted(queue_times)
                metrics["avg_queue_ms"] = round(statistics.mean(sorted_q) * 1000, 2)
                metrics["p50_queue_ms"] = round(percentile(sorted_q, 50) * 1000, 2)
                metrics["p99_queue_ms"] = round(percentile(sorted_q, 99) * 1000, 2)
            if per_user_stats:
                metrics["per_user_stats"] = per_user_stats

        log.info(
            "llama-server benchmark: %d/%d prompts, %d tokens, "
            "TTFT p50=%.1fms p99=%.1fms, ITL p50=%.1fms p99=%.1fms%s",
            successful_prompts,
            attempted,
            total_tokens,
            metrics["p50_ttft_ms"],
            metrics["p99_ttft_ms"],
            metrics["p50_itl_ms"],
            metrics["p99_itl_ms"],
            f", {concurrent_users} concurrent users" if concurrent_users > 1 else "",
        )

        return {"results": metrics}

    def teardown(self) -> None:
        """Safety-net: kill any lingering server process."""
        if self._process is not None and self._process.poll() is None:
            log.warning("teardown: killing lingering llama-server (pid %d)", self._process.pid)
            self.stop_server(self._process)
            self._process = None

    # ---- optional: probe_ctx ------------------------------------------------

    def probe_ctx(self, model_path: Path, n_ctx: int) -> bool:
        """Return *True* if llama-server can start with *n_ctx*.

        Attempts to launch the server, waits for ``/health`` to return
        a success response, then shuts the server down.

        Raises
        ------
        TimeoutError
            If the server was still running but did not become healthy
            within the configured timeout.  This is distinct from an
            OOM crash (returns *False*) and lets callers decide whether
            to retry with a longer timeout.
        """
        try:
            proc = self.start_server(model_path, n_ctx)
        except TimeoutError:
            raise  # server alive but not healthy — let caller decide
        except OSError:
            return False  # genuine crash (OOM or bad model)
        else:
            self.stop_server(proc)
            return True

    # ---- internal (delegate to mixin) ---------------------------------------

    def _start_server(self, model_path: Path, n_ctx: int) -> subprocess.Popen[str]:
        """Delegate to :meth:`ServerMixin.start_server`."""
        return self.start_server(model_path, n_ctx)

    def _stop_server(self, proc: subprocess.Popen[str]) -> None:
        """Delegate to :meth:`ServerMixin.stop_server`."""
        self.stop_server(proc)

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
                        event = _json.loads(body)
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
