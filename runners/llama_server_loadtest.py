"""
llama-server load-test runner — auto-discover maximum sustainable concurrency.

Starts ``llama-server`` once then escalates the number of concurrent users
(1 → 2 → 4 → 8 → …) sending real prompts simultaneously.  At each level it
records TTFT, ITL, queue time and throughput.  The escalation stops when the
error rate exceeds a configurable threshold or ``max_users`` is reached.

The result includes the full **concurrency curve** and the maximum number
of concurrent users the system can sustain.


runner_params
-------------
max_users : int
    Maximum concurrent users to test (default: 64).
user_steps : list[int]
    Explicit list of concurrency levels to test.
    Overrides the default power-of-two progression.
error_threshold : float
    Fraction (0–1) of prompts that may fail before stopping
    escalation (default: 0.1 — i.e. 10 %).
ramp_delay_s : float
    Seconds to wait between concurrency levels (default: 1.0).
num_prompts : int
    Prompts to send **per user** at each level (default: 5).
n_predict : int
    Max tokens per prompt (default: 256).
health_timeout : int | float
    Seconds to wait for server readiness (default: 120).
llama_server_cmd : str
    Path or name of the ``llama-server`` binary.
prompt_distribution : str
    ``"shared"`` (each user gets all *num_prompts*) or
    ``"split"`` (prompts are round-robined across users).

Plus all standard dataset params (``dataset_dir``, ``dataset_repo``,
``dataset_filename``, ``shuffle``, ``seed``).
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

from datasets import download_dataset, load_sharegpt_prompts
from datasets.sharegpt import SHAREGPT_FILENAME, SHAREGPT_REPO

from .base import BaseRunner
from ._server_mixin import (
    ServerMixin,
    percentile,
    _HEALTH_TIMEOUT_S,
    _SERVER_STOP_TIMEOUT_S,
    _DEFAULT_N_PREDICT,
)

log = logging.getLogger("ppb")

# ---------------------------------------------------------------------------
# Default load-test settings
# ---------------------------------------------------------------------------

_DEFAULT_MAX_USERS = 64
_DEFAULT_ERROR_THRESHOLD = 0.10  # 10 %
_DEFAULT_RAMP_DELAY_S = 1.0
_DEFAULT_NUM_PROMPTS_LT = 5  # per user per level


def _default_user_steps(max_users: int) -> list[int]:
    """Return powers of two up to *max_users*: [1, 2, 4, 8, …]."""
    steps: list[int] = []
    n = 1
    while n <= max_users:
        steps.append(n)
        n *= 2
    if steps and steps[-1] != max_users:
        steps.append(max_users)
    return steps


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class LlamaServerLoadTestRunner(ServerMixin, BaseRunner):
    """Escalating-concurrency benchmark runner.

    Starts ``llama-server`` once per ``run()`` call, then tests
    increasing concurrency levels and reports the full curve plus the
    maximum sustainable user count.
    """

    runner_type: str = "llama-server-loadtest"

    def __init__(self) -> None:
        self._cmd: str = ""
        self._prompts: list[str] = []
        self._n_predict: int = _DEFAULT_N_PREDICT
        self._health_timeout: float = _HEALTH_TIMEOUT_S
        self._stop_timeout: float = _SERVER_STOP_TIMEOUT_S
        self._process = None
        self._port: int = 0
        self._max_users: int = _DEFAULT_MAX_USERS
        self._user_steps: list[int] | None = None
        self._error_threshold: float = _DEFAULT_ERROR_THRESHOLD
        self._ramp_delay_s: float = _DEFAULT_RAMP_DELAY_S
        self._prompt_distribution: str = "shared"
        # Per-level error counters (reset before each level).
        self._ctx_exceeded_count: int = 0
        self._disconnect_count: int = 0

    # ---- lifecycle ----------------------------------------------------------

    def setup(self, runner_params: dict[str, Any]) -> None:
        """Resolve binary, download dataset, load prompts, store settings."""
        self._cmd = self.resolve_server_cmd(runner_params)
        self._n_predict = int(runner_params.get("n_predict", _DEFAULT_N_PREDICT))
        self._health_timeout = float(
            runner_params.get("health_timeout", _HEALTH_TIMEOUT_S)
        )
        self._max_users = int(runner_params.get("max_users", _DEFAULT_MAX_USERS))
        self._error_threshold = float(
            runner_params.get("error_threshold", _DEFAULT_ERROR_THRESHOLD)
        )
        self._ramp_delay_s = float(
            runner_params.get("ramp_delay_s", _DEFAULT_RAMP_DELAY_S)
        )
        self._prompt_distribution = str(
            runner_params.get("prompt_distribution", "shared")
        )
        self._stop_timeout = float(
            runner_params.get("stop_timeout", _SERVER_STOP_TIMEOUT_S)
        )

        raw_steps = runner_params.get("user_steps")
        if raw_steps is not None:
            self._user_steps = [int(s) for s in raw_steps]
        else:
            self._user_steps = None  # computed at run-time

        num_prompts = int(runner_params.get("num_prompts", _DEFAULT_NUM_PROMPTS_LT))
        dataset_dir_str = runner_params.get("dataset_dir")
        dataset_dir = Path(dataset_dir_str) if dataset_dir_str else None

        repo_id = str(runner_params.get("dataset_repo", SHAREGPT_REPO))
        filename = str(runner_params.get("dataset_filename", SHAREGPT_FILENAME))
        shuffle = bool(runner_params.get("shuffle", False))
        seed_val = runner_params.get("seed")
        seed = int(seed_val) if seed_val is not None else None

        dataset_path = download_dataset(
            repo_id=repo_id,
            filename=filename,
            dataset_dir=dataset_dir,
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
        """Escalate concurrency levels against a single server instance.

        Returns a result containing the concurrency curve and the max
        sustainable user count.
        """
        model_path = config["model_path"]
        n_ctx = config["n_ctx"]

        # Start server once for the whole escalation
        try:
            proc = self.start_server(Path(model_path), n_ctx)
        except TimeoutError as exc:
            log.error("Server health-check timed out: %s", exc)
            return None
        except OSError as exc:
            log.error("Failed to start llama-server: %s", exc)
            return None

        steps = self._user_steps or _default_user_steps(self._max_users)
        curve: list[dict[str, Any]] = []
        max_sustainable: int = 0

        try:
            for level in steps:
                log.info("Load-test: testing %d concurrent user(s)", level)
                level_result = asyncio.run(self._test_concurrency_level(level))

                if level_result is None:
                    log.warning("Level %d: all prompts failed", level)
                    break

                curve.append(level_result)

                # Check error rate
                attempted = level_result["num_prompts_attempted"]
                succeeded = level_result["num_prompts_succeeded"]
                error_rate = 1.0 - (succeeded / attempted) if attempted > 0 else 1.0

                if error_rate <= self._error_threshold:
                    max_sustainable = level

                if error_rate > self._error_threshold:
                    log.info(
                        "Level %d: error rate %.1f%% exceeds threshold %.1f%% — stopping",
                        level,
                        error_rate * 100,
                        self._error_threshold * 100,
                    )
                    break

                if level < steps[-1]:
                    time.sleep(self._ramp_delay_s)

        finally:
            self.stop_server(proc)

        if not curve:
            log.error("Load-test produced no results")
            return None

        results: dict[str, Any] = {
            "max_sustainable_users": max_sustainable,
            "error_threshold": self._error_threshold,
            "concurrency_curve": curve,
        }

        log.info(
            "Load-test complete: max sustainable users = %d (tested levels: %s)",
            max_sustainable,
            [c["concurrent_users"] for c in curve],
        )

        return {"results": results}

    def teardown(self) -> None:
        """Safety-net: kill any lingering server process."""
        if self._process is not None and self._process.poll() is None:
            log.warning(
                "teardown: killing lingering llama-server (pid %d)", self._process.pid
            )
            self.stop_server(self._process)
            self._process = None

    def probe_ctx(self, model_path: Path, n_ctx: int) -> bool:
        """Delegate to standard server start/stop probe."""
        try:
            proc = self.start_server(model_path, n_ctx)
        except (TimeoutError, OSError):
            return False
        else:
            self.stop_server(proc)
            return True

    # ---- internal -----------------------------------------------------------

    async def _test_concurrency_level(
        self,
        concurrent_users: int,
    ) -> dict[str, Any] | None:
        """Test one concurrency level and return aggregated metrics."""
        self._ctx_exceeded_count = 0
        self._disconnect_count = 0
        base_url = f"http://127.0.0.1:{self._port}"
        user_prompts = self._distribute_prompts(concurrent_users)
        t_start = time.monotonic()

        async with httpx.AsyncClient(
            base_url=base_url,
            timeout=300.0,
        ) as client:
            tasks = [
                self._async_user_session(client, prompts, uid)
                for uid, prompts in enumerate(user_prompts)
            ]
            user_results = await asyncio.gather(*tasks)

        total_duration = time.monotonic() - t_start

        # Flatten
        all_ttft: list[float] = []
        all_itl: list[float] = []
        all_queue: list[float] = []
        total_tokens = 0
        successful_prompts = 0
        total_attempted = 0

        for uresult in user_results:
            all_ttft.extend(uresult["ttft"])
            all_itl.extend(uresult["itl"])
            all_queue.extend(uresult["queue"])
            total_tokens += uresult["tokens"]
            successful_prompts += uresult["succeeded"]
            total_attempted += uresult["attempted"]

        if successful_prompts == 0:
            return None

        sorted_ttft = sorted(all_ttft)
        sorted_itl = sorted(all_itl) if all_itl else [0.0]
        sorted_q = sorted(all_queue) if all_queue else [0.0]

        metrics: dict[str, Any] = {
            "concurrent_users": concurrent_users,
            "num_prompts_attempted": total_attempted,
            "num_prompts_succeeded": successful_prompts,
            "n_predict": self._n_predict,
            "total_tokens": total_tokens,
            "total_duration_s": round(total_duration, 3),
            "aggregate_throughput_tok_s": round(total_tokens / total_duration, 2)
            if total_duration > 0
            else 0.0,
            "per_user_throughput_tok_s": round(
                total_tokens / total_duration / concurrent_users, 2
            )
            if total_duration > 0
            else 0.0,
            "avg_ttft_ms": round(statistics.mean(sorted_ttft) * 1000, 2),
            "p50_ttft_ms": round(percentile(sorted_ttft, 50) * 1000, 2),
            "p99_ttft_ms": round(percentile(sorted_ttft, 99) * 1000, 2),
            "avg_itl_ms": round(statistics.mean(sorted_itl) * 1000, 2)
            if sorted_itl
            else 0.0,
            "p50_itl_ms": round(percentile(sorted_itl, 50) * 1000, 2),
            "p99_itl_ms": round(percentile(sorted_itl, 99) * 1000, 2),
            "avg_queue_ms": round(statistics.mean(sorted_q) * 1000, 2),
            "p50_queue_ms": round(percentile(sorted_q, 50) * 1000, 2),
            "p99_queue_ms": round(percentile(sorted_q, 99) * 1000, 2),
        }

        # Summarise non-fatal errors (if any) in a single log line each.
        if self._ctx_exceeded_count:
            log.info(
                "Level %d: %d prompt(s) skipped: prompt exceeds per-slot context size",
                concurrent_users,
                self._ctx_exceeded_count,
            )
        if self._disconnect_count:
            log.info(
                "Level %d: %d request(s) failed due to server disconnect",
                concurrent_users,
                self._disconnect_count,
            )

        return metrics

    def _distribute_prompts(self, concurrent_users: int) -> list[list[str]]:
        """Split prompts among users."""
        if self._prompt_distribution == "split":
            buckets: list[list[str]] = [[] for _ in range(concurrent_users)]
            for i, p in enumerate(self._prompts):
                buckets[i % concurrent_users].append(p)
            return buckets
        return [list(self._prompts) for _ in range(concurrent_users)]

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
            log.debug(
                "User %d prompt %d/%d (%d chars)",
                user_id,
                i + 1,
                len(prompts),
                len(prompt),
            )
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
        """Async streaming completion with queue-time tracking."""
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
                    body_text = body.decode()[:200]
                    if "exceed" in body_text.lower() and "context" in body_text.lower():
                        self._ctx_exceeded_count += 1
                        log.debug(
                            "/completion returned %d (context exceeded): %s",
                            response.status_code,
                            body_text,
                        )
                    else:
                        log.warning(
                            "/completion returned %d: %s",
                            response.status_code,
                            body_text,
                        )
                    return None

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue

                    body_str = line[len("data:") :].strip()
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
            self._disconnect_count += 1
            log.debug("Completion request failed: %s", exc)
            return None

        if t_first_token is None or n_tokens == 0:
            log.debug("No tokens received for prompt (len=%d)", len(prompt))
            return None

        ttft = t_first_token - t_request
        queue_time = (t_first_byte - t_request) if t_first_byte else ttft

        itl_list: list[float] = []
        prev = t_first_token
        for ts in token_timestamps:
            itl_list.append(ts - prev)
            prev = ts

        return ttft, itl_list, n_tokens, queue_time
