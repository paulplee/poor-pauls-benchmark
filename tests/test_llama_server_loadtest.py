"""Tests for the llama-server-loadtest runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from runners import _REGISTRY, get_runner
from runners.llama_server_loadtest import (
    LlamaServerLoadTestRunner,
    _default_user_steps,
)


# ==========================================================================
# Registry integration
# ==========================================================================


class TestLoadTestRegistration:
    """Ensure llama-server-loadtest is auto-registered."""

    def test_registered(self) -> None:
        assert "llama-server-loadtest" in _REGISTRY

    def test_get_runner_returns_correct_type(self) -> None:
        r = get_runner("llama-server-loadtest")
        assert isinstance(r, LlamaServerLoadTestRunner)

    def test_get_runner_returns_fresh_instance(self) -> None:
        r1 = get_runner("llama-server-loadtest")
        r2 = get_runner("llama-server-loadtest")
        assert r1 is not r2


# ==========================================================================
# Helpers
# ==========================================================================


class TestDefaultUserSteps:
    """Verify the power-of-two step generation."""

    def test_small(self) -> None:
        assert _default_user_steps(1) == [1]

    def test_power_of_two(self) -> None:
        assert _default_user_steps(8) == [1, 2, 4, 8]

    def test_non_power_of_two(self) -> None:
        steps = _default_user_steps(10)
        assert steps == [1, 2, 4, 8, 10]

    def test_large(self) -> None:
        steps = _default_user_steps(64)
        assert steps == [1, 2, 4, 8, 16, 32, 64]


# ==========================================================================
# LlamaServerLoadTestRunner — setup
# ==========================================================================


class TestLoadTestSetup:
    """Test setup() parameter parsing."""

    def test_runner_type(self) -> None:
        r = LlamaServerLoadTestRunner()
        assert r.runner_type == "llama-server-loadtest"

    @patch("runners.llama_server_loadtest.download_dataset")
    @patch("runners.llama_server_loadtest.load_sharegpt_prompts")
    def test_setup_defaults(self, mock_load: Mock, mock_download: Mock) -> None:
        mock_download.return_value = Path("/fake/data.json")
        mock_load.return_value = ["prompt1"]

        r = LlamaServerLoadTestRunner()
        r.setup({})
        assert r._max_users == 64
        assert r._error_threshold == 0.10
        assert r._ramp_delay_s == 1.0
        assert r._n_predict == 256
        assert r._user_steps is None
        assert r._prompt_distribution == "shared"

    @patch("runners.llama_server_loadtest.download_dataset")
    @patch("runners.llama_server_loadtest.load_sharegpt_prompts")
    def test_setup_custom_params(self, mock_load: Mock, mock_download: Mock) -> None:
        mock_download.return_value = Path("/fake/data.json")
        mock_load.return_value = ["a", "b"]

        r = LlamaServerLoadTestRunner()
        r.setup({
            "max_users": 32,
            "error_threshold": 0.05,
            "ramp_delay_s": 3.0,
            "user_steps": [1, 4, 8],
            "n_predict": 128,
            "prompt_distribution": "split",
        })
        assert r._max_users == 32
        assert r._error_threshold == 0.05
        assert r._ramp_delay_s == 3.0
        assert r._user_steps == [1, 4, 8]
        assert r._n_predict == 128
        assert r._prompt_distribution == "split"

    @patch("runners.llama_server_loadtest.download_dataset")
    @patch("runners.llama_server_loadtest.load_sharegpt_prompts")
    def test_setup_cmd_env_fallback(
        self, mock_load: Mock, mock_download: Mock, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("PPB_LLAMA_SERVER", "/env/llama-server")
        mock_download.return_value = Path("/fake/data.json")
        mock_load.return_value = ["a"]

        r = LlamaServerLoadTestRunner()
        r.setup({})
        assert r._cmd == "/env/llama-server"

    @patch("runners.llama_server_loadtest.download_dataset")
    @patch("runners.llama_server_loadtest.load_sharegpt_prompts")
    def test_setup_cmd_runner_params_override(
        self, mock_load: Mock, mock_download: Mock, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("PPB_LLAMA_SERVER", "/env/llama-server")
        mock_download.return_value = Path("/fake/data.json")
        mock_load.return_value = ["a"]

        r = LlamaServerLoadTestRunner()
        r.setup({"llama_server_cmd": "/custom/bin"})
        assert r._cmd == "/custom/bin"


# ==========================================================================
# LlamaServerLoadTestRunner — distribute prompts
# ==========================================================================


class TestDistributePrompts:

    def test_shared_mode(self) -> None:
        r = LlamaServerLoadTestRunner()
        r._prompts = ["a", "b", "c"]
        r._prompt_distribution = "shared"

        result = r._distribute_prompts(3)
        assert len(result) == 3
        for bucket in result:
            assert bucket == ["a", "b", "c"]

    def test_split_mode(self) -> None:
        r = LlamaServerLoadTestRunner()
        r._prompts = ["a", "b", "c", "d", "e", "f"]
        r._prompt_distribution = "split"

        result = r._distribute_prompts(3)
        assert len(result) == 3
        assert result[0] == ["a", "d"]
        assert result[1] == ["b", "e"]
        assert result[2] == ["c", "f"]

    def test_split_uneven(self) -> None:
        r = LlamaServerLoadTestRunner()
        r._prompts = ["a", "b", "c", "d", "e"]
        r._prompt_distribution = "split"

        result = r._distribute_prompts(3)
        # Round-robin: 0->a, 1->b, 2->c, 0->d, 1->e
        assert result[0] == ["a", "d"]
        assert result[1] == ["b", "e"]
        assert result[2] == ["c"]


# ==========================================================================
# LlamaServerLoadTestRunner — run (mocked escalation)
# ==========================================================================


class TestLoadTestRun:
    """Test run() with fully mocked server and async."""

    def _make_runner(self, prompts: list[str] | None = None) -> LlamaServerLoadTestRunner:
        r = LlamaServerLoadTestRunner()
        r._cmd = "/fake/llama-server"
        r._prompts = prompts or ["Hello, how are you?"]
        r._n_predict = 64
        r._health_timeout = 5.0
        r._max_users = 4
        r._user_steps = [1, 2, 4]
        r._error_threshold = 0.1
        r._ramp_delay_s = 0.0  # no delay in tests
        r._prompt_distribution = "shared"
        return r

    def _make_level_result(
        self,
        concurrent_users: int,
        succeeded: int = 5,
        attempted: int = 5,
        total_tokens: int = 35,
    ) -> dict[str, Any]:
        """Build a fake level result (matching _test_concurrency_level output)."""
        return {
            "concurrent_users": concurrent_users,
            "num_prompts_attempted": attempted,
            "num_prompts_succeeded": succeeded,
            "n_predict": 64,
            "total_tokens": total_tokens,
            "total_duration_s": 1.0,
            "aggregate_throughput_tok_s": 35.0,
            "per_user_throughput_tok_s": round(35.0 / concurrent_users, 2),
            "avg_ttft_ms": 50.0,
            "p50_ttft_ms": 48.0,
            "p99_ttft_ms": 55.0,
            "avg_itl_ms": 10.0,
            "p50_itl_ms": 9.0,
            "p99_itl_ms": 15.0,
            "avg_queue_ms": 5.0,
            "p50_queue_ms": 4.0,
            "p99_queue_ms": 8.0,
        }

    @patch("runners.llama_server_loadtest.time.sleep")
    @patch.object(LlamaServerLoadTestRunner, "stop_server")
    @patch.object(LlamaServerLoadTestRunner, "start_server")
    def test_run_success_all_levels_pass(
        self, mock_start: Mock, mock_stop: Mock, mock_sleep: Mock,
    ) -> None:
        """All concurrency levels pass → max_sustainable_users = highest level."""
        proc = MagicMock()
        mock_start.return_value = proc

        r = self._make_runner()

        results_by_level = {
            1: self._make_level_result(1),
            2: self._make_level_result(2),
            4: self._make_level_result(4),
        }

        async def fake_test_level(level: int):
            return results_by_level[level]

        with patch.object(r, "_test_concurrency_level", side_effect=fake_test_level):
            result = r.run({"model_path": "/fake.gguf", "n_ctx": 4096})

        assert result is not None
        data = result["results"]
        assert data["max_sustainable_users"] == 4
        assert len(data["concurrency_curve"]) == 3

    @patch("runners.llama_server_loadtest.time.sleep")
    @patch.object(LlamaServerLoadTestRunner, "stop_server")
    @patch.object(LlamaServerLoadTestRunner, "start_server")
    def test_run_stops_at_threshold(
        self, mock_start: Mock, mock_stop: Mock, mock_sleep: Mock,
    ) -> None:
        """Escalation stops when error rate exceeds threshold."""
        proc = MagicMock()
        mock_start.return_value = proc

        r = self._make_runner()

        level2_bad = self._make_level_result(2, succeeded=4, attempted=5)

        results_by_level = {
            1: self._make_level_result(1),
            2: level2_bad,  # 20% error rate > 10%
        }

        async def fake_test_level(level: int):
            return results_by_level.get(level)

        with patch.object(r, "_test_concurrency_level", side_effect=fake_test_level):
            result = r.run({"model_path": "/fake.gguf", "n_ctx": 4096})

        assert result is not None
        data = result["results"]
        assert data["max_sustainable_users"] == 1  # only level 1 passed
        assert len(data["concurrency_curve"]) == 2  # tested 1 and 2

    @patch.object(LlamaServerLoadTestRunner, "stop_server")
    @patch.object(LlamaServerLoadTestRunner, "start_server")
    def test_run_server_start_failure(
        self, mock_start: Mock, mock_stop: Mock,
    ) -> None:
        """If server fails to start, returns None."""
        mock_start.side_effect = TimeoutError("health-check timed out")

        r = self._make_runner()
        result = r.run({"model_path": "/fake.gguf", "n_ctx": 4096})
        assert result is None

    @patch("runners.llama_server_loadtest.time.sleep")
    @patch.object(LlamaServerLoadTestRunner, "stop_server")
    @patch.object(LlamaServerLoadTestRunner, "start_server")
    def test_run_first_level_fails(
        self, mock_start: Mock, mock_stop: Mock, mock_sleep: Mock,
    ) -> None:
        """If even the first level returns None, result is None."""
        proc = MagicMock()
        mock_start.return_value = proc

        r = self._make_runner()

        async def fake_test_level(level: int):
            return None

        with patch.object(r, "_test_concurrency_level", side_effect=fake_test_level):
            result = r.run({"model_path": "/fake.gguf", "n_ctx": 4096})

        assert result is None

    @patch("runners.llama_server_loadtest.time.sleep")
    @patch.object(LlamaServerLoadTestRunner, "stop_server")
    @patch.object(LlamaServerLoadTestRunner, "start_server")
    def test_server_stopped_even_on_failure(
        self, mock_start: Mock, mock_stop: Mock, mock_sleep: Mock,
    ) -> None:
        """stop_server is always called, even when escalation blows up."""
        proc = MagicMock()
        mock_start.return_value = proc

        r = self._make_runner()

        async def fail_level(level: int):
            raise RuntimeError("boom")

        with patch.object(r, "_test_concurrency_level", side_effect=fail_level):
            with pytest.raises(RuntimeError, match="boom"):
                r.run({"model_path": "/fake.gguf", "n_ctx": 4096})

        mock_stop.assert_called_once_with(proc)

    @patch("runners.llama_server_loadtest.time.sleep")
    @patch.object(LlamaServerLoadTestRunner, "stop_server")
    @patch.object(LlamaServerLoadTestRunner, "start_server")
    def test_result_structure(
        self, mock_start: Mock, mock_stop: Mock, mock_sleep: Mock,
    ) -> None:
        """Verify top-level keys in the returned result."""
        proc = MagicMock()
        mock_start.return_value = proc

        r = self._make_runner()
        r._user_steps = [1]

        async def fake_test_level(level: int):
            return self._make_level_result(1)

        with patch.object(r, "_test_concurrency_level", side_effect=fake_test_level):
            result = r.run({"model_path": "/fake.gguf", "n_ctx": 4096})

        assert result is not None
        data = result["results"]
        assert "max_sustainable_users" in data
        assert "error_threshold" in data
        assert "concurrency_curve" in data
        assert isinstance(data["concurrency_curve"], list)


# ==========================================================================
# LlamaServerLoadTestRunner — teardown
# ==========================================================================


class TestLoadTestTeardown:

    def test_teardown_no_process(self) -> None:
        r = LlamaServerLoadTestRunner()
        r.teardown()  # no-op

    def test_teardown_kills_lingering(self) -> None:
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 1234

        r = LlamaServerLoadTestRunner()
        r._process = proc
        r.teardown()

        proc.send_signal.assert_called()

    def test_teardown_already_exited(self) -> None:
        proc = MagicMock()
        proc.poll.return_value = 0

        r = LlamaServerLoadTestRunner()
        r._process = proc
        r.teardown()

        proc.send_signal.assert_not_called()
