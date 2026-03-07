"""Tests for the llama-server runner and ShareGPT dataset module."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest

from runners import _REGISTRY, get_runner
from runners.llama_server import (
    LlamaServerRunner,
    _find_free_port,
    _parse_sse_lines,
    _percentile,
)


# ==========================================================================
# Registry integration
# ==========================================================================


class TestLlamaServerRegistration:
    """Ensure llama-server is auto-registered."""

    def test_registered(self) -> None:
        assert "llama-server" in _REGISTRY

    def test_get_runner_returns_correct_type(self) -> None:
        r = get_runner("llama-server")
        assert isinstance(r, LlamaServerRunner)

    def test_get_runner_returns_fresh_instance(self) -> None:
        r1 = get_runner("llama-server")
        r2 = get_runner("llama-server")
        assert r1 is not r2


# ==========================================================================
# Helpers
# ==========================================================================


class TestFindFreePort:
    """Verify dynamic port allocation."""

    def test_returns_int(self) -> None:
        port = _find_free_port()
        assert isinstance(port, int)

    def test_port_in_range(self) -> None:
        port = _find_free_port()
        assert 1024 <= port <= 65535

    def test_different_ports(self) -> None:
        """Two consecutive calls should (almost certainly) return different ports."""
        ports = {_find_free_port() for _ in range(5)}
        # At least 2 distinct ports out of 5
        assert len(ports) >= 2


class TestPercentile:
    """Verify the simplistic percentile helper."""

    def test_empty(self) -> None:
        assert _percentile([], 50) == 0.0

    def test_single(self) -> None:
        assert _percentile([1.0], 50) == 1.0

    def test_median(self) -> None:
        assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0

    def test_p99(self) -> None:
        data = list(range(100))
        assert _percentile(data, 99) == 99


class TestParseSSELines:
    """Test SSE line parsing."""

    def test_basic(self) -> None:
        lines = [
            'data: {"content": "Hello", "stop": false}',
            'data: {"content": " world", "stop": false}',
            'data: {"content": "", "stop": true}',
        ]
        payloads = _parse_sse_lines(lines)
        assert len(payloads) == 3
        assert payloads[0]["content"] == "Hello"
        assert payloads[1]["content"] == " world"

    def test_skip_comments_and_blanks(self) -> None:
        lines = ["", ": comment", 'data: {"content": "ok", "stop": false}']
        payloads = _parse_sse_lines(lines)
        assert len(payloads) == 1

    def test_skip_done(self) -> None:
        lines = ["data: [DONE]"]
        assert _parse_sse_lines(lines) == []

    def test_skip_bad_json(self) -> None:
        lines = ["data: NOT_JSON"]
        assert _parse_sse_lines(lines) == []


# ==========================================================================
# LlamaServerRunner — setup
# ==========================================================================


class TestLlamaServerSetup:
    """Test setup() binary resolution and prompt loading."""

    def test_runner_type(self) -> None:
        r = LlamaServerRunner()
        assert r.runner_type == "llama-server"

    @patch("runners.llama_server.download_sharegpt")
    @patch("runners.llama_server.load_sharegpt_prompts")
    def test_setup_default_cmd(
        self,
        mock_load: Mock,
        mock_download: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without params or env-var, falls back to 'llama-server'."""
        monkeypatch.delenv("PPB_LLAMA_SERVER", raising=False)
        mock_download.return_value = Path("/fake/data.json")
        mock_load.return_value = ["prompt1"]

        r = LlamaServerRunner()
        r.setup({})
        assert r._cmd == "llama-server"

    @patch("runners.llama_server.download_sharegpt")
    @patch("runners.llama_server.load_sharegpt_prompts")
    def test_setup_env_var(
        self,
        mock_load: Mock,
        mock_download: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """PPB_LLAMA_SERVER env-var overrides the default."""
        monkeypatch.setenv("PPB_LLAMA_SERVER", "/opt/llama-server")
        mock_download.return_value = Path("/fake/data.json")
        mock_load.return_value = ["prompt1"]

        r = LlamaServerRunner()
        r.setup({})
        assert r._cmd == "/opt/llama-server"

    @patch("runners.llama_server.download_sharegpt")
    @patch("runners.llama_server.load_sharegpt_prompts")
    def test_setup_runner_params_override(
        self,
        mock_load: Mock,
        mock_download: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """runner_params['llama_server_cmd'] takes highest precedence."""
        monkeypatch.setenv("PPB_LLAMA_SERVER", "/opt/llama-server")
        mock_download.return_value = Path("/fake/data.json")
        mock_load.return_value = ["prompt1"]

        r = LlamaServerRunner()
        r.setup({"llama_server_cmd": "/custom/llama-server"})
        assert r._cmd == "/custom/llama-server"

    @patch("runners.llama_server.download_sharegpt")
    @patch("runners.llama_server.load_sharegpt_prompts")
    def test_setup_num_prompts(
        self,
        mock_load: Mock,
        mock_download: Mock,
    ) -> None:
        """num_prompts param is forwarded to load_sharegpt_prompts."""
        mock_download.return_value = Path("/fake/data.json")
        mock_load.return_value = ["a", "b", "c"]

        r = LlamaServerRunner()
        r.setup({"num_prompts": 3})

        mock_load.assert_called_once_with(Path("/fake/data.json"), max_prompts=3)

    @patch("runners.llama_server.download_sharegpt")
    @patch("runners.llama_server.load_sharegpt_prompts")
    def test_setup_n_predict(
        self,
        mock_load: Mock,
        mock_download: Mock,
    ) -> None:
        """n_predict is stored from runner_params."""
        mock_download.return_value = Path("/fake/data.json")
        mock_load.return_value = ["a"]

        r = LlamaServerRunner()
        r.setup({"n_predict": 128})
        assert r._n_predict == 128


# ==========================================================================
# LlamaServerRunner — run (mocked)
# ==========================================================================


def _make_sse_response(tokens: list[str]) -> str:
    """Build a fake SSE response body from a list of token strings."""
    lines = []
    for tok in tokens:
        lines.append(f'data: {json.dumps({"content": tok, "stop": False})}')
    lines.append(f'data: {json.dumps({"content": "", "stop": True})}')
    return "\n".join(lines)


class FakeStreamResponse:
    """Minimal mock for httpx streaming response context manager."""

    def __init__(self, tokens: list[str], status_code: int = 200) -> None:
        self.status_code = status_code
        self._lines = _make_sse_response(tokens).split("\n")

    def iter_lines(self):
        yield from self._lines

    def read(self):
        return b"error"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeHealthResponse:
    """Minimal mock for httpx.get health check response."""

    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code


class TestLlamaServerRun:
    """Test run() with fully mocked subprocess and HTTP."""

    def _make_runner(self, prompts: list[str] | None = None) -> LlamaServerRunner:
        """Return a runner with pre-loaded prompts (skips setup)."""
        r = LlamaServerRunner()
        r._cmd = "/fake/llama-server"
        r._prompts = prompts or ["Hello, how are you?", "What is AI?"]
        r._n_predict = 64
        r._health_timeout = 5.0
        return r

    @patch("runners.llama_server._find_free_port", return_value=12345)
    @patch("runners.llama_server.httpx.get")
    @patch("subprocess.Popen")
    def test_run_success(
        self,
        mock_popen: Mock,
        mock_get: Mock,
        mock_port: Mock,
    ) -> None:
        """Successful run returns metrics dict with all expected keys."""
        # Mock the subprocess
        proc = MagicMock()
        proc.poll.return_value = None  # Server "running"
        proc.pid = 9999
        mock_popen.return_value = proc

        # Mock health check
        mock_get.return_value = FakeHealthResponse(200)

        r = self._make_runner(["Hello world"])

        # Mock the httpx.Client.stream
        tokens = ["Hello", " there", "!", " How", " are", " you", "?"]
        fake_stream = FakeStreamResponse(tokens)

        with patch("runners.llama_server.httpx.Client") as MockClient:
            client_inst = MagicMock()
            MockClient.return_value.__enter__ = Mock(return_value=client_inst)
            MockClient.return_value.__exit__ = Mock(return_value=False)
            client_inst.stream.return_value = fake_stream

            result = r.run({"model_path": "/fake.gguf", "n_ctx": 4096})

        assert result is not None
        metrics = result["results"]

        # Check all required metric keys exist
        expected_keys = {
            "num_prompts_attempted",
            "num_prompts_succeeded",
            "n_predict",
            "total_tokens",
            "total_duration_s",
            "throughput_tok_s",
            "avg_ttft_ms",
            "p50_ttft_ms",
            "p99_ttft_ms",
            "avg_itl_ms",
            "p50_itl_ms",
            "p99_itl_ms",
        }
        assert expected_keys.issubset(metrics.keys())

        assert metrics["num_prompts_attempted"] == 1
        assert metrics["num_prompts_succeeded"] == 1
        assert metrics["total_tokens"] == 7
        assert metrics["avg_ttft_ms"] >= 0
        assert metrics["avg_itl_ms"] >= 0

    @patch("runners.llama_server._find_free_port", return_value=12345)
    @patch("runners.llama_server.httpx.get")
    @patch("subprocess.Popen")
    def test_run_server_crash_returns_none(
        self,
        mock_popen: Mock,
        mock_get: Mock,
        mock_port: Mock,
    ) -> None:
        """If the server crashes during health check, run() returns None."""
        proc = MagicMock()
        proc.poll.return_value = 1  # already exited
        proc.returncode = 1
        proc.stderr = MagicMock()
        proc.stderr.read.return_value = "segfault"
        proc.pid = 9999
        mock_popen.return_value = proc

        r = self._make_runner()
        result = r.run({"model_path": "/fake.gguf", "n_ctx": 4096})
        assert result is None

    @patch("runners.llama_server._find_free_port", return_value=12345)
    @patch("runners.llama_server.httpx.get")
    @patch("subprocess.Popen")
    def test_run_no_prompts_returns_none(
        self,
        mock_popen: Mock,
        mock_get: Mock,
        mock_port: Mock,
    ) -> None:
        """If all prompts fail, run() returns None."""
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 9999
        mock_popen.return_value = proc
        mock_get.return_value = FakeHealthResponse(200)

        # No prompts at all
        r = self._make_runner(prompts=[])

        with patch("runners.llama_server.httpx.Client") as MockClient:
            client_inst = MagicMock()
            MockClient.return_value.__enter__ = Mock(return_value=client_inst)
            MockClient.return_value.__exit__ = Mock(return_value=False)

            result = r.run({"model_path": "/fake.gguf", "n_ctx": 4096})

        assert result is None


# ==========================================================================
# LlamaServerRunner — probe_ctx
# ==========================================================================


class TestLlamaServerProbeCtx:
    """Test probe_ctx with mocked server lifecycle."""

    @patch("runners.llama_server._find_free_port", return_value=12345)
    @patch("runners.llama_server.httpx.get")
    @patch("subprocess.Popen")
    def test_probe_ctx_pass(
        self,
        mock_popen: Mock,
        mock_get: Mock,
        mock_port: Mock,
    ) -> None:
        """Successful health check → True."""
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 9999
        mock_popen.return_value = proc
        mock_get.return_value = FakeHealthResponse(200)

        r = LlamaServerRunner()
        r._cmd = "/fake/llama-server"
        r._health_timeout = 5.0

        assert r.probe_ctx(Path("/fake.gguf"), 8192) is True

    @patch("runners.llama_server._find_free_port", return_value=12345)
    @patch("subprocess.Popen")
    def test_probe_ctx_crash(
        self,
        mock_popen: Mock,
        mock_port: Mock,
    ) -> None:
        """Server crash during startup → False."""
        proc = MagicMock()
        proc.poll.return_value = 1
        proc.returncode = 1
        proc.stderr = MagicMock()
        proc.stderr.read.return_value = "out of memory"
        proc.pid = 9999
        mock_popen.return_value = proc

        r = LlamaServerRunner()
        r._cmd = "/fake/llama-server"
        r._health_timeout = 5.0

        assert r.probe_ctx(Path("/fake.gguf"), 131072) is False


# ==========================================================================
# LlamaServerRunner — teardown
# ==========================================================================


class TestLlamaServerTeardown:
    """Test teardown() process cleanup."""

    def test_teardown_no_process(self) -> None:
        """teardown() with no process should not raise."""
        r = LlamaServerRunner()
        r.teardown()  # no-op

    def test_teardown_kills_lingering(self) -> None:
        """teardown() should stop a lingering process."""
        proc = MagicMock()
        proc.poll.return_value = None  # still running
        proc.pid = 1234

        r = LlamaServerRunner()
        r._process = proc
        r.teardown()

        proc.send_signal.assert_called()

    def test_teardown_already_exited(self) -> None:
        """teardown() with already-exited process is a no-op."""
        proc = MagicMock()
        proc.poll.return_value = 0  # already exited

        r = LlamaServerRunner()
        r._process = proc
        r.teardown()

        proc.send_signal.assert_not_called()


# ==========================================================================
# LlamaServerRunner — _stop_server
# ==========================================================================


class TestStopServer:
    """Test graceful shutdown flow."""

    def test_sigterm_then_wait(self) -> None:
        """Normal shutdown: SIGTERM → wait."""
        import signal

        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 42

        r = LlamaServerRunner()
        r._stop_server(proc)

        proc.send_signal.assert_called_once_with(signal.SIGTERM)
        proc.wait.assert_called_once()

    def test_sigkill_on_timeout(self) -> None:
        """If SIGTERM wait times out, fall back to SIGKILL."""
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 42
        # First wait (after SIGTERM) times out; second wait (after SIGKILL) succeeds.
        proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="llama-server", timeout=5),
            None,
        ]

        r = LlamaServerRunner()
        r._stop_server(proc)

        proc.kill.assert_called_once()

    def test_already_exited(self) -> None:
        """If process already exited, do nothing."""
        proc = MagicMock()
        proc.poll.return_value = 0

        r = LlamaServerRunner()
        r._stop_server(proc)

        proc.send_signal.assert_not_called()


# ==========================================================================
# LlamaServerRunner — _stream_completion
# ==========================================================================


class TestStreamCompletion:
    """Test single-prompt streaming and TTFT/ITL measurement."""

    def test_basic_metrics(self) -> None:
        """Normal stream returns valid TTFT and ITL values."""
        tokens = ["A", "B", "C", "D"]
        fake_stream = FakeStreamResponse(tokens)

        client = MagicMock()
        client.stream.return_value = fake_stream

        r = LlamaServerRunner()
        r._n_predict = 64
        result = r._stream_completion(client, "Test prompt")

        assert result is not None
        ttft, itl_list, n_tokens = result
        assert n_tokens == 4
        assert ttft >= 0
        assert len(itl_list) == 3  # N-1 inter-token gaps
        assert all(x >= 0 for x in itl_list)

    def test_http_error_returns_none(self) -> None:
        """HTTP error → None."""
        import httpx

        client = MagicMock()
        client.stream.side_effect = httpx.HTTPError("timeout")

        r = LlamaServerRunner()
        r._n_predict = 64
        result = r._stream_completion(client, "Test prompt")
        assert result is None

    def test_no_tokens_returns_none(self) -> None:
        """Empty response → None."""
        fake_stream = FakeStreamResponse([])  # only the stop event

        client = MagicMock()
        client.stream.return_value = fake_stream

        r = LlamaServerRunner()
        r._n_predict = 64
        result = r._stream_completion(client, "prompt")

        # The only event has content="" (stop=true), so no real tokens
        assert result is None

    def test_non_200_returns_none(self) -> None:
        """Non-200 status code → None."""
        fake_stream = FakeStreamResponse([], status_code=500)

        client = MagicMock()
        client.stream.return_value = fake_stream

        r = LlamaServerRunner()
        r._n_predict = 64
        result = r._stream_completion(client, "prompt")
        assert result is None


# ==========================================================================
# ShareGPT dataset module
# ==========================================================================


class TestShareGPTDataset:
    """Test the datasets module (mocked HF download)."""

    def test_load_sharegpt_prompts_filters_short(self, tmp_path: Path) -> None:
        """Prompts shorter than _MIN_PROMPT_LENGTH are filtered out."""
        from datasets.sharegpt import load_sharegpt_prompts

        data = [
            {"conversations": [{"from": "human", "value": "Hi"}]},  # too short
            {"conversations": [{"from": "human", "value": "Tell me about quantum computing in detail please"}]},
        ]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))

        prompts = load_sharegpt_prompts(f, max_prompts=10)
        assert len(prompts) == 1
        assert "quantum" in prompts[0]

    def test_load_sharegpt_prompts_max(self, tmp_path: Path) -> None:
        """max_prompts limits the number of returned prompts."""
        from datasets.sharegpt import load_sharegpt_prompts

        data = [
            {"conversations": [{"from": "human", "value": f"Prompt number {i} with enough length to be valid"}]}
            for i in range(20)
        ]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))

        prompts = load_sharegpt_prompts(f, max_prompts=5)
        assert len(prompts) == 5

    def test_load_sharegpt_prompts_skips_non_human(self, tmp_path: Path) -> None:
        """Conversations with no human turn are skipped."""
        from datasets.sharegpt import load_sharegpt_prompts

        data = [
            {"conversations": [{"from": "gpt", "value": "I am a bot response only, no human turn here."}]},
        ]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))

        prompts = load_sharegpt_prompts(f, max_prompts=10)
        assert len(prompts) == 0

    @patch("datasets.sharegpt.hf_hub_download")
    def test_download_sharegpt(self, mock_dl: Mock, tmp_path: Path) -> None:
        """download_sharegpt calls hf_hub_download and returns a Path."""
        from datasets.sharegpt import download_sharegpt

        mock_dl.return_value = str(tmp_path / "data.json")
        result = download_sharegpt(dataset_dir=tmp_path)

        assert isinstance(result, Path)
        mock_dl.assert_called_once()


# ==========================================================================
# download-dataset CLI command
# ==========================================================================


class TestDownloadDatasetCommand:
    """Test the ``ppb download-dataset`` CLI command."""

    def test_success(self, tmp_path: Path) -> None:
        """Successful download prints the cached path."""
        from typer.testing import CliRunner
        from ppb import app

        runner = CliRunner()
        fake_path = tmp_path / "data.json"

        with patch("ppb.download_sharegpt", return_value=fake_path) as mock_dl:
            result = runner.invoke(app, ["download-dataset"])

        assert result.exit_code == 0
        assert "Dataset ready" in result.output
        mock_dl.assert_called_once()

    def test_success_with_custom_dir(self, tmp_path: Path) -> None:
        """--dataset-dir is forwarded to download_sharegpt."""
        from typer.testing import CliRunner
        from ppb import app

        runner = CliRunner()
        custom_dir = tmp_path / "custom"
        fake_path = custom_dir / "data.json"

        with patch("ppb.download_sharegpt", return_value=fake_path) as mock_dl:
            result = runner.invoke(
                app, ["download-dataset", "--dataset-dir", str(custom_dir)]
            )

        assert result.exit_code == 0
        mock_dl.assert_called_once_with(dataset_dir=custom_dir)

    def test_download_failure(self) -> None:
        """Network/HF error → exit code 1 with error message."""
        from typer.testing import CliRunner
        from ppb import app

        runner = CliRunner()

        with patch(
            "ppb.download_sharegpt",
            side_effect=OSError("Connection refused"),
        ):
            result = runner.invoke(app, ["download-dataset"])

        assert result.exit_code == 1
        assert "download failed" in result.output.lower()

    def test_help_text(self) -> None:
        """--help should mention ShareGPT and llama-server."""
        from typer.testing import CliRunner
        from ppb import app

        runner = CliRunner()
        result = runner.invoke(app, ["download-dataset", "--help"])

        assert result.exit_code == 0
        assert "ShareGPT" in result.output
        assert "llama-server" in result.output
