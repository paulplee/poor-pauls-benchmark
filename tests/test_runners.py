"""Tests for the runner registry and base class contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from runners import _REGISTRY, get_runner, register_runner
from runners.base import BaseRunner
from runners.llama_bench import LlamaBenchRunner


# ==========================================================================
# BaseRunner ABC
# ==========================================================================


class TestBaseRunnerContract:
    """Verify the abstract contract cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self) -> None:
        """BaseRunner itself is abstract — instantiation must fail."""
        with pytest.raises(TypeError):
            BaseRunner()  # type: ignore[abstract]

    def test_subclass_must_implement_all_methods(self) -> None:
        """A subclass missing any abstract method cannot be instantiated."""

        class Incomplete(BaseRunner):
            runner_type = "incomplete"
            def setup(self, runner_params: dict[str, Any]) -> None: ...
            # run() and teardown() intentionally missing

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_probe_ctx_default_raises(self) -> None:
        """The default probe_ctx must raise NotImplementedError."""

        class Minimal(BaseRunner):
            runner_type = "minimal"
            def setup(self, rp: dict[str, Any]) -> None: ...
            def run(self, config: dict[str, Any]) -> dict | None:
                return None
            def teardown(self) -> None: ...

        r = Minimal()
        with pytest.raises(NotImplementedError, match="minimal"):
            r.probe_ctx(Path("/fake.gguf"), 1024)


# ==========================================================================
# Runner registry
# ==========================================================================


class TestRunnerRegistry:
    """Verify the register → get round-trip and error handling."""

    def test_llama_bench_auto_registered(self) -> None:
        """The built-in llama-bench runner must be available by default."""
        assert "llama-bench" in _REGISTRY

    def test_get_runner_returns_correct_type(self) -> None:
        r = get_runner("llama-bench")
        assert isinstance(r, LlamaBenchRunner)

    def test_get_runner_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown runner_type"):
            get_runner("does-not-exist")

    def test_error_message_lists_available(self) -> None:
        """The ValueError message should list available runners."""
        with pytest.raises(ValueError, match="llama-bench"):
            get_runner("nope")

    def test_register_custom_runner(self, fake_runner) -> None:
        """A custom runner can be registered and then retrieved."""
        from tests.conftest import FakeRunner

        register_runner("fake", FakeRunner)
        try:
            r = get_runner("fake")
            assert isinstance(r, FakeRunner)
        finally:
            # Clean up to avoid polluting other tests
            _REGISTRY.pop("fake", None)

    def test_get_runner_returns_fresh_instance(self) -> None:
        """Each get_runner call should return a new instance."""
        r1 = get_runner("llama-bench")
        r2 = get_runner("llama-bench")
        assert r1 is not r2


# ==========================================================================
# LlamaBenchRunner
# ==========================================================================


class TestLlamaBenchRunner:
    """Unit tests for the llama-bench runner (mocked subprocess)."""

    def test_runner_type(self) -> None:
        r = LlamaBenchRunner()
        assert r.runner_type == "llama-bench"

    def test_setup_default_cmd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without params or env-var, falls back to 'llama-bench'."""
        monkeypatch.delenv("PPB_LLAMA_BENCH", raising=False)
        r = LlamaBenchRunner()
        r.setup({})
        assert r._cmd == "llama-bench"

    def test_setup_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PPB_LLAMA_BENCH env-var overrides the default."""
        monkeypatch.setenv("PPB_LLAMA_BENCH", "/opt/llama-bench")
        r = LlamaBenchRunner()
        r.setup({})
        assert r._cmd == "/opt/llama-bench"

    def test_setup_runner_params_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """runner_params['llama_bench_cmd'] takes highest precedence."""
        monkeypatch.setenv("PPB_LLAMA_BENCH", "/opt/llama-bench")
        r = LlamaBenchRunner()
        r.setup({"llama_bench_cmd": "/custom/llama-bench"})
        assert r._cmd == "/custom/llama-bench"

    def test_run_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Successful subprocess → parsed results dict."""
        import subprocess as sp

        fake_output = [{"avg_ts": 100.0, "model_filename": "test.gguf"}]

        def mock_run(cmd, **kw):
            m = type("Proc", (), {"returncode": 0, "stdout": __import__("json").dumps(fake_output), "stderr": ""})()
            return m

        monkeypatch.setattr(sp, "run", mock_run)

        r = LlamaBenchRunner()
        r.setup({})
        result = r.run({"model_path": "/fake.gguf", "n_ctx": 8192, "n_batch": 512})

        assert result is not None
        assert result["results"] == fake_output

    def test_run_nonzero_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-zero exit code → returns None."""
        import subprocess as sp

        def mock_run(cmd, **kw):
            return type("Proc", (), {"returncode": 1, "stdout": "", "stderr": "segfault"})()

        monkeypatch.setattr(sp, "run", mock_run)

        r = LlamaBenchRunner()
        r.setup({})
        assert r.run({"model_path": "/f.gguf", "n_ctx": 1, "n_batch": 1}) is None

    def test_run_bad_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Un-parseable stdout → returns None."""
        import subprocess as sp

        def mock_run(cmd, **kw):
            return type("Proc", (), {"returncode": 0, "stdout": "NOT JSON", "stderr": ""})()

        monkeypatch.setattr(sp, "run", mock_run)

        r = LlamaBenchRunner()
        r.setup({})
        assert r.run({"model_path": "/f.gguf", "n_ctx": 1, "n_batch": 1}) is None

    def test_probe_ctx_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Successful allocation → True."""
        import subprocess as sp

        def mock_run(cmd, **kw):
            return type("Proc", (), {"returncode": 0, "stdout": "[]", "stderr": ""})()

        monkeypatch.setattr(sp, "run", mock_run)

        r = LlamaBenchRunner()
        r.setup({})
        assert r.probe_ctx(Path("/fake.gguf"), 8192) is True

    def test_probe_ctx_oom_marker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OOM marker in output → False."""
        import subprocess as sp

        def mock_run(cmd, **kw):
            return type("Proc", (), {"returncode": 0, "stdout": "", "stderr": "CUDA error: out of memory"})()

        monkeypatch.setattr(sp, "run", mock_run)

        r = LlamaBenchRunner()
        r.setup({})
        assert r.probe_ctx(Path("/fake.gguf"), 131072) is False

    def test_probe_ctx_nonzero_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-zero exit → False."""
        import subprocess as sp

        def mock_run(cmd, **kw):
            return type("Proc", (), {"returncode": 1, "stdout": "", "stderr": ""})()

        monkeypatch.setattr(sp, "run", mock_run)

        r = LlamaBenchRunner()
        r.setup({})
        assert r.probe_ctx(Path("/fake.gguf"), 65536) is False

    def test_oom_markers_coverage(self) -> None:
        """All six known OOM markers must be present."""
        markers = LlamaBenchRunner.OOM_MARKERS
        assert len(markers) == 6
        assert "bad_alloc" in markers
        assert "cudaerroroutofmemory" in markers
        assert "rocm out of memory" in markers

    def test_teardown_is_noop(self) -> None:
        """teardown() must not raise."""
        r = LlamaBenchRunner()
        r.setup({})
        r.teardown()  # should not raise
