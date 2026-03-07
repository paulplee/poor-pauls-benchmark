"""Shared fixtures for the PPB test suite."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from runners.base import BaseRunner


# ---------------------------------------------------------------------------
# Fake runner for integration tests (doesn't need llama-bench installed)
# ---------------------------------------------------------------------------


class FakeRunner(BaseRunner):
    """Deterministic in-memory runner for testing the orchestration layer."""

    runner_type: str = "fake"

    def __init__(self) -> None:
        self.setup_called = False
        self.teardown_called = False
        self.run_calls: list[dict[str, Any]] = []
        self.probe_calls: list[tuple[Path, int]] = []

        # Test hooks — override these to control behaviour:
        self.run_return: dict | None = {
            "results": [
                {
                    "avg_ts": 42.0,
                    "model_filename": "fake.gguf",
                    "n_prompt": 8192,
                    "n_gen": 0,
                }
            ]
        }
        self.probe_return: bool = True

    def setup(self, runner_params: dict[str, Any]) -> None:
        self.setup_called = True
        self._params = runner_params

    def run(self, config: dict[str, Any]) -> dict | None:
        self.run_calls.append(config)
        return self.run_return

    def teardown(self) -> None:
        self.teardown_called = True

    def probe_ctx(self, model_path: Path, n_ctx: int) -> bool:
        self.probe_calls.append((model_path, n_ctx))
        return self.probe_return


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_runner() -> FakeRunner:
    """Return a fresh :class:`FakeRunner` instance."""
    return FakeRunner()


@pytest.fixture()
def tmp_model(tmp_path: Path) -> Path:
    """Create a tiny dummy ``.gguf`` file and return its path."""
    model = tmp_path / "test-model.gguf"
    model.write_bytes(b"\x00" * 64)
    return model


@pytest.fixture()
def tmp_model_dir(tmp_path: Path) -> Path:
    """Create a directory with two dummy ``.gguf`` files."""
    d = tmp_path / "models"
    d.mkdir()
    for name in ("alpha.gguf", "beta.gguf"):
        (d / name).write_bytes(b"\x00" * 64)
    return d


@pytest.fixture()
def sweep_toml(tmp_path: Path, tmp_model: Path) -> Path:
    """Write a minimal sweep TOML and return the path."""
    cfg = tmp_path / "sweep.toml"
    cfg.write_text(
        textwrap.dedent(f"""\
        [sweep]
        model_path = "{tmp_model}"
        n_ctx = [512, 1024]
        n_batch = [256]
        """)
    )
    return cfg


@pytest.fixture()
def sweep_toml_with_runner(tmp_path: Path, tmp_model: Path) -> Path:
    """Write a sweep TOML that specifies ``runner_type = "fake"``."""
    cfg = tmp_path / "sweep_fake.toml"
    cfg.write_text(
        textwrap.dedent(f"""\
        [sweep]
        runner_type = "fake"
        model_path = "{tmp_model}"
        n_ctx = [512, 1024]
        n_batch = [256]

        [sweep.runner_params]
        custom_key = "custom_value"
        """)
    )
    return cfg
