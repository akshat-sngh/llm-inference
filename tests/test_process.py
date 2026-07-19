from __future__ import annotations

import sys
from pathlib import Path

import pytest

from llm_inference_experiments.commands import CommandSpec
from llm_inference_experiments.errors import ReadinessError
from llm_inference_experiments.process import ProcessRunner, wait_for_readiness

from .conftest import free_port


def test_readiness_timeout(tmp_path: Path) -> None:
    spec = CommandSpec(
        "server",
        [sys.executable, "-c", "import time; time.sleep(3)"],
        tmp_path,
        {},
        None,
    )
    runner = ProcessRunner()
    process = runner.start(spec, tmp_path / "out.log", tmp_path / "err.log")
    try:
        with pytest.raises(ReadinessError, match="did not become ready"):
            wait_for_readiness("http://127.0.0.1:9/health", process, 0.1, 0.02)
    finally:
        process.stop(1)


def test_server_exit_before_readiness(tmp_path: Path) -> None:
    spec = CommandSpec("server", [sys.executable, "-c", "raise SystemExit(4)"], tmp_path, {}, None)
    runner = ProcessRunner()
    process = runner.start(spec, tmp_path / "out.log", tmp_path / "err.log")
    try:
        with pytest.raises(ReadinessError, match="return code 4"):
            wait_for_readiness("http://127.0.0.1:9/health", process, 1, 0.02)
    finally:
        process.stop(1)


def test_non_success_http_status_is_not_ready(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures/fake_server.py"
    port = free_port()
    spec = CommandSpec(
        "server",
        [
            sys.executable,
            str(fixture),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--health-status",
            "503",
        ],
        tmp_path,
        {},
        None,
    )
    process = ProcessRunner().start(spec, tmp_path / "out.log", tmp_path / "err.log")
    try:
        with pytest.raises(ReadinessError, match="did not become ready"):
            wait_for_readiness(f"http://127.0.0.1:{port}/health", process, 0.15, 0.02)
    finally:
        process.stop(1)


def test_subprocess_inherits_parent_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_EXP_PARENT_VALUE", "inherited")
    spec = CommandSpec(
        "environment",
        [sys.executable, "-c", "import os; print(os.environ['LLM_EXP_PARENT_VALUE'])"],
        tmp_path,
        {},
        1,
    )
    result = ProcessRunner().run(spec, tmp_path / "out.log", tmp_path / "err.log")
    assert result.return_code == 0
    assert (tmp_path / "out.log").read_text() == "inherited\n"


def test_subprocess_environment_overrides_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_EXP_PARENT_VALUE", "inherited")
    spec = CommandSpec(
        "environment",
        [sys.executable, "-c", "import os; print(os.environ['LLM_EXP_PARENT_VALUE'])"],
        tmp_path,
        {"LLM_EXP_PARENT_VALUE": "overridden"},
        1,
    )
    result = ProcessRunner().run(spec, tmp_path / "out.log", tmp_path / "err.log")
    assert result.return_code == 0
    assert (tmp_path / "out.log").read_text() == "overridden\n"
