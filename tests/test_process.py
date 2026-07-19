from __future__ import annotations

import sys
from pathlib import Path

import pytest

from llm_inference_experiments.commands import CommandSpec
from llm_inference_experiments.errors import ReadinessError
from llm_inference_experiments.process import ProcessRunner, wait_for_readiness


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
