from __future__ import annotations

import socket
import sys
from pathlib import Path

import yaml


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_config(tmp_path: Path, *, repeats: int = 2, fail_benchmark: bool = False) -> Path:
    fixture_directory = Path(__file__).parent / "fixtures"
    port = free_port()
    benchmark_arguments = ["--label", "test"]
    if fail_benchmark:
        benchmark_arguments.append("--fail")
    document = {
        "schema_version": 1,
        "experiment": {"name": "test-run", "repeats": repeats},
        "paths": {"results_root": "./results", "working_directory": "."},
        "server": {
            "command": [sys.executable, str(fixture_directory / "fake_server.py")],
            "arguments": ["--host", "127.0.0.1", "--port", str(port)],
            "host": "127.0.0.1",
            "port": port,
            "readiness_path": "/health",
            "readiness_timeout_seconds": 3,
            "readiness_poll_interval_seconds": 0.03,
            "shutdown_timeout_seconds": 2,
        },
        "benchmark": {
            "command": [sys.executable, str(fixture_directory / "fake_benchmark.py")],
            "arguments": benchmark_arguments,
            "timeout_seconds": 3,
        },
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path
