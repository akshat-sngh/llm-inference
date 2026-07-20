from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

import yaml


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_config(
    tmp_path: Path,
    *,
    repeats: int = 2,
    fail_benchmark: bool = False,
    benchmark_timeout: float = 3,
    benchmark_sleep_seconds: float = 0.02,
    enable_warmup: bool = False,
    fail_warmup: bool = False,
    warmup_timeout: float = 3,
    warmup_sleep_seconds: float = 0.02,
    telemetry: dict[str, object] | None = None,
    server_health_status: int = 200,
    readiness_timeout: float = 3,
    vllm: dict[str, object] | None = None,
) -> Path:
    fixture_directory = Path(__file__).parent / "fixtures"
    port = free_port()
    benchmark_arguments = ["--label", "test", "--sleep-seconds", str(benchmark_sleep_seconds)]
    if fail_benchmark:
        benchmark_arguments.append("--fail")
    warmup_arguments = ["--label", "warmup", "--sleep-seconds", str(warmup_sleep_seconds)]
    if fail_warmup:
        warmup_arguments.append("--fail")
    document = {
        "schema_version": 1,
        "experiment": {"name": "test-run", "repeats": repeats},
        "paths": {"results_root": "./results", "working_directory": "."},
        "server": {
            "command": [sys.executable, str(fixture_directory / "fake_server.py")],
            "arguments": [
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--health-status",
                str(server_health_status),
            ],
            "host": "127.0.0.1",
            "port": port,
            "readiness_path": "/health",
            "readiness_timeout_seconds": readiness_timeout,
            "readiness_poll_interval_seconds": 0.03,
            "shutdown_timeout_seconds": 2,
        },
        "benchmark": {
            "command": [sys.executable, str(fixture_directory / "fake_benchmark.py")],
            "arguments": benchmark_arguments,
            "timeout_seconds": benchmark_timeout,
        },
    }
    if enable_warmup:
        document["warmup"] = {
            "enabled": True,
            "command": [sys.executable, str(fixture_directory / "fake_benchmark.py")],
            "arguments": warmup_arguments,
            "timeout_seconds": warmup_timeout,
        }
    if telemetry is not None:
        document["telemetry"] = {"nvidia": telemetry}
    if vllm is not None:
        document["vllm"] = vllm
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def create_fake_vllm_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "fake-vllm-repository"
    repository.mkdir()
    (repository / "marker.txt").write_text("fake vllm repository\n")
    for arguments in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "tests@example.invalid"],
        ["git", "config", "user.name", "Test User"],
        ["git", "add", "marker.txt"],
        ["git", "commit", "-qm", "fake vllm"],
    ):
        subprocess.run(arguments, cwd=repository, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    return repository, head
