"""Portable subprocess and HTTP readiness management."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO
from urllib.error import URLError
from urllib.request import urlopen

from .commands import CommandSpec
from .errors import ProcessError, ProcessTimeoutError, ReadinessError


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ProcessResult:
    command: list[str]
    started_at: str
    ended_at: str
    duration_seconds: float
    return_code: int
    stdout_path: str
    stderr_path: str
    timed_out: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "command": self.command,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "return_code": self.return_code,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "timed_out": self.timed_out,
        }


class ManagedProcess:
    def __init__(
        self,
        process: subprocess.Popen[bytes],
        command: list[str],
        started_at: str,
        started_monotonic: float,
        stdout_file: IO[bytes],
        stderr_file: IO[bytes],
        stdout_path: Path,
        stderr_path: Path,
    ) -> None:
        self.process = process
        self.command = command
        self.started_at = started_at
        self.started_monotonic = started_monotonic
        self.stdout_file = stdout_file
        self.stderr_file = stderr_file
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self._result: ProcessResult | None = None

    @property
    def pid(self) -> int:
        return self.process.pid

    def poll(self) -> int | None:
        return self.process.poll()

    def result(self, timed_out: bool = False) -> ProcessResult:
        if self._result is not None:
            return self._result
        return_code = self.process.wait()
        self.stdout_file.close()
        self.stderr_file.close()
        self._result = ProcessResult(
            command=self.command,
            started_at=self.started_at,
            ended_at=_timestamp(),
            duration_seconds=time.monotonic() - self.started_monotonic,
            return_code=return_code,
            stdout_path=str(self.stdout_path),
            stderr_path=str(self.stderr_path),
            timed_out=timed_out,
        )
        return self._result

    def stop(self, timeout_seconds: float, *, timed_out: bool = False) -> ProcessResult:
        if self.poll() is not None:
            return self.result(timed_out=timed_out)
        _terminate_process_group(self.process)
        try:
            self.process.wait(timeout=timeout_seconds)
            return self.result(timed_out=timed_out)
        except subprocess.TimeoutExpired:
            _kill_process_group(self.process)
            self.process.wait()
            return self.result(timed_out=True)


class ProcessRunner:
    def start(self, spec: CommandSpec, stdout_path: Path, stderr_path: Path) -> ManagedProcess:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_file = stdout_path.open("wb")
        stderr_file = stderr_path.open("wb")
        environment = os.environ.copy()
        environment.update(spec.environment)
        kwargs: dict[str, object] = {
            "cwd": str(spec.cwd),
            "env": environment,
            "stdout": stdout_file,
            "stderr": stderr_file,
        }
        if os.name == "posix":
            kwargs["start_new_session"] = True
        elif os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            process = subprocess.Popen(spec.args, **kwargs)
        except OSError as exc:
            stdout_file.close()
            stderr_file.close()
            raise ProcessError(f"Could not start {spec.name}: {exc}") from exc
        return ManagedProcess(
            process=process,
            command=spec.args,
            started_at=_timestamp(),
            started_monotonic=time.monotonic(),
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    def run(self, spec: CommandSpec, stdout_path: Path, stderr_path: Path) -> ProcessResult:
        process = self.start(spec, stdout_path, stderr_path)
        try:
            process.process.wait(timeout=spec.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            result = process.stop(timeout_seconds=1, timed_out=True)
            raise ProcessTimeoutError(
                f"{spec.name} exceeded its {spec.timeout_seconds}s timeout; logs: "
                f"{result.stdout_path}, {result.stderr_path}",
                result,
            ) from exc
        return process.result()


def wait_for_readiness(
    url: str,
    server: ManagedProcess,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: str | None = None
    while time.monotonic() < deadline:
        return_code = server.poll()
        if return_code is not None:
            raise ReadinessError(
                f"Server exited before readiness (return code {return_code}); logs: "
                f"{server.stdout_path}, {server.stderr_path}"
            )
        try:
            with urlopen(url, timeout=min(2.0, poll_interval_seconds + 1.0)) as response:  # noqa: S310
                if 200 <= response.status < 300:
                    return
                last_error = f"HTTP {response.status}"
        except (URLError, OSError) as exc:
            last_error = str(exc)
        time.sleep(poll_interval_seconds)
    suffix = f" Last error: {last_error}" if last_error else ""
    raise ReadinessError(f"Server did not become ready within {timeout_seconds}s.{suffix}")


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:  # pragma: no cover - Windows behavior is exercised by platform users.
        process.terminate()


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGKILL)
    else:  # pragma: no cover - Windows behavior is exercised by platform users.
        process.kill()
