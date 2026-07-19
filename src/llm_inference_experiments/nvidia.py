"""Small, dependency-free NVIDIA command construction and CSV parsing helpers."""

from __future__ import annotations

import csv
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import NvidiaTelemetrySettings

METADATA_FIELDS = "index,uuid,name,driver_version,memory.total,power.limit,pstate"
SAMPLE_FIELDS = (
    "timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,"
    "power.draw,power.limit,temperature.gpu,clocks.current.sm,clocks.current.memory,pstate"
)
NULL_VALUES = {"", "n/a", "[not supported]", "not supported", "unknown"}


class NvidiaCommandError(Exception):
    """A failed one-shot NVIDIA command with a stable error type for artifacts."""

    def __init__(
        self,
        error_type: str,
        message: str,
        duration_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.duration_seconds = duration_seconds


def metadata_command(executable: Path, device_index: int) -> list[str]:
    return [
        str(executable),
        f"--id={device_index}",
        f"--query-gpu={METADATA_FIELDS}",
        "--format=csv,noheader,nounits",
    ]


def sample_command(executable: Path, device_index: int) -> list[str]:
    return [
        str(executable),
        f"--id={device_index}",
        f"--query-gpu={SAMPLE_FIELDS}",
        "--format=csv,noheader,nounits",
    ]


def collect_metadata(settings: NvidiaTelemetrySettings, executable: Path) -> dict[str, Any]:
    """Collect one device row, returning an unavailable artifact instead of raising."""
    data: dict[str, Any] = {
        "enabled": True,
        "required": settings.required,
        "available": False,
        "executable": str(executable),
        "device_index": settings.device_index,
        "collected_at": utc_timestamp(),
        "gpu": None,
        "command_return_code": None,
        "raw_stdout": None,
        "raw_stderr": None,
        "error": None,
    }
    try:
        result, _duration = _run_command(
            metadata_command(executable, settings.device_index), settings.command_timeout_seconds
        )
        data["command_return_code"] = result.returncode
        data["raw_stdout"] = result.stdout
        data["raw_stderr"] = result.stderr
        if result.returncode != 0:
            raise NvidiaCommandError(
                "NvidiaCommandError",
                f"nvidia-smi metadata command exited with return code {result.returncode}",
            )
        row = _single_csv_row(result.stdout)
        if row is None or len(row) < 7:
            raise NvidiaCommandError("NvidiaProbeError", "nvidia-smi returned no device metadata")
        data["gpu"] = {
            "index": _as_int(row[0]),
            "uuid": _as_string(row[1]),
            "name": _as_string(row[2]),
            "driver_version": _as_string(row[3]),
            "memory_total_mib": _as_int(row[4]),
            "power_limit_watts": _as_float(row[5]),
            "performance_state": _as_string(row[6]),
        }
        data["available"] = True
    except NvidiaCommandError as exc:
        data["error"] = {"type": exc.error_type, "message": str(exc)}
    return data


def collect_sample(
    settings: NvidiaTelemetrySettings,
    executable: Path,
) -> tuple[dict[str, Any], float]:
    """Collect and parse one lightweight metric sample or raise ``NvidiaCommandError``."""
    result, duration_seconds = _run_command(
        sample_command(executable, settings.device_index), settings.command_timeout_seconds
    )
    if result.returncode != 0:
        raise NvidiaCommandError(
            "NvidiaCommandError",
            f"nvidia-smi sample command exited with return code {result.returncode}",
            duration_seconds,
        )
    row = _single_csv_row(result.stdout)
    if row is None or len(row) < 13:
        raise NvidiaCommandError("NvidiaProbeError", "nvidia-smi returned no telemetry sample")
    metrics = {
        "gpu_utilization_percent": _as_int(row[3]),
        "memory_utilization_percent": _as_int(row[4]),
        "memory_used_mib": _as_int(row[5]),
        "memory_total_mib": _as_int(row[6]),
        "power_draw_watts": _as_float(row[7]),
        "power_limit_watts": _as_float(row[8]),
        "temperature_celsius": _as_int(row[9]),
        "sm_clock_mhz": _as_int(row[10]),
        "memory_clock_mhz": _as_int(row[11]),
        "performance_state": _as_string(row[12]),
    }
    return metrics, duration_seconds


def unavailable_metadata(
    settings: NvidiaTelemetrySettings,
    message: str,
    error_type: str = "NvidiaUnavailable",
) -> dict[str, Any]:
    return {
        "enabled": True,
        "required": settings.required,
        "available": False,
        "executable": settings.executable,
        "device_index": settings.device_index,
        "collected_at": utc_timestamp(),
        "gpu": None,
        "command_return_code": None,
        "raw_stdout": None,
        "raw_stderr": None,
        "error": {"type": error_type, "message": message},
    }


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _run_command(
    command: list[str], timeout_seconds: float
) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise NvidiaCommandError(
            "ProcessError",
            f"nvidia-smi executable was not found: {exc}",
            time.monotonic() - started,
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise NvidiaCommandError(
            "ProcessTimeoutError",
            f"nvidia-smi command exceeded its {timeout_seconds}s timeout",
            time.monotonic() - started,
        ) from exc
    return result, time.monotonic() - started


def _single_csv_row(text: str) -> list[str] | None:
    rows = [row for row in csv.reader(text.splitlines()) if row]
    if not rows or len(rows[0]) < 1:
        return None
    return [cell.strip() for cell in rows[0]]


def _as_string(value: str) -> str | None:
    return None if value.strip().lower() in NULL_VALUES else value.strip()


def _as_int(value: str) -> int | None:
    text = _as_string(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _as_float(value: str) -> float | None:
    text = _as_string(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None
