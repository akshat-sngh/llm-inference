"""Background NVIDIA sampling with durable JSONL records."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .manifest import atomic_write_json
from .models import NvidiaTelemetrySettings
from .nvidia import NvidiaCommandError, collect_sample, utc_timestamp


def unavailable_status(
    settings: NvidiaTelemetrySettings,
    status: str,
    error: dict[str, str] | None,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "required": settings.required,
        "available": False,
        "started_at": None,
        "stopped_at": utc_timestamp(),
        "sample_interval_seconds": settings.sample_interval_seconds,
        "device_index": settings.device_index,
        "successful_samples": 0,
        "failed_samples": 0,
        "total_samples": 0,
        "sampler_status": status,
        "last_error": error,
        "output_file_path": None,
    }


class NvidiaSampler:
    """A single-device sampler that tolerates individual command failures."""

    def __init__(
        self,
        settings: NvidiaTelemetrySettings,
        executable: Path,
        output_path: Path,
        status_path: Path,
    ) -> None:
        self.settings = settings
        self.executable = executable
        self.output_path = output_path
        self.status_path = status_path
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._started_at: str | None = None
        self._stopped_at: str | None = None
        self._successful_samples = 0
        self._failed_samples = 0
        self._last_error: dict[str, str] | None = None
        self._sampler_status = "failed_to_start"

    @property
    def thread(self) -> threading.Thread | None:
        return self._thread

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._started_at = utc_timestamp()
        self._sampler_status = "running"
        self._write_status()
        self._thread = threading.Thread(target=self._run, name="nvidia-telemetry", daemon=False)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            self._stopped_at = utc_timestamp()
            if self._sampler_status == "running":
                self._sampler_status = "completed"
            self._write_status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": True,
                "required": self.settings.required,
                "available": True,
                "started_at": self._started_at,
                "stopped_at": self._stopped_at,
                "sample_interval_seconds": self.settings.sample_interval_seconds,
                "device_index": self.settings.device_index,
                "successful_samples": self._successful_samples,
                "failed_samples": self._failed_samples,
                "total_samples": self._successful_samples + self._failed_samples,
                "sampler_status": self._sampler_status,
                "last_error": self._last_error,
                "output_file_path": str(self.output_path),
            }

    def _run(self) -> None:
        sample_number = 0
        while not self._stop_event.is_set():
            observed_at = utc_timestamp()
            try:
                metrics, duration_seconds = collect_sample(self.settings, self.executable)
                record: dict[str, Any] = {
                    "observed_at": observed_at,
                    "sample_number": sample_number,
                    "device_index": self.settings.device_index,
                    "query_duration_seconds": duration_seconds,
                    "status": "ok",
                    "metrics": metrics,
                    "error": None,
                }
                with self._lock:
                    self._successful_samples += 1
            except NvidiaCommandError as exc:
                record = {
                    "observed_at": observed_at,
                    "sample_number": sample_number,
                    "device_index": self.settings.device_index,
                    "query_duration_seconds": exc.duration_seconds
                    if exc.duration_seconds is not None
                    else self.settings.command_timeout_seconds,
                    "status": "error",
                    "metrics": None,
                    "error": {"type": exc.error_type, "message": str(exc)},
                }
                with self._lock:
                    self._failed_samples += 1
                    self._last_error = record["error"]
            self._append_record(record)
            with self._lock:
                self._write_status()
            sample_number += 1
            self._stop_event.wait(self.settings.sample_interval_seconds)

    def _append_record(self, record: dict[str, Any]) -> None:
        with self._lock:
            with self.output_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()

    def _write_status(self) -> None:
        atomic_write_json(self.status_path, self.status())
