"""Atomic JSON output and the compact run manifest."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def atomic_write_text(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(data, encoding="utf-8")
    os.replace(temporary_path, path)


class RunStatus(StrEnum):
    CREATED = "created"
    STARTING_SERVER = "starting_server"
    WAITING_FOR_READINESS = "waiting_for_readiness"
    WARMING_UP = "warming_up"
    BENCHMARKING = "benchmarking"
    STOPPING_SERVER = "stopping_server"
    COMPLETED = "completed"
    FAILED = "failed"


class Manifest:
    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        self.path = path
        self.data = data
        self.write()

    def update(self, **changes: Any) -> None:
        self.data.update(changes)
        self.write()

    def set_phase(self, phase: RunStatus) -> None:
        self.update(status=phase.value, phase=phase.value)

    def write(self) -> None:
        atomic_write_json(self.path, self.data)
