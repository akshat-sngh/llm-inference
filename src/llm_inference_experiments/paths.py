"""Filesystem helpers for resolved configuration and run directories."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from pathlib import Path


def resolve_from_config(config_path: Path, value: str) -> Path:
    """Resolve a path relative to the directory that contains the YAML file."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def safe_experiment_name(name: str) -> str:
    """Make an experiment name safe for a single directory component."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-") or "experiment"


def create_run_directory(results_root: Path, experiment_name: str) -> tuple[str, Path]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
    run_directory = results_root / safe_experiment_name(experiment_name) / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_id, run_directory
