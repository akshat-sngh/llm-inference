"""Lightweight, best-effort metadata collection."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

from . import __version__


def system_metadata() -> dict[str, object]:
    return {
        "operating_system": platform.system(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "current_working_directory": str(Path.cwd()),
    }


def python_metadata() -> dict[str, str]:
    return {
        "version": sys.version,
        "executable": sys.executable,
        "package_version": __version__,
    }


def git_metadata(directory: Path) -> dict[str, object]:
    root = _git(directory, "rev-parse", "--show-toplevel")
    if root is None:
        return {"available": False, "repository_path": str(directory)}
    status = _git(Path(root), "status", "--porcelain")
    return {
        "available": True,
        "repository_path": root,
        "commit_sha": _git(Path(root), "rev-parse", "HEAD"),
        "branch": _git(Path(root), "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


def _git(directory: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
