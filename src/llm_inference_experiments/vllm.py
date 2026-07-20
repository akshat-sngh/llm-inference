"""Optional vLLM repository validation, probes, and native-result indexing."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import ExperimentError
from .manifest import atomic_write_json, atomic_write_text
from .models import VllmSettings
from .paths import resolve_from_config


class VllmError(ExperimentError):
    """Raised for a vLLM reproducibility or probe failure."""


def resolve_vllm_path(config_path: Path, value: str) -> Path:
    if not Path(value).is_absolute() and os.sep not in value:
        found = shutil.which(value)
        if found is not None:
            return Path(found).resolve()
    return resolve_from_config(config_path, value)


def collect_repository_metadata(config_path: Path, settings: VllmSettings) -> dict[str, Any]:
    repository = resolve_vllm_path(config_path, settings.repository_path)
    root = _git(repository, "rev-parse", "--show-toplevel")
    if root is None:
        raise VllmError(f"vLLM repository is not a usable Git worktree: {repository}")
    repository_root = Path(root)
    head = _git_required(repository_root, "rev-parse", "HEAD")
    porcelain = _git_required(repository_root, "status", "--porcelain")
    metadata = {
        "repository_path": str(repository),
        "repository_root": str(repository_root),
        "head_commit": head,
        "expected_commit": settings.expected_commit,
        "head_matches_expected": head.lower() == settings.expected_commit.lower(),
        "branch": _git(repository_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git(repository_root, "describe", "--tags", "--always", "--dirty"),
        "dirty": bool(porcelain),
        "porcelain_status": porcelain,
        "origin_url": _git(repository_root, "remote", "get-url", "origin"),
        "commit_timestamp": _git(repository_root, "show", "-s", "--format=%cI", "HEAD"),
        "commit_subject": _git(repository_root, "show", "-s", "--format=%s", "HEAD"),
        "executable": str(resolve_vllm_path(config_path, settings.executable)),
        "python_executable": str(resolve_vllm_path(config_path, settings.python_executable)),
        "model": {
            "id": settings.model.id,
            "revision": settings.model.revision,
            "served_name": settings.model.served_name,
        },
    }
    return metadata


def validate_repository_metadata(metadata: dict[str, Any], settings: VllmSettings) -> None:
    if not metadata["head_matches_expected"]:
        raise VllmError(
            "vLLM commit mismatch: expected "
            f"{metadata['expected_commit']}, actual {metadata['head_commit']}"
        )
    if settings.require_clean_worktree and metadata["dirty"]:
        raise VllmError("vLLM repository is dirty but require_clean_worktree is true")


def write_dirty_artifacts(repository_root: Path, run_directory: Path) -> None:
    atomic_write_text(
        run_directory / "metadata/vllm-git-status.txt",
        _git_required(repository_root, "status", "--porcelain") + "\n",
    )
    atomic_write_text(
        run_directory / "metadata/vllm-diff.patch",
        _git_required(repository_root, "diff"),
    )
    atomic_write_text(
        run_directory / "metadata/vllm-diff-stat.txt",
        _git_required(repository_root, "diff", "--stat") + "\n",
    )


def run_probes(
    metadata: dict[str, Any],
    settings: VllmSettings,
    run_directory: Path,
) -> dict[str, Any]:
    executable = str(metadata["executable"])
    python_executable = str(metadata["python_executable"])
    probe_script = Path(__file__).parent / "probes/vllm_environment.py"
    version = _run_probe([executable, "--version"])
    _write_text_probe(run_directory / "metadata/vllm-version.txt", version)
    if version["return_code"] != 0:
        raise VllmError(f"vLLM version probe failed with return code {version['return_code']}")
    environment = _run_probe([python_executable, str(probe_script), "environment"])
    if environment["return_code"] != 0:
        raise VllmError(
            f"vLLM Python environment probe failed with return code {environment['return_code']}"
        )
    try:
        environment_data = json.loads(environment["stdout"])
    except json.JSONDecodeError as exc:
        raise VllmError("vLLM Python environment probe did not emit valid JSON") from exc
    atomic_write_json(run_directory / "metadata/vllm-environment.json", environment_data)

    packages = _run_probe([python_executable, str(probe_script), "packages"])
    if packages["return_code"] != 0:
        raise VllmError(
            f"vLLM package inventory probe failed with return code {packages['return_code']}"
        )
    try:
        packages_data = json.loads(packages["stdout"])
    except json.JSONDecodeError as exc:
        raise VllmError("vLLM package inventory probe did not emit valid JSON") from exc
    atomic_write_json(run_directory / "metadata/vllm-packages.json", packages_data)

    collect_env: dict[str, Any] | None = None
    if settings.collect_environment_report:
        collect_env = _run_probe([executable, "collect-env"])
        _write_text_probe(run_directory / "metadata/vllm-collect-env.txt", collect_env)
    probes = {
        "version": version,
        "environment": environment,
        "packages": packages,
        "collect_env": collect_env,
    }
    atomic_write_json(
        run_directory / "commands/vllm-probes.json",
        {
            "version": version["command"],
            "environment": environment["command"],
            "packages": packages["command"],
            "collect_env": None if collect_env is None else collect_env["command"],
        },
    )
    return probes


def help_check(executable: Path, arguments: list[str]) -> dict[str, Any]:
    return _run_probe([str(executable), *arguments, "--help"])


def probe_command(command: list[str]) -> dict[str, Any]:
    """Run one bounded diagnostic command and return captured raw output."""
    return _run_probe(command)


def index_native_result(path: Path, *, partial: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise VllmError(f"required native vLLM result is missing: {path}")
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        if partial:
            return {
                "format": "vllm-bench-serve-json",
                "raw_result_path": path.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
                "valid_json": False,
                "top_level_keys": [],
                "partial": True,
            }
        raise VllmError(f"native vLLM result is invalid JSON: {path}") from exc
    if not isinstance(document, dict):
        raise VllmError(f"native vLLM result must contain a JSON object: {path}")
    result = {
        "format": "vllm-bench-serve-json",
        "raw_result_path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "valid_json": True,
        "top_level_keys": sorted(document),
    }
    if partial:
        result["partial"] = True
    return result


def _run_probe(command: list[str]) -> dict[str, Any]:
    started_at = _timestamp()
    started = time.monotonic()
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": command,
            "started_at": started_at,
            "ended_at": _timestamp(),
            "duration_seconds": time.monotonic() - started,
            "return_code": None,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "command": command,
        "started_at": started_at,
        "ended_at": _timestamp(),
        "duration_seconds": time.monotonic() - started,
        "return_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def _write_text_probe(path: Path, probe: dict[str, Any]) -> None:
    text = probe["stdout"]
    if probe["stderr"]:
        text += f"\n[stderr]\n{probe['stderr']}"
    atomic_write_text(path, text)


def _git(directory: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments], cwd=directory, capture_output=True, text=True, check=False, timeout=10
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_required(directory: Path, *arguments: str) -> str:
    value = _git(directory, *arguments)
    if value is None:
        raise VllmError(f"Git command failed in {directory}: git {' '.join(arguments)}")
    return value


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
