"""Non-executing checks for the local command and filesystem prerequisites."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import LoadedConfig
from .errors import PreflightError

if TYPE_CHECKING:
    from .commands import ExecutionPlan


@dataclass(frozen=True)
class PreflightReport:
    warnings: tuple[str, ...] = ()
    nvidia_executable: Path | None = None


def validate_preflight(loaded: LoadedConfig, plan: ExecutionPlan) -> PreflightReport:
    """Raise a readable error if this plan cannot be started locally.

    The checks intentionally do not create directories or execute any command, so they
    are safe for ``plan`` and ``run --dry-run``.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not loaded.working_directory.is_dir():
        errors.append(f"working_directory is not an existing directory: {loaded.working_directory}")
    _check_results_root(loaded.results_root, errors)
    _check_executable(
        "server", plan.server.args[0], loaded.working_directory, plan.server.environment, errors
    )
    _check_executable(
        "benchmark",
        plan.benchmark.args[0],
        loaded.working_directory,
        plan.benchmark.environment,
        errors,
    )
    if plan.warmup is not None:
        _check_executable(
            "warm-up",
            plan.warmup.args[0],
            loaded.working_directory,
            plan.warmup.environment,
            errors,
        )
    nvidia_executable: Path | None = None
    nvidia = loaded.config.telemetry.nvidia
    if nvidia.enabled:
        nvidia_executable = resolve_executable(nvidia.executable, loaded.working_directory)
        if nvidia_executable is None:
            message = f"NVIDIA executable cannot be located: {nvidia.executable}"
            if nvidia.required:
                errors.append(message)
            else:
                warnings.append(message)
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise PreflightError(f"Preflight validation failed:\n{details}")
    return PreflightReport(warnings=tuple(warnings), nvidia_executable=nvidia_executable)


def _check_results_root(results_root: Path, errors: list[str]) -> None:
    if results_root.exists() and not results_root.is_dir():
        errors.append(f"results_root exists but is not a directory: {results_root}")
        return

    candidate = results_root.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.is_dir():
        errors.append(f"results_root parent cannot be created: {results_root.parent}")
    elif not os.access(candidate, os.W_OK | os.X_OK):
        errors.append(f"results_root parent is not writable: {candidate}")


def _check_executable(
    command_name: str,
    executable: str,
    working_directory: Path,
    environment: dict[str, str],
    errors: list[str],
) -> None:
    resolved = resolve_executable(executable, working_directory, environment)
    if resolved is None:
        errors.append(f"{command_name} executable cannot be located: {executable}")


def resolve_executable(
    executable: str,
    working_directory: Path,
    environment: dict[str, str] | None = None,
) -> Path | None:
    """Resolve an executable from an absolute/relative path or the inherited PATH."""
    candidate = Path(executable).expanduser()
    has_path_component = candidate.is_absolute() or os.sep in executable
    if os.altsep is not None:
        has_path_component = has_path_component or os.altsep in executable
    if has_path_component:
        if not candidate.is_absolute():
            candidate = working_directory / candidate
        candidate = candidate.resolve()
        return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None

    merged_environment = os.environ.copy()
    if environment is not None:
        merged_environment.update(environment)
    found = shutil.which(executable, path=merged_environment.get("PATH"))
    return Path(found).resolve() if found else None
