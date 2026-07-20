"""Non-executing checks for the local command and filesystem prerequisites."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import LoadedConfig
from .errors import PreflightError
from .placeholders import substitute
from .vllm import resolve_vllm_path

if TYPE_CHECKING:
    from .commands import ExecutionPlan


@dataclass(frozen=True)
class PreflightReport:
    warnings: tuple[str, ...] = ()
    nvidia_executable: Path | None = None
    vllm_executable: Path | None = None
    vllm_python_executable: Path | None = None
    vllm_repository_path: Path | None = None


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
    vllm_executable: Path | None = None
    vllm_python_executable: Path | None = None
    vllm_repository_path: Path | None = None
    vllm = loaded.config.vllm
    if vllm.enabled:
        vllm_repository_path = resolve_vllm_path(loaded.config_path, vllm.repository_path)
        if not vllm_repository_path.is_dir() or not (vllm_repository_path / ".git").exists():
            errors.append(f"vLLM repository is not a Git worktree: {vllm_repository_path}")
        vllm_executable = resolve_executable(vllm.executable, loaded.config_path.parent)
        vllm_python_executable = resolve_executable(
            vllm.python_executable, loaded.config_path.parent
        )
        if vllm_executable is None:
            errors.append(f"vLLM executable cannot be located: {vllm.executable}")
        if vllm_python_executable is None:
            errors.append(f"vLLM Python executable cannot be located: {vllm.python_executable}")
        _check_vllm_commands(loaded, plan, vllm.benchmark_result.filename, errors)
        _check_vllm_single_gpu(loaded, plan, errors)
    _check_placeholders(loaded, plan, errors)
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise PreflightError(f"Preflight validation failed:\n{details}")
    return PreflightReport(
        warnings=tuple(warnings),
        nvidia_executable=nvidia_executable,
        vllm_executable=vllm_executable,
        vllm_python_executable=vllm_python_executable,
        vllm_repository_path=vllm_repository_path,
    )


def _check_vllm_commands(
    loaded: LoadedConfig, plan: ExecutionPlan, result_filename: str, errors: list[str]
) -> None:
    if "serve" not in plan.server.args:
        errors.append("vLLM server command must invoke 'vllm serve'")
    try:
        bench_index = plan.benchmark.args.index("bench")
        if plan.benchmark.args[bench_index + 1] != "serve":
            raise ValueError
    except (ValueError, IndexError):
        errors.append("vLLM benchmark command must invoke 'vllm bench serve'")
    if "--save-result" not in plan.benchmark.args:
        errors.append("vLLM benchmark command must include --save-result")
    result_dir = _argument_value(plan.benchmark.args, "--result-dir")
    filename = _argument_value(plan.benchmark.args, "--result-filename")
    if result_dir is None or "{trial_dir}" not in result_dir:
        errors.append("vLLM benchmark --result-dir must use {trial_dir}")
    if filename != result_filename:
        errors.append(f"vLLM benchmark --result-filename must be {result_filename}")
    if not _is_loopback(loaded.config.server.host):
        errors.append("vLLM server host must be loopback-only")


def _check_vllm_single_gpu(loaded: LoadedConfig, plan: ExecutionPlan, errors: list[str]) -> None:
    cuda_devices = plan.server.environment.get("CUDA_VISIBLE_DEVICES")
    if (
        cuda_devices is not None
        and len([item for item in cuda_devices.split(",") if item.strip()]) != 1
    ):
        errors.append("CUDA_VISIBLE_DEVICES must identify exactly one device")
    for arguments in (plan.server.args, plan.benchmark.args):
        for flag in ("--tensor-parallel-size", "--pipeline-parallel-size", "--data-parallel-size"):
            value = _argument_value(arguments, flag)
            if value is not None:
                try:
                    if int(value) > 1:
                        errors.append(f"{flag} must not exceed one")
                except ValueError:
                    errors.append(f"{flag} must be an integer")


def _check_placeholders(loaded: LoadedConfig, plan: ExecutionPlan, errors: list[str]) -> None:
    context = {
        "run_id": "run",
        "run_dir": "/run",
        "trial_dir": "/trial",
        "trial_index": "0",
        "experiment_name": loaded.config.experiment.name,
        "working_directory": str(loaded.working_directory),
        "results_root": str(loaded.results_root),
        "server_host": loaded.config.server.host,
        "server_port": str(loaded.config.server.port),
        "vllm_commit": "commit",
        "model_id": loaded.config.vllm.model.id,
        "model_revision": loaded.config.vllm.model.revision,
        "served_model_name": loaded.config.vllm.model.served_name,
    }
    for spec, allow_trial in ((plan.server, False), (plan.warmup, False), (plan.benchmark, True)):
        if spec is None:
            continue
        try:
            for value in [*spec.args, *spec.environment.values()]:
                substitute(value, context, allow_trial=allow_trial)
        except Exception as exc:
            errors.append(str(exc))


def _argument_value(arguments: list[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for index, argument in enumerate(arguments):
        if argument.startswith(prefix):
            return argument[len(prefix) :]
        if argument == flag and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def _is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "::1", "localhost"}


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
