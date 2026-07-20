"""Executing diagnostics that do not start an experiment server or benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .commands import build_plan
from .config import LoadedConfig
from .errors import ExperimentError
from .manifest import atomic_write_json
from .nvidia import collect_metadata
from .preflight import validate_preflight
from .vllm import (
    VllmError,
    collect_repository_metadata,
    help_check,
    probe_command,
    validate_repository_metadata,
)


def run_doctor(loaded: LoadedConfig) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        preflight = validate_preflight(loaded, build_plan(loaded))
        checks.append(
            {"name": "preflight", "status": "passed", "warnings": list(preflight.warnings)}
        )
    except ExperimentError as exc:
        checks.append({"name": "preflight", "status": "failed", "error": str(exc)})
        return {"passed": False, "checks": checks}

    nvidia = loaded.config.telemetry.nvidia
    if nvidia.enabled:
        if preflight.nvidia_executable is None:
            checks.append(
                {"name": "nvidia", "status": "warning", "error": "executable unavailable"}
            )
        else:
            metadata = collect_metadata(nvidia, preflight.nvidia_executable)
            checks.append(
                {
                    "name": "nvidia",
                    "status": "passed" if metadata["available"] else "failed",
                    "metadata": metadata,
                }
            )

    settings = loaded.config.vllm
    if settings.enabled:
        try:
            metadata = collect_repository_metadata(loaded.config_path, settings)
            validate_repository_metadata(metadata, settings)
            checks.append({"name": "vllm_repository", "status": "passed", "metadata": metadata})
            executable = preflight.vllm_executable
            python_executable = preflight.vllm_python_executable
            assert executable is not None and python_executable is not None
            probe_script = Path(__file__).parent / "probes/vllm_environment.py"
            for name, command in (
                ("vllm_version", [str(executable), "--version"]),
                ("vllm_environment", [str(python_executable), str(probe_script), "environment"]),
                ("vllm_packages", [str(python_executable), str(probe_script), "packages"]),
            ):
                result = probe_command(command)
                checks.append(
                    {
                        "name": name,
                        "status": "passed" if result["return_code"] == 0 else "failed",
                        "result": result,
                    }
                )
            for name, arguments in (
                ("vllm_serve_help", ["serve"]),
                ("vllm_bench_serve_help", ["bench", "serve"]),
            ):
                result = help_check(executable, arguments)
                checks.append(
                    {
                        "name": name,
                        "status": "passed" if result["return_code"] == 0 else "failed",
                        "result": result,
                    }
                )
        except VllmError as exc:
            checks.append({"name": "vllm", "status": "failed", "error": str(exc)})
    passed = all(check["status"] != "failed" for check in checks)
    return {"passed": passed, "checks": checks}


def write_doctor_report(path: Path, report: dict[str, Any]) -> None:
    atomic_write_json(path, report)


def format_doctor_report(report: dict[str, Any]) -> str:
    lines = ["Doctor: passed" if report["passed"] else "Doctor: failed"]
    for check in report["checks"]:
        suffix = check.get("error", "")
        lines.append(f"- {check['name']}: {check['status']}{f' ({suffix})' if suffix else ''}")
    return "\n".join(lines)
