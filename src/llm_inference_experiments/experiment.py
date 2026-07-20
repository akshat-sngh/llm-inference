"""Experiment lifecycle orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .commands import CommandSpec, ExecutionPlan, build_plan
from .config import LoadedConfig
from .errors import ExperimentExecutionError, ProcessTimeoutError
from .manifest import Manifest, RunStatus, atomic_write_json, atomic_write_text, utc_now
from .metadata import git_metadata, python_metadata, system_metadata
from .nvidia import collect_metadata, metadata_command, sample_command, unavailable_metadata
from .paths import create_run_directory
from .placeholders import substitute
from .preflight import PreflightReport, validate_preflight
from .process import ManagedProcess, ProcessRunner, wait_for_readiness
from .telemetry import NvidiaSampler, unavailable_status
from .vllm import (
    VllmError,
    collect_repository_metadata,
    index_native_result,
    run_probes,
    validate_repository_metadata,
    write_dirty_artifacts,
)


class ExperimentRunner:
    def __init__(self, process_runner: ProcessRunner | None = None) -> None:
        self.process_runner = process_runner or ProcessRunner()

    def run(self, loaded: LoadedConfig) -> Path:
        plan = build_plan(loaded)
        preflight = validate_preflight(loaded, plan)
        run_id, run_directory = create_run_directory(
            loaded.results_root, loaded.config.experiment.name
        )
        manifest = self._prepare_run(loaded, plan, preflight, run_id, run_directory)
        server: ManagedProcess | None = None
        sampler: NvidiaSampler | None = None
        error: BaseException | None = None
        failed_phase = RunStatus.CREATED
        try:
            vllm_metadata = self._prepare_vllm(loaded, run_directory, manifest)
            context = self._placeholder_context(loaded, run_id, run_directory, vllm_metadata)
            server_spec = self._resolve_command(plan.server, context, allow_trial=False)
            warmup_spec = (
                None
                if plan.warmup is None
                else self._resolve_command(plan.warmup, context, allow_trial=False)
            )
            atomic_write_json(run_directory / "commands/server.json", server_spec.as_dict())
            if warmup_spec is not None:
                atomic_write_json(run_directory / "commands/warmup.json", warmup_spec.as_dict())
            sampler = self._prepare_nvidia_telemetry(loaded, preflight, run_directory, manifest)
            manifest.set_phase(RunStatus.STARTING_SERVER)
            server = self.process_runner.start(
                server_spec,
                run_directory / "logs/server.stdout.log",
                run_directory / "logs/server.stderr.log",
            )
            manifest.update(server_process={"pid": server.pid, "command": server_spec.args})

            manifest.set_phase(RunStatus.WAITING_FOR_READINESS)
            wait_for_readiness(
                plan.readiness_url,
                server,
                plan.readiness_timeout_seconds,
                plan.readiness_poll_interval_seconds,
            )

            if warmup_spec is not None:
                manifest.set_phase(RunStatus.WARMING_UP)
                try:
                    warmup = self.process_runner.run(
                        warmup_spec,
                        run_directory / "logs/warmup.stdout.log",
                        run_directory / "logs/warmup.stderr.log",
                    )
                except ProcessTimeoutError as exc:
                    manifest.update(warmup_process=exc.result.as_dict())
                    raise
                manifest.update(warmup_process=warmup.as_dict())
                if warmup.return_code != 0:
                    raise ExperimentExecutionError(
                        f"Warm-up exited with return code {warmup.return_code}; logs: "
                        f"{warmup.stdout_path}, {warmup.stderr_path}"
                    )

            manifest.set_phase(RunStatus.BENCHMARKING)
            for trial_index in range(plan.repeats):
                trial_directory = run_directory / "trials" / f"trial_{trial_index:03d}"
                trial_directory.mkdir(parents=True)
                manifest.data["trial_directories"].append(str(trial_directory))
                manifest.write()
                trial_context = {
                    **context,
                    "trial_dir": str(trial_directory),
                    "trial_index": str(trial_index),
                }
                benchmark_spec = self._resolve_command(
                    plan.benchmark, trial_context, allow_trial=True
                )
                try:
                    result = self.process_runner.run(
                        benchmark_spec,
                        trial_directory / "benchmark.stdout.log",
                        trial_directory / "benchmark.stderr.log",
                    )
                except ProcessTimeoutError as exc:
                    status = {
                        "trial": trial_index,
                        "status": "failed",
                        **exc.result.as_dict(),
                    }
                    atomic_write_json(trial_directory / "status.json", status)
                    self._index_partial_vllm_result(loaded, trial_directory, manifest)
                    manifest.update(failed_trials=manifest.data["failed_trials"] + 1)
                    raise
                status: dict[str, Any] = {
                    "trial": trial_index,
                    "status": "completed",
                    **result.as_dict(),
                }
                if result.return_code != 0:
                    status["status"] = "failed"
                    atomic_write_json(trial_directory / "status.json", status)
                    self._index_partial_vllm_result(loaded, trial_directory, manifest)
                    manifest.update(failed_trials=manifest.data["failed_trials"] + 1)
                    raise ExperimentExecutionError(
                        f"Benchmark trial {trial_index} exited with return code "
                        f"{result.return_code}; logs: "
                        f"{result.stdout_path}, {result.stderr_path}"
                    )
                if loaded.config.vllm.enabled:
                    try:
                        result_index = self._index_vllm_result(loaded, trial_directory)
                    except VllmError as exc:
                        status["status"] = "failed"
                        status["result_error"] = str(exc)
                        atomic_write_json(trial_directory / "status.json", status)
                        manifest.update(failed_trials=manifest.data["failed_trials"] + 1)
                        raise ExperimentExecutionError(str(exc)) from exc
                    status["native_result"] = result_index
                    native_result_path = (
                        trial_directory / loaded.config.vllm.benchmark_result.filename
                    )
                    if native_result_path.is_file():
                        manifest.data["native_result_files"].append(str(native_result_path))
                atomic_write_json(trial_directory / "status.json", status)
                manifest.update(completed_trials=manifest.data["completed_trials"] + 1)
        except BaseException as exc:
            error = exc
            failed_phase = RunStatus(manifest.data["phase"])
            manifest.update(
                status=RunStatus.FAILED.value,
                phase=RunStatus.FAILED.value,
                error={
                    "failed_phase": failed_phase.value,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
        finally:
            if server is not None:
                manifest.set_phase(RunStatus.STOPPING_SERVER)
                server_result = server.stop(plan.shutdown_timeout_seconds)
                manifest.update(server_process={"pid": server.pid, **server_result.as_dict()})
            if sampler is not None:
                sampler.stop()
                sampler_status = sampler.status()
                manifest.update(
                    telemetry={
                        **manifest.data["telemetry"],
                        "successful_samples": sampler_status["successful_samples"],
                        "failed_samples": sampler_status["failed_samples"],
                        "available": True,
                    }
                )

        if error is not None:
            manifest.update(
                status=RunStatus.FAILED.value,
                phase=RunStatus.FAILED.value,
                completed_at=utc_now(),
            )
            raise error
        manifest.update(
            status=RunStatus.COMPLETED.value,
            phase=RunStatus.COMPLETED.value,
            completed_at=utc_now(),
        )
        return run_directory

    def _prepare_run(
        self,
        loaded: LoadedConfig,
        plan: ExecutionPlan,
        preflight: PreflightReport,
        run_id: str,
        run_directory: Path,
    ) -> Manifest:
        atomic_write_text(run_directory / "config/original.yaml", loaded.original_yaml)
        atomic_write_json(run_directory / "config/resolved.json", loaded.resolved_dict())
        atomic_write_json(run_directory / "metadata/system.json", system_metadata())
        atomic_write_json(run_directory / "metadata/python.json", python_metadata())
        atomic_write_json(
            run_directory / "metadata/git.json", git_metadata(loaded.config_path.parent)
        )
        atomic_write_json(run_directory / "commands/server.json", plan.server.as_dict())
        atomic_write_json(run_directory / "commands/benchmark.json", plan.benchmark.as_dict())
        if plan.warmup is not None:
            atomic_write_json(run_directory / "commands/warmup.json", plan.warmup.as_dict())
        metadata_files = ["metadata/system.json", "metadata/python.json", "metadata/git.json"]
        command_files = ["commands/server.json", "commands/benchmark.json"]
        if plan.warmup is not None:
            command_files.append("commands/warmup.json")
        return Manifest(
            run_directory / "manifest.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "experiment_name": loaded.config.experiment.name,
                "run_directory": str(run_directory),
                "status": RunStatus.CREATED.value,
                "phase": RunStatus.CREATED.value,
                "created_at": utc_now(),
                "completed_at": None,
                "original_config_path": str(loaded.config_path),
                "resolved_config_path": "config/resolved.json",
                "metadata_files": metadata_files,
                "command_files": command_files,
                "trial_directories": [],
                "completed_trials": 0,
                "failed_trials": 0,
                "server_process": None,
                "warmup_process": None,
                "telemetry": {
                    "enabled": loaded.config.telemetry.nvidia.enabled,
                    "required": loaded.config.telemetry.nvidia.required,
                    "available": False,
                    "warnings": list(preflight.warnings),
                    "nvidia_metadata_path": None,
                    "nvidia_samples_path": None,
                    "nvidia_status_path": None,
                    "successful_samples": 0,
                    "failed_samples": 0,
                },
                "error": None,
                "native_result_files": [],
            },
        )

    def _prepare_vllm(
        self,
        loaded: LoadedConfig,
        run_directory: Path,
        manifest: Manifest,
    ) -> dict[str, Any] | None:
        settings = loaded.config.vllm
        if not settings.enabled:
            return None
        metadata = collect_repository_metadata(loaded.config_path, settings)
        metadata_path = run_directory / "metadata/vllm.json"
        atomic_write_json(metadata_path, metadata)
        manifest.update(metadata_files=[*manifest.data["metadata_files"], "metadata/vllm.json"])
        repository_root = Path(metadata["repository_root"])
        if (
            metadata["dirty"]
            and not settings.require_clean_worktree
            and settings.capture_dirty_diff
        ):
            write_dirty_artifacts(repository_root, run_directory)
        validate_repository_metadata(metadata, settings)
        probes = run_probes(metadata, settings, run_directory)
        metadata["probes"] = probes
        if probes["collect_env"] is not None and probes["collect_env"]["return_code"] != 0:
            metadata["warnings"] = ["vllm collect-env probe failed"]
        atomic_write_json(metadata_path, metadata)
        manifest.update(
            metadata_files=[
                *manifest.data["metadata_files"],
                "metadata/vllm-version.txt",
                "metadata/vllm-environment.json",
                "metadata/vllm-packages.json",
                *(["metadata/vllm-collect-env.txt"] if settings.collect_environment_report else []),
            ],
            command_files=[*manifest.data["command_files"], "commands/vllm-probes.json"],
        )
        return metadata

    def _placeholder_context(
        self,
        loaded: LoadedConfig,
        run_id: str,
        run_directory: Path,
        vllm_metadata: dict[str, Any] | None,
    ) -> dict[str, str]:
        settings = loaded.config.vllm
        return {
            "run_id": run_id,
            "run_dir": str(run_directory),
            "trial_dir": "",
            "trial_index": "",
            "experiment_name": loaded.config.experiment.name,
            "working_directory": str(loaded.working_directory),
            "results_root": str(loaded.results_root),
            "server_host": loaded.config.server.host,
            "server_port": str(loaded.config.server.port),
            "vllm_commit": "" if vllm_metadata is None else str(vllm_metadata["head_commit"]),
            "model_id": settings.model.id,
            "model_revision": settings.model.revision,
            "served_model_name": settings.model.served_name,
        }

    def _resolve_command(
        self,
        spec: CommandSpec,
        context: dict[str, str],
        *,
        allow_trial: bool,
    ) -> CommandSpec:
        return replace(
            spec,
            args=[substitute(argument, context, allow_trial=allow_trial) for argument in spec.args],
            environment={
                key: substitute(value, context, allow_trial=allow_trial)
                for key, value in spec.environment.items()
            },
        )

    def _index_vllm_result(self, loaded: LoadedConfig, trial_directory: Path) -> dict[str, Any]:
        path = trial_directory / loaded.config.vllm.benchmark_result.filename
        if not path.is_file() and not loaded.config.vllm.benchmark_result.required:
            return {"raw_result_path": path.name, "required": False, "present": False}
        index = index_native_result(path)
        atomic_write_json(trial_directory / "result-index.json", index)
        return index

    def _index_partial_vllm_result(
        self,
        loaded: LoadedConfig,
        trial_directory: Path,
        manifest: Manifest,
    ) -> None:
        if not loaded.config.vllm.enabled:
            return
        path = trial_directory / loaded.config.vllm.benchmark_result.filename
        if not path.is_file():
            return
        try:
            index = index_native_result(path, partial=True)
        except VllmError:
            return
        atomic_write_json(trial_directory / "result-index.json", index)
        manifest.data["native_result_files"].append(str(path))
        manifest.write()

    def _prepare_nvidia_telemetry(
        self,
        loaded: LoadedConfig,
        preflight: PreflightReport,
        run_directory: Path,
        manifest: Manifest,
    ) -> NvidiaSampler | None:
        settings = loaded.config.telemetry.nvidia
        if not settings.enabled:
            return None

        metadata_path = run_directory / "metadata/nvidia.json"
        status_path = run_directory / "telemetry/status.json"
        samples_path = run_directory / "telemetry/nvidia.jsonl"
        telemetry_paths = {
            "nvidia_metadata_path": "metadata/nvidia.json",
            "nvidia_samples_path": None,
            "nvidia_status_path": "telemetry/status.json",
        }
        metadata_files = [*manifest.data["metadata_files"], "metadata/nvidia.json"]
        executable = preflight.nvidia_executable
        if executable is None:
            message = f"NVIDIA executable cannot be located: {settings.executable}"
            metadata = unavailable_metadata(settings, message)
            atomic_write_json(metadata_path, metadata)
            atomic_write_json(
                status_path, unavailable_status(settings, "unavailable", metadata["error"])
            )
            manifest.update(
                telemetry={
                    **manifest.data["telemetry"],
                    **telemetry_paths,
                    "available": False,
                },
                metadata_files=metadata_files,
            )
            return None

        atomic_write_json(
            run_directory / "commands/nvidia-metadata.json",
            {
                "name": "nvidia-metadata",
                "command": metadata_command(executable, settings.device_index),
                "timeout_seconds": settings.command_timeout_seconds,
            },
        )
        atomic_write_json(
            run_directory / "commands/nvidia-sample.json",
            {
                "name": "nvidia-sample",
                "command": sample_command(executable, settings.device_index),
                "timeout_seconds": settings.command_timeout_seconds,
            },
        )
        command_files = [
            *manifest.data["command_files"],
            "commands/nvidia-metadata.json",
            "commands/nvidia-sample.json",
        ]
        metadata = collect_metadata(settings, executable)
        atomic_write_json(metadata_path, metadata)
        if not metadata["available"]:
            error = metadata["error"]
            atomic_write_json(status_path, unavailable_status(settings, "unavailable", error))
            manifest.update(
                telemetry={
                    **manifest.data["telemetry"],
                    **telemetry_paths,
                    "available": False,
                    "warnings": [
                        *manifest.data["telemetry"]["warnings"],
                        metadata["error"]["message"] if metadata["error"] is not None else "",
                    ],
                },
                metadata_files=metadata_files,
                command_files=command_files,
            )
            if settings.required:
                message = error["message"] if error is not None else "unknown NVIDIA probe failure"
                raise ExperimentExecutionError(f"NVIDIA metadata probe failed: {message}")
            return None

        sampler = NvidiaSampler(settings, executable, samples_path, status_path)
        sampler.start()
        manifest.update(
            telemetry={
                **manifest.data["telemetry"],
                **telemetry_paths,
                "nvidia_samples_path": "telemetry/nvidia.jsonl",
                "available": True,
            },
            metadata_files=metadata_files,
            command_files=command_files,
        )
        return sampler
