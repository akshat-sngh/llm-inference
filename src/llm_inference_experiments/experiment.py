"""Experiment lifecycle orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .commands import ExecutionPlan, build_plan
from .config import LoadedConfig
from .errors import ExperimentExecutionError, ProcessTimeoutError
from .manifest import Manifest, RunStatus, atomic_write_json, atomic_write_text, utc_now
from .metadata import git_metadata, python_metadata, system_metadata
from .paths import create_run_directory
from .preflight import validate_preflight
from .process import ManagedProcess, ProcessRunner, wait_for_readiness


class ExperimentRunner:
    def __init__(self, process_runner: ProcessRunner | None = None) -> None:
        self.process_runner = process_runner or ProcessRunner()

    def run(self, loaded: LoadedConfig) -> Path:
        plan = build_plan(loaded)
        validate_preflight(loaded, plan)
        run_id, run_directory = create_run_directory(
            loaded.results_root, loaded.config.experiment.name
        )
        manifest = self._prepare_run(loaded, plan, run_id, run_directory)
        server: ManagedProcess | None = None
        error: BaseException | None = None
        failed_phase = RunStatus.CREATED
        try:
            manifest.set_phase(RunStatus.STARTING_SERVER)
            server = self.process_runner.start(
                plan.server,
                run_directory / "logs/server.stdout.log",
                run_directory / "logs/server.stderr.log",
            )
            manifest.update(server_process={"pid": server.pid, "command": plan.server.args})

            manifest.set_phase(RunStatus.WAITING_FOR_READINESS)
            wait_for_readiness(
                plan.readiness_url,
                server,
                plan.readiness_timeout_seconds,
                plan.readiness_poll_interval_seconds,
            )

            if plan.warmup is not None:
                manifest.set_phase(RunStatus.WARMING_UP)
                try:
                    warmup = self.process_runner.run(
                        plan.warmup,
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
                try:
                    result = self.process_runner.run(
                        plan.benchmark,
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
                    manifest.update(failed_trials=manifest.data["failed_trials"] + 1)
                    raise ExperimentExecutionError(
                        f"Benchmark trial {trial_index} exited with return code "
                        f"{result.return_code}; logs: "
                        f"{result.stdout_path}, {result.stderr_path}"
                    )
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
                "error": None,
            },
        )
