"""Construction and presentation of the commands an experiment will execute."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .config import LoadedConfig


@dataclass(frozen=True)
class CommandSpec:
    name: str
    args: list[str]
    cwd: Path
    environment: dict[str, str]
    timeout_seconds: float | None

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["cwd"] = str(self.cwd)
        data["command"] = data.pop("args")
        return data


@dataclass(frozen=True)
class ExecutionPlan:
    server: CommandSpec
    warmup: CommandSpec | None
    benchmark: CommandSpec
    readiness_url: str
    repeats: int
    readiness_timeout_seconds: float
    readiness_poll_interval_seconds: float
    shutdown_timeout_seconds: float


def build_plan(loaded: LoadedConfig) -> ExecutionPlan:
    config = loaded.config
    server = CommandSpec(
        name="server",
        args=[*config.server.command, *config.server.arguments],
        cwd=loaded.working_directory,
        environment=config.server.environment,
        timeout_seconds=None,
    )
    warmup = None
    if config.warmup.enabled:
        warmup = CommandSpec(
            name="warmup",
            args=[*config.warmup.command, *config.warmup.arguments],
            cwd=loaded.working_directory,
            environment=config.warmup.environment,
            timeout_seconds=config.warmup.timeout_seconds,
        )
    benchmark = CommandSpec(
        name="benchmark",
        args=[*config.benchmark.command, *config.benchmark.arguments],
        cwd=loaded.working_directory,
        environment=config.benchmark.environment,
        timeout_seconds=config.benchmark.timeout_seconds,
    )
    return ExecutionPlan(
        server=server,
        warmup=warmup,
        benchmark=benchmark,
        readiness_url=(
            f"http://{config.server.host}:{config.server.port}{config.server.readiness_path}"
        ),
        repeats=config.experiment.repeats,
        readiness_timeout_seconds=config.server.readiness_timeout_seconds,
        readiness_poll_interval_seconds=config.server.readiness_poll_interval_seconds,
        shutdown_timeout_seconds=config.server.shutdown_timeout_seconds,
    )


def format_plan(loaded: LoadedConfig, plan: ExecutionPlan) -> str:
    warmup = "disabled" if plan.warmup is None else " ".join(plan.warmup.args)
    return "\n".join(
        [
            f"Experiment: {loaded.config.experiment.name}",
            f"Working directory: {loaded.working_directory}",
            f"Results root: {loaded.results_root}",
            f"Server command: {' '.join(plan.server.args)}",
            f"Readiness URL: {plan.readiness_url}",
            f"Warm-up: {warmup}",
            f"Benchmark command: {' '.join(plan.benchmark.args)}",
            f"Trials: {plan.repeats}",
            "Timeouts: "
            f"readiness={plan.readiness_timeout_seconds}s, "
            f"poll={plan.readiness_poll_interval_seconds}s, "
            f"shutdown={plan.shutdown_timeout_seconds}s, "
            f"benchmark={plan.benchmark.timeout_seconds}s",
        ]
    )
