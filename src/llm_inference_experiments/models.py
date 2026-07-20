"""Pydantic models for the version 1 experiment file."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

PositiveFloat = Annotated[float, Field(gt=0)]
PositiveInt = Annotated[int, Field(gt=0)]
Command = Annotated[list[str], Field(min_length=1)]


class ExperimentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    seed: int | None = None
    repeats: PositiveInt = 1


class PathsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results_root: str = "./results"
    working_directory: str = "."


class ServerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Command
    arguments: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    readiness_path: str = Field(min_length=1)
    readiness_timeout_seconds: PositiveFloat
    readiness_poll_interval_seconds: PositiveFloat
    shutdown_timeout_seconds: PositiveFloat

    @model_validator(mode="after")
    def readiness_path_is_absolute(self) -> ServerSettings:
        if not self.readiness_path.startswith("/"):
            raise ValueError("readiness_path must begin with '/'")
        return self


class WarmupSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    command: list[str] = Field(default_factory=list)
    arguments: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: PositiveFloat = 30

    @model_validator(mode="after")
    def enabled_requires_command(self) -> WarmupSettings:
        if self.enabled and not self.command:
            raise ValueError("warmup.command cannot be empty when warmup is enabled")
        return self


class BenchmarkSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Command
    arguments: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: PositiveFloat


class NvidiaTelemetrySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    required: bool = False
    executable: str = Field(default="nvidia-smi", min_length=1)
    device_index: int = Field(default=0, ge=0)
    sample_interval_seconds: PositiveFloat = 1.0
    command_timeout_seconds: PositiveFloat = 5.0


class TelemetrySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nvidia: NvidiaTelemetrySettings = Field(default_factory=NvidiaTelemetrySettings)


class VllmModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = ""
    revision: str = ""
    served_name: str = ""


class VllmBenchmarkResultSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: bool = True
    filename: str = "vllm-result.json"

    @model_validator(mode="after")
    def filename_is_plain(self) -> VllmBenchmarkResultSettings:
        path = Path(self.filename)
        if (
            not self.filename
            or path.is_absolute()
            or path.name != self.filename
            or ".." in path.parts
        ):
            raise ValueError("filename must be a plain filename without path traversal")
        return self


class VllmSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    repository_path: str = ""
    executable: str = ""
    python_executable: str = ""
    expected_commit: str = ""
    require_clean_worktree: bool = True
    capture_dirty_diff: bool = True
    collect_environment_report: bool = True
    model: VllmModelSettings = Field(default_factory=VllmModelSettings)
    benchmark_result: VllmBenchmarkResultSettings = Field(
        default_factory=VllmBenchmarkResultSettings
    )

    @model_validator(mode="after")
    def enabled_settings_are_complete(self) -> VllmSettings:
        if not self.enabled:
            return self
        required_fields = {
            "repository_path": self.repository_path,
            "executable": self.executable,
            "python_executable": self.python_executable,
            "model.id": self.model.id,
            "model.revision": self.model.revision,
            "model.served_name": self.model.served_name,
        }
        missing = [name for name, value in required_fields.items() if not value]
        if missing:
            raise ValueError(f"vLLM enabled configuration requires: {', '.join(missing)}")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", self.expected_commit):
            raise ValueError("expected_commit must be exactly 40 hexadecimal characters")
        return self


class ExperimentConfig(BaseModel):
    """The validated, un-resolved versioned YAML document."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    experiment: ExperimentSettings
    paths: PathsSettings = Field(default_factory=PathsSettings)
    server: ServerSettings
    warmup: WarmupSettings = Field(default_factory=WarmupSettings)
    benchmark: BenchmarkSettings
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    vllm: VllmSettings = Field(default_factory=VllmSettings)

    @model_validator(mode="after")
    def supported_schema_version(self) -> ExperimentConfig:
        if self.schema_version != 1:
            raise ValueError("only schema_version 1 is supported")
        return self
