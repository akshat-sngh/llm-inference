"""Pydantic models for the version 1 experiment file."""

from __future__ import annotations

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


class ExperimentConfig(BaseModel):
    """The validated, un-resolved versioned YAML document."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    experiment: ExperimentSettings
    paths: PathsSettings = Field(default_factory=PathsSettings)
    server: ServerSettings
    warmup: WarmupSettings = Field(default_factory=WarmupSettings)
    benchmark: BenchmarkSettings

    @model_validator(mode="after")
    def supported_schema_version(self) -> ExperimentConfig:
        if self.schema_version != 1:
            raise ValueError("only schema_version 1 is supported")
        return self
