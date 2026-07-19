"""YAML loading and path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .errors import ConfigurationError
from .models import ExperimentConfig
from .paths import resolve_from_config


@dataclass(frozen=True)
class LoadedConfig:
    config: ExperimentConfig
    config_path: Path
    original_yaml: str
    working_directory: Path
    results_root: Path

    def resolved_dict(self) -> dict[str, Any]:
        document = self.config.model_dump(mode="json")
        document["paths"] = {
            "results_root": str(self.results_root),
            "working_directory": str(self.working_directory),
        }
        return document


def load_config(path: str | Path) -> LoadedConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        original_yaml = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"Unable to read configuration {config_path}: {exc}") from exc

    try:
        document = yaml.safe_load(original_yaml)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigurationError("Configuration root must be a mapping")

    try:
        config = ExperimentConfig.model_validate(document)
    except ValidationError as exc:
        raise ConfigurationError(format_validation_error(exc)) from exc

    return LoadedConfig(
        config=config,
        config_path=config_path,
        original_yaml=original_yaml,
        working_directory=resolve_from_config(config_path, config.paths.working_directory),
        results_root=resolve_from_config(config_path, config.paths.results_root),
    )


def format_validation_error(error: ValidationError) -> str:
    lines = ["Configuration validation failed:"]
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "configuration"
        lines.append(f"  {location}: {item['msg']}")
    return "\n".join(lines)
