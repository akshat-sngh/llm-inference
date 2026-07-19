from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llm_inference_experiments.config import load_config
from llm_inference_experiments.errors import ConfigurationError

from .conftest import write_config


def test_loads_valid_configuration(tmp_path: Path) -> None:
    loaded = load_config(write_config(tmp_path))
    assert loaded.config.experiment.repeats == 2
    assert loaded.resolved_dict()["paths"]["results_root"] == str(tmp_path / "results")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("port", 0, "server.port"), ("repeats", 0, "experiment.repeats")],
)
def test_rejects_invalid_values(tmp_path: Path, field: str, value: int, message: str) -> None:
    path = write_config(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if field == "port":
        document["server"]["port"] = value
    else:
        document["experiment"]["repeats"] = value
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        load_config(path)


def test_paths_are_resolved_from_configuration_directory(tmp_path: Path) -> None:
    config_directory = tmp_path / "nested"
    config_directory.mkdir()
    source = write_config(tmp_path)
    destination = config_directory / "experiment.yaml"
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    loaded = load_config(destination)
    assert loaded.working_directory == config_directory
    assert loaded.results_root == config_directory / "results"


def test_example_results_root_is_at_repository_level() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    loaded = load_config(repository_root / "configs/examples/local-smoke.yaml")
    assert loaded.results_root == repository_root / "results"
