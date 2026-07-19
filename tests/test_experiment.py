from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_inference_experiments.cli import app
from llm_inference_experiments.config import load_config
from llm_inference_experiments.errors import ExperimentExecutionError
from llm_inference_experiments.experiment import ExperimentRunner

from .conftest import write_config


def test_successful_experiment_writes_expected_output(tmp_path: Path) -> None:
    run_directory = ExperimentRunner().run(load_config(write_config(tmp_path)))
    manifest = json.loads((run_directory / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["completed_trials"] == 2
    assert (run_directory / "config/original.yaml").exists()
    assert (run_directory / "config/resolved.json").exists()
    assert (run_directory / "metadata/system.json").exists()
    assert (run_directory / "logs/server.stdout.log").exists()
    assert (run_directory / "trials/trial_001/status.json").exists()


def test_benchmark_failure_is_preserved_and_server_is_stopped(tmp_path: Path) -> None:
    with pytest.raises(ExperimentExecutionError):
        ExperimentRunner().run(load_config(write_config(tmp_path, fail_benchmark=True)))
    manifest_path = next((tmp_path / "results/test-run").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["failed_trials"] == 1
    assert manifest["server_process"]["return_code"] is not None
    assert (manifest_path.parent / "trials/trial_000/benchmark.stderr.log").exists()


def test_dry_run_creates_no_processes_or_run_directory(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    result = CliRunner().invoke(app, ["run", str(config), "--dry-run"])
    assert result.exit_code == 0
    assert "nothing was executed" in result.stdout
    assert not (tmp_path / "results").exists()
