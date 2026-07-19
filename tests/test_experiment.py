from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llm_inference_experiments.cli import app
from llm_inference_experiments.config import load_config
from llm_inference_experiments.errors import ExperimentExecutionError, ProcessTimeoutError
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


def test_warmup_success_is_recorded(tmp_path: Path) -> None:
    run_directory = ExperimentRunner().run(load_config(write_config(tmp_path, enable_warmup=True)))
    manifest = json.loads((run_directory / "manifest.json").read_text())
    assert manifest["warmup_process"]["return_code"] == 0
    assert manifest["warmup_process"]["timed_out"] is False
    assert (run_directory / "logs/warmup.stdout.log").exists()


def test_warmup_failure_preserves_result_and_stops_server(tmp_path: Path) -> None:
    with pytest.raises(ExperimentExecutionError):
        ExperimentRunner().run(
            load_config(write_config(tmp_path, enable_warmup=True, fail_warmup=True))
        )
    manifest_path = next((tmp_path / "results/test-run").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["failed_phase"] == "warming_up"
    assert manifest["warmup_process"]["return_code"] == 7
    assert manifest["server_process"]["return_code"] is not None


def test_benchmark_timeout_writes_status_json(tmp_path: Path) -> None:
    with pytest.raises(ProcessTimeoutError):
        ExperimentRunner().run(
            load_config(write_config(tmp_path, benchmark_timeout=0.05, benchmark_sleep_seconds=1))
        )
    manifest_path = next((tmp_path / "results/test-run").glob("*/manifest.json"))
    status = json.loads((manifest_path.parent / "trials/trial_000/status.json").read_text())
    manifest = json.loads(manifest_path.read_text())
    assert status["status"] == "failed"
    assert status["timed_out"] is True
    assert status["ended_at"]
    assert status["duration_seconds"] >= 0
    assert manifest["status"] == "failed"


def test_warmup_timeout_is_recorded_in_manifest(tmp_path: Path) -> None:
    with pytest.raises(ProcessTimeoutError):
        ExperimentRunner().run(
            load_config(
                write_config(
                    tmp_path,
                    enable_warmup=True,
                    warmup_timeout=0.05,
                    warmup_sleep_seconds=1,
                )
            )
        )
    manifest_path = next((tmp_path / "results/test-run").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["warmup_process"]["timed_out"] is True
    assert manifest["warmup_process"]["ended_at"]
    assert manifest["status"] == "failed"


def test_dry_run_creates_no_processes_or_run_directory(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    result = CliRunner().invoke(app, ["run", str(config), "--dry-run"])
    assert result.exit_code == 0
    assert "nothing was executed" in result.stdout
    assert not (tmp_path / "results").exists()


def test_validate_rejects_invalid_configuration(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    document = yaml.safe_load(config.read_text())
    document["server"]["port"] = 0
    config.write_text(yaml.safe_dump(document))
    result = CliRunner().invoke(app, ["validate", str(config)])
    assert result.exit_code == 1
    assert "server.port" in result.output


def test_plan_creates_no_run_directory_or_processes(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    result = CliRunner().invoke(app, ["plan", str(config)])
    assert result.exit_code == 0
    assert "Experiment: test-run" in result.output
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("command", ["plan", "run"])
def test_plan_and_dry_run_report_preflight_failure(tmp_path: Path, command: str) -> None:
    config = write_config(tmp_path)
    document = yaml.safe_load(config.read_text())
    document["server"]["command"] = ["missing-llm-exp-executable"]
    config.write_text(yaml.safe_dump(document))
    arguments = [command, str(config)]
    if command == "run":
        arguments.append("--dry-run")
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 1
    assert "Preflight validation failed" in result.output
    assert "server executable cannot be located" in result.output
    assert not (tmp_path / "results").exists()
