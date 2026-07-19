from __future__ import annotations

import json
import stat
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from llm_inference_experiments.cli import app
from llm_inference_experiments.config import load_config
from llm_inference_experiments.errors import (
    ExperimentExecutionError,
    PreflightError,
    ReadinessError,
)
from llm_inference_experiments.experiment import ExperimentRunner
from llm_inference_experiments.models import ExperimentConfig, NvidiaTelemetrySettings
from llm_inference_experiments.nvidia import collect_metadata

from .conftest import write_config


def fake_nvidia_executable() -> Path:
    path = Path(__file__).parent / "fixtures/fake_nvidia_smi.py"
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def nvidia_settings(executable: Path, **overrides: object) -> dict[str, object]:
    settings: dict[str, object] = {
        "enabled": True,
        "executable": str(executable),
        "sample_interval_seconds": 0.02,
        "command_timeout_seconds": 0.5,
    }
    settings.update(overrides)
    return settings


def manifest_for(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    path = next((tmp_path / "results/test-run").glob("*/manifest.json"))
    return path, json.loads(path.read_text())


def jsonl_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def assert_no_sampler_thread() -> None:
    assert not any(
        thread.name == "nvidia-telemetry" and thread.is_alive() for thread in threading.enumerate()
    )


def test_existing_configuration_without_telemetry_is_valid(tmp_path: Path) -> None:
    loaded = load_config(write_config(tmp_path))
    assert loaded.config.telemetry.nvidia.enabled is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device_index", -1),
        ("sample_interval_seconds", 0),
        ("command_timeout_seconds", 0),
        ("executable", ""),
    ],
)
def test_nvidia_telemetry_validation(field: str, value: object) -> None:
    document = {
        "schema_version": 1,
        "experiment": {"name": "validation"},
        "server": {
            "command": ["python3"],
            "host": "localhost",
            "port": 1,
            "readiness_path": "/health",
            "readiness_timeout_seconds": 1,
            "readiness_poll_interval_seconds": 1,
            "shutdown_timeout_seconds": 1,
        },
        "benchmark": {"command": ["python3"], "timeout_seconds": 1},
        "telemetry": {"nvidia": {field: value}},
    }
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(document)


def test_disabled_telemetry_creates_no_nvidia_artifacts(tmp_path: Path) -> None:
    run_directory = ExperimentRunner().run(load_config(write_config(tmp_path)))
    assert not (run_directory / "metadata/nvidia.json").exists()
    assert not (run_directory / "telemetry").exists()


def test_optional_missing_nvidia_executable_continues(tmp_path: Path) -> None:
    config = write_config(
        tmp_path,
        telemetry=nvidia_settings(Path("missing-nvidia-smi")),
    )
    run_directory = ExperimentRunner().run(load_config(config))
    metadata = json.loads((run_directory / "metadata/nvidia.json").read_text())
    manifest = json.loads((run_directory / "manifest.json").read_text())
    assert metadata["available"] is False
    assert manifest["status"] == "completed"
    assert manifest["telemetry"]["available"] is False


def test_required_missing_nvidia_executable_fails_preflight(tmp_path: Path) -> None:
    config = write_config(
        tmp_path,
        telemetry=nvidia_settings(Path("missing-nvidia-smi"), required=True),
    )
    with pytest.raises(PreflightError, match="NVIDIA executable"):
        ExperimentRunner().run(load_config(config))
    assert not (tmp_path / "results").exists()


def test_dry_run_does_not_invoke_nvidia(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counter = tmp_path / "nvidia-count"
    monkeypatch.setenv("FAKE_NVIDIA_COUNTER_FILE", str(counter))
    config = write_config(tmp_path, telemetry=nvidia_settings(fake_nvidia_executable()))
    result = CliRunner().invoke(app, ["run", str(config), "--dry-run"])
    assert result.exit_code == 0
    assert not counter.exists()


def test_successful_nvidia_metadata_parsing() -> None:
    settings = NvidiaTelemetrySettings(enabled=True, executable=str(fake_nvidia_executable()))
    metadata = collect_metadata(settings, fake_nvidia_executable())
    assert metadata["available"] is True
    assert metadata["gpu"]["name"] == "Fake NVIDIA GPU"
    assert metadata["gpu"]["memory_total_mib"] == 24564


def test_unavailable_nvidia_csv_values_become_null(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_NVIDIA_MODE", "unavailable-fields")
    settings = NvidiaTelemetrySettings(enabled=True, executable=str(fake_nvidia_executable()))
    metadata = collect_metadata(settings, fake_nvidia_executable())
    assert metadata["available"] is True
    assert metadata["gpu"]["uuid"] is None
    assert metadata["gpu"]["memory_total_mib"] is None


def test_periodic_telemetry_writes_numeric_jsonl_and_manifest_paths(tmp_path: Path) -> None:
    config = write_config(
        tmp_path,
        benchmark_sleep_seconds=0.12,
        telemetry=nvidia_settings(fake_nvidia_executable()),
    )
    run_directory = ExperimentRunner().run(load_config(config))
    records = jsonl_records(run_directory / "telemetry/nvidia.jsonl")
    manifest = json.loads((run_directory / "manifest.json").read_text())
    assert records
    assert isinstance(records[0]["metrics"]["gpu_utilization_percent"], int)
    assert isinstance(records[0]["metrics"]["power_draw_watts"], float)
    assert manifest["telemetry"]["nvidia_samples_path"] == "telemetry/nvidia.jsonl"
    assert manifest["telemetry"]["successful_samples"] >= 1
    assert_no_sampler_thread()


def test_intermittent_sample_failure_is_recorded_without_failing_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_NVIDIA_FAIL_AFTER", "1")
    monkeypatch.setenv("FAKE_NVIDIA_COUNTER_FILE", str(tmp_path / "counter"))
    config = write_config(
        tmp_path,
        benchmark_sleep_seconds=0.12,
        telemetry=nvidia_settings(fake_nvidia_executable()),
    )
    run_directory = ExperimentRunner().run(load_config(config))
    records = jsonl_records(run_directory / "telemetry/nvidia.jsonl")
    status = json.loads((run_directory / "telemetry/status.json").read_text())
    assert any(record["status"] == "error" for record in records)
    assert status["successful_samples"] >= 1
    assert status["failed_samples"] >= 1
    assert_no_sampler_thread()


@pytest.mark.parametrize("failure", ["readiness", "warmup", "benchmark"])
def test_telemetry_stops_after_experiment_failures(tmp_path: Path, failure: str) -> None:
    kwargs: dict[str, object] = {"telemetry": nvidia_settings(fake_nvidia_executable())}
    expected_exception: type[Exception] = ExperimentExecutionError
    if failure == "readiness":
        kwargs.update({"server_health_status": 503, "readiness_timeout": 0.1})
        expected_exception = ReadinessError
    elif failure == "warmup":
        kwargs.update({"enable_warmup": True, "fail_warmup": True})
    else:
        kwargs.update({"fail_benchmark": True})
    with pytest.raises(expected_exception):
        ExperimentRunner().run(load_config(write_config(tmp_path, **kwargs)))
    manifest_path, manifest = manifest_for(tmp_path)
    status = json.loads((manifest_path.parent / "telemetry/status.json").read_text())
    assert manifest["status"] == "failed"
    assert status["sampler_status"] == "completed"
    assert status["stopped_at"]
    assert_no_sampler_thread()


def test_required_probe_failure_prevents_server_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_NVIDIA_MODE", "metadata-fail")
    config = write_config(
        tmp_path,
        telemetry=nvidia_settings(fake_nvidia_executable(), required=True),
    )
    with pytest.raises(ExperimentExecutionError, match="NVIDIA metadata probe failed"):
        ExperimentRunner().run(load_config(config))
    manifest_path, manifest = manifest_for(tmp_path)
    assert manifest["status"] == "failed"
    assert not (manifest_path.parent / "logs/server.stdout.log").exists()


def test_required_missing_device_prevents_server_start(tmp_path: Path) -> None:
    config = write_config(
        tmp_path,
        telemetry=nvidia_settings(fake_nvidia_executable(), required=True, device_index=9),
    )
    with pytest.raises(ExperimentExecutionError, match="NVIDIA metadata probe failed"):
        ExperimentRunner().run(load_config(config))
    manifest_path, manifest = manifest_for(tmp_path)
    assert manifest["status"] == "failed"
    assert not (manifest_path.parent / "logs/server.stdout.log").exists()


def test_optional_probe_failure_allows_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_NVIDIA_MODE", "metadata-fail")
    config = write_config(tmp_path, telemetry=nvidia_settings(fake_nvidia_executable()))
    run_directory = ExperimentRunner().run(load_config(config))
    metadata = json.loads((run_directory / "metadata/nvidia.json").read_text())
    assert metadata["available"] is False
    assert (run_directory / "trials/trial_001/status.json").exists()


def test_nvidia_command_records_are_argument_arrays(tmp_path: Path) -> None:
    config = write_config(tmp_path, telemetry=nvidia_settings(fake_nvidia_executable()))
    run_directory = ExperimentRunner().run(load_config(config))
    command = json.loads((run_directory / "commands/nvidia-sample.json").read_text())
    assert isinstance(command["command"], list)
    assert command["command"][1] == "--id=0"


def test_jsonl_survives_benchmark_failure(tmp_path: Path) -> None:
    config = write_config(
        tmp_path,
        benchmark_sleep_seconds=0.08,
        fail_benchmark=True,
        telemetry=nvidia_settings(fake_nvidia_executable()),
    )
    with pytest.raises(ExperimentExecutionError):
        ExperimentRunner().run(load_config(config))
    manifest_path, _manifest = manifest_for(tmp_path)
    assert jsonl_records(manifest_path.parent / "telemetry/nvidia.jsonl")
    assert_no_sampler_thread()
