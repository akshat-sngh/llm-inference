from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llm_inference_experiments.cli import app
from llm_inference_experiments.config import load_config
from llm_inference_experiments.errors import (
    ConfigurationError,
    ExperimentExecutionError,
    PreflightError,
)
from llm_inference_experiments.experiment import ExperimentRunner
from llm_inference_experiments.vllm import VllmError

from .conftest import create_fake_vllm_repository, write_config


def fake_vllm_executable() -> Path:
    path = Path(__file__).parent / "fixtures/fake_vllm.py"
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def vllm_settings(repository: Path, commit: str, **overrides: object) -> dict[str, object]:
    settings: dict[str, object] = {
        "enabled": True,
        "repository_path": str(repository),
        "executable": str(fake_vllm_executable()),
        "python_executable": __import__("sys").executable,
        "expected_commit": commit,
        "require_clean_worktree": True,
        "capture_dirty_diff": True,
        "collect_environment_report": True,
        "model": {"id": "fake/model", "revision": "fake-revision", "served_name": "fake-model"},
        "benchmark_result": {"required": True, "filename": "vllm-result.json"},
    }
    settings.update(overrides)
    return settings


def write_vllm_config(tmp_path: Path, **overrides: object) -> Path:
    repository, commit = create_fake_vllm_repository(tmp_path)
    config = write_config(tmp_path, vllm=vllm_settings(repository, commit, **overrides))
    document = yaml.safe_load(config.read_text())
    fake = str(fake_vllm_executable())
    port = document["server"]["port"]
    document["server"]["command"] = [fake, "serve"]
    document["server"]["arguments"] = ["--host", "127.0.0.1", "--port", str(port)]
    document["warmup"] = {
        "enabled": True,
        "command": [fake, "bench", "serve"],
        "arguments": ["--model", "fake/model"],
        "timeout_seconds": 2,
    }
    document["benchmark"]["command"] = [fake, "bench", "serve"]
    document["benchmark"]["arguments"] = [
        "--model",
        "fake/model",
        "--save-result",
        "--result-dir",
        "{trial_dir}",
        "--result-filename",
        "vllm-result.json",
        "--tag",
        '{"kept": true}',
    ]
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    return config


def manifest_path(tmp_path: Path) -> Path:
    return next((tmp_path / "results/test-run").glob("*/manifest.json"))


def test_vllm_disabled_existing_config_still_runs(tmp_path: Path) -> None:
    run_directory = ExperimentRunner().run(load_config(write_config(tmp_path)))
    assert not (run_directory / "metadata/vllm.json").exists()


@pytest.mark.parametrize("expected_commit", ["short", "z" * 40])
def test_invalid_vllm_expected_commit_is_rejected(tmp_path: Path, expected_commit: str) -> None:
    repository, commit = create_fake_vllm_repository(tmp_path)
    config = write_config(tmp_path, vllm=vllm_settings(repository, expected_commit))
    with pytest.raises(ConfigurationError, match="expected_commit"):
        load_config(config)


def test_vllm_dry_run_does_not_invoke_fake_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_vllm_config(tmp_path)
    monkeypatch.setenv("FAKE_VLLM_MODE", "bench-fail")
    result = CliRunner().invoke(app, ["run", str(config), "--dry-run"])
    assert result.exit_code == 0
    assert not (tmp_path / "results").exists()


def test_doctor_runs_vllm_probes_without_starting_server(tmp_path: Path) -> None:
    config = write_vllm_config(tmp_path)
    output = tmp_path / "doctor.json"
    result = CliRunner().invoke(app, ["doctor", str(config), "--output", str(output)])
    report = json.loads(output.read_text())
    assert result.exit_code == 0
    assert report["passed"] is True
    assert any(check["name"] == "vllm_serve_help" for check in report["checks"])
    assert not (tmp_path / "results").exists()


def test_commit_mismatch_prevents_server_start(tmp_path: Path) -> None:
    config = write_vllm_config(tmp_path, expected_commit="a" * 40)
    with pytest.raises(VllmError, match="vLLM commit mismatch"):
        ExperimentRunner().run(load_config(config))
    path = manifest_path(tmp_path)
    assert not (path.parent / "logs/server.stdout.log").exists()


def test_dirty_allowed_repository_writes_diff_artifacts(tmp_path: Path) -> None:
    config = write_vllm_config(tmp_path, require_clean_worktree=False)
    document = yaml.safe_load(config.read_text())
    repository = Path(document["vllm"]["repository_path"])
    (repository / "marker.txt").write_text("dirty\n")
    run_directory = ExperimentRunner().run(load_config(config))
    assert (run_directory / "metadata/vllm-diff.patch").exists()
    assert (run_directory / "metadata/vllm-git-status.txt").exists()


def test_vllm_metadata_and_environment_probes_are_saved(tmp_path: Path) -> None:
    run_directory = ExperimentRunner().run(load_config(write_vllm_config(tmp_path)))
    metadata = json.loads((run_directory / "metadata/vllm.json").read_text())
    packages = json.loads((run_directory / "metadata/vllm-packages.json").read_text())
    assert metadata["head_matches_expected"] is True
    assert (run_directory / "metadata/vllm-version.txt").read_text().startswith("fake-vllm")
    assert (run_directory / "metadata/vllm-environment.json").exists()
    assert packages == sorted(packages, key=lambda item: item["name"].lower().replace("-", "_"))


def test_placeholder_substitution_and_trial_results_are_preserved(tmp_path: Path) -> None:
    run_directory = ExperimentRunner().run(load_config(write_vllm_config(tmp_path)))
    status_zero = json.loads((run_directory / "trials/trial_000/status.json").read_text())
    status_one = json.loads((run_directory / "trials/trial_001/status.json").read_text())
    index = json.loads((run_directory / "trials/trial_000/result-index.json").read_text())
    assert str(run_directory / "trials/trial_000") in status_zero["command"]
    assert str(run_directory / "trials/trial_001") in status_one["command"]
    assert '{"kept": true}' in status_zero["command"]
    assert index["sha256"] and index["size_bytes"] > 0
    assert (
        json.loads((run_directory / "trials/trial_000/vllm-result.json").read_text())["completed"]
        == 64
    )


@pytest.mark.parametrize("mode", ["missing-result", "invalid-result"])
def test_required_missing_or_invalid_native_result_fails_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("FAKE_VLLM_MODE", mode)
    with pytest.raises(ExperimentExecutionError):
        ExperimentRunner().run(load_config(write_vllm_config(tmp_path)))
    manifest = json.loads(manifest_path(tmp_path).read_text())
    assert manifest["status"] == "failed"
    assert manifest["failed_trials"] == 1


def test_partial_native_result_is_indexed_after_benchmark_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_VLLM_MODE", "partial-result")
    with pytest.raises(ExperimentExecutionError):
        ExperimentRunner().run(load_config(write_vllm_config(tmp_path)))
    path = manifest_path(tmp_path).parent / "trials/trial_000/result-index.json"
    assert json.loads(path.read_text())["partial"] is True


@pytest.mark.parametrize(
    ("target", "value", "message"),
    [
        ("server_placeholder", "{trial_dir}", "only valid"),
        ("benchmark_placeholder", "{unknown_value}", "Unknown placeholder"),
        ("cuda", "0,1", "CUDA_VISIBLE_DEVICES"),
        ("parallel", "2", "tensor-parallel-size"),
    ],
)
def test_vllm_preflight_rejects_unsafe_templates_and_single_gpu_violations(
    tmp_path: Path, target: str, value: str, message: str
) -> None:
    config = write_vllm_config(tmp_path)
    document = yaml.safe_load(config.read_text())
    if target == "server_placeholder":
        document["server"]["arguments"].append(value)
    elif target == "benchmark_placeholder":
        document["benchmark"]["arguments"].append(value)
    elif target == "cuda":
        document["server"].setdefault("environment", {})["CUDA_VISIBLE_DEVICES"] = value
    else:
        document["server"]["arguments"].extend(["--tensor-parallel-size", value])
    config.write_text(yaml.safe_dump(document))
    with pytest.raises(PreflightError, match=message):
        ExperimentRunner().run(load_config(config))
