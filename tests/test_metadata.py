from __future__ import annotations

from pathlib import Path

from llm_inference_experiments.metadata import git_metadata, python_metadata, system_metadata


def test_metadata_handles_non_git_directory(tmp_path: Path) -> None:
    assert git_metadata(tmp_path)["available"] is False
    assert "executable" in python_metadata()
    assert "hostname" in system_metadata()
