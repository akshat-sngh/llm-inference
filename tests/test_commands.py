from __future__ import annotations

from pathlib import Path

from llm_inference_experiments.commands import build_plan
from llm_inference_experiments.config import load_config

from .conftest import write_config


def test_constructs_exact_commands(tmp_path: Path) -> None:
    loaded = load_config(write_config(tmp_path))
    plan = build_plan(loaded)
    assert plan.server.args[-4:] == [
        "--host",
        "127.0.0.1",
        "--port",
        str(loaded.config.server.port),
    ]
    assert plan.benchmark.args[-4:] == ["--label", "test", "--sleep-seconds", "0.02"]
    assert plan.warmup is None
