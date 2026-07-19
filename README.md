# llm-inference-experiments

This project is a reproducible experiment runner for benchmarking and profiling LLM inference on rented, single-GPU machines. Its first milestone is deliberately local and fully testable: it controls fake server and benchmark programs without installing vLLM, CUDA, PyTorch, or GPU dependencies on the development laptop.

Real vLLM integration and GPU telemetry are later milestones. The runner currently records raw benchmark output only; it does not invent benchmark request generation, metric parsing, or aggregation.

## Setup

Python 3.11+ is required. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest
uv run ruff check .
uv run llm-exp validate configs/examples/local-smoke.yaml
uv run llm-exp plan configs/examples/local-smoke.yaml
uv run llm-exp run configs/examples/local-smoke.yaml --dry-run
uv run llm-exp run configs/examples/local-smoke.yaml
```

The same commands work after installing the package with another Python environment manager. No GPU packages are dependencies of this project.

## Configuration and output

Experiment YAML files describe server, optional warm-up, and benchmark commands as explicit argument lists. Relative `paths.results_root` and `paths.working_directory` values are resolved from the YAML file's directory. Environment entries are merged with the parent environment; shell interpolation is never performed.

A real run creates `results/<experiment>/<UTC timestamp_run id>/`. It contains the original and resolved configuration, lightweight system/Python/Git metadata, planned command JSON, separate stdout/stderr logs, per-trial status files, and a `manifest.json` that indexes the run and captures failure details. Dry runs create no directory and launch no process.
