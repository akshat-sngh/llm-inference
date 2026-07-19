"""Typer command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer

from .commands import build_plan, format_plan
from .config import load_config
from .errors import ExperimentError
from .experiment import ExperimentRunner
from .preflight import validate_preflight

app = typer.Typer(help="Run reproducible local LLM inference experiments.", no_args_is_help=True)


def _load_or_exit(config: Path):
    try:
        return load_config(config)
    except ExperimentError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def validate(config: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Validate a versioned experiment YAML file."""
    _load_or_exit(config)
    typer.echo(f"Configuration is valid: {config}")


@app.command()
def plan(config: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Print the resolved execution plan without changing anything."""
    loaded = _load_or_exit(config)
    execution_plan = build_plan(loaded)
    try:
        preflight = validate_preflight(loaded, execution_plan)
    except ExperimentError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(format_plan(loaded, execution_plan, preflight))


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, readable=True),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan without executing it."),
) -> None:
    """Execute an experiment lifecycle."""
    loaded = _load_or_exit(config)
    execution_plan = build_plan(loaded)
    if dry_run:
        try:
            preflight = validate_preflight(loaded, execution_plan)
        except ExperimentError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(format_plan(loaded, execution_plan, preflight))
        typer.echo("Dry run: nothing was executed and no run directory was created.")
        return
    try:
        run_directory = ExperimentRunner().run(loaded)
    except ExperimentError as exc:
        typer.echo(f"Experiment failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Experiment completed: {run_directory}")
