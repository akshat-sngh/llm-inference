"""Project-specific exceptions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .process import ProcessResult


class ExperimentError(Exception):
    """Base exception for an experiment lifecycle failure."""


class ConfigurationError(ExperimentError):
    """Raised when an experiment configuration cannot be loaded or validated."""


class ProcessError(ExperimentError):
    """Raised when a managed subprocess cannot complete as requested."""


class ProcessTimeoutError(ProcessError):
    """Raised when a subprocess exceeds its configured timeout."""

    def __init__(self, message: str, result: ProcessResult) -> None:
        super().__init__(message)
        self.result = result


class ReadinessError(ExperimentError):
    """Raised when the server does not become ready."""


class PreflightError(ExperimentError):
    """Raised when a configuration cannot be executed on this machine."""


class ExperimentExecutionError(ExperimentError):
    """Raised for a failed command during the experiment lifecycle."""
