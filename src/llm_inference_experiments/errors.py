"""Project-specific exceptions."""

from __future__ import annotations


class ExperimentError(Exception):
    """Base exception for an experiment lifecycle failure."""


class ConfigurationError(ExperimentError):
    """Raised when an experiment configuration cannot be loaded or validated."""


class ProcessError(ExperimentError):
    """Raised when a managed subprocess cannot complete as requested."""


class ProcessTimeoutError(ProcessError):
    """Raised when a subprocess exceeds its configured timeout."""


class ReadinessError(ExperimentError):
    """Raised when the server does not become ready."""


class ExperimentExecutionError(ExperimentError):
    """Raised for a failed command during the experiment lifecycle."""
