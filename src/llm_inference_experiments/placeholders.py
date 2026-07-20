"""Safe, context-aware substitution for individual command and environment values."""

from __future__ import annotations

import re

from .errors import ConfigurationError

PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
TRIAL_ONLY = {"trial_dir", "trial_index"}


def substitute(value: str, context: dict[str, str], *, allow_trial: bool) -> str:
    """Replace known placeholders while preserving unrelated JSON-style braces."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in context:
            raise ConfigurationError(f"Unknown placeholder: {{{name}}}")
        if name in TRIAL_ONLY and not allow_trial:
            raise ConfigurationError(
                f"Placeholder {{{name}}} is only valid in benchmark trial commands"
            )
        return context[name]

    return PLACEHOLDER_PATTERN.sub(replace, value)
