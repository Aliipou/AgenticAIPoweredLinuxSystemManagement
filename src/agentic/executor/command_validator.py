"""Command validator — deterministic pre-flight safety check before execution.

Runs AFTER policy evaluation and BEFORE executor.execute_many().
Unlike risk levels (coarse-grained per action type), this inspects the
actual command and target strings for patterns that must never execute
regardless of approved risk level, --force flags, or dry-run mode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agentic.models.action import ActionCandidate

# Each entry: (compiled regex, human-readable reason)
_DANGEROUS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"rm\s+.*-[rf]*f[rf]*\s+/", re.IGNORECASE), "rm -rf / detected"),
    (re.compile(r">\s*/dev/sd[a-z]", re.IGNORECASE), "raw block device write detected"),
    (re.compile(r"\bdd\b.*\bof=/dev/[sh]d", re.IGNORECASE), "dd to disk device detected"),
    (re.compile(r"\bmkfs\b", re.IGNORECASE), "filesystem format command detected"),
    (re.compile(r"\bshred\b", re.IGNORECASE), "shred command detected"),
    (re.compile(r":\(\)\s*\{.*:\|:", re.IGNORECASE | re.DOTALL), "fork bomb pattern detected"),
    (re.compile(r"[;|&]\s*rm\s+.*-[rf]", re.IGNORECASE), "chained destructive rm detected"),
]


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str = ""


class CommandValidator:
    """Deterministic, pattern-based validator for generated command strings."""

    def validate(self, action: ActionCandidate) -> ValidationResult:
        for pattern, reason in _DANGEROUS:
            if pattern.search(action.command) or pattern.search(action.target):
                return ValidationResult(valid=False, reason=reason)
        return ValidationResult(valid=True, reason="")

    def validate_many(
        self, actions: list[ActionCandidate]
    ) -> list[tuple[ActionCandidate, ValidationResult]]:
        return [(action, self.validate(action)) for action in actions]
