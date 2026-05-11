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
    # --- syntactic patterns ---
    (re.compile(r"rm\s+.*-[rf]*f[rf]*\s+/", re.IGNORECASE), "rm -rf / detected"),
    (re.compile(r">\s*/dev/sd[a-z]", re.IGNORECASE), "raw block device write detected"),
    (re.compile(r"\bdd\b.*\bof=/dev/[sh]d", re.IGNORECASE), "dd to disk device detected"),
    (re.compile(r"\bmkfs\b", re.IGNORECASE), "filesystem format command detected"),
    (re.compile(r"\bshred\b", re.IGNORECASE), "shred command detected"),
    (re.compile(r":\(\)\s*\{.*:\|:", re.IGNORECASE | re.DOTALL), "fork bomb pattern detected"),
    (re.compile(r"[;|&]\s*rm\s+.*-[rf]", re.IGNORECASE), "chained destructive rm detected"),
    # --- semantic patterns — catastrophic by effect, not by syntax ---
    (re.compile(r"\bfind\s+/[\s/][^|;&\n]*--?delete\b", re.IGNORECASE), "find -delete on filesystem root detected"),
    (re.compile(r"\bfind\s+/[\s/][^|;&\n]*-exec\s+rm\b", re.IGNORECASE), "find -exec rm on filesystem root detected"),
    (re.compile(r"\bchmod\s+-[Rr]\w*\s+[0-7]*7{2,}\s+/(?=\s|$|[;&|])", re.IGNORECASE), "chmod recursive world-writable on filesystem root detected"),
    (re.compile(r"\bchown\s+-[Rr]\w*\s+\S+\s+/(?:etc|boot|usr|bin|sbin|lib)\b", re.IGNORECASE), "chown recursive on critical system path detected"),
    (re.compile(r"\brm\b[^|;&\n]*\s/(?:etc/passwd|etc/shadow|etc/sudoers|boot/\S+)\b", re.IGNORECASE), "deletion of critical system file detected"),
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
