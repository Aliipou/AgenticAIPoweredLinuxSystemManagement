"""Confidence gate — rejects or force-downgrades low-confidence intents before execution.

LLM confidence scores are not a security guarantee. This gate translates
confidence thresholds into deterministic gate decisions so that the rest of
the pipeline never has to reason about probability.

Three zones:
  - [0, min_confidence)        → rejected entirely  (LowConfidenceError)
  - [min_confidence, dry_run_below) → approved, but forced into dry-run mode
  - [dry_run_below, 1.0]       → full approval
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic.models.intent import IntentType, ParsedIntent


@dataclass(frozen=True)
class ConfidenceDecision:
    passed: bool
    force_dry_run: bool = False
    reason: str = ""


class ConfidenceGate:
    def __init__(
        self,
        min_confidence: float = 0.70,
        dry_run_below: float = 0.85,
    ) -> None:
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
        if not (0.0 <= dry_run_below <= 1.0):
            raise ValueError(f"dry_run_below must be in [0, 1], got {dry_run_below}")
        if dry_run_below < min_confidence:
            raise ValueError(
                f"dry_run_below ({dry_run_below}) must be >= min_confidence ({min_confidence})"
            )
        self._min = min_confidence
        self._dry_run_below = dry_run_below

    def evaluate(self, intent: ParsedIntent) -> ConfidenceDecision:
        if intent.intent_type == IntentType.UNKNOWN:
            return ConfidenceDecision(
                passed=False,
                reason="Intent classified as UNKNOWN — refusing to act",
            )
        if intent.confidence < self._min:
            return ConfidenceDecision(
                passed=False,
                reason=(
                    f"Confidence {intent.confidence:.2f} below minimum "
                    f"{self._min:.2f} — refusing to act on ambiguous intent"
                ),
            )
        if intent.confidence < self._dry_run_below:
            return ConfidenceDecision(
                passed=True,
                force_dry_run=True,
                reason=(
                    f"Confidence {intent.confidence:.2f} below dry-run threshold "
                    f"{self._dry_run_below:.2f} — forcing dry-run mode"
                ),
            )
        return ConfidenceDecision(
            passed=True,
            force_dry_run=False,
            reason=f"Confidence {intent.confidence:.2f} passed all thresholds",
        )
