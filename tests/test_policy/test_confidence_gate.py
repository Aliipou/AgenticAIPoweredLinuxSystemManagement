"""Tests for ConfidenceGate — full branch coverage."""

from __future__ import annotations

import pytest

from agentic.models.intent import IntentType, ParsedIntent
from agentic.policy.confidence_gate import ConfidenceDecision, ConfidenceGate


def _make_intent(intent_type: IntentType, confidence: float) -> ParsedIntent:
    return ParsedIntent(
        raw_query="test",
        intent_type=intent_type,
        confidence=confidence,
    )


class TestConfidenceGateInit:
    def test_valid_defaults(self):
        gate = ConfidenceGate()
        assert gate._min == 0.70
        assert gate._dry_run_below == 0.85

    def test_valid_custom(self):
        gate = ConfidenceGate(min_confidence=0.5, dry_run_below=0.9)
        assert gate._min == 0.5
        assert gate._dry_run_below == 0.9

    def test_min_confidence_below_zero_raises(self):
        with pytest.raises(ValueError, match="min_confidence"):
            ConfidenceGate(min_confidence=-0.1)

    def test_min_confidence_above_one_raises(self):
        with pytest.raises(ValueError, match="min_confidence"):
            ConfidenceGate(min_confidence=1.1)

    def test_dry_run_below_below_zero_raises(self):
        with pytest.raises(ValueError, match="dry_run_below"):
            ConfidenceGate(min_confidence=0.5, dry_run_below=-0.1)

    def test_dry_run_below_above_one_raises(self):
        with pytest.raises(ValueError, match="dry_run_below"):
            ConfidenceGate(min_confidence=0.5, dry_run_below=1.1)

    def test_dry_run_below_less_than_min_raises(self):
        with pytest.raises(ValueError, match="dry_run_below.*must be >="):
            ConfidenceGate(min_confidence=0.8, dry_run_below=0.7)

    def test_equal_thresholds_valid(self):
        gate = ConfidenceGate(min_confidence=0.8, dry_run_below=0.8)
        assert gate._min == gate._dry_run_below


class TestConfidenceGateEvaluate:
    def test_unknown_intent_always_rejected(self):
        gate = ConfidenceGate()
        intent = _make_intent(IntentType.UNKNOWN, confidence=0.99)
        result = gate.evaluate(intent)
        assert result.passed is False
        assert "UNKNOWN" in result.reason

    def test_below_min_confidence_rejected(self):
        gate = ConfidenceGate(min_confidence=0.70, dry_run_below=0.85)
        intent = _make_intent(IntentType.FOCUS, confidence=0.50)
        result = gate.evaluate(intent)
        assert result.passed is False
        assert result.force_dry_run is False
        assert "0.50" in result.reason
        assert "0.70" in result.reason

    def test_exactly_at_min_confidence_forced_dry_run(self):
        gate = ConfidenceGate(min_confidence=0.70, dry_run_below=0.85)
        intent = _make_intent(IntentType.FOCUS, confidence=0.70)
        result = gate.evaluate(intent)
        assert result.passed is True
        assert result.force_dry_run is True

    def test_between_thresholds_force_dry_run(self):
        gate = ConfidenceGate(min_confidence=0.70, dry_run_below=0.85)
        intent = _make_intent(IntentType.UPDATE, confidence=0.78)
        result = gate.evaluate(intent)
        assert result.passed is True
        assert result.force_dry_run is True
        assert "dry-run" in result.reason

    def test_at_dry_run_threshold_full_approval(self):
        gate = ConfidenceGate(min_confidence=0.70, dry_run_below=0.85)
        intent = _make_intent(IntentType.CLEAN_MEMORY, confidence=0.85)
        result = gate.evaluate(intent)
        assert result.passed is True
        assert result.force_dry_run is False

    def test_above_dry_run_threshold_full_approval(self):
        gate = ConfidenceGate(min_confidence=0.70, dry_run_below=0.85)
        intent = _make_intent(IntentType.FOCUS, confidence=0.95)
        result = gate.evaluate(intent)
        assert result.passed is True
        assert result.force_dry_run is False
        assert "passed" in result.reason

    def test_new_intent_types_evaluated(self):
        gate = ConfidenceGate(min_confidence=0.70, dry_run_below=0.85)
        for intent_type in (IntentType.OBSERVE, IntentType.NETWORK, IntentType.STORAGE):
            result = gate.evaluate(_make_intent(intent_type, 0.90))
            assert result.passed is True
            assert result.force_dry_run is False

    def test_confidence_zero_rejected(self):
        gate = ConfidenceGate()
        result = gate.evaluate(_make_intent(IntentType.FOCUS, confidence=0.0))
        assert result.passed is False

    def test_confidence_one_approved(self):
        gate = ConfidenceGate()
        result = gate.evaluate(_make_intent(IntentType.FOCUS, confidence=1.0))
        assert result.passed is True
        assert result.force_dry_run is False


class TestConfidenceDecisionDataclass:
    def test_frozen(self):
        cd = ConfidenceDecision(passed=True, force_dry_run=False, reason="ok")
        with pytest.raises((AttributeError, TypeError)):
            cd.passed = False  # type: ignore[misc]

    def test_defaults(self):
        cd = ConfidenceDecision(passed=True)
        assert cd.force_dry_run is False
        assert cd.reason == ""
