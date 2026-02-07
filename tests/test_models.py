"""Brutal tests for all Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic.models.action import ActionCandidate, ActionPlan, ActionResult, ActionType
from agentic.models.intent import Entity, IntentType, ParsedIntent
from agentic.models.policy import PolicyDecision, RiskLevel


class TestIntentType:
    def test_all_members(self):
        assert IntentType.FOCUS == "FOCUS"
        assert IntentType.UPDATE == "UPDATE"
        assert IntentType.CLEAN_MEMORY == "CLEAN_MEMORY"
        assert IntentType.UNKNOWN == "UNKNOWN"

    def test_from_value(self):
        assert IntentType("FOCUS") == IntentType.FOCUS

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            IntentType("INVALID")


class TestEntity:
    def test_basic(self):
        e = Entity(name="process", value="firefox")
        assert e.name == "process"
        assert e.source == ""

    def test_with_source(self):
        e = Entity(name="package", value="vim", source="install vim")
        assert e.source == "install vim"


class TestParsedIntent:
    def test_defaults(self):
        intent = ParsedIntent(
            raw_query="test",
            intent_type=IntentType.FOCUS,
            confidence=0.9,
        )
        assert intent.id is not None
        assert len(intent.id) == 32
        assert intent.entities == []
        assert intent.reasoning == ""
        assert intent.created_at is not None

    def test_confidence_boundary_zero(self):
        intent = ParsedIntent(
            raw_query="test",
            intent_type=IntentType.UNKNOWN,
            confidence=0.0,
        )
        assert intent.confidence == 0.0

    def test_confidence_boundary_one(self):
        intent = ParsedIntent(
            raw_query="test",
            intent_type=IntentType.FOCUS,
            confidence=1.0,
        )
        assert intent.confidence == 1.0

    def test_confidence_below_zero_invalid(self):
        with pytest.raises(ValidationError):
            ParsedIntent(
                raw_query="test",
                intent_type=IntentType.FOCUS,
                confidence=-0.1,
            )

    def test_confidence_above_one_invalid(self):
        with pytest.raises(ValidationError):
            ParsedIntent(
                raw_query="test",
                intent_type=IntentType.FOCUS,
                confidence=1.1,
            )

    def test_unique_ids(self):
        i1 = ParsedIntent(raw_query="a", intent_type=IntentType.FOCUS, confidence=0.5)
        i2 = ParsedIntent(raw_query="b", intent_type=IntentType.FOCUS, confidence=0.5)
        assert i1.id != i2.id


class TestActionType:
    def test_all_members_exist(self):
        expected = {
            "KILL_PROCESS", "SUSPEND_PROCESS", "RENICE_PROCESS",
            "APT_INSTALL", "APT_UPGRADE",
            "DROP_CACHES", "KILL_BY_MEMORY",
            "SYSTEMCTL_START", "SYSTEMCTL_STOP", "SYSTEMCTL_RESTART",
        }
        actual = {m.value for m in ActionType}
        assert actual == expected


class TestActionCandidate:
    def test_defaults(self):
        a = ActionCandidate(
            action_type=ActionType.KILL_PROCESS,
            description="Kill chrome",
        )
        assert a.id is not None
        assert a.command == ""
        assert a.target == ""
        assert a.parameters == {}
        assert a.rollback_command == ""

    def test_uuid_generation(self):
        a1 = ActionCandidate(action_type=ActionType.KILL_PROCESS, description="a")
        a2 = ActionCandidate(action_type=ActionType.KILL_PROCESS, description="b")
        assert a1.id != a2.id


class TestActionPlan:
    def test_defaults(self):
        p = ActionPlan(intent_id="i1")
        assert p.actions == []
        assert p.reasoning == ""
        assert p.created_at is not None

    def test_with_actions(self):
        a = ActionCandidate(action_type=ActionType.KILL_PROCESS, description="Kill")
        p = ActionPlan(intent_id="i1", actions=[a])
        assert len(p.actions) == 1


class TestActionResult:
    def test_success(self):
        r = ActionResult(action_id="a1", success=True, output="done")
        assert r.rolled_back is False
        assert r.error == ""

    def test_failure(self):
        r = ActionResult(action_id="a1", success=False, error="failed")
        assert r.output == ""


class TestRiskLevel:
    def test_values(self):
        assert RiskLevel.SAFE.value == 1
        assert RiskLevel.LOW.value == 2
        assert RiskLevel.MEDIUM.value == 3
        assert RiskLevel.HIGH.value == 4
        assert RiskLevel.CRITICAL.value == 5

    def test_ordering(self):
        assert RiskLevel.SAFE < RiskLevel.LOW < RiskLevel.MEDIUM < RiskLevel.HIGH < RiskLevel.CRITICAL


class TestPolicyDecision:
    def test_defaults(self):
        d = PolicyDecision(
            action_id="a1",
            risk_level=RiskLevel.LOW,
            approved=True,
        )
        assert d.requires_sudo is False
        assert d.reason == ""
        assert d.requires_confirmation is False

    def test_all_fields(self):
        d = PolicyDecision(
            action_id="a1",
            risk_level=RiskLevel.HIGH,
            approved=False,
            requires_sudo=True,
            reason="Too risky",
            requires_confirmation=True,
        )
        assert d.requires_sudo is True
        assert d.reason == "Too risky"
