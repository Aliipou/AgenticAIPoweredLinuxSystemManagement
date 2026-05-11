"""Tests for EnvironmentGate and ENVIRONMENT_RISK_CAPS."""

from __future__ import annotations

import pytest

from agentic.models.action import ActionCandidate, ActionPlan, ActionType
from agentic.models.environment import Environment
from agentic.models.policy import RiskLevel
from agentic.policy.environment_gate import EnvironmentGate
from agentic.policy.permissions import ENVIRONMENT_RISK_CAPS


class TestEnvironmentRiskCaps:
    def test_all_environments_present(self):
        for env in Environment:
            assert env in ENVIRONMENT_RISK_CAPS

    def test_production_is_medium(self):
        assert ENVIRONMENT_RISK_CAPS[Environment.PRODUCTION] == "MEDIUM"

    def test_staging_is_high(self):
        assert ENVIRONMENT_RISK_CAPS[Environment.STAGING] == "HIGH"

    def test_development_is_critical(self):
        assert ENVIRONMENT_RISK_CAPS[Environment.DEVELOPMENT] == "CRITICAL"


class TestEnvironmentGateProperties:
    def test_environment_property(self):
        gate = EnvironmentGate(Environment.PRODUCTION)
        assert gate.environment == Environment.PRODUCTION

    def test_cap_production(self):
        gate = EnvironmentGate(Environment.PRODUCTION)
        assert gate.cap == RiskLevel.MEDIUM

    def test_cap_staging(self):
        gate = EnvironmentGate(Environment.STAGING)
        assert gate.cap == RiskLevel.HIGH

    def test_cap_development(self):
        gate = EnvironmentGate(Environment.DEVELOPMENT)
        assert gate.cap == RiskLevel.CRITICAL


class TestEnvironmentGateEvaluate:
    def test_production_blocks_high_risk(self):
        gate = EnvironmentGate(Environment.PRODUCTION)
        action = ActionCandidate(
            action_type=ActionType.APT_UPGRADE,
            description="Upgrade all",
        )
        decision = gate.evaluate(action)
        assert decision is not None
        assert decision.approved is False
        assert "PRODUCTION" in decision.reason
        assert "HIGH" in decision.reason
        assert "MEDIUM" in decision.reason

    def test_production_permits_low_risk(self):
        gate = EnvironmentGate(Environment.PRODUCTION)
        action = ActionCandidate(
            action_type=ActionType.SUSPEND_PROCESS,
            description="Suspend chrome",
        )
        decision = gate.evaluate(action)
        assert decision is None

    def test_production_permits_medium_risk(self):
        gate = EnvironmentGate(Environment.PRODUCTION)
        action = ActionCandidate(
            action_type=ActionType.KILL_PROCESS,
            description="Kill process",
        )
        decision = gate.evaluate(action)
        assert decision is None

    def test_production_blocks_systemctl_stop(self):
        gate = EnvironmentGate(Environment.PRODUCTION)
        action = ActionCandidate(
            action_type=ActionType.SYSTEMCTL_STOP,
            description="Stop service",
            target="nginx",
        )
        decision = gate.evaluate(action)
        assert decision is not None
        assert decision.approved is False

    def test_staging_blocks_critical_but_permits_high(self):
        gate = EnvironmentGate(Environment.STAGING)
        high_action = ActionCandidate(
            action_type=ActionType.APT_UPGRADE,
            description="Upgrade",
        )
        decision = gate.evaluate(high_action)
        assert decision is None  # HIGH is within STAGING cap (HIGH)

    def test_development_permits_everything(self):
        gate = EnvironmentGate(Environment.DEVELOPMENT)
        for action_type in ActionType:
            action = ActionCandidate(action_type=action_type, description="test")
            decision = gate.evaluate(action)
            assert decision is None, f"Expected None for {action_type} in DEVELOPMENT"

    def test_denied_decision_has_risk_level(self):
        gate = EnvironmentGate(Environment.PRODUCTION)
        action = ActionCandidate(
            action_type=ActionType.KILL_BY_MEMORY,
            description="Kill memory hog",
        )
        decision = gate.evaluate(action)
        assert decision is not None
        assert decision.risk_level == RiskLevel.HIGH


class TestEnvironmentGateFilterApproved:
    def test_empty_plan_returns_empty(self):
        gate = EnvironmentGate(Environment.PRODUCTION)
        plan = ActionPlan(intent_id="i1", actions=[])
        permitted, denied = gate.filter_approved(plan)
        assert permitted == []
        assert denied == []

    def test_all_permitted_in_development(self):
        gate = EnvironmentGate(Environment.DEVELOPMENT)
        actions = [
            ActionCandidate(action_type=ActionType.APT_UPGRADE, description="up"),
            ActionCandidate(action_type=ActionType.SYSTEMCTL_STOP, description="stop"),
        ]
        plan = ActionPlan(intent_id="i1", actions=actions)
        permitted, denied = gate.filter_approved(plan)
        assert len(permitted) == 2
        assert len(denied) == 0

    def test_mixed_in_production(self):
        gate = EnvironmentGate(Environment.PRODUCTION)
        safe = ActionCandidate(
            id="s1", action_type=ActionType.SUSPEND_PROCESS, description="safe"
        )
        risky = ActionCandidate(
            id="r1", action_type=ActionType.APT_UPGRADE, description="risky"
        )
        plan = ActionPlan(intent_id="i1", actions=[safe, risky])
        permitted, denied = gate.filter_approved(plan)
        assert len(permitted) == 1
        assert permitted[0].id == "s1"
        assert len(denied) == 1
        assert denied[0].action_id == "r1"

    def test_all_denied_in_production(self):
        gate = EnvironmentGate(Environment.PRODUCTION)
        actions = [
            ActionCandidate(action_type=ActionType.APT_UPGRADE, description="up"),
            ActionCandidate(action_type=ActionType.KILL_BY_MEMORY, description="kill"),
        ]
        plan = ActionPlan(intent_id="i1", actions=actions)
        permitted, denied = gate.filter_approved(plan)
        assert len(permitted) == 0
        assert len(denied) == 2
