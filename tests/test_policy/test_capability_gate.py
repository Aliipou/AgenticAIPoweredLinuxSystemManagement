"""Tests for CapabilityGate and ACTION_CAPABILITIES."""

from __future__ import annotations

import pytest

from agentic.models.action import ActionCandidate, ActionPlan, ActionType
from agentic.models.capability import Capability
from agentic.models.policy import RiskLevel
from agentic.policy.capability_gate import ACTION_CAPABILITIES, CapabilityGate


class TestCapability:
    def test_all_members(self):
        expected = {
            "KILL_PROCESS", "SUSPEND_PROCESS", "RENICE_PROCESS",
            "PACKAGE_MANAGEMENT", "MEMORY_MANAGEMENT", "SERVICE_MANAGEMENT",
        }
        assert {c.value for c in Capability} == expected

    def test_from_value(self):
        assert Capability("SERVICE_MANAGEMENT") == Capability.SERVICE_MANAGEMENT


class TestActionCapabilities:
    def test_all_action_types_mapped(self):
        for action_type in ActionType:
            assert action_type in ACTION_CAPABILITIES, f"{action_type} missing from ACTION_CAPABILITIES"

    def test_kill_process_requires_kill(self):
        assert ACTION_CAPABILITIES[ActionType.KILL_PROCESS] == Capability.KILL_PROCESS

    def test_kill_by_memory_requires_kill(self):
        assert ACTION_CAPABILITIES[ActionType.KILL_BY_MEMORY] == Capability.KILL_PROCESS

    def test_apt_install_requires_package(self):
        assert ACTION_CAPABILITIES[ActionType.APT_INSTALL] == Capability.PACKAGE_MANAGEMENT

    def test_apt_upgrade_requires_package(self):
        assert ACTION_CAPABILITIES[ActionType.APT_UPGRADE] == Capability.PACKAGE_MANAGEMENT

    def test_systemctl_actions_require_service(self):
        for at in (ActionType.SYSTEMCTL_START, ActionType.SYSTEMCTL_STOP, ActionType.SYSTEMCTL_RESTART):
            assert ACTION_CAPABILITIES[at] == Capability.SERVICE_MANAGEMENT


class TestCapabilityGateEvaluate:
    def test_granted_capability_returns_none(self):
        gate = CapabilityGate(frozenset({Capability.KILL_PROCESS}))
        action = ActionCandidate(action_type=ActionType.KILL_PROCESS, description="kill")
        assert gate.evaluate(action) is None

    def test_missing_capability_returns_decision(self):
        gate = CapabilityGate(frozenset())  # no capabilities granted
        action = ActionCandidate(action_type=ActionType.KILL_PROCESS, description="kill")
        decision = gate.evaluate(action)
        assert decision is not None
        assert decision.approved is False
        assert "KILL_PROCESS" in decision.reason

    def test_denied_decision_has_correct_risk(self):
        gate = CapabilityGate(frozenset())
        action = ActionCandidate(action_type=ActionType.APT_UPGRADE, description="upgrade")
        decision = gate.evaluate(action)
        assert decision is not None
        assert decision.risk_level == RiskLevel.HIGH

    def test_unknown_action_type_not_in_map_permitted(self, monkeypatch):
        from agentic.policy import capability_gate as cg
        original = cg.ACTION_CAPABILITIES.copy()
        cg.ACTION_CAPABILITIES.clear()
        try:
            gate = CapabilityGate(frozenset())
            action = ActionCandidate(action_type=ActionType.KILL_PROCESS, description="kill")
            assert gate.evaluate(action) is None
        finally:
            cg.ACTION_CAPABILITIES.update(original)

    def test_all_capabilities_gate_permits_all(self):
        gate = CapabilityGate.all_capabilities()
        for action_type in ActionType:
            action = ActionCandidate(action_type=action_type, description="test")
            assert gate.evaluate(action) is None

    def test_granted_property(self):
        caps = frozenset({Capability.SERVICE_MANAGEMENT})
        gate = CapabilityGate(caps)
        assert gate.granted == caps


class TestCapabilityGateFilterApproved:
    def test_empty_plan(self):
        gate = CapabilityGate.all_capabilities()
        plan = ActionPlan(intent_id="i1", actions=[])
        permitted, denied = gate.filter_approved(plan)
        assert permitted == []
        assert denied == []

    def test_all_permitted(self):
        gate = CapabilityGate.all_capabilities()
        actions = [
            ActionCandidate(id="a1", action_type=ActionType.KILL_PROCESS, description="k"),
            ActionCandidate(id="a2", action_type=ActionType.SUSPEND_PROCESS, description="s"),
        ]
        plan = ActionPlan(intent_id="i1", actions=actions)
        permitted, denied = gate.filter_approved(plan)
        assert len(permitted) == 2
        assert len(denied) == 0

    def test_all_denied(self):
        gate = CapabilityGate(frozenset())
        actions = [
            ActionCandidate(id="a1", action_type=ActionType.KILL_PROCESS, description="k"),
            ActionCandidate(id="a2", action_type=ActionType.APT_UPGRADE, description="u"),
        ]
        plan = ActionPlan(intent_id="i1", actions=actions)
        permitted, denied = gate.filter_approved(plan)
        assert len(permitted) == 0
        assert len(denied) == 2

    def test_mixed(self):
        gate = CapabilityGate(frozenset({Capability.SUSPEND_PROCESS}))
        safe = ActionCandidate(id="s1", action_type=ActionType.SUSPEND_PROCESS, description="safe")
        risky = ActionCandidate(id="r1", action_type=ActionType.APT_UPGRADE, description="risky")
        plan = ActionPlan(intent_id="i1", actions=[safe, risky])
        permitted, denied = gate.filter_approved(plan)
        assert len(permitted) == 1
        assert permitted[0].id == "s1"
        assert len(denied) == 1
        assert denied[0].action_id == "r1"
