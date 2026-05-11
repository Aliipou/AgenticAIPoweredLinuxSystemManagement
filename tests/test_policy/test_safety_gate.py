"""Brutal tests for safety gate, permissions, and risk levels."""

from __future__ import annotations

import pytest

from agentic.models.action import ActionCandidate, ActionPlan, ActionType, RollbackSupport
from agentic.models.policy import PolicyDecision, RiskLevel
from agentic.policy.permissions import CRITICAL_SERVICES, PERMISSION_MATRIX, ROLLBACK_CAPABILITIES
from agentic.policy.risk_levels import is_above_threshold, requires_user_confirmation, risk_from_string
from agentic.policy.safety_gate import SafetyGate


class TestRiskLevels:
    def test_risk_from_string_all_levels(self):
        assert risk_from_string("SAFE") == RiskLevel.SAFE
        assert risk_from_string("LOW") == RiskLevel.LOW
        assert risk_from_string("MEDIUM") == RiskLevel.MEDIUM
        assert risk_from_string("HIGH") == RiskLevel.HIGH
        assert risk_from_string("CRITICAL") == RiskLevel.CRITICAL

    def test_risk_from_string_case_insensitive(self):
        assert risk_from_string("safe") == RiskLevel.SAFE
        assert risk_from_string("Safe") == RiskLevel.SAFE
        assert risk_from_string("SAFE") == RiskLevel.SAFE

    def test_risk_from_string_invalid(self):
        with pytest.raises(ValueError, match="Unknown risk level"):
            risk_from_string("EXTREME")

    def test_is_above_threshold(self):
        assert is_above_threshold(RiskLevel.HIGH, RiskLevel.MEDIUM) is True
        assert is_above_threshold(RiskLevel.MEDIUM, RiskLevel.HIGH) is False
        assert is_above_threshold(RiskLevel.HIGH, RiskLevel.HIGH) is False
        assert is_above_threshold(RiskLevel.CRITICAL, RiskLevel.HIGH) is True

    def test_requires_user_confirmation(self):
        assert requires_user_confirmation(RiskLevel.SAFE) is False
        assert requires_user_confirmation(RiskLevel.LOW) is False
        assert requires_user_confirmation(RiskLevel.MEDIUM) is True
        assert requires_user_confirmation(RiskLevel.HIGH) is True
        assert requires_user_confirmation(RiskLevel.CRITICAL) is True


class TestPermissions:
    def test_all_action_types_in_matrix(self):
        for action_type in ActionType:
            assert action_type in PERMISSION_MATRIX, f"{action_type} missing from matrix"

    def test_matrix_values_are_tuples(self):
        for action_type, value in PERMISSION_MATRIX.items():
            assert isinstance(value, tuple)
            assert len(value) == 2
            assert isinstance(value[0], RiskLevel)
            assert isinstance(value[1], bool)

    def test_suspend_is_low_risk(self):
        risk, sudo = PERMISSION_MATRIX[ActionType.SUSPEND_PROCESS]
        assert risk == RiskLevel.LOW
        assert sudo is False

    def test_apt_upgrade_is_high_risk(self):
        risk, sudo = PERMISSION_MATRIX[ActionType.APT_UPGRADE]
        assert risk == RiskLevel.HIGH
        assert sudo is True

    def test_drop_caches_requires_sudo(self):
        _, sudo = PERMISSION_MATRIX[ActionType.DROP_CACHES]
        assert sudo is True

    def test_kill_by_memory_is_high_risk(self):
        risk, _ = PERMISSION_MATRIX[ActionType.KILL_BY_MEMORY]
        assert risk == RiskLevel.HIGH


class TestSafetyGate:
    def test_evaluate_low_risk_approved(self):
        gate = SafetyGate(max_risk_level="HIGH")
        action = ActionCandidate(
            action_type=ActionType.SUSPEND_PROCESS,
            description="Suspend firefox",
            target="firefox",
        )
        decision = gate.evaluate(action)
        assert decision.approved is True
        assert decision.risk_level == RiskLevel.LOW

    def test_evaluate_high_risk_within_threshold(self):
        gate = SafetyGate(max_risk_level="HIGH")
        action = ActionCandidate(
            action_type=ActionType.APT_UPGRADE,
            description="Upgrade all",
            target="system",
        )
        decision = gate.evaluate(action)
        assert decision.approved is True
        assert decision.requires_sudo is True

    def test_evaluate_exceeds_max_risk(self):
        gate = SafetyGate(max_risk_level="LOW")
        action = ActionCandidate(
            action_type=ActionType.KILL_BY_MEMORY,
            description="Kill memory hog",
            target="chrome",
        )
        decision = gate.evaluate(action)
        assert decision.approved is False
        assert "exceeds" in decision.reason

    def test_evaluate_medium_requires_confirmation(self):
        gate = SafetyGate(max_risk_level="HIGH")
        action = ActionCandidate(
            action_type=ActionType.KILL_PROCESS,
            description="Kill process",
            target="chrome",
        )
        decision = gate.evaluate(action)
        assert decision.requires_confirmation is True

    def test_force_skips_confirmation(self):
        gate = SafetyGate(max_risk_level="HIGH", force=True)
        action = ActionCandidate(
            action_type=ActionType.APT_UPGRADE,
            description="Upgrade",
            target="system",
        )
        decision = gate.evaluate(action)
        assert decision.approved is True
        assert decision.requires_confirmation is False

    def test_evaluate_plan_returns_list(self, sample_action_plan):
        gate = SafetyGate(max_risk_level="HIGH")
        decisions = gate.evaluate_plan(sample_action_plan)
        assert len(decisions) == len(sample_action_plan.actions)
        for d in decisions:
            assert isinstance(d, PolicyDecision)

    def test_filter_approved_returns_only_approved(self):
        gate = SafetyGate(max_risk_level="LOW")
        action_low = ActionCandidate(
            id="a1",
            action_type=ActionType.SUSPEND_PROCESS,
            description="Suspend",
            target="firefox",
        )
        action_high = ActionCandidate(
            id="a2",
            action_type=ActionType.KILL_BY_MEMORY,
            description="Kill",
            target="chrome",
        )
        plan = ActionPlan(
            intent_id="i1",
            actions=[action_low, action_high],
        )
        decisions = gate.evaluate_plan(plan)
        approved_actions, approved_decisions = gate.filter_approved(plan, decisions)
        assert len(approved_actions) == 1
        assert approved_actions[0].id == "a1"

    def test_filter_approved_all_denied(self):
        gate = SafetyGate(max_risk_level="SAFE")
        action = ActionCandidate(
            action_type=ActionType.KILL_BY_MEMORY,
            description="Kill",
            target="chrome",
        )
        plan = ActionPlan(intent_id="i1", actions=[action])
        decisions = gate.evaluate_plan(plan)
        approved_actions, approved_decisions = gate.filter_approved(plan, decisions)
        assert len(approved_actions) == 0
        assert len(approved_decisions) == 0

    def test_filter_approved_empty_plan(self):
        gate = SafetyGate(max_risk_level="HIGH")
        plan = ActionPlan(intent_id="i1", actions=[])
        decisions = gate.evaluate_plan(plan)
        approved_actions, approved_decisions = gate.filter_approved(plan, decisions)
        assert len(approved_actions) == 0

    def test_evaluate_unknown_action_type_defaults_to_medium(self):
        """Test fallback for action types not in the permission matrix."""
        gate = SafetyGate(max_risk_level="HIGH")
        action = ActionCandidate(
            action_type=ActionType.RENICE_PROCESS,
            description="Renice",
            target="proc",
        )
        decision = gate.evaluate(action)
        # RENICE_PROCESS is in the matrix (LOW), so this should work
        assert decision.approved is True

    def test_critical_risk_blocked_without_force(self):
        """CRITICAL actions are blocked unless --force is used."""
        gate = SafetyGate(max_risk_level="CRITICAL")
        action = ActionCandidate(
            id="crit-1",
            action_type=ActionType.KILL_PROCESS,
            description="Kill critical",
            target="test",
        )
        # Temporarily patch the permission matrix to make this CRITICAL
        from agentic.policy import permissions
        original = permissions.PERMISSION_MATRIX[ActionType.KILL_PROCESS]
        permissions.PERMISSION_MATRIX[ActionType.KILL_PROCESS] = (RiskLevel.CRITICAL, True)
        try:
            decision = gate.evaluate(action)
            assert decision.approved is False
            assert "CRITICAL" in decision.reason
            assert "--force" in decision.reason
        finally:
            permissions.PERMISSION_MATRIX[ActionType.KILL_PROCESS] = original

    def test_critical_risk_allowed_with_force(self):
        gate = SafetyGate(max_risk_level="CRITICAL", force=True)
        action = ActionCandidate(
            id="crit-2",
            action_type=ActionType.KILL_PROCESS,
            description="Kill critical",
            target="test",
        )
        from agentic.policy import permissions
        original = permissions.PERMISSION_MATRIX[ActionType.KILL_PROCESS]
        permissions.PERMISSION_MATRIX[ActionType.KILL_PROCESS] = (RiskLevel.CRITICAL, True)
        try:
            decision = gate.evaluate(action)
            assert decision.approved is True
        finally:
            permissions.PERMISSION_MATRIX[ActionType.KILL_PROCESS] = original


class TestRollbackCapabilities:
    def test_all_action_types_present(self):
        from agentic.policy.permissions import ROLLBACK_CAPABILITIES
        for at in ActionType:
            assert at in ROLLBACK_CAPABILITIES, f"{at} missing from ROLLBACK_CAPABILITIES"

    def test_values_are_rollback_support_instances(self):
        from agentic.policy.permissions import ROLLBACK_CAPABILITIES
        for at, rs in ROLLBACK_CAPABILITIES.items():
            assert isinstance(rs, RollbackSupport), f"{at} has non-RollbackSupport value"

    def test_kill_process_is_none(self):
        assert ROLLBACK_CAPABILITIES[ActionType.KILL_PROCESS] == RollbackSupport.NONE

    def test_kill_by_memory_is_none(self):
        assert ROLLBACK_CAPABILITIES[ActionType.KILL_BY_MEMORY] == RollbackSupport.NONE

    def test_apt_upgrade_is_none(self):
        assert ROLLBACK_CAPABILITIES[ActionType.APT_UPGRADE] == RollbackSupport.NONE

    def test_suspend_process_is_full(self):
        assert ROLLBACK_CAPABILITIES[ActionType.SUSPEND_PROCESS] == RollbackSupport.FULL

    def test_renice_process_is_full(self):
        assert ROLLBACK_CAPABILITIES[ActionType.RENICE_PROCESS] == RollbackSupport.FULL

    def test_drop_caches_is_full(self):
        assert ROLLBACK_CAPABILITIES[ActionType.DROP_CACHES] == RollbackSupport.FULL

    def test_systemctl_restart_is_partial(self):
        assert ROLLBACK_CAPABILITIES[ActionType.SYSTEMCTL_RESTART] == RollbackSupport.PARTIAL

    def test_apt_install_is_partial(self):
        assert ROLLBACK_CAPABILITIES[ActionType.APT_INSTALL] == RollbackSupport.PARTIAL


class TestCriticalServices:
    def test_critical_services_frozenset_not_empty(self):
        assert len(CRITICAL_SERVICES) > 0
        assert "postgresql" in CRITICAL_SERVICES
        assert "nginx" in CRITICAL_SERVICES
        assert "docker" in CRITICAL_SERVICES

    def test_systemctl_stop_postgresql_escalates_to_critical(self):
        gate = SafetyGate(max_risk_level="CRITICAL")
        action = ActionCandidate(
            action_type=ActionType.SYSTEMCTL_STOP,
            description="Stop postgres",
            target="postgresql",
        )
        decision = gate.evaluate(action)
        assert decision.risk_level == RiskLevel.CRITICAL
        assert decision.approved is False
        assert "--force" in decision.reason

    def test_systemctl_restart_nginx_escalates_to_critical(self):
        gate = SafetyGate(max_risk_level="CRITICAL")
        action = ActionCandidate(
            action_type=ActionType.SYSTEMCTL_RESTART,
            description="Restart nginx",
            target="nginx",
        )
        decision = gate.evaluate(action)
        assert decision.risk_level == RiskLevel.CRITICAL
        assert decision.approved is False

    def test_systemctl_stop_critical_service_allowed_with_force(self):
        gate = SafetyGate(max_risk_level="CRITICAL", force=True)
        action = ActionCandidate(
            action_type=ActionType.SYSTEMCTL_STOP,
            description="Stop postgres",
            target="postgresql",
        )
        decision = gate.evaluate(action)
        assert decision.risk_level == RiskLevel.CRITICAL
        assert decision.approved is True

    def test_systemctl_stop_non_critical_service_stays_high(self):
        gate = SafetyGate(max_risk_level="HIGH")
        action = ActionCandidate(
            action_type=ActionType.SYSTEMCTL_STOP,
            description="Stop cups",
            target="cups",
        )
        decision = gate.evaluate(action)
        assert decision.risk_level == RiskLevel.HIGH
        assert decision.approved is True

    def test_systemctl_start_critical_service_not_escalated(self):
        # Only STOP and RESTART escalate; START does not
        gate = SafetyGate(max_risk_level="HIGH")
        action = ActionCandidate(
            action_type=ActionType.SYSTEMCTL_START,
            description="Start postgresql",
            target="postgresql",
        )
        decision = gate.evaluate(action)
        assert decision.risk_level == RiskLevel.MEDIUM
        assert decision.approved is True

    def test_critical_service_target_case_insensitive(self):
        gate = SafetyGate(max_risk_level="CRITICAL")
        action = ActionCandidate(
            action_type=ActionType.SYSTEMCTL_STOP,
            description="Stop docker",
            target="Docker",
        )
        decision = gate.evaluate(action)
        assert decision.risk_level == RiskLevel.CRITICAL
        assert decision.approved is False
