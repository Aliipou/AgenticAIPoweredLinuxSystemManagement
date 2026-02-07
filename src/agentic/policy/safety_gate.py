"""Safety gate — evaluates and filters action plans by risk."""

from __future__ import annotations

from agentic.models.action import ActionCandidate, ActionPlan
from agentic.models.policy import PolicyDecision, RiskLevel
from agentic.policy.permissions import PERMISSION_MATRIX
from agentic.policy.risk_levels import is_above_threshold, requires_user_confirmation, risk_from_string


class SafetyGate:
    def __init__(self, max_risk_level: str = "HIGH", force: bool = False) -> None:
        self._max_risk = risk_from_string(max_risk_level)
        self._force = force

    def evaluate(self, action: ActionCandidate) -> PolicyDecision:
        risk_level, requires_sudo = PERMISSION_MATRIX.get(
            action.action_type, (RiskLevel.MEDIUM, False)
        )

        if risk_level == RiskLevel.CRITICAL and not self._force:
            return PolicyDecision(
                action_id=action.id,
                risk_level=risk_level,
                approved=False,
                requires_sudo=requires_sudo,
                reason=f"CRITICAL risk action blocked: {action.action_type.value}. Use --force to override.",
                requires_confirmation=False,
            )

        if is_above_threshold(risk_level, self._max_risk):
            return PolicyDecision(
                action_id=action.id,
                risk_level=risk_level,
                approved=False,
                requires_sudo=requires_sudo,
                reason=f"Risk level {risk_level.name} exceeds maximum allowed {self._max_risk.name}.",
                requires_confirmation=False,
            )

        needs_confirmation = requires_user_confirmation(risk_level) and not self._force

        return PolicyDecision(
            action_id=action.id,
            risk_level=risk_level,
            approved=True,
            requires_sudo=requires_sudo,
            reason=f"Action approved at {risk_level.name} risk.",
            requires_confirmation=needs_confirmation,
        )

    def evaluate_plan(self, plan: ActionPlan) -> list[PolicyDecision]:
        return [self.evaluate(action) for action in plan.actions]

    def filter_approved(
        self, plan: ActionPlan, decisions: list[PolicyDecision]
    ) -> tuple[list[ActionCandidate], list[PolicyDecision]]:
        approved_actions: list[ActionCandidate] = []
        approved_decisions: list[PolicyDecision] = []

        decision_map = {d.action_id: d for d in decisions}
        for action in plan.actions:
            decision = decision_map.get(action.id)
            if decision and decision.approved:
                approved_actions.append(action)
                approved_decisions.append(decision)

        return approved_actions, approved_decisions
