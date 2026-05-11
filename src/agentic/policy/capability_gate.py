"""Capability gate — least-privilege enforcement per action type."""

from __future__ import annotations

from agentic.models.action import ActionCandidate, ActionPlan, ActionType
from agentic.models.capability import Capability
from agentic.models.policy import PolicyDecision, RiskLevel
from agentic.policy.permissions import PERMISSION_MATRIX

# Maps each action type to the single capability it requires.
ACTION_CAPABILITIES: dict[ActionType, Capability] = {
    ActionType.KILL_PROCESS: Capability.KILL_PROCESS,
    ActionType.SUSPEND_PROCESS: Capability.SUSPEND_PROCESS,
    ActionType.RENICE_PROCESS: Capability.RENICE_PROCESS,
    ActionType.APT_INSTALL: Capability.PACKAGE_MANAGEMENT,
    ActionType.APT_UPGRADE: Capability.PACKAGE_MANAGEMENT,
    ActionType.DROP_CACHES: Capability.MEMORY_MANAGEMENT,
    ActionType.KILL_BY_MEMORY: Capability.KILL_PROCESS,
    ActionType.SYSTEMCTL_START: Capability.SERVICE_MANAGEMENT,
    ActionType.SYSTEMCTL_STOP: Capability.SERVICE_MANAGEMENT,
    ActionType.SYSTEMCTL_RESTART: Capability.SERVICE_MANAGEMENT,
}


class CapabilityGate:
    """Enforces least-privilege access: each action type requires a declared capability.

    Construct with the frozenset of capabilities this runtime instance has been
    granted. Actions requiring a capability not in the set are denied.
    """

    def __init__(self, granted: frozenset[Capability]) -> None:
        self._granted = granted

    @property
    def granted(self) -> frozenset[Capability]:
        return self._granted

    def evaluate(self, action: ActionCandidate) -> PolicyDecision | None:
        """Return None if action is permitted, or a denied PolicyDecision."""
        required = ACTION_CAPABILITIES.get(action.action_type)
        if required is None or required in self._granted:
            return None
        risk_level, requires_sudo = PERMISSION_MATRIX.get(
            action.action_type, (RiskLevel.MEDIUM, False)
        )
        return PolicyDecision(
            action_id=action.id,
            risk_level=risk_level,
            approved=False,
            requires_sudo=requires_sudo,
            reason=f"Required capability {required.value} not granted.",
            requires_confirmation=False,
        )

    def filter_approved(
        self, plan: ActionPlan
    ) -> tuple[list[ActionCandidate], list[PolicyDecision]]:
        """Return (permitted actions, denied decisions) for a plan."""
        permitted: list[ActionCandidate] = []
        denied: list[PolicyDecision] = []
        for action in plan.actions:
            decision = self.evaluate(action)
            if decision is None:
                permitted.append(action)
            else:
                denied.append(decision)
        return permitted, denied

    @classmethod
    def all_capabilities(cls) -> CapabilityGate:
        """Grants every defined capability — use in development/testing."""
        return cls(frozenset(Capability))
