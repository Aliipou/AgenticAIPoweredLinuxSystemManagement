"""Environment gate — caps maximum allowed risk by deployment context."""

from __future__ import annotations

from agentic.models.action import ActionCandidate, ActionPlan
from agentic.models.environment import Environment
from agentic.models.policy import PolicyDecision, RiskLevel
from agentic.policy.permissions import ENVIRONMENT_RISK_CAPS, PERMISSION_MATRIX
from agentic.policy.risk_levels import is_above_threshold, risk_from_string


class EnvironmentGate:
    """Applies a hard risk ceiling based on deployment environment.

    PRODUCTION → max MEDIUM (APT_UPGRADE, SYSTEMCTL_STOP, etc. are blocked)
    STAGING    → max HIGH   (same as default SafetyGate)
    DEVELOPMENT → max CRITICAL (unrestricted)
    """

    def __init__(self, environment: Environment) -> None:
        self._env = environment
        self._cap: RiskLevel = risk_from_string(ENVIRONMENT_RISK_CAPS[environment])

    @property
    def environment(self) -> Environment:
        return self._env

    @property
    def cap(self) -> RiskLevel:
        return self._cap

    def evaluate(self, action: ActionCandidate) -> PolicyDecision | None:
        """Return a denied PolicyDecision if the action exceeds the env cap, else None."""
        risk_level, requires_sudo = PERMISSION_MATRIX.get(
            action.action_type, (RiskLevel.MEDIUM, False)
        )
        if is_above_threshold(risk_level, self._cap):
            return PolicyDecision(
                action_id=action.id,
                risk_level=risk_level,
                approved=False,
                requires_sudo=requires_sudo,
                reason=(
                    f"{self._env.value} environment cap is {self._cap.name}; "
                    f"{risk_level.name} risk actions require explicit override."
                ),
                requires_confirmation=False,
            )
        return None

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
