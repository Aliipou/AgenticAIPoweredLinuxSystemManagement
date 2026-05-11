"""Simulation engine — predicts action effects before execution touches the OS."""

from __future__ import annotations

from agentic.models.action import (
    ActionCandidate,
    ActionPlan,
    ActionScope,
    ActionSimulation,
    ActionType,
)
from agentic.policy.permissions import PERMISSION_MATRIX

_ACTION_SCOPES: dict[ActionType, ActionScope] = {
    ActionType.KILL_PROCESS: ActionScope.PROCESS,
    ActionType.SUSPEND_PROCESS: ActionScope.PROCESS,
    ActionType.RENICE_PROCESS: ActionScope.PROCESS,
    ActionType.APT_INSTALL: ActionScope.PACKAGE,
    ActionType.APT_UPGRADE: ActionScope.PACKAGE,
    ActionType.DROP_CACHES: ActionScope.MEMORY,
    ActionType.KILL_BY_MEMORY: ActionScope.PROCESS,
    ActionType.SYSTEMCTL_START: ActionScope.SERVICE,
    ActionType.SYSTEMCTL_STOP: ActionScope.SERVICE,
    ActionType.SYSTEMCTL_RESTART: ActionScope.SERVICE,
}

_HIGH_IMPACT: frozenset[ActionType] = frozenset({
    ActionType.APT_UPGRADE,
    ActionType.KILL_BY_MEMORY,
    ActionType.SYSTEMCTL_STOP,
    ActionType.SYSTEMCTL_RESTART,
})

_AVAILABILITY_IMPACT: frozenset[ActionType] = frozenset({
    ActionType.SYSTEMCTL_STOP,
    ActionType.SYSTEMCTL_RESTART,
})


class SimulationEngine:
    """Predicts the effect of an action or plan without executing anything.

    Uses the action's declared ActionEffect when present; otherwise derives
    predictions from action type via static lookup tables.
    """

    def simulate(self, action: ActionCandidate) -> ActionSimulation:
        if action.effect is not None:
            scope = action.effect.scope
            reversible = action.effect.reversible
            data_loss = action.effect.data_loss_risk
            availability = action.effect.availability_impact
        else:
            scope = _ACTION_SCOPES[action.action_type]
            reversible = action.action_type not in _HIGH_IMPACT
            data_loss = False
            availability = action.action_type in _AVAILABILITY_IMPACT

        _, requires_sudo = PERMISSION_MATRIX.get(action.action_type, (None, False))

        warnings: list[str] = []
        if not reversible:
            warnings.append(f"{action.action_type.value} may not be easily reversible.")
        if data_loss:
            warnings.append("Action carries data loss risk.")
        if availability:
            warnings.append(
                f"Action may affect availability of {action.target or 'the service'}."
            )

        cmd_label = action.command if action.command else action.action_type.value
        target_label = f" on {action.target}" if action.target else ""
        simulated_output = f"[SIMULATED] Would execute: {cmd_label}{target_label}"

        return ActionSimulation(
            action_id=action.id,
            predicted_scope=scope,
            reversible=reversible,
            data_loss_risk=data_loss,
            availability_impact=availability,
            would_require_sudo=requires_sudo,
            simulated_output=simulated_output,
            warnings=warnings,
        )

    def simulate_plan(self, plan: ActionPlan) -> list[ActionSimulation]:
        return [self.simulate(action) for action in plan.actions]
