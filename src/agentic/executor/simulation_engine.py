"""Rule-based effect prediction — estimates action impact before execution.

This is NOT a full runtime simulator. It is a static prediction engine that
derives expected effects from declared ActionEffect metadata or from static
lookup tables keyed on ActionType. Predictions are deterministic but cannot
account for runtime state (running processes, disk usage, service health).

Use predictions for: pre-flight warnings, audit logging, operator visibility.
Do NOT use predictions as: execution guards, security controls, or correctness
proofs. All actual safety enforcement happens in the gate chain before this runs.
"""

from __future__ import annotations

from agentic.models.action import (
    ActionCandidate,
    ActionPlan,
    ActionScope,
    ActionSimulation,
    ActionType,
    RollbackSupport,
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
    """Static effect predictor.

    Uses declared ActionEffect when present. Otherwise derives predictions
    from RollbackSupport declaration and action type lookup tables.
    """

    def simulate(self, action: ActionCandidate) -> ActionSimulation:
        if action.effect is not None:
            scope = action.effect.scope
            reversible = action.effect.reversible
            data_loss = action.effect.data_loss_risk
            availability = action.effect.availability_impact
        else:
            scope = _ACTION_SCOPES[action.action_type]
            data_loss = False
            availability = action.action_type in _AVAILABILITY_IMPACT
            rs = action.rollback_support
            if rs == RollbackSupport.NONE:
                reversible = False
            elif rs in (RollbackSupport.FULL, RollbackSupport.PARTIAL):
                reversible = True
            else:  # UNKNOWN — fall back to static high-impact table
                reversible = action.action_type not in _HIGH_IMPACT

        _, requires_sudo = PERMISSION_MATRIX.get(action.action_type, (None, False))

        warnings: list[str] = []
        if action.effect is None and action.rollback_support == RollbackSupport.PARTIAL:
            warnings.append(
                "Rollback is partial — residual effects may remain after recovery."
            )
        if not reversible:
            warnings.append(f"{action.action_type.value} is not reversible.")
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
