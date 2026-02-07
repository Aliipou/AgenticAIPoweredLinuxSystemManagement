"""UPDATE strategy — install or upgrade system packages."""

from __future__ import annotations

from agentic.engine.strategies.base import IntentStrategy
from agentic.models.action import ActionCandidate, ActionType
from agentic.models.intent import ParsedIntent


class UpdateStrategy(IntentStrategy):
    async def generate_actions(self, intent: ParsedIntent) -> list[ActionCandidate]:
        packages = [e.value for e in intent.entities if e.name == "package"]

        actions: list[ActionCandidate] = []
        if packages:
            for pkg in packages:
                actions.append(
                    ActionCandidate(
                        action_type=ActionType.APT_INSTALL,
                        description=f"Install package: {pkg}",
                        command=f"apt install -y {pkg}",
                        target=pkg,
                        rollback_command=f"apt remove -y {pkg}",
                    )
                )
        else:
            actions.append(
                ActionCandidate(
                    action_type=ActionType.APT_UPGRADE,
                    description="Upgrade all system packages",
                    command="apt update && apt upgrade -y",
                    target="system",
                    rollback_command="",
                )
            )
        return actions
