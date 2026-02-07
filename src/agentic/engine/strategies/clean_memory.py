"""CLEAN_MEMORY strategy — free RAM and kill memory hogs."""

from __future__ import annotations

from agentic.engine.strategies.base import IntentStrategy
from agentic.models.action import ActionCandidate, ActionType
from agentic.models.intent import ParsedIntent


class CleanMemoryStrategy(IntentStrategy):
    async def generate_actions(self, intent: ParsedIntent) -> list[ActionCandidate]:
        actions: list[ActionCandidate] = []

        # Always drop caches
        actions.append(
            ActionCandidate(
                action_type=ActionType.DROP_CACHES,
                description="Drop filesystem caches",
                command="sync && echo 3 > /proc/sys/vm/drop_caches",
                target="caches",
                rollback_command="",
            )
        )

        # Kill specific memory hogs if mentioned
        targets = [e.value for e in intent.entities if e.name == "process"]
        if targets:
            for target in targets:
                actions.append(
                    ActionCandidate(
                        action_type=ActionType.KILL_BY_MEMORY,
                        description=f"Kill memory-hogging process: {target}",
                        command=f"pkill -f {target}",
                        target=target,
                        rollback_command="",
                    )
                )
        else:
            # Generic: kill top memory consumers
            actions.append(
                ActionCandidate(
                    action_type=ActionType.KILL_BY_MEMORY,
                    description="Kill top memory-consuming processes (>500MB)",
                    command="ps aux --sort=-%mem | head -5",
                    target="memory_hogs",
                    rollback_command="",
                )
            )

        return actions
