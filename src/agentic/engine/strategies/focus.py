"""FOCUS strategy — suspend or kill distracting processes."""

from __future__ import annotations

from agentic.engine.strategies.base import IntentStrategy
from agentic.models.action import ActionCandidate, ActionType
from agentic.models.intent import ParsedIntent

DEFAULT_DISTRACTIONS = [
    "firefox",
    "chrome",
    "chromium",
    "slack",
    "discord",
    "spotify",
    "telegram",
    "signal",
]


class FocusStrategy(IntentStrategy):
    async def generate_actions(self, intent: ParsedIntent) -> list[ActionCandidate]:
        targets = [
            e.value for e in intent.entities if e.name == "process"
        ]
        if not targets:
            targets = DEFAULT_DISTRACTIONS

        actions: list[ActionCandidate] = []
        for target in targets:
            actions.append(
                ActionCandidate(
                    action_type=ActionType.SUSPEND_PROCESS,
                    description=f"Suspend process: {target}",
                    command=f"kill -STOP $(pgrep -f {target})",
                    target=target,
                    rollback_command=f"kill -CONT $(pgrep -f {target})",
                )
            )
        return actions
