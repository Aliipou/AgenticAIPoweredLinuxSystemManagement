"""Decision engine — converts parsed intents into action plans."""

from __future__ import annotations

from agentic.engine.action_registry import ActionRegistry
from agentic.models.action import ActionPlan
from agentic.models.intent import IntentType, ParsedIntent


class DecisionEngine:
    def __init__(self, registry: ActionRegistry | None = None) -> None:
        self._registry = registry or ActionRegistry()

    async def decide(self, intent: ParsedIntent) -> ActionPlan:
        if intent.intent_type == IntentType.UNKNOWN:
            return ActionPlan(
                intent_id=intent.id,
                actions=[],
                reasoning="Intent is UNKNOWN — no actions generated.",
            )

        strategy = self._registry.get(intent.intent_type)
        if strategy is None:
            return ActionPlan(
                intent_id=intent.id,
                actions=[],
                reasoning=f"No strategy registered for {intent.intent_type.value}.",
            )

        actions = await strategy.generate_actions(intent)
        return ActionPlan(
            intent_id=intent.id,
            actions=actions,
            reasoning=f"Generated {len(actions)} action(s) for {intent.intent_type.value}.",
        )
