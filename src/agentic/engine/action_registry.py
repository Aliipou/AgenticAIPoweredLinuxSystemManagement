"""Maps IntentType to its corresponding strategy."""

from __future__ import annotations

from agentic.engine.strategies.base import IntentStrategy
from agentic.engine.strategies.clean_memory import CleanMemoryStrategy
from agentic.engine.strategies.focus import FocusStrategy
from agentic.engine.strategies.update import UpdateStrategy
from agentic.models.intent import IntentType


class ActionRegistry:
    def __init__(self) -> None:
        self._strategies: dict[IntentType, IntentStrategy] = {
            IntentType.FOCUS: FocusStrategy(),
            IntentType.UPDATE: UpdateStrategy(),
            IntentType.CLEAN_MEMORY: CleanMemoryStrategy(),
        }

    def get(self, intent_type: IntentType) -> IntentStrategy | None:
        return self._strategies.get(intent_type)

    def register(self, intent_type: IntentType, strategy: IntentStrategy) -> None:
        self._strategies[intent_type] = strategy
