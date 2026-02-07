"""Abstract base for intent strategies."""

from __future__ import annotations

import abc

from agentic.models.action import ActionCandidate
from agentic.models.intent import ParsedIntent


class IntentStrategy(abc.ABC):
    @abc.abstractmethod
    async def generate_actions(self, intent: ParsedIntent) -> list[ActionCandidate]:
        ...  # pragma: no cover
