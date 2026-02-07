"""Abstract base for action runners."""

from __future__ import annotations

import abc

from agentic.models.action import ActionCandidate, ActionResult


class BaseRunner(abc.ABC):
    @abc.abstractmethod
    async def run(self, action: ActionCandidate, dry_run: bool = False) -> ActionResult:
        ...  # pragma: no cover

    @abc.abstractmethod
    async def rollback(self, action: ActionCandidate) -> ActionResult:
        ...  # pragma: no cover
