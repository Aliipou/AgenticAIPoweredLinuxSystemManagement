"""Transaction manager — sequential execution with automatic rollback on failure."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic.executor.action_executor import ActionExecutor
from agentic.models.action import ActionCandidate, ActionResult, RollbackSupport


@dataclass(frozen=True)
class TransactionResult:
    success: bool
    results: list[ActionResult]
    rolled_back_ids: list[str] = field(default_factory=list)
    rollback_errors: list[str] = field(default_factory=list)


class TransactionManager:
    """Executes actions one-by-one. On the first failure, rolls back all
    previously succeeded actions in reverse order using their rollback_command."""

    async def execute_with_rollback(
        self,
        actions: list[ActionCandidate],
        executor: ActionExecutor,
        dry_run: bool = False,
    ) -> TransactionResult:
        executed: list[ActionCandidate] = []
        results: list[ActionResult] = []

        for action in actions:
            result = await executor.execute(action, dry_run=dry_run)
            results.append(result)
            if not result.success:
                rolled_back_ids, rollback_errors = await self._rollback(executed, executor)
                return TransactionResult(
                    success=False,
                    results=results,
                    rolled_back_ids=rolled_back_ids,
                    rollback_errors=rollback_errors,
                )
            executed.append(action)

        return TransactionResult(success=True, results=results)

    async def _rollback(
        self,
        executed: list[ActionCandidate],
        executor: ActionExecutor,
    ) -> tuple[list[str], list[str]]:
        rolled_back_ids: list[str] = []
        rollback_errors: list[str] = []

        for action in reversed(executed):
            if not action.rollback_command:
                continue
            if action.rollback_support == RollbackSupport.NONE:
                continue
            rb_result = await executor.rollback(action)
            if rb_result.success:
                rolled_back_ids.append(action.id)
            else:
                rollback_errors.append(f"{action.id}: {rb_result.error}")

        return rolled_back_ids, rollback_errors
