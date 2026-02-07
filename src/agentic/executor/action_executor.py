"""Action executor — dispatches actions to the appropriate runner."""

from __future__ import annotations

from agentic.exceptions import ExecutionError
from agentic.executor.runners.base import BaseRunner
from agentic.executor.runners.memory_runner import MemoryRunner
from agentic.executor.runners.package_runner import PackageRunner
from agentic.executor.runners.process_runner import ProcessRunner
from agentic.executor.runners.systemctl_runner import SystemctlRunner
from agentic.models.action import ActionCandidate, ActionResult, ActionType

_RUNNER_MAP: dict[ActionType, type[BaseRunner]] = {
    ActionType.KILL_PROCESS: ProcessRunner,
    ActionType.SUSPEND_PROCESS: ProcessRunner,
    ActionType.RENICE_PROCESS: ProcessRunner,
    ActionType.APT_INSTALL: PackageRunner,
    ActionType.APT_UPGRADE: PackageRunner,
    ActionType.DROP_CACHES: MemoryRunner,
    ActionType.KILL_BY_MEMORY: MemoryRunner,
    ActionType.SYSTEMCTL_START: SystemctlRunner,
    ActionType.SYSTEMCTL_STOP: SystemctlRunner,
    ActionType.SYSTEMCTL_RESTART: SystemctlRunner,
}


class ActionExecutor:
    def __init__(self) -> None:
        self._runners: dict[ActionType, BaseRunner] = {}

    def _get_runner(self, action_type: ActionType) -> BaseRunner:
        if action_type not in self._runners:
            runner_cls = _RUNNER_MAP.get(action_type)
            if runner_cls is None:
                raise ExecutionError(
                    f"No runner registered for {action_type.value}",
                    action_id="",
                )
            self._runners[action_type] = runner_cls()
        return self._runners[action_type]

    async def execute(
        self, action: ActionCandidate, dry_run: bool = False
    ) -> ActionResult:
        runner = self._get_runner(action.action_type)
        return await runner.run(action, dry_run=dry_run)

    async def execute_many(
        self, actions: list[ActionCandidate], dry_run: bool = False
    ) -> list[ActionResult]:
        results: list[ActionResult] = []
        for action in actions:
            result = await self.execute(action, dry_run=dry_run)
            results.append(result)
        return results

    async def rollback(self, action: ActionCandidate) -> ActionResult:
        runner = self._get_runner(action.action_type)
        return await runner.rollback(action)
