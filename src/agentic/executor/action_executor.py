"""Action executor — dispatches actions to the appropriate runner."""

from __future__ import annotations

from agentic.exceptions import ExecutionError
from agentic.executor.runners.base import BaseRunner
from agentic.executor.runners.memory_runner import MemoryRunner
from agentic.executor.runners.package_runner import PackageRunner
from agentic.executor.runners.process_runner import ProcessRunner
from agentic.executor.runners.systemctl_runner import SystemctlRunner
from agentic.executor.sandbox.manager import SandboxManager
from agentic.models.action import ActionCandidate, ActionResult, ActionScope, ActionType

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


class ActionExecutor:
    """Dispatches approved actions to runners, optionally via a Docker sandbox.

    Without a sandbox (default): actions execute as direct subprocess calls in
    the host process space — same behaviour as before this class existed.

    With a sandbox: the action's command string runs inside a fresh Docker
    container constrained to the ActionScope's seccomp whitelist. The runner
    is bypassed — the sandbox IS the execution environment.
    """

    def __init__(self, sandbox: SandboxManager | None = None) -> None:
        self._runners: dict[ActionType, BaseRunner] = {}
        self._sandbox = sandbox

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
        if dry_run:
            runner = self._get_runner(action.action_type)
            return await runner.run(action, dry_run=True)

        if self._sandbox is not None:
            scope = _ACTION_SCOPES.get(action.action_type, ActionScope.SYSTEM)
            return await self._sandbox.run(
                action.command,
                scope=scope,
                action_id=action.id,
            )

        runner = self._get_runner(action.action_type)
        return await runner.run(action, dry_run=False)

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
