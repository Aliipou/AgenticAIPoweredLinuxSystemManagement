"""Memory cleanup runner — drop caches and kill memory hogs."""

from __future__ import annotations

import asyncio

import psutil

from agentic.exceptions import ExecutionError
from agentic.executor.runners.base import BaseRunner
from agentic.models.action import ActionCandidate, ActionResult, ActionType

MEMORY_THRESHOLD_MB = 500


class MemoryRunner(BaseRunner):
    async def run(self, action: ActionCandidate, dry_run: bool = False) -> ActionResult:
        if action.action_type == ActionType.DROP_CACHES:
            return await self._drop_caches(action, dry_run)
        if action.action_type == ActionType.KILL_BY_MEMORY:
            return await self._kill_by_memory(action, dry_run)

        raise ExecutionError(
            f"MemoryRunner cannot handle {action.action_type.value}",
            action_id=action.id,
        )

    async def _drop_caches(
        self, action: ActionCandidate, dry_run: bool
    ) -> ActionResult:
        if dry_run:
            return ActionResult(
                action_id=action.id,
                success=True,
                output="[DRY RUN] Would drop filesystem caches",
            )

        try:
            proc = await asyncio.create_subprocess_shell(
                "sync && echo 3 > /proc/sys/vm/drop_caches",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        except OSError as exc:
            raise ExecutionError(
                f"Failed to drop caches: {exc}",
                action_id=action.id,
            ) from exc

        if proc.returncode != 0:
            raise ExecutionError(
                f"Drop caches failed: {stderr.decode()}",
                action_id=action.id,
            )

        return ActionResult(
            action_id=action.id,
            success=True,
            output="Filesystem caches dropped",
        )

    async def _kill_by_memory(
        self, action: ActionCandidate, dry_run: bool
    ) -> ActionResult:
        hogs = self._find_memory_hogs(action.target)

        if not hogs:
            return ActionResult(
                action_id=action.id,
                success=True,
                output="No memory-hogging processes found",
            )

        if dry_run:
            desc = ", ".join(f"{name}(PID {pid}, {mb:.0f}MB)" for pid, name, mb in hogs)
            return ActionResult(
                action_id=action.id,
                success=True,
                output=f"[DRY RUN] Would kill: {desc}",
            )

        killed: list[str] = []
        for pid, name, mb in hogs:
            try:
                psutil.Process(pid).terminate()
                killed.append(f"{name}(PID {pid})")
            except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                raise ExecutionError(
                    f"Failed to kill PID {pid}: {exc}",
                    action_id=action.id,
                ) from exc

        return ActionResult(
            action_id=action.id,
            success=True,
            output=f"Killed: {', '.join(killed)}",
        )

    @staticmethod
    def _find_memory_hogs(target: str) -> list[tuple[int, str, float]]:
        hogs: list[tuple[int, str, float]] = []
        for proc in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                mem = proc.info.get("memory_info")
                if mem is None:
                    continue
                mb = mem.rss / (1024 * 1024)
                name = proc.info.get("name", "") or ""

                if target and target != "memory_hogs":
                    if target.lower() in name.lower():
                        hogs.append((proc.pid, name, mb))
                elif mb > MEMORY_THRESHOLD_MB:
                    hogs.append((proc.pid, name, mb))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return hogs

    async def rollback(self, action: ActionCandidate) -> ActionResult:
        return ActionResult(
            action_id=action.id,
            success=False,
            error="Cannot rollback memory cleanup actions",
            rolled_back=False,
        )
