"""Process management runner using psutil."""

from __future__ import annotations

import os
import signal
import sys

import psutil

from agentic.exceptions import ExecutionError
from agentic.executor.runners.base import BaseRunner
from agentic.models.action import ActionCandidate, ActionResult, ActionType

ESSENTIAL_PROCESSES = frozenset({
    "init", "systemd", "kernel", "kthreadd", "sshd", "login",
})

# Windows lacks SIGSTOP/SIGCONT — use SIGTERM as fallback
_SIGSTOP = getattr(signal, "SIGSTOP", signal.SIGTERM)
_SIGCONT = getattr(signal, "SIGCONT", signal.SIGTERM)


class ProcessRunner(BaseRunner):
    async def run(self, action: ActionCandidate, dry_run: bool = False) -> ActionResult:
        target = action.target
        if target in ESSENTIAL_PROCESSES:
            raise ExecutionError(
                f"Refusing to touch essential process: {target}",
                action_id=action.id,
            )

        if dry_run:
            return ActionResult(
                action_id=action.id,
                success=True,
                output=f"[DRY RUN] Would {action.action_type.value} process: {target}",
            )

        pids = self._find_pids(target)
        if not pids:
            return ActionResult(
                action_id=action.id,
                success=True,
                output=f"No matching processes found for: {target}",
            )

        sig = self._get_signal(action.action_type)
        killed: list[int] = []
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                proc.send_signal(sig)
                killed.append(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError) as exc:
                raise ExecutionError(
                    f"Failed to signal PID {pid}: {exc}",
                    action_id=action.id,
                ) from exc

        return ActionResult(
            action_id=action.id,
            success=True,
            output=f"Sent signal to PIDs: {killed}",
        )

    async def rollback(self, action: ActionCandidate) -> ActionResult:
        if action.action_type == ActionType.SUSPEND_PROCESS:
            pids = self._find_pids(action.target)
            for pid in pids:
                try:
                    psutil.Process(pid).send_signal(_SIGCONT)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return ActionResult(
                action_id=action.id,
                success=True,
                output=f"Resumed {action.target}",
                rolled_back=True,
            )
        return ActionResult(
            action_id=action.id,
            success=False,
            error=f"Rollback not supported for {action.action_type.value}",
            rolled_back=False,
        )

    @staticmethod
    def _find_pids(name: str) -> list[int]:
        pids: list[int] = []
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                pname = proc.info.get("name", "") or ""
                cmdline = proc.info.get("cmdline") or []
                if name.lower() in pname.lower() or any(
                    name.lower() in c.lower() for c in cmdline
                ):
                    pids.append(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return pids

    @staticmethod
    def _get_signal(action_type: ActionType) -> signal.Signals:
        mapping = {
            ActionType.KILL_PROCESS: signal.SIGTERM,
            ActionType.SUSPEND_PROCESS: _SIGSTOP,
            ActionType.RENICE_PROCESS: signal.SIGTERM,
        }
        return mapping.get(action_type, signal.SIGTERM)
