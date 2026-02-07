"""Systemd service management runner."""

from __future__ import annotations

import asyncio

from agentic.exceptions import ExecutionError
from agentic.executor.runners.base import BaseRunner
from agentic.models.action import ActionCandidate, ActionResult, ActionType

_REVERSE: dict[ActionType, str] = {
    ActionType.SYSTEMCTL_START: "stop",
    ActionType.SYSTEMCTL_STOP: "start",
    ActionType.SYSTEMCTL_RESTART: "restart",
}


class SystemctlRunner(BaseRunner):
    async def run(self, action: ActionCandidate, dry_run: bool = False) -> ActionResult:
        verb = action.action_type.value.replace("SYSTEMCTL_", "").lower()
        service = action.target
        cmd = f"systemctl {verb} {service}"

        if dry_run:
            return ActionResult(
                action_id=action.id,
                success=True,
                output=f"[DRY RUN] Would execute: {cmd}",
            )

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        except OSError as exc:
            raise ExecutionError(
                f"systemctl failed: {exc}",
                action_id=action.id,
            ) from exc

        if proc.returncode != 0:
            raise ExecutionError(
                f"systemctl {verb} {service} failed: {stderr.decode()}",
                action_id=action.id,
            )

        return ActionResult(
            action_id=action.id,
            success=True,
            output=stdout.decode() or f"Service {service} {verb}ed successfully",
        )

    async def rollback(self, action: ActionCandidate) -> ActionResult:
        reverse_verb = _REVERSE.get(action.action_type)
        if not reverse_verb:
            return ActionResult(
                action_id=action.id,
                success=False,
                error=f"No rollback for {action.action_type.value}",
                rolled_back=False,
            )

        service = action.target
        cmd = f"systemctl {reverse_verb} {service}"

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        except OSError as exc:
            raise ExecutionError(
                f"Rollback failed: {exc}",
                action_id=action.id,
            ) from exc

        if proc.returncode != 0:
            return ActionResult(
                action_id=action.id,
                success=False,
                error=f"Rollback failed: {stderr.decode()}",
                rolled_back=False,
            )

        return ActionResult(
            action_id=action.id,
            success=True,
            output=f"Service {service} {reverse_verb}ed (rollback)",
            rolled_back=True,
        )
