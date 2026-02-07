"""Package management runner using subprocess (apt)."""

from __future__ import annotations

import asyncio
import shlex

from agentic.exceptions import ExecutionError
from agentic.executor.runners.base import BaseRunner
from agentic.models.action import ActionCandidate, ActionResult


class PackageRunner(BaseRunner):
    async def run(self, action: ActionCandidate, dry_run: bool = False) -> ActionResult:
        if dry_run:
            return ActionResult(
                action_id=action.id,
                success=True,
                output=f"[DRY RUN] Would execute: {action.command}",
            )

        try:
            proc = await asyncio.create_subprocess_shell(
                action.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        except OSError as exc:
            raise ExecutionError(
                f"Failed to run package command: {exc}",
                action_id=action.id,
            ) from exc

        if proc.returncode != 0:
            raise ExecutionError(
                f"Package command failed (rc={proc.returncode}): {stderr.decode()}",
                action_id=action.id,
            )

        return ActionResult(
            action_id=action.id,
            success=True,
            output=stdout.decode(),
        )

    async def rollback(self, action: ActionCandidate) -> ActionResult:
        if not action.rollback_command:
            return ActionResult(
                action_id=action.id,
                success=False,
                error="No rollback command available",
                rolled_back=False,
            )

        try:
            proc = await asyncio.create_subprocess_shell(
                action.rollback_command,
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
                error=f"Rollback command failed: {stderr.decode()}",
                rolled_back=False,
            )

        return ActionResult(
            action_id=action.id,
            success=True,
            output=stdout.decode(),
            rolled_back=True,
        )
