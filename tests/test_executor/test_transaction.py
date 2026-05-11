"""Tests for TransactionManager and TransactionResult."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from agentic.executor.transaction import TransactionManager, TransactionResult
from agentic.models.action import ActionCandidate, ActionResult, ActionType, RollbackSupport


def _action(action_id: str = "a1", rollback: str = "") -> ActionCandidate:
    return ActionCandidate(
        id=action_id,
        action_type=ActionType.SUSPEND_PROCESS,
        description="test",
        command="kill -STOP 1",
        target="chrome",
        rollback_command=rollback,
    )


def _ok(action_id: str = "a1") -> ActionResult:
    return ActionResult(action_id=action_id, success=True, output="done")


def _fail(action_id: str = "a1", error: str = "oops") -> ActionResult:
    return ActionResult(action_id=action_id, success=False, error=error)


class TestTransactionResultDataclass:
    def test_frozen(self):
        tr = TransactionResult(success=True, results=[])
        with pytest.raises((AttributeError, TypeError)):
            tr.success = False  # type: ignore[misc]

    def test_defaults(self):
        tr = TransactionResult(success=True, results=[])
        assert tr.rolled_back_ids == []
        assert tr.rollback_errors == []

    def test_all_fields(self):
        r = _ok()
        tr = TransactionResult(
            success=False,
            results=[r],
            rolled_back_ids=["a1"],
            rollback_errors=["a2: failed"],
        )
        assert tr.success is False
        assert len(tr.results) == 1
        assert tr.rolled_back_ids == ["a1"]
        assert tr.rollback_errors == ["a2: failed"]


class TestTransactionManagerSuccess:
    @pytest.mark.asyncio
    async def test_empty_actions_returns_success(self):
        executor = AsyncMock()
        tm = TransactionManager()
        result = await tm.execute_with_rollback([], executor)
        assert result.success is True
        assert result.results == []
        executor.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_succeed_no_rollback(self):
        a1 = _action("a1", rollback="kill -CONT 1")
        a2 = _action("a2", rollback="kill -CONT 2")

        executor = AsyncMock()
        executor.execute = AsyncMock(side_effect=[_ok("a1"), _ok("a2")])

        tm = TransactionManager()
        result = await tm.execute_with_rollback([a1, a2], executor)

        assert result.success is True
        assert len(result.results) == 2
        assert result.rolled_back_ids == []
        executor.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_propagated_to_executor(self):
        a1 = _action("a1")
        executor = AsyncMock()
        executor.execute = AsyncMock(return_value=_ok("a1"))

        tm = TransactionManager()
        await tm.execute_with_rollback([a1], executor, dry_run=True)

        executor.execute.assert_called_once_with(a1, dry_run=True)


class TestTransactionManagerRollback:
    @pytest.mark.asyncio
    async def test_first_action_fails_no_rollback_needed(self):
        a1 = _action("a1", rollback="restore")
        executor = AsyncMock()
        executor.execute = AsyncMock(return_value=_fail("a1"))

        tm = TransactionManager()
        result = await tm.execute_with_rollback([a1], executor)

        assert result.success is False
        assert len(result.results) == 1
        assert result.rolled_back_ids == []  # nothing executed yet
        executor.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_second_fails_first_rolled_back(self):
        a1 = _action("a1", rollback="restore a1")
        a2 = _action("a2", rollback="restore a2")

        executor = AsyncMock()
        executor.execute = AsyncMock(side_effect=[_ok("a1"), _fail("a2")])
        executor.rollback = AsyncMock(return_value=ActionResult(action_id="a1", success=True))

        tm = TransactionManager()
        result = await tm.execute_with_rollback([a1, a2], executor)

        assert result.success is False
        assert "a1" in result.rolled_back_ids
        executor.rollback.assert_called_once_with(a1)

    @pytest.mark.asyncio
    async def test_skips_action_without_rollback_command(self):
        a1 = _action("a1", rollback="")  # no rollback
        a2 = _action("a2", rollback="restore a2")

        executor = AsyncMock()
        executor.execute = AsyncMock(side_effect=[_ok("a1"), _ok("a2"), _fail("a3")])
        executor.rollback = AsyncMock(return_value=ActionResult(action_id="a2", success=True))

        # Third action fails: a1 (no rollback), a2 (has rollback)
        a3 = _action("a3")
        tm = TransactionManager()
        result = await tm.execute_with_rollback([a1, a2, a3], executor)

        assert result.success is False
        # a1 has no rollback_command, a2 does
        assert "a2" in result.rolled_back_ids
        assert "a1" not in result.rolled_back_ids

    @pytest.mark.asyncio
    async def test_rollback_failure_recorded_in_errors(self):
        a1 = _action("a1", rollback="restore")
        a2 = _action("a2")

        executor = AsyncMock()
        executor.execute = AsyncMock(side_effect=[_ok("a1"), _fail("a2")])
        executor.rollback = AsyncMock(
            return_value=ActionResult(action_id="a1", success=False, error="restore failed")
        )

        tm = TransactionManager()
        result = await tm.execute_with_rollback([a1, a2], executor)

        assert result.success is False
        assert result.rolled_back_ids == []
        assert any("a1" in e and "restore failed" in e for e in result.rollback_errors)

    @pytest.mark.asyncio
    async def test_rollback_none_skips_even_with_rollback_command(self):
        a1 = ActionCandidate(
            id="a1",
            action_type=ActionType.KILL_PROCESS,
            description="Kill",
            command="kill -9 123",
            rollback_command="restore",
            rollback_support=RollbackSupport.NONE,
        )
        a2 = _action("a2")

        executor = AsyncMock()
        executor.execute = AsyncMock(side_effect=[_ok("a1"), _fail("a2")])

        tm = TransactionManager()
        result = await tm.execute_with_rollback([a1, a2], executor)

        assert result.success is False
        assert "a1" not in result.rolled_back_ids
        executor.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollback_in_reverse_order(self):
        a1 = _action("a1", rollback="r1")
        a2 = _action("a2", rollback="r2")
        a3 = _action("a3")  # fails

        call_order: list[str] = []

        async def execute_side_effect(action, dry_run=False):
            if action.id == "a3":
                return _fail("a3")
            return _ok(action.id)

        async def rollback_side_effect(action):
            call_order.append(action.id)
            return ActionResult(action_id=action.id, success=True)

        executor = AsyncMock()
        executor.execute = AsyncMock(side_effect=execute_side_effect)
        executor.rollback = AsyncMock(side_effect=rollback_side_effect)

        tm = TransactionManager()
        result = await tm.execute_with_rollback([a1, a2, a3], executor)

        # Should rollback a2 then a1 (reverse order)
        assert call_order == ["a2", "a1"]
        assert result.rolled_back_ids == ["a2", "a1"]
