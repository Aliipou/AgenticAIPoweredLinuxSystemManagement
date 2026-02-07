"""Brutal tests for memory store, models, migrations, and context."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from agentic.memory.context import ContextRetriever
from agentic.memory.migrations import TABLES
from agentic.memory.models import ActionRecord, ExecutionRecord, RequestRecord
from agentic.memory.store import MemoryStore


class TestMemoryModels:
    def test_request_record(self):
        r = RequestRecord(
            id="req-1",
            raw_query="focus mode",
            intent_type="FOCUS",
            confidence=0.9,
        )
        assert r.id == "req-1"
        assert r.raw_query == "focus mode"
        assert r.created_at is not None

    def test_action_record(self):
        a = ActionRecord(
            id="act-1",
            request_id="req-1",
            action_type="SUSPEND_PROCESS",
            description="Suspend firefox",
        )
        assert a.approved is False
        assert a.risk_level == 1
        assert a.command == ""

    def test_execution_record(self):
        e = ExecutionRecord(
            id="exec-1",
            action_id="act-1",
            success=True,
            output="done",
        )
        assert e.rolled_back is False
        assert e.error == ""
        assert e.executed_at is not None

    def test_request_record_custom_timestamp(self):
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        r = RequestRecord(
            id="req-2",
            raw_query="test",
            intent_type="UPDATE",
            confidence=0.5,
            created_at=ts,
        )
        assert r.created_at == ts


class TestMigrations:
    def test_tables_list_not_empty(self):
        assert len(TABLES) == 5

    def test_all_tables_are_create_statements(self):
        for sql in TABLES:
            assert "CREATE TABLE IF NOT EXISTS" in sql

    def test_table_names(self):
        expected = ["requests", "actions", "policy_decisions", "execution_results", "embeddings_cache"]
        for name in expected:
            found = any(name in sql for sql in TABLES)
            assert found, f"Table {name} not found in migrations"


class TestMemoryStore:
    @pytest.mark.asyncio
    async def test_initialize_and_close(self):
        store = MemoryStore(":memory:")
        await store.initialize()
        assert store._db is not None
        await store.close()
        assert store._db is None

    @pytest.mark.asyncio
    async def test_initialize_with_file_path(self, tmp_path):
        db_file = tmp_path / "subdir" / "test.db"
        store = MemoryStore(db_path=db_file)
        await store.initialize()
        assert store._db is not None
        assert db_file.parent.exists()
        await store.close()

    @pytest.mark.asyncio
    async def test_get_db_raises_when_not_initialized(self):
        store = MemoryStore(":memory:")
        with pytest.raises(RuntimeError, match="not initialized"):
            store._get_db()

    @pytest.mark.asyncio
    async def test_log_and_retrieve_request(self, temp_db):
        record = RequestRecord(
            id="req-test-1",
            raw_query="focus",
            intent_type="FOCUS",
            confidence=0.95,
        )
        await temp_db.log_request(record)
        result = await temp_db.get_request("req-test-1")
        assert result is not None
        assert result.raw_query == "focus"

    @pytest.mark.asyncio
    async def test_get_request_not_found(self, temp_db):
        result = await temp_db.get_request("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_log_and_retrieve_action(self, temp_db):
        # First log a request
        req = RequestRecord(id="req-a", raw_query="test", intent_type="FOCUS", confidence=0.9)
        await temp_db.log_request(req)

        action = ActionRecord(
            id="act-a",
            request_id="req-a",
            action_type="SUSPEND_PROCESS",
            description="Suspend firefox",
            approved=True,
        )
        await temp_db.log_action(action)

        actions = await temp_db.get_actions_for_request("req-a")
        assert len(actions) == 1
        assert actions[0].approved is True

    @pytest.mark.asyncio
    async def test_log_and_retrieve_execution(self, temp_db):
        record = ExecutionRecord(
            id="exec-1",
            action_id="act-1",
            success=True,
            output="done",
        )
        await temp_db.log_execution(record)
        result = await temp_db.get_execution("act-1")
        assert result is not None
        assert result.success is True

    @pytest.mark.asyncio
    async def test_get_execution_not_found(self, temp_db):
        result = await temp_db.get_execution("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_recent_context(self, temp_db):
        for i in range(5):
            await temp_db.log_request(
                RequestRecord(
                    id=f"req-ctx-{i}",
                    raw_query=f"query {i}",
                    intent_type="FOCUS",
                    confidence=0.8,
                )
            )
        recent = await temp_db.get_recent_context(limit=3)
        assert len(recent) == 3

    @pytest.mark.asyncio
    async def test_get_recent_context_empty_db(self, temp_db):
        recent = await temp_db.get_recent_context()
        assert recent == []

    @pytest.mark.asyncio
    async def test_search_similar_delegates_to_recent(self, temp_db):
        await temp_db.log_request(
            RequestRecord(id="req-s", raw_query="focus", intent_type="FOCUS", confidence=0.9)
        )
        results = await temp_db.search_similar("focus", limit=5)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_history(self, temp_db):
        req = RequestRecord(id="req-h", raw_query="test", intent_type="UPDATE", confidence=0.8)
        await temp_db.log_request(req)
        action = ActionRecord(
            id="act-h", request_id="req-h", action_type="APT_UPGRADE",
            description="Upgrade", approved=True,
        )
        await temp_db.log_action(action)
        history = await temp_db.get_history(limit=10)
        assert len(history) >= 1
        assert history[0]["raw_query"] == "test"

    @pytest.mark.asyncio
    async def test_get_history_empty(self, temp_db):
        history = await temp_db.get_history()
        assert history == []

    @pytest.mark.asyncio
    async def test_log_policy_decision(self, temp_db):
        await temp_db.log_policy_decision(
            action_id="act-pol",
            risk_level=3,
            approved=True,
            requires_sudo=True,
            reason="Approved by gate",
        )
        # No direct retrieval method, but should not raise
        db = temp_db._get_db()
        cursor = await db.execute("SELECT * FROM policy_decisions WHERE action_id = ?", ("act-pol",))
        row = await cursor.fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_get_rollback_command_no_execution(self, temp_db):
        result = await temp_db.get_rollback_command("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_rollback_command_with_execution(self, temp_db):
        await temp_db.log_execution(
            ExecutionRecord(
                id="exec-rb",
                action_id="act-rb",
                success=True,
                output="done",
            )
        )
        result = await temp_db.get_rollback_command("act-rb")
        # Currently returns None (stub)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_actions_for_request_empty(self, temp_db):
        actions = await temp_db.get_actions_for_request("nonexistent")
        assert actions == []

    @pytest.mark.asyncio
    async def test_close_when_already_closed(self, temp_db):
        await temp_db.close()
        await temp_db.close()  # Should not raise


class TestContextRetriever:
    @pytest.mark.asyncio
    async def test_get_context_empty(self, temp_db):
        retriever = ContextRetriever(temp_db)
        records = await retriever.get_context("test")
        assert records == []

    @pytest.mark.asyncio
    async def test_format_context_empty(self, temp_db):
        retriever = ContextRetriever(temp_db)
        result = await retriever.format_context("test")
        assert "No previous context" in result

    @pytest.mark.asyncio
    async def test_format_context_with_records(self, temp_db):
        await temp_db.log_request(
            RequestRecord(id="req-fc", raw_query="focus now", intent_type="FOCUS", confidence=0.9)
        )
        retriever = ContextRetriever(temp_db)
        result = await retriever.format_context("focus")
        assert "FOCUS" in result
        assert "focus now" in result

    @pytest.mark.asyncio
    async def test_get_context_respects_limit(self, temp_db):
        for i in range(10):
            await temp_db.log_request(
                RequestRecord(id=f"req-lim-{i}", raw_query=f"q{i}", intent_type="FOCUS", confidence=0.8)
            )
        retriever = ContextRetriever(temp_db)
        records = await retriever.get_context("test", limit=3)
        assert len(records) == 3
