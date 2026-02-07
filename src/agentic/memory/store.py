"""SQLite-backed memory store for audit logging and context retrieval."""

from __future__ import annotations

import uuid
from pathlib import Path

import aiosqlite

from agentic.memory.migrations import TABLES
from agentic.memory.models import ActionRecord, ExecutionRecord, RequestRecord


class MemoryStore:
    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self.db_path = str(db_path)
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        for table_sql in TABLES:
            await self._db.execute(table_sql)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    def _get_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("MemoryStore not initialized — call initialize() first")
        return self._db

    async def log_request(self, record: RequestRecord) -> None:
        db = self._get_db()
        await db.execute(
            "INSERT INTO requests (id, raw_query, intent_type, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                record.id,
                record.raw_query,
                record.intent_type,
                record.confidence,
                record.created_at.isoformat(),
            ),
        )
        await db.commit()

    async def log_action(self, record: ActionRecord) -> None:
        db = self._get_db()
        await db.execute(
            "INSERT INTO actions (id, request_id, action_type, description, command, risk_level, approved) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.request_id,
                record.action_type,
                record.description,
                record.command,
                record.risk_level,
                int(record.approved),
            ),
        )
        await db.commit()

    async def log_execution(self, record: ExecutionRecord) -> None:
        db = self._get_db()
        await db.execute(
            "INSERT INTO execution_results (id, action_id, success, output, error, rolled_back, executed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.action_id,
                int(record.success),
                record.output,
                record.error,
                int(record.rolled_back),
                record.executed_at.isoformat(),
            ),
        )
        await db.commit()

    async def get_recent_context(self, limit: int = 10) -> list[RequestRecord]:
        db = self._get_db()
        cursor = await db.execute(
            "SELECT id, raw_query, intent_type, confidence, created_at "
            "FROM requests ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            RequestRecord(
                id=r[0],
                raw_query=r[1],
                intent_type=r[2],
                confidence=r[3],
                created_at=r[4],
            )
            for r in rows
        ]

    async def get_request(self, request_id: str) -> RequestRecord | None:
        db = self._get_db()
        cursor = await db.execute(
            "SELECT id, raw_query, intent_type, confidence, created_at "
            "FROM requests WHERE id = ?",
            (request_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return RequestRecord(
            id=row[0],
            raw_query=row[1],
            intent_type=row[2],
            confidence=row[3],
            created_at=row[4],
        )

    async def get_actions_for_request(self, request_id: str) -> list[ActionRecord]:
        db = self._get_db()
        cursor = await db.execute(
            "SELECT id, request_id, action_type, description, command, risk_level, approved "
            "FROM actions WHERE request_id = ?",
            (request_id,),
        )
        rows = await cursor.fetchall()
        return [
            ActionRecord(
                id=r[0],
                request_id=r[1],
                action_type=r[2],
                description=r[3],
                command=r[4],
                risk_level=r[5],
                approved=bool(r[6]),
            )
            for r in rows
        ]

    async def get_execution(self, action_id: str) -> ExecutionRecord | None:
        db = self._get_db()
        cursor = await db.execute(
            "SELECT id, action_id, success, output, error, rolled_back, executed_at "
            "FROM execution_results WHERE action_id = ?",
            (action_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return ExecutionRecord(
            id=row[0],
            action_id=row[1],
            success=bool(row[2]),
            output=row[3],
            error=row[4],
            rolled_back=bool(row[5]),
            executed_at=row[6],
        )

    async def search_similar(self, query: str, limit: int = 5) -> list[RequestRecord]:
        # Stub — future embedding-based similarity search
        return await self.get_recent_context(limit=limit)

    async def get_history(self, limit: int = 20) -> list[dict]:
        db = self._get_db()
        cursor = await db.execute(
            "SELECT r.id, r.raw_query, r.intent_type, r.confidence, r.created_at, "
            "       a.id, a.action_type, a.description, a.approved "
            "FROM requests r LEFT JOIN actions a ON a.request_id = r.id "
            "ORDER BY r.created_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        results: list[dict] = []
        for r in rows:
            results.append(
                {
                    "request_id": r[0],
                    "raw_query": r[1],
                    "intent_type": r[2],
                    "confidence": r[3],
                    "created_at": r[4],
                    "action_id": r[5],
                    "action_type": r[6],
                    "description": r[7],
                    "approved": bool(r[8]) if r[8] is not None else None,
                }
            )
        return results

    async def log_policy_decision(
        self,
        action_id: str,
        risk_level: int,
        approved: bool,
        requires_sudo: bool = False,
        reason: str = "",
    ) -> None:
        db = self._get_db()
        await db.execute(
            "INSERT INTO policy_decisions (action_id, risk_level, approved, requires_sudo, reason) "
            "VALUES (?, ?, ?, ?, ?)",
            (action_id, risk_level, int(approved), int(requires_sudo), reason),
        )
        await db.commit()

    async def get_rollback_command(self, action_id: str) -> str | None:
        db = self._get_db()
        # Check execution_results to see if it was executed
        exec_cursor = await db.execute(
            "SELECT success FROM execution_results WHERE action_id = ?",
            (action_id,),
        )
        exec_row = await exec_cursor.fetchone()
        if exec_row is None:
            return None
        # For now, return None — rollback commands stored on ActionCandidate
        return None
