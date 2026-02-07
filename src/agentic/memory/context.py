"""Context retrieval for enriching intent parsing with history."""

from __future__ import annotations

from agentic.memory.models import RequestRecord
from agentic.memory.store import MemoryStore


class ContextRetriever:
    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def get_context(self, query: str, limit: int = 5) -> list[RequestRecord]:
        return await self._store.search_similar(query, limit=limit)

    async def format_context(self, query: str, limit: int = 5) -> str:
        records = await self.get_context(query, limit=limit)
        if not records:
            return "No previous context available."
        lines: list[str] = []
        for r in records:
            lines.append(f"- [{r.intent_type}] {r.raw_query} (confidence: {r.confidence})")
        return "Recent history:\n" + "\n".join(lines)
