"""SQLite CREATE TABLE statements."""

from __future__ import annotations

TABLES: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS requests (
        id TEXT PRIMARY KEY,
        raw_query TEXT NOT NULL,
        intent_type TEXT NOT NULL,
        confidence REAL NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS actions (
        id TEXT PRIMARY KEY,
        request_id TEXT NOT NULL,
        action_type TEXT NOT NULL,
        description TEXT NOT NULL,
        command TEXT DEFAULT '',
        risk_level INTEGER DEFAULT 1,
        approved INTEGER DEFAULT 0,
        FOREIGN KEY (request_id) REFERENCES requests(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS policy_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        action_id TEXT NOT NULL,
        risk_level INTEGER NOT NULL,
        approved INTEGER NOT NULL,
        requires_sudo INTEGER DEFAULT 0,
        reason TEXT DEFAULT '',
        FOREIGN KEY (action_id) REFERENCES actions(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS execution_results (
        id TEXT PRIMARY KEY,
        action_id TEXT NOT NULL,
        success INTEGER NOT NULL,
        output TEXT DEFAULT '',
        error TEXT DEFAULT '',
        rolled_back INTEGER DEFAULT 0,
        executed_at TEXT NOT NULL,
        FOREIGN KEY (action_id) REFERENCES actions(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS embeddings_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        embedding BLOB NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
]
