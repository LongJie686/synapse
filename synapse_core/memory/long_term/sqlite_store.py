"""SQLite-based long-term memory store - zero external dependencies."""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from synapse_core.memory import BaseMemoryStore, MemoryEntry, MemorySearchOptions, MemoryType


class SqliteMemoryStore(BaseMemoryStore):
    """Persistent memory store backed by SQLite with full-text search."""

    def __init__(self, db_path: str = "") -> None:
        if not db_path:
            project_root = Path(__file__).resolve().parent.parent.parent
            db_path = str(project_root / "data" / "memory.db")
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL DEFAULT 'episodic',
                importance REAL NOT NULL DEFAULT 0.0,
                session_id TEXT,
                agent_id TEXT,
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id UNINDEXED, content, memory_type,
                content='memories', content_rowid='rowid'
            );
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, id, content, memory_type)
                VALUES (new.rowid, new.id, new.content, new.memory_type);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, id, content, memory_type)
                VALUES ('delete', old.rowid, old.id, old.content, old.memory_type);
            END;
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
        """)
        conn.commit()
        conn.close()

    async def add(self, entry: MemoryEntry) -> str:
        entry_id = entry.id or str(uuid.uuid4())
        now = time.time()
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO memories (id, content, memory_type, importance, session_id, agent_id, metadata, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry_id,
                entry.content,
                entry.memory_type.value if isinstance(entry.memory_type, MemoryType) else entry.memory_type,
                entry.importance,
                entry.session_id,
                entry.agent_id,
                json.dumps(entry.metadata),
                entry.created_at or now,
                now,
            ),
        )
        conn.commit()
        conn.close()
        return entry_id

    async def search(self, query: str, options: MemorySearchOptions) -> list[MemoryEntry]:
        # FTS5 safe: escape special characters, split into words
        import re
        safe_query = " ".join(
            f'"{w}"' for w in re.findall(r'\w+', query) if len(w) > 1
        ) or "placeholder_no_match"

        conn = self._get_conn()

        sql = """
            SELECT m.* FROM memories m
            JOIN memories_fts fts ON m.id = fts.id
            WHERE memories_fts MATCH ?
        """
        params: list[Any] = [safe_query]

        if options.memory_type:
            sql += " AND m.memory_type = ?"
            params.append(options.memory_type.value if isinstance(options.memory_type, MemoryType) else options.memory_type)

        sql += " ORDER BY m.importance DESC, m.created_at DESC LIMIT ?"
        params.append(options.top_k)

        rows = conn.execute(sql, params).fetchall()
        conn.close()

        return [self._row_to_entry(r) for r in rows]

    async def get(self, entry_id: str) -> MemoryEntry | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (entry_id,)).fetchone()
        conn.close()
        return self._row_to_entry(row) if row else None

    async def delete(self, entry_id: str) -> bool:
        conn = self._get_conn()
        deleted = conn.execute("DELETE FROM memories WHERE id = ?", (entry_id,)).rowcount
        conn.commit()
        conn.close()
        return deleted > 0

    async def update(self, entry_id: str, content: str, metadata: dict[str, Any] | None = None) -> bool:
        conn = self._get_conn()
        if metadata:
            conn.execute(
                "UPDATE memories SET content = ?, metadata = ?, updated_at = ? WHERE id = ?",
                (content, json.dumps(metadata), time.time(), entry_id),
            )
        else:
            conn.execute(
                "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
                (content, time.time(), entry_id),
            )
        updated = conn.total_changes > 0
        conn.commit()
        conn.close()
        return updated

    async def get_recent(self, memory_type: MemoryType | None = None, limit: int = 20) -> list[MemoryEntry]:
        conn = self._get_conn()
        if memory_type:
            rows = conn.execute(
                "SELECT * FROM memories WHERE memory_type = ? ORDER BY created_at DESC LIMIT ?",
                (memory_type.value, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()
        return [self._row_to_entry(r) for r in rows]

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            importance=row["importance"],
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
