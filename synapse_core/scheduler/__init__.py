"""Scheduled tasks system - cron-based task execution."""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field, asdict


@dataclass
class ScheduledTask:
    """A scheduled task definition."""
    id: str = ""
    name: str = ""
    description: str = ""
    agent_id: str = "general-assistant"
    message: str = ""
    cron: str = ""  # cron expression: "min hour day month dow"
    enabled: bool = True
    last_run: float = 0.0
    next_run: float = 0.0
    run_count: int = 0
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskScheduler:
    """SQLite-backed scheduled task manager."""

    def __init__(self, db_path: str = "") -> None:
        if not db_path:
            project_root = Path(__file__).resolve().parent.parent.parent
            db_path = str(project_root / "data" / "scheduler.db")
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()
        self._running = False

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                agent_id TEXT NOT NULL DEFAULT 'general-assistant',
                message TEXT NOT NULL,
                cron TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                last_run REAL DEFAULT 0,
                next_run REAL DEFAULT 0,
                run_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            );
        """)
        conn.commit()
        conn.close()

    def create_task(self, name: str, message: str, cron: str, agent_id: str = "general-assistant", description: str = "") -> ScheduledTask:
        """Create a new scheduled task."""
        task = ScheduledTask(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            agent_id=agent_id,
            message=message,
            cron=cron,
            enabled=True,
            created_at=time.time(),
        )
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO scheduled_tasks (id, name, description, agent_id, message, cron, enabled, last_run, next_run, run_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (task.id, task.name, task.description, task.agent_id, task.message, task.cron, 1, 0.0, 0.0, 0, task.created_at),
        )
        conn.commit()
        conn.close()
        return task

    def get_task(self, task_id: str) -> ScheduledTask | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)).fetchone()
        conn.close()
        if not row:
            return None
        return ScheduledTask(**dict(row))

    def list_tasks(self) -> list[ScheduledTask]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM scheduled_tasks ORDER BY created_at DESC").fetchall()
        conn.close()
        return [ScheduledTask(**dict(r)) for r in rows]

    def update_task(self, task_id: str, **kwargs: Any) -> bool:
        conn = self._get_conn()
        sets = []
        vals = []
        for k, v in kwargs.items():
            if k in ("name", "description", "agent_id", "message", "cron", "enabled", "last_run", "next_run", "run_count"):
                sets.append(f"{k} = ?")
                vals.append(v)
        if not sets:
            return False
        vals.append(task_id)
        conn.execute(f"UPDATE scheduled_tasks SET {', '.join(sets)} WHERE id = ?", vals)
        conn.commit()
        conn.close()
        return True

    def delete_task(self, task_id: str) -> bool:
        conn = self._get_conn()
        deleted = conn.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,)).rowcount
        conn.commit()
        conn.close()
        return deleted > 0

    def record_run(self, task_id: str) -> None:
        """Record that a task was just executed."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE scheduled_tasks SET last_run = ?, run_count = run_count + 1 WHERE id = ?",
            (time.time(), task_id),
        )
        conn.commit()
        conn.close()
