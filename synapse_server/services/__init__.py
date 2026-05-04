"""Business services for synapse server."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from synapse_core.agent import AgentDefinition, AgentRegistry
from synapse_core.llm import LLMConfig
from synapse_core.tools import ToolRegistry
from synapse_core.tools.builtin import register_all_builtin_tools
from synapse_core.observability import get_hub
from synapse_core.skills import load_skills_from_directory, SkillRegistry

# Global reference for cross-module access
_tool_service_instance: ToolService | None = None


def _get_tool_service() -> ToolService | None:
    return _tool_service_instance


def _default_llm() -> LLMConfig:
    """Build LLMConfig from environment, defaults to ZhiPu GLM."""
    return LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "anthropic"),
        model=os.getenv("LLM_MODEL", "glm-4-flash"),
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
        temperature=0.7,
        max_tokens=4096,
    )


# ── Agent Service ───────────────────────────────────────────────────────────

class AgentService:
    """Manages agent definitions and execution."""

    def __init__(self) -> None:
        self.registry = AgentRegistry()
        self.skill_registry = SkillRegistry()
        self._setup_default_agents()
        self._load_skills()

    def _setup_default_agents(self) -> None:
        llm = _default_llm()

        # General Assistant -- broad capability, all tools
        self.registry.register(AgentDefinition(
            id="general-assistant",
            name="Synapse Assistant",
            role="General-purpose AI assistant with memory",
            goal="Help users accomplish a wide range of tasks, remembering context across conversations",
            backstory=(
                "You are Synapse Assistant, a versatile AI helper with memory capabilities. "
                "You can remember previous conversations and user preferences across sessions. "
                "Your memory system includes:\n"
                "- Short-term memory: remembers the current conversation context\n"
                "- Long-term memory: stores important facts and user preferences permanently\n"
                "- Auto memory: automatically saves important interactions for future reference\n\n"
                "When a user asks about your memory, explain that you CAN remember conversations and learn their preferences. "
                "You can perform calculations, search the web, scrape web pages, make HTTP requests, query databases, execute code, and read/write local files. "
                "For file operations: use file_read to read files, file_write to create or modify files, file_list to list directories, file_search to search within files. "
                "These file operations work within the workspace directory. "
                "Always explain your reasoning clearly. When a task involves math, use the calculator tool. "
                "When a task requires external information, use web_search or web_scrape to read web pages. "
                "When a task requires running code, use code_execute to run Python code and see the output. "
                "Respond in the same language the user writes in (Chinese or English)."
            ),
            llm=llm.model_copy(),
            tools=["calculator", "web_search", "web_scrape", "http_request", "code_execute", "file_read", "file_write", "file_list", "file_search"],
        ))

        # Code Expert -- focused on software engineering
        self.registry.register(AgentDefinition(
            id="code-expert",
            name="Code Expert",
            role="Software engineering specialist with memory",
            goal="Write, review, debug, and explain code, remembering past context",
            backstory=(
                "You are Code Expert, a senior software engineer with 15 years of experience. "
                "You have memory capabilities -- you can remember previous conversations, code reviews, and the user's coding preferences. "
                "Your memory system automatically stores important technical decisions and user preferences for future reference.\n\n"
                "You specialize in Python, JavaScript/TypeScript, Go, Rust, SQL, and system design. "
                "When asked to write code:\n"
                "1. Analyze requirements first\n"
                "2. Write clean, well-structured code with proper error handling\n"
                "3. Explain key design decisions\n"
                "4. Suggest testing strategies\n\n"
                "When asked to debug:\n"
                "1. Read the error message carefully\n"
                "2. Identify the root cause\n"
                "3. Provide a minimal fix with explanation\n"
                "4. Suggest how to prevent similar issues\n\n"
                "Use the calculator tool for algorithmic complexity analysis or numeric verification. "
                "Use code_execute to run code snippets and verify their behavior. "
                "Use web_scrape to read documentation or code examples from web pages. "
                "Use file_read, file_write, file_list, file_search to work with local files in the workspace. "
                "Always respond in the same language the user writes in."
            ),
            llm=llm.model_copy(),
            tools=["calculator", "code_execute", "web_scrape", "file_read", "file_write", "file_list", "file_search"],
        ))

        # Data Analyst -- focused on data analysis and SQL
        self.registry.register(AgentDefinition(
            id="data-analyst",
            name="Data Analyst",
            role="Data analysis and visualization specialist with memory",
            goal="Help users analyze data, write SQL queries, and extract actionable insights, remembering past analyses",
            backstory=(
                "You are Data Analyst, an expert in data analysis, SQL, statistics, and data visualization. "
                "You have memory capabilities -- you can remember previous analyses, the user's data preferences, and past query results. "
                "Your memory system automatically stores important analytical findings and user preferences.\n\n"
                "Your workflow:\n"
                "1. Understand the business question behind the data request\n"
                "2. Design the appropriate query or analysis approach\n"
                "3. Execute calculations precisely using the calculator tool\n"
                "4. Present results with clear interpretation and actionable insights\n\n"
                "For SQL queries:\n"
                "- Always explain what the query does\n"
                "- Suggest optimizations if the query is complex\n"
                "- Consider edge cases (NULL values, duplicates, etc.)\n\n"
                "For statistical analysis:\n"
                "- Choose the right statistical method\n"
                "- Explain assumptions and limitations\n"
                "- Present confidence intervals when relevant\n\n"
                "Use code_execute to run statistical analysis or data processing code in Python. "
                "Always respond in the same language the user writes in."
            ),
            llm=llm.model_copy(),
            tools=["calculator", "sql_query", "code_execute"],
        ))

    def get_agent(self, agent_id: str) -> AgentDefinition | None:
        return self.registry.get(agent_id)

    def _load_skills(self) -> None:
        """Load user-defined skills from the skills/ directory."""
        project_root = Path(__file__).resolve().parent.parent.parent
        skills_dir = project_root / "skills"

        skills = load_skills_from_directory(skills_dir)
        llm = _default_llm()

        for skill in skills:
            self.skill_registry.register(skill)
            agent = skill.to_agent(llm)
            self.registry.register(agent)

        if skills:
            print(f"Loaded {len(skills)} skills from {skills_dir}")

    def list_agents(self) -> list[AgentDefinition]:
        return self.registry.list_all()

    def register_agent(self, agent: AgentDefinition) -> None:
        self.registry.register(agent)

    def remove_agent(self, agent_id: str) -> bool:
        try:
            self.registry.remove(agent_id)
            return True
        except KeyError:
            return False


# ── Tool Service ────────────────────────────────────────────────────────────

class ToolService:
    """Manages tool registry and execution."""

    def __init__(self) -> None:
        self.registry = ToolRegistry()
        register_all_builtin_tools(self.registry)

    def list_tools(self) -> list[dict[str, Any]]:
        tools = self.registry.list_all()
        return [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "parameters": [
                    {"name": p.name, "type": p.type, "description": p.description, "required": p.required}
                    for p in t.parameters
                ],
                "safety": {
                    "requires_approval": t.safety.requires_approval,
                    "timeout_ms": t.safety.timeout_ms,
                    "sandboxed": t.safety.sandboxed,
                },
            }
            for t in tools
        ]


# ── Session Service ─────────────────────────────────────────────────────────

class SessionService:
    """Manages conversation sessions with SQLite persistence."""

    def __init__(self, db_path: str = "") -> None:
        if not db_path:
            project_root = Path(__file__).resolve().parent.parent.parent
            db_path = str(project_root / "data" / "sessions.db")
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()
        self._sessions_cache: dict[str, dict[str, Any]] = {}

    def _get_conn(self):
        import sqlite3
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL DEFAULT 'general-assistant',
                title TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
        """)
        conn.commit()
        conn.close()

    def create_session(self, user_id: str = "", agent_id: str = "general-assistant", title: str = "") -> dict[str, Any]:
        import time
        session_id = str(uuid.uuid4())
        now = time.time()
        title = title or f"Session {session_id[:8]}"
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO sessions (session_id, agent_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, agent_id, title, now, now),
        )
        conn.commit()
        conn.close()
        return {"session_id": session_id, "agent_id": agent_id, "title": title}

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        conn.close()
        if not row:
            return None
        return dict(row)

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        import time
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time()),
        )
        conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (time.time(), session_id))
        conn.commit()
        conn.close()

    def list_sessions(self, user_id: str = "") -> list[dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT session_id, agent_id, title, "
            "(SELECT COUNT(*) FROM messages WHERE messages.session_id = sessions.session_id) as message_count "
            "FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def update_session_title(self, session_id: str, title: str) -> None:
        import time
        conn = self._get_conn()
        conn.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?", (title, time.time(), session_id))
        conn.commit()
        conn.close()

    def delete_session(self, session_id: str) -> bool:
        conn = self._get_conn()
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        deleted = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,)).rowcount
        conn.commit()
        conn.close()
        return deleted > 0


# ── Run Service ─────────────────────────────────────────────────────────────

class RunService:
    """Manages run execution with observability."""

    def __init__(self, agent_service: AgentService, tool_service: ToolService) -> None:
        self.agent_service = agent_service
        self.tool_service = tool_service
        self._runs: dict[str, dict[str, Any]] = {}

    def create_run(self, session_id: str, agent_id: str, message: str) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        run = {
            "run_id": run_id,
            "session_id": session_id,
            "agent_id": agent_id,
            "message": message,
            "status": "created",
            "created_at": __import__("time").time(),
        }
        self._runs[run_id] = run
        return run

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._runs.get(run_id)

    def list_runs(self, session_id: str = "") -> list[dict[str, Any]]:
        runs = list(self._runs.values())
        if session_id:
            runs = [r for r in runs if r.get("session_id") == session_id]
        return runs

    def update_run_status(self, run_id: str, status: str) -> None:
        run = self._runs.get(run_id)
        if run:
            run["status"] = status
