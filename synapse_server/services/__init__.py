"""Business services for synapse server."""

from __future__ import annotations

import json
import uuid
from typing import Any

from synapse_core.agent import AgentDefinition, AgentRegistry
from synapse_core.llm import LLMConfig
from synapse_core.tools import ToolRegistry
from synapse_core.tools.builtin import register_all_builtin_tools
from synapse_core.memory import MemoryManager
from synapse_core.observability import get_hub


# ── Agent Service ───────────────────────────────────────────────────────────

class AgentService:
    """Manages agent definitions and execution."""

    def __init__(self) -> None:
        self.registry = AgentRegistry()
        self._setup_default_agents()

    def _setup_default_agents(self) -> None:
        self.registry.register(AgentDefinition(
            id="general-assistant",
            name="Synapse Assistant",
            role="General-purpose AI assistant",
            goal="Help users accomplish tasks using available tools",
            backstory="You are a helpful and capable AI assistant with access to tools.",
            llm=LLMConfig(),
            tools=["calculator", "web_search", "http_request", "sql_query"],
        ))
        self.registry.register(AgentDefinition(
            id="code-expert",
            name="Code Expert",
            role="Software engineering specialist",
            goal="Write, review, and debug code",
            backstory="You are an experienced software engineer with deep knowledge of multiple programming languages.",
            llm=LLMConfig(),
            tools=["calculator"],
        ))
        self.registry.register(AgentDefinition(
            id="data-analyst",
            name="Data Analyst",
            role="Data analysis and visualization specialist",
            goal="Help users analyze data and extract insights",
            backstory="You are a data analyst expert in SQL, statistics, and data visualization.",
            llm=LLMConfig(),
            tools=["calculator", "sql_query"],
        ))

    def get_agent(self, agent_id: str) -> AgentDefinition | None:
        return self.registry.get(agent_id)

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
    """Manages conversation sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def create_session(self, user_id: str = "", agent_id: str = "general-assistant") -> dict[str, Any]:
        session_id = str(uuid.uuid4())
        session = {
            "session_id": session_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "created_at": __import__("time").time(),
            "message_count": 0,
            "messages": [],
        }
        self._sessions[session_id] = session
        return {"session_id": session_id, "agent_id": agent_id}

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return self._sessions.get(session_id)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        session = self._sessions.get(session_id)
        if session:
            session["messages"].append({"role": role, "content": content})
            session["message_count"] += 1

    def list_sessions(self, user_id: str = "") -> list[dict[str, Any]]:
        sessions = list(self._sessions.values())
        if user_id:
            sessions = [s for s in sessions if s.get("user_id") == user_id]
        return [{"session_id": s["session_id"], "agent_id": s["agent_id"], "message_count": s["message_count"]} for s in sessions]

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False


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
