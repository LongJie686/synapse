"""Run execution endpoints with streaming."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from synapse_core.agent import AgentDefinition
from synapse_core.llm import LLMConfig
from synapse_core.tools import ToolRegistry
from synapse_core.tools.builtin import register_calculator
from synapse_core.graph import GraphBuilder
from synapse_core.observability import get_hub

router = APIRouter()


class RunRequest(BaseModel):
    message: str
    agent_id: str | None = None
    session_id: str | None = None
    images: list[dict] | None = None  # [{"mime_type": "image/png", "data": "base64..."}]


_agent_service = None
_tool_service = None
_session_service = None


def configure(
    agent_service,
    tool_service,
    session_service,
) -> None:
    global _agent_service, _tool_service, _session_service
    _agent_service = agent_service
    _tool_service = tool_service
    _session_service = session_service


def _get_agent(agent_id: str) -> AgentDefinition:
    if _agent_service:
        agent = _agent_service.get_agent(agent_id)
        if agent:
            return agent
    return _get_default_agent()


def _get_default_agent() -> AgentDefinition:
    if _agent_service:
        agent = _agent_service.get_agent("general-assistant")
        if agent:
            return agent
    return AgentDefinition(
        id="default",
        name="Synapse Assistant",
        role="General-purpose AI assistant",
        goal="Help users accomplish tasks using available tools",
        backstory="You are a helpful and capable AI assistant with access to tools.",
        llm=LLMConfig(),
        tools=["calculator"],
    )


def _get_tool_registry() -> ToolRegistry:
    if _tool_service:
        return _tool_service.registry
    registry = ToolRegistry()
    register_calculator(registry)
    return registry


def _ensure_session(session_id: str | None, agent_id: str) -> str:
    """Get or create a session, return session_id."""
    if not _session_service:
        return session_id or str(uuid.uuid4())

    if session_id:
        existing = _session_service.get_session(session_id)
        if existing:
            return session_id

    result = _session_service.create_session(agent_id=agent_id)
    return result["session_id"]


async def _generate_title(message: str, response: str) -> str:
    """Use LLM to generate a short conversation title."""
    try:
        from synapse_core.llm.providers import create_provider
        from synapse_core.llm import LLMConfig
        import os

        config = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "anthropic"),
            model=os.getenv("LLM_MODEL", "glm-5-turbo"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url=os.getenv("ANTHROPIC_BASE_URL"),
            temperature=0.3,
            max_tokens=50,
        )
        provider = create_provider(config)
        from synapse_core.llm import LLMMessage
        resp = await provider.invoke([
            LLMMessage(role="system", content="Generate a very short title (max 10 chars) for this conversation. Output ONLY the title, no quotes, no explanation. Use the same language as the user."),
            LLMMessage(role="user", content=f"User: {message}\nAssistant: {response[:200]}"),
        ])
        title = resp.content.strip().strip('"').strip("'")
        return title[:30] if title else ""
    except Exception:
        return ""


@router.post("/runs")
async def create_run(request: RunRequest) -> dict:
    hub = get_hub()
    run_id = str(uuid.uuid4())
    agent_id = request.agent_id or "general-assistant"
    session_id = _ensure_session(request.session_id, agent_id)

    if _session_service:
        _session_service.add_message(session_id, "user", request.message)

    hub.track_run_start(run_id, agent_id, request.message)

    return {
        "run_id": run_id,
        "session_id": session_id,
        "agent_id": agent_id,
        "status": "created",
    }


@router.post("/runs/stream")
async def stream_run(request: RunRequest) -> StreamingResponse:
    agent_id = request.agent_id or "general-assistant"
    agent = _get_agent(agent_id)
    registry = _get_tool_registry()
    builder = GraphBuilder(agent, registry)
    hub = get_hub()

    session_id = _ensure_session(request.session_id, agent_id)

    run_id = str(uuid.uuid4())
    hub.track_run_start(run_id, agent.id, request.message)

    if _session_service:
        _session_service.add_message(session_id, "user", request.message)

    final_content = ""

    async def ndjson_generator():
        nonlocal final_content
        last_token_usage = {}
        try:
            async for event in builder.run(request.message, images=request.images):
                if event.type == "agent:message" and event.data.get("content"):
                    final_content = event.data["content"]
                if event.type == "run:end" and event.data.get("tokenUsage"):
                    last_token_usage = event.data["tokenUsage"]

                line = json.dumps(
                    {"type": event.type, "data": event.data},
                    ensure_ascii=False,
                )
                yield line + "\n"
        finally:
            if _session_service and final_content:
                _session_service.add_message(session_id, "assistant", final_content)

                # Auto-generate title for new sessions
                session = _session_service.get_session(session_id)
                if session and (not session.get("title") or session["title"].startswith("Session ")):
                    title = await _generate_title(request.message, final_content)
                    if title:
                        _session_service.update_session_title(session_id, title)
                        yield json.dumps({"type": "session:title_update", "data": {"session_id": session_id, "title": title}}, ensure_ascii=False) + "\n"

            hub.track_run_end(run_id, token_usage=last_token_usage)

    return StreamingResponse(
        ndjson_generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    return {"run_id": run_id, "status": "completed"}
