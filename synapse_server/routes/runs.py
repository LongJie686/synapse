"""Run execution endpoints with SSE streaming."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from synapse_core.agent import AgentDefinition
from synapse_core.llm import LLMConfig
from synapse_core.tools import ToolRegistry
from synapse_core.tools.builtin import register_calculator
from synapse_core.graph import GraphBuilder

router = APIRouter()


class RunRequest(BaseModel):
    message: str
    agent_id: str | None = None
    session_id: str | None = None


# Temporary default agent for Phase 1
def _get_default_agent() -> AgentDefinition:
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
    registry = ToolRegistry()
    register_calculator(registry)
    return registry


@router.post("/runs")
async def create_run(request: RunRequest) -> dict[str, str]:
    run_id = str(uuid.uuid4())
    return {"runId": run_id, "status": "started"}


@router.post("/runs/stream")
async def stream_run(request: RunRequest) -> EventSourceResponse:
    agent = _get_default_agent()
    registry = _get_tool_registry()
    builder = GraphBuilder(agent, registry)

    async def event_generator():
        async for event in builder.run(request.message):
            yield {
                "event": event.type,
                "data": json.dumps(event.data, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())
