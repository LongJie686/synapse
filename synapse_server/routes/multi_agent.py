"""Multi-agent pattern endpoints with streaming."""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from synapse_core.agent import AgentDefinition
from synapse_core.llm import LLMConfig
from synapse_core.tools import ToolRegistry
from synapse_core.observability import get_hub

router = APIRouter()

_agent_service = None
_tool_service = None


def configure(agent_service, tool_service) -> None:
    global _agent_service, _tool_service
    _agent_service = agent_service
    _tool_service = tool_service


class MultiAgentRequest(BaseModel):
    message: str
    pattern: str = "supervisor"  # supervisor | parallel | collaboration
    agent_ids: list[str] | None = None


def _get_llm() -> LLMConfig:
    import os
    return LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "anthropic"),
        model=os.getenv("LLM_MODEL", "glm-4-flash"),
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
        temperature=0.7,
        max_tokens=4096,
    )


def _get_agents(agent_ids: list[str] | None) -> list[AgentDefinition]:
    llm = _get_llm()
    if _agent_service:
        if agent_ids:
            agents = []
            for aid in agent_ids:
                a = _agent_service.get_agent(aid)
                if a:
                    agents.append(a)
            if agents:
                return agents
        return _agent_service.list_agents()[:3]
    return [
        AgentDefinition(id="general-assistant", name="Synapse Assistant", role="General-purpose AI assistant", goal="Help users", backstory="You are a helpful assistant.", llm=llm, tools=["calculator"]),
        AgentDefinition(id="code-expert", name="Code Expert", role="Software engineering specialist", goal="Help with code", backstory="You are a coding expert.", llm=llm.model_copy(), tools=["calculator"]),
        AgentDefinition(id="data-analyst", name="Data Analyst", role="Data analysis specialist", goal="Analyze data", backstory="You are a data expert.", llm=llm.model_copy(), tools=["calculator"]),
    ]


def _get_tool_registry() -> ToolRegistry:
    if _tool_service:
        return _tool_service.registry
    registry = ToolRegistry()
    from synapse_core.tools.builtin import register_calculator
    register_calculator(registry)
    return registry


@router.post("/multi-agent/stream")
async def stream_multi_agent(request: MultiAgentRequest) -> StreamingResponse:
    hub = get_hub()
    run_id = str(uuid.uuid4())
    agents = _get_agents(request.agent_ids)
    registry = _get_tool_registry()
    pattern = request.pattern.lower()

    hub.track_run_start(run_id, f"multi:{pattern}", request.message)

    async def ndjson_generator():
        last_token_usage = None
        try:
            if pattern == "supervisor":
                from synapse_core.patterns import SupervisorPattern
                supervisor_agent = agents[0]
                workers = agents[1:] if len(agents) > 1 else agents
                pat = SupervisorPattern(supervisor=supervisor_agent, workers=workers, tool_registry=registry)
                event_stream = pat.run(request.message, run_id=run_id)

            elif pattern == "parallel":
                from synapse_core.patterns import ParallelPattern
                pat = ParallelPattern(agents=agents, tool_registry=registry)
                event_stream = pat.run(request.message, run_id=run_id)

            elif pattern == "collaboration":
                from synapse_core.patterns import CollaborationPattern
                pat = CollaborationPattern(agents=agents, max_rounds=2)
                event_stream = pat.run(request.message, run_id=run_id)

            else:
                yield json.dumps({"type": "run:error", "data": {"runId": run_id, "error": f"Unknown pattern: {pattern}"}}, ensure_ascii=False) + "\n"
                return

            async for event in event_stream:
                if event.type == "run:end":
                    last_token_usage = event.data.get("tokenUsage", event.data.get("token_usage"))
                line = json.dumps({"type": event.type, "data": event.data}, ensure_ascii=False)
                yield line + "\n"

        except Exception:
            logger.exception("Multi-agent run %s failed", run_id)
            yield json.dumps({"type": "run:error", "data": {"runId": run_id, "error": "Internal error occurred"}}, ensure_ascii=False) + "\n"
        finally:
            hub.track_run_end(run_id, token_usage=last_token_usage)

    return StreamingResponse(
        ndjson_generator(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/multi-agent/patterns")
async def list_patterns() -> dict:
    return {
        "patterns": [
            {"id": "supervisor", "name": "Supervisor", "description": "Central coordinator delegates tasks to specialists"},
            {"id": "parallel", "name": "Parallel", "description": "Multiple agents work simultaneously on subtasks"},
            {"id": "collaboration", "name": "Collaboration", "description": "Agents discuss in rounds to refine answers"},
        ]
    }
