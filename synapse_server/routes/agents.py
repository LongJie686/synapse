"""Agent management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

_agent_service = None


def configure(agent_service) -> None:
    global _agent_service
    _agent_service = agent_service


class AgentCreateRequest(BaseModel):
    id: str
    name: str
    role: str
    goal: str = ""
    backstory: str = ""
    model: str = "gpt-4o-mini"
    tools: list[str] = []
    max_iterations: int = 10


@router.get("/agents")
async def list_agents() -> list[dict]:
    if not _agent_service:
        return []
    agents = _agent_service.list_agents()
    skills = _agent_service.skill_registry.list_all() if hasattr(_agent_service, "skill_registry") else []
    skill_names = {f"skill-{s.name}" for s in skills}

    result = []
    for a in agents:
        entry = {
            "id": a.id,
            "name": a.name,
            "role": a.role,
            "goal": a.goal,
            "tools": a.tools,
            "is_skill": a.id in skill_names,
        }
        result.append(entry)
    return result


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str) -> dict:
    if not _agent_service:
        raise HTTPException(status_code=404, detail="Agent service not available")
    agent = _agent_service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return {
        "id": agent.id,
        "name": agent.name,
        "role": agent.role,
        "goal": agent.goal,
        "backstory": agent.backstory,
        "tools": agent.tools,
        "max_iterations": agent.max_iterations,
    }


@router.post("/agents")
async def create_agent(request: AgentCreateRequest) -> dict:
    from synapse_core.agent import AgentDefinition
    from synapse_core.llm import LLMConfig

    if not _agent_service:
        raise HTTPException(status_code=500, detail="Agent service not available")

    agent = AgentDefinition(
        id=request.id,
        name=request.name,
        role=request.role,
        goal=request.goal,
        backstory=request.backstory,
        llm=LLMConfig(model=request.model),
        tools=request.tools,
        max_iterations=request.max_iterations,
    )
    _agent_service.register_agent(agent)
    return {"id": agent.id, "name": agent.name, "status": "created"}


@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str) -> dict:
    if not _agent_service:
        raise HTTPException(status_code=500, detail="Agent service not available")
    if not _agent_service.remove_agent(agent_id):
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return {"id": agent_id, "status": "deleted"}
