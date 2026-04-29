"""Session management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

_session_service = None


def configure(session_service) -> None:
    global _session_service
    _session_service = session_service


class SessionCreateRequest(BaseModel):
    agent_id: str = "general-assistant"
    user_id: str = ""
    title: str = ""


@router.get("/sessions")
async def list_sessions(user_id: str = "") -> list[dict]:
    if not _session_service:
        return []
    return _session_service.list_sessions(user_id)


@router.post("/sessions")
async def create_session(request: SessionCreateRequest) -> dict:
    if not _session_service:
        raise HTTPException(status_code=500, detail="Session service not available")
    return _session_service.create_session(
        user_id=request.user_id,
        agent_id=request.agent_id,
        title=request.title,
    )


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    if not _session_service:
        raise HTTPException(status_code=500, detail="Session service not available")
    session = _session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    messages = _session_service.get_messages(session_id)
    return {
        "session_id": session["session_id"],
        "agent_id": session["agent_id"],
        "title": session.get("title", ""),
        "message_count": len(messages),
        "messages": messages[-100:],
    }


@router.patch("/sessions/{session_id}")
async def update_session(session_id: str, body: dict) -> dict:
    if not _session_service:
        raise HTTPException(status_code=500, detail="Session service not available")
    title = body.get("title")
    if title:
        _session_service.update_session_title(session_id, title)
    return {"session_id": session_id, "status": "updated"}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    if not _session_service:
        raise HTTPException(status_code=500, detail="Session service not available")
    if not _session_service.delete_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"session_id": session_id, "status": "deleted"}
