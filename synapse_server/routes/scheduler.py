"""Scheduled task management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

_scheduler = None


def configure(scheduler) -> None:
    global _scheduler
    _scheduler = scheduler


class TaskCreateRequest(BaseModel):
    name: str
    message: str
    cron: str
    agent_id: str = "general-assistant"
    description: str = ""


class TaskUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    message: str | None = None
    cron: str | None = None
    agent_id: str | None = None
    enabled: bool | None = None


@router.get("/scheduler/tasks")
async def list_tasks() -> list[dict]:
    if not _scheduler:
        return []
    return [t.to_dict() for t in _scheduler.list_tasks()]


@router.post("/scheduler/tasks")
async def create_task(req: TaskCreateRequest) -> dict:
    if not _scheduler:
        raise HTTPException(status_code=500, detail="Scheduler not available")
    task = _scheduler.create_task(
        name=req.name,
        message=req.message,
        cron=req.cron,
        agent_id=req.agent_id,
        description=req.description,
    )
    return task.to_dict()


@router.get("/scheduler/tasks/{task_id}")
async def get_task(task_id: str) -> dict:
    if not _scheduler:
        raise HTTPException(status_code=500, detail="Scheduler not available")
    task = _scheduler.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.to_dict()


@router.put("/scheduler/tasks/{task_id}")
async def update_task(task_id: str, req: TaskUpdateRequest) -> dict:
    if not _scheduler:
        raise HTTPException(status_code=500, detail="Scheduler not available")
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    if not _scheduler.update_task(task_id, **updates):
        raise HTTPException(status_code=404, detail="Task not found")
    task = _scheduler.get_task(task_id)
    return task.to_dict()


@router.delete("/scheduler/tasks/{task_id}")
async def delete_task(task_id: str) -> dict:
    if not _scheduler:
        raise HTTPException(status_code=500, detail="Scheduler not available")
    if not _scheduler.delete_task(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    return {"id": task_id, "status": "deleted"}


@router.post("/scheduler/tasks/{task_id}/run")
async def run_task(task_id: str) -> dict:
    """Manually trigger a scheduled task."""
    if not _scheduler:
        raise HTTPException(status_code=500, detail="Scheduler not available")
    task = _scheduler.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    _scheduler.record_run(task_id)
    return {"id": task_id, "status": "triggered", "message": task.message}
