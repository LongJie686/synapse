"""Skills management endpoints."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

_skill_registry = None


def configure(skill_registry) -> None:
    global _skill_registry
    _skill_registry = skill_registry


class SkillCreateRequest(BaseModel):
    name: str
    display_name: str
    description: str = ""
    tools: list[str] = []
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096


@router.get("/skills")
async def list_skills() -> list[dict]:
    if not _skill_registry:
        return []
    return [
        {
            "name": s.name,
            "display_name": s.display_name,
            "description": s.description,
            "tools": s.tools,
            "temperature": s.temperature,
            "max_tokens": s.max_tokens,
        }
        for s in _skill_registry.list_all()
    ]


@router.post("/skills")
async def create_skill(req: SkillCreateRequest) -> dict:
    """Create a new skill and save as YAML file."""
    import yaml

    skill_data = {
        "name": req.name,
        "display_name": req.display_name,
        "description": req.description,
        "tools": req.tools,
        "system_prompt": req.system_prompt,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }

    project_root = Path(__file__).resolve().parent.parent.parent
    skills_dir = project_root / "skills"
    skills_dir.mkdir(exist_ok=True)
    yaml_path = skills_dir / f"{req.name}.yaml"

    if yaml_path.exists():
        raise HTTPException(status_code=409, detail=f"Skill '{req.name}' already exists")

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(skill_data, f, allow_unicode=True, default_flow_style=False)

    # Reload skills into registry
    if _skill_registry:
        from synapse_core.skills import load_skill_from_yaml
        skill = load_skill_from_yaml(yaml_path)
        if skill:
            _skill_registry.register(skill)

    return {"name": req.name, "status": "created", "path": str(yaml_path)}


@router.delete("/skills/{skill_name}")
async def delete_skill(skill_name: str) -> dict:
    """Delete a skill by removing its YAML file."""
    project_root = Path(__file__).resolve().parent.parent.parent
    yaml_path = project_root / "skills" / f"{skill_name}.yaml"

    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")

    os.remove(yaml_path)

    if _skill_registry:
        _skill_registry.unregister(skill_name)

    return {"name": skill_name, "status": "deleted"}


@router.post("/skills/reload")
async def reload_skills() -> dict:
    """Reload all skills from the skills directory."""
    if not _skill_registry:
        raise HTTPException(status_code=500, detail="Skill registry not available")

    project_root = Path(__file__).resolve().parent.parent.parent
    skills_dir = project_root / "skills"
    count = _skill_registry.load_from_directory(skills_dir)
    return {"status": "reloaded", "count": count}
