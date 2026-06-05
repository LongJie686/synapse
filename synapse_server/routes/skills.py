"""Skills management endpoints."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

router = APIRouter()

_skill_registry = None

_SKILLS_ROOT = Path(__file__).resolve().parent.parent.parent / "skills"

# Only allow alphanumeric, underscore, hyphen — no slashes or dots
_SAFE_SKILL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def configure(skill_registry) -> None:
    global _skill_registry
    _skill_registry = skill_registry


def _resolve_skill_path(skill_name: str) -> Path:
    """Validate skill name and return a path guaranteed within skills directory."""
    if not _SAFE_SKILL_NAME.match(skill_name):
        raise HTTPException(status_code=400, detail="Invalid skill name format")
    resolved = (_SKILLS_ROOT / f"{skill_name}.yaml").resolve()
    try:
        resolved.relative_to(_SKILLS_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid skill path")
    return resolved


class SkillCreateRequest(BaseModel):
    name: str
    display_name: str
    description: str = ""
    tools: list[str] = []
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not _SAFE_SKILL_NAME.match(v):
            raise ValueError("name must be 1-64 alphanumeric/underscore/hyphen characters")
        return v


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

    _SKILLS_ROOT.mkdir(exist_ok=True)
    yaml_path = _resolve_skill_path(req.name)

    if yaml_path.exists():
        raise HTTPException(status_code=409, detail=f"Skill '{req.name}' already exists")

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(skill_data, f, allow_unicode=True, default_flow_style=False)

    if _skill_registry:
        from synapse_core.skills import load_skill_from_yaml
        skill = load_skill_from_yaml(yaml_path)
        if skill:
            _skill_registry.register(skill)

    return {"name": req.name, "status": "created", "path": str(yaml_path)}


@router.delete("/skills/{skill_name}")
async def delete_skill(skill_name: str) -> dict:
    """Delete a skill by removing its YAML file."""
    yaml_path = _resolve_skill_path(skill_name)

    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")

    yaml_path.unlink()

    if _skill_registry:
        _skill_registry.unregister(skill_name)

    return {"name": skill_name, "status": "deleted"}


@router.post("/skills/reload")
async def reload_skills() -> dict:
    """Reload all skills from the skills directory."""
    if not _skill_registry:
        raise HTTPException(status_code=500, detail="Skill registry not available")

    count = _skill_registry.load_from_directory(_SKILLS_ROOT)
    return {"status": "reloaded", "count": count}
