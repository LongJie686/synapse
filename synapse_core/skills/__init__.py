"""Custom Skills system - Hermes-style user-defined agents."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from synapse_core.agent import AgentDefinition
from synapse_core.llm import LLMConfig


class SkillDefinition:
    """A user-defined skill that creates a specialized agent."""

    def __init__(
        self,
        name: str,
        display_name: str = "",
        description: str = "",
        tools: list[str] | None = None,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        source_path: str = "",
    ) -> None:
        self.name = name
        self.display_name = display_name or name.replace("-", " ").replace("_", " ").title()
        self.description = description
        self.tools = tools or ["calculator"]
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.source_path = source_path

    def to_agent(self, llm: LLMConfig) -> AgentDefinition:
        """Convert skill to an AgentDefinition for the registry."""
        agent_llm = llm.model_copy()
        agent_llm.temperature = self.temperature
        agent_llm.max_tokens = self.max_tokens

        return AgentDefinition(
            id=f"skill-{self.name}",
            name=self.display_name,
            role=self.description,
            goal=self.description,
            backstory=self.system_prompt,
            llm=agent_llm,
            tools=self.tools,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": f"skill-{self.name}",
            "name": self.display_name,
            "description": self.description,
            "tools": self.tools,
            "source": "skill",
            "source_path": self.source_path,
        }


class SkillRegistry:
    """Registry for user-defined skills."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}

    def register(self, skill: SkillDefinition) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillDefinition | None:
        return self._skills.get(name)

    def list_all(self) -> list[SkillDefinition]:
        return list(self._skills.values())

    def remove(self, name: str) -> None:
        self._skills.pop(name, None)


def load_skill_from_yaml(path: Path | str) -> SkillDefinition:
    """Load a skill definition from a YAML file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "name" not in data:
        raise ValueError(f"Invalid skill file: {path} - missing 'name' field")

    return SkillDefinition(
        name=data["name"],
        display_name=data.get("display_name", ""),
        description=data.get("description", ""),
        tools=data.get("tools", ["calculator"]),
        system_prompt=data.get("system_prompt", ""),
        temperature=data.get("temperature", 0.7),
        max_tokens=data.get("max_tokens", 4096),
        source_path=str(path),
    )


def load_skills_from_directory(directory: Path | str) -> list[SkillDefinition]:
    """Load all skill YAML files from a directory."""
    directory = Path(directory)
    skills: list[SkillDefinition] = []

    if not directory.exists():
        return skills

    for yaml_file in sorted(directory.glob("*.yaml")):
        try:
            skill = load_skill_from_yaml(yaml_file)
            skills.append(skill)
        except Exception as e:
            print(f"Warning: Failed to load skill from {yaml_file}: {e}")

    for yaml_file in sorted(directory.glob("*.yml")):
        try:
            skill = load_skill_from_yaml(yaml_file)
            skills.append(skill)
        except Exception as e:
            print(f"Warning: Failed to load skill from {yaml_file}: {e}")

    return skills
