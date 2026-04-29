"""Agent definition and registry."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from synapse_core.llm import LLMConfig


class GuardrailConfig(BaseModel):
    input_filters: list[str] = Field(default_factory=lambda: ["prompt-injection"])
    output_filters: list[str] = Field(default_factory=list)
    max_tool_calls_per_run: int = 20
    max_tokens_per_run: int = 100_000
    blocked_tools: list[str] = Field(default_factory=list)
    blocked_topics: list[str] = Field(default_factory=list)


class MemoryConfig(BaseModel):
    short_term_type: str = "summary-buffer"
    long_term_enabled: bool = True
    long_term_provider: str = "chroma"
    auto_save_importance: float = 0.6
    max_context_tokens: int = 8000


class AgentDefinition(BaseModel):
    """Defines an agent's identity, capabilities, and constraints."""

    id: str
    name: str
    role: str
    goal: str
    backstory: str
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tools: list[str] = Field(default_factory=list)
    max_iterations: int = 10
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRegistry:
    """Registry for agent definitions."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentDefinition] = {}

    def register(self, agent: AgentDefinition) -> None:
        self._agents[agent.id] = agent

    def get(self, agent_id: str) -> AgentDefinition:
        if agent_id not in self._agents:
            raise KeyError(f"Agent not found: {agent_id}")
        return self._agents[agent_id]

    def list_all(self) -> list[AgentDefinition]:
        return list(self._agents.values())

    def remove(self, agent_id: str) -> None:
        if agent_id not in self._agents:
            raise KeyError(f"Agent not found: {agent_id}")
        del self._agents[agent_id]
