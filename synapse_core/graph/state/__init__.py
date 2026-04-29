"""Graph state definition for LangGraph orchestration."""

from __future__ import annotations

from typing import Any, Annotated

from pydantic import BaseModel, Field


def _replace(old: Any, new: Any) -> Any:
    return new


def _append_list(old: list[Any], new: list[Any]) -> list[Any]:
    return old + new


def _merge_dict(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    return {**old, **new}


class Message(BaseModel):
    role: str
    content: str
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class RunState(BaseModel):
    """State that flows through the graph nodes."""

    messages: list[Message] = Field(default_factory=list)
    current_agent: str = "default"
    next_agents: list[str] = Field(default_factory=list)
    tool_results: list[ToolCallRecord] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    session_id: str = ""
    user_id: str = ""
    run_id: str = ""
    step_count: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    metadata: dict[str, Any] = Field(default_factory=dict)
