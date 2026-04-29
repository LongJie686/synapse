"""Tool system with registry and safety wrapper."""

from __future__ import annotations

import time
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class ToolSafetyConfig(BaseModel):
    requires_approval: bool = False
    rate_limit_max_calls: int | None = None
    rate_limit_window_ms: int = 60000
    timeout_ms: int = 30000
    input_validation: bool = True
    sandboxed: bool = False


class ToolDefinition(BaseModel):
    name: str
    description: str
    category: str = "custom"
    parameters: list[ToolParameter] = Field(default_factory=list)
    safety: ToolSafetyConfig = Field(default_factory=ToolSafetyConfig)


class ToolResult(BaseModel):
    tool_name: str
    input: dict[str, Any]
    output: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0


ToolHandler = Callable[..., Awaitable[Any]]


class ToolRegistry:
    """Registry and executor for tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(
        self,
        definition: ToolDefinition,
        handler: ToolHandler,
    ) -> None:
        self._tools[definition.name] = definition
        self._handlers[definition.name] = handler

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]

    def list_all(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        if name not in self._handlers:
            return ToolResult(tool_name=name, input=kwargs, error=f"Tool not found: {name}")

        start = time.monotonic()
        try:
            result = await self._handlers[name](**kwargs)
            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(tool_name=name, input=kwargs, output=result, execution_time_ms=elapsed)
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(tool_name=name, input=kwargs, error=str(e), execution_time_ms=elapsed)
