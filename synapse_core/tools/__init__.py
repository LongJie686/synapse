"""Tool system with registry, safety wrapper, fallback chains, and rate limiting."""

from __future__ import annotations

import asyncio
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
    fallback_used: str | None = None


ToolHandler = Callable[..., Awaitable[Any]]


class _RateLimiter:
    """Token-bucket rate limiter for per-tool call limiting."""

    def __init__(self, max_calls: int, window_ms: int) -> None:
        self._max_calls = max_calls
        self._window_ms = window_ms
        self._timestamps: list[float] = []

    def allow(self) -> bool:
        now = time.monotonic()
        cutoff = now - (self._window_ms / 1000.0)
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        if len(self._timestamps) >= self._max_calls:
            return False
        self._timestamps.append(now)
        return True


class ToolRegistry:
    """Registry and executor for tools with fallback chains and rate limiting."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, ToolHandler] = {}
        self._fallback_chains: dict[str, list[str]] = {}
        self._rate_limiters: dict[str, _RateLimiter] = {}

    def register(
        self,
        definition: ToolDefinition,
        handler: ToolHandler,
        fallback_chain: list[str] | None = None,
    ) -> None:
        self._tools[definition.name] = definition
        self._handlers[definition.name] = handler
        if fallback_chain:
            self._fallback_chains[definition.name] = fallback_chain
        if definition.safety.rate_limit_max_calls:
            self._rate_limiters[definition.name] = _RateLimiter(
                max_calls=definition.safety.rate_limit_max_calls,
                window_ms=definition.safety.rate_limit_window_ms,
            )

    def set_fallback_chain(self, tool_name: str, chain: list[str]) -> None:
        """Define fallback tools to try if the primary tool fails."""
        self._fallback_chains[tool_name] = chain

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]

    def list_all(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        if name not in self._handlers:
            return ToolResult(tool_name=name, input=kwargs, error=f"Tool not found: {name}")

        # Rate limit check
        limiter = self._rate_limiters.get(name)
        if limiter and not limiter.allow():
            fallback = self._fallback_chains.get(name, [])
            if fallback:
                return await self._execute_fallback(name, kwargs, "Rate limit exceeded")
            return ToolResult(
                tool_name=name, input=kwargs,
                error=f"Rate limit exceeded for {name}. Try again later.",
            )

        # Primary execution
        start = time.monotonic()
        try:
            result = await self._handlers[name](**kwargs)
            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(tool_name=name, input=kwargs, output=result, execution_time_ms=elapsed)
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            # Try fallback chain
            fallback = self._fallback_chains.get(name, [])
            if fallback:
                return await self._execute_fallback(name, kwargs, str(e))
            return ToolResult(tool_name=name, input=kwargs, error=str(e), execution_time_ms=elapsed)

    async def _execute_fallback(
        self,
        primary_name: str,
        kwargs: dict[str, Any],
        primary_error: str,
    ) -> ToolResult:
        """Try each fallback tool in order until one succeeds."""
        for fb_name in self._fallback_chains.get(primary_name, []):
            if fb_name not in self._handlers:
                continue
            fb_limiter = self._rate_limiters.get(fb_name)
            if fb_limiter and not fb_limiter.allow():
                continue
            start = time.monotonic()
            try:
                result = await self._handlers[fb_name](**kwargs)
                elapsed = (time.monotonic() - start) * 1000
                return ToolResult(
                    tool_name=fb_name, input=kwargs, output=result,
                    execution_time_ms=elapsed, fallback_used=fb_name,
                )
            except Exception:
                continue

        return ToolResult(
            tool_name=primary_name, input=kwargs,
            error=f"Primary failed ({primary_error}) and all fallbacks exhausted",
        )

    def to_openai_tools(self, tool_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Convert registered tools to OpenAI Function Calling format.

        Args:
            tool_names: Optional filter. If None, exports all tools.
        """
        tools = []
        names = tool_names or list(self._tools.keys())
        for name in names:
            tool_def = self._tools.get(name)
            if not tool_def:
                continue
            properties: dict[str, Any] = {}
            required: list[str] = []
            for p in tool_def.parameters:
                properties[p.name] = {
                    "type": p.type,
                    "description": p.description,
                }
                if p.required:
                    required.append(p.name)
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return tools
