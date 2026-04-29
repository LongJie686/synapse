"""LLM provider abstractions."""

from __future__ import annotations

from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel


class LLMConfig(BaseModel):
    provider: Literal["openai", "anthropic", "ollama", "azure-openai"] = "openai"
    model: str = "gpt-4o"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class LLMResponse(BaseModel):
    content: str
    tool_calls: list[dict[str, Any]] = []
    usage: dict[str, int] = {}
    model: str = ""


class BaseLLMProvider:
    """Abstract base for LLM providers."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    async def invoke(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        raise NotImplementedError

    async def stream(self, messages: list[LLMMessage], **kwargs: Any) -> AsyncIterator[str]:
        raise NotImplementedError
        # make this an async generator
        yield ""  # type: ignore[unreachable]
