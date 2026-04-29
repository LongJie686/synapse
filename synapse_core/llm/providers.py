"""LLM provider implementations."""

from __future__ import annotations

from typing import Any, AsyncIterator

from . import BaseLLMProvider, LLMConfig, LLMMessage, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """OpenAI and Azure OpenAI provider."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            kwargs: dict[str, Any] = {}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def invoke(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.config.model,
            messages=[m.model_dump(exclude_none=True) for m in messages],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs,
        )
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=[
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in (choice.message.tool_calls or [])
            ],
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            model=response.model,
        )

    async def stream(self, messages: list[LLMMessage], **kwargs: Any) -> AsyncIterator[str]:
        client = self._get_client()
        stream = await client.chat.completions.create(
            model=self.config.model,
            messages=[m.model_dump(exclude_none=True) for m in messages],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from anthropic import AsyncAnthropic

            kwargs: dict[str, Any] = {}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = AsyncAnthropic(**kwargs)
        return self._client

    async def invoke(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        client = self._get_client()
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        response = await client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system_msg if system_msg else anthropic.NOT_GIVEN,  # type: ignore[name-defined]  # noqa: F821
            messages=chat_messages,
            **kwargs,
        )
        text_content = ""
        for block in response.content:
            if hasattr(block, "text"):
                text_content += block.text

        return LLMResponse(
            content=text_content,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            model=response.model,
        )

    async def stream(self, messages: list[LLMMessage], **kwargs: Any) -> AsyncIterator[str]:
        client = self._get_client()
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        async with client.messages.stream(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system_msg if system_msg else anthropic.NOT_GIVEN,  # type: ignore[name-defined]  # noqa: F821
            messages=chat_messages,
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield text


def create_provider(config: LLMConfig) -> BaseLLMProvider:
    """Factory function to create the appropriate LLM provider."""
    match config.provider:
        case "openai" | "azure-openai":
            return OpenAIProvider(config)
        case "anthropic":
            return AnthropicProvider(config)
        case _:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
