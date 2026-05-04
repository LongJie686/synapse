"""LLM provider implementations."""

from __future__ import annotations

from typing import Any, AsyncIterator

from . import BaseLLMProvider, LLMConfig, LLMMessage, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """OpenAI and Azure OpenAI provider."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client: Any = None
        self.last_usage: dict[str, int] = {}
        self.last_tool_calls: list[dict[str, Any]] = []

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

    def _extra_params(self) -> dict[str, Any]:
        """Build optional generation params from config."""
        params: dict[str, Any] = {}
        if self.config.frequency_penalty:
            params["frequency_penalty"] = self.config.frequency_penalty
        if self.config.presence_penalty:
            params["presence_penalty"] = self.config.presence_penalty
        if self.config.stop:
            params["stop"] = self.config.stop
        return params

    async def invoke(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        client = self._get_client()
        serialized = []
        for m in messages:
            entry: dict[str, Any] = {"role": m.role}
            if isinstance(m.content, list):
                # Convert Anthropic-style content blocks to OpenAI format
                parts = []
                for block in m.content:
                    if block.get("type") == "text":
                        parts.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image":
                        src = block.get("source", {})
                        if src.get("type") == "base64":
                            parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{src['media_type']};base64,{src['data']}"},
                            })
                entry["content"] = parts
            else:
                entry["content"] = m.content
            if m.name:
                entry["name"] = m.name
            serialized.append(entry)

        api_kwargs = {**self._extra_params(), **kwargs}
        response = await client.chat.completions.create(
            model=self.config.model,
            messages=serialized,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **api_kwargs,
        )
        choice = response.choices[0]
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }
        self.last_usage = usage
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
            usage=usage,
            model=response.model,
        )

    async def stream(self, messages: list[LLMMessage], **kwargs: Any) -> AsyncIterator[str]:
        client = self._get_client()
        self.last_usage = {}
        self.last_tool_calls = []
        tools = kwargs.pop("tools", None)
        api_kwargs = {**self._extra_params(), **kwargs}
        create_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            **api_kwargs,
        }
        if tools:
            create_kwargs["tools"] = tools
        stream = await client.chat.completions.create(**create_kwargs)

        # Accumulate streaming tool calls (OpenAI sends them incrementally)
        tool_call_accum: dict[int, dict[str, str]] = {}

        async for chunk in stream:
            # Capture usage from the final chunk
            if chunk.usage:
                self.last_usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens or 0,
                    "completion_tokens": chunk.usage.completion_tokens or 0,
                    "total_tokens": chunk.usage.total_tokens or 0,
                }
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
                # Accumulate tool call fragments
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_call_accum:
                            tool_call_accum[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc.id:
                            tool_call_accum[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_call_accum[idx]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_call_accum[idx]["arguments"] += tc.function.arguments

        # Finalize native tool calls
        if tool_call_accum:
            import json as _json
            for idx in sorted(tool_call_accum.keys()):
                tc_data = tool_call_accum[idx]
                try:
                    args = _json.loads(tc_data["arguments"])
                except _json.JSONDecodeError:
                    args = {"expression": tc_data["arguments"]}
                self.last_tool_calls.append({
                    "name": tc_data["name"],
                    "arguments": args,
                })


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client: Any = None
        self.last_usage: dict[str, int] = {}

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic as _anthropic

            kwargs: dict[str, Any] = {}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = _anthropic.AsyncAnthropic(**kwargs)
        return self._client

    @staticmethod
    def _not_given() -> Any:
        import anthropic as _anthropic
        return _anthropic.NOT_GIVEN

    def _extra_params(self) -> dict[str, Any]:
        """Build optional generation params from config."""
        params: dict[str, Any] = {}
        if self.config.stop:
            params["stop_sequences"] = self.config.stop
        return params

    async def invoke(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        client = self._get_client()
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content if isinstance(m.content, str) else str(m.content)
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        kwargs.pop("tools", None)
        api_kwargs = {**self._extra_params(), **kwargs}
        response = await client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p if self.config.top_p < 1.0 else self._not_given(),
            system=system_msg if system_msg else self._not_given(),
            messages=chat_messages,
            **api_kwargs,
        )
        text_content = ""
        for block in response.content:
            if hasattr(block, "text"):
                text_content += block.text

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }
        self.last_usage = usage
        return LLMResponse(
            content=text_content,
            usage=usage,
            model=response.model,
        )

    async def stream(self, messages: list[LLMMessage], **kwargs: Any) -> AsyncIterator[str]:
        client = self._get_client()
        self.last_usage = {}
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content if isinstance(m.content, str) else str(m.content)
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        kwargs.pop("tools", None)
        api_kwargs = {**self._extra_params(), **kwargs}
        async with client.messages.stream(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p if self.config.top_p < 1.0 else self._not_given(),
            system=system_msg if system_msg else self._not_given(),
            messages=chat_messages,
            **api_kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield text
            # Capture usage from the completed stream
            final = await stream.get_final_message()
            self.last_usage = {
                "prompt_tokens": final.usage.input_tokens,
                "completion_tokens": final.usage.output_tokens,
                "total_tokens": final.usage.input_tokens + final.usage.output_tokens,
            }


def create_provider(config: LLMConfig) -> BaseLLMProvider:
    """Factory function to create the appropriate LLM provider."""
    if config.provider in ("openai", "azure-openai"):
        return OpenAIProvider(config)
    elif config.provider == "anthropic":
        return AnthropicProvider(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
