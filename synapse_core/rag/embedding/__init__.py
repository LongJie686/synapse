"""Embedding providers."""

from __future__ import annotations

from typing import Any

from synapse_core.rag import BaseEmbeddingProvider


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI text-embedding provider."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI
            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def embed(self, text: str) -> list[float]:
        client = self._get_client()
        response = await client.embeddings.create(input=text, model=self._model)
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        response = await client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]
