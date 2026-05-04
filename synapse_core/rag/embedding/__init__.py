"""Embedding providers: local (BGE) and API-based (OpenAI-compatible)."""

from __future__ import annotations

from typing import Any

from synapse_core.rag import BaseEmbeddingProvider


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI-compatible text-embedding provider."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None, base_url: str | None = None) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI
            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
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


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """
    Local embedding using BAAI/bge-small-zh-v1.5 via PyTorch + transformers.
    Free, offline, Chinese-optimized, 512 dimensions.
    """

    _MODEL_ID = "BAAI/bge-small-zh-v1.5"

    def __init__(self, model_path: str | None = None, device: str = "cpu") -> None:
        self._model_path = model_path or self._find_local_path()
        self._device = device
        self._tokenizer: Any = None
        self._model: Any = None

    def _find_local_path(self) -> str:
        import os
        cache = os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{self._MODEL_ID.replace('/', '--')}/snapshots/local"
        )
        if os.path.isdir(cache) and os.path.isfile(os.path.join(cache, "config.json")):
            return cache
        return self._MODEL_ID

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModel

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._model = AutoModel.from_pretrained(self._model_path)
        self._model.to(self._device)
        self._model.eval()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        import torch
        import numpy as np

        self._load_model()
        inputs = self._tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = (embeddings / norms).tolist()
        return normalized

    async def embed(self, text: str) -> list[float]:
        return self._encode([text])[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        batch_size = 32
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            results.extend(self._encode(texts[i:i + batch_size]))
        return results


# Domestic embedding model profiles (OpenAI-compatible API)
EMBEDDING_PROFILES = {
    "openai-small": {
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "provider": "openai",
    },
    "openai-large": {
        "model": "text-embedding-3-large",
        "dimensions": 3072,
        "provider": "openai",
    },
    "bge-large-zh": {
        "model": "bge-large-zh-v1.5",
        "dimensions": 1024,
        "provider": "local",
    },
    "glm-embedding": {
        "model": "embedding-3",
        "dimensions": 2048,
        "provider": "zhipu",
    },
    "bge-small-zh": {
        "model": "BAAI/bge-small-zh-v1.5",
        "dimensions": 512,
        "provider": "local",
    },
}


def create_embedding_provider(
    profile: str = "openai-small",
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAIEmbeddingProvider | LocalEmbeddingProvider:
    """Create embedding provider by profile name."""
    config = EMBEDDING_PROFILES.get(profile)
    if not config:
        raise ValueError(f"Unknown embedding profile: {profile}. Available: {list(EMBEDDING_PROFILES.keys())}")
    if config["provider"] == "local":
        return LocalEmbeddingProvider(model_path=config["model"])
    return OpenAIEmbeddingProvider(
        model=config["model"],
        api_key=api_key,
        base_url=base_url,
    )
