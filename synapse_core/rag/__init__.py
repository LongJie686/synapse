"""RAG pipeline types and interfaces."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    page: int | None = None
    chunk_index: int = 0


class DocumentChunk(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    embedding: list[float] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    chunk: DocumentChunk
    score: float = 0.0
    strategy: str = ""


class KnowledgeCollection(BaseModel):
    id: str
    name: str
    description: str = ""
    document_count: int = 0
    chunk_count: int = 0
    embedding_dimension: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGPipelineConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50
    splitter_type: str = "recursive"
    embedding_provider: str = "openai"
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.5
    retrieval_method: str = "vector"
    strategy: str = "basic"


class BaseDocumentLoader:
    async def load(self, source: str, **kwargs: Any) -> list[Document]:
        raise NotImplementedError


class BaseTextSplitter:
    def split(self, document: Document, chunk_size: int = 500, chunk_overlap: int = 50) -> list[DocumentChunk]:
        raise NotImplementedError


class BaseEmbeddingProvider:
    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class BaseRetriever:
    async def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> list[RetrievalResult]:
        raise NotImplementedError
