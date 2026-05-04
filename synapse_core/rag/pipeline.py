"""RAG pipeline - orchestrates the full retrieval-augmented generation flow."""

from __future__ import annotations

from typing import Any

from synapse_core.rag import (
    BaseDocumentLoader,
    BaseEmbeddingProvider,
    BaseTextSplitter,
    Document,
    DocumentChunk,
    KnowledgeCollection,
    RetrievalResult,
    RAGPipelineConfig,
)
from synapse_core.rag.loaders import TextLoader, MarkdownLoader, PDFLoader, DocxLoader, DirectoryLoader
from synapse_core.rag.splitters import RecursiveTextSplitter, CodeSplitter, SemanticSplitter
from synapse_core.rag.embedding import LocalEmbeddingProvider
from synapse_core.rag.strategies import BasicRAG, SelfRAG, CorrectiveRAG, AdaptiveRAG


class RAGPipeline:
    """
    End-to-end RAG pipeline that orchestrates:
    1. Document loading
    2. Text splitting / chunking
    3. Embedding generation
    4. Storage in vector store
    5. Retrieval with configurable strategy
    """

    def __init__(
        self,
        config: RAGPipelineConfig | None = None,
        embedding_provider: BaseEmbeddingProvider | None = None,
    ) -> None:
        self._config = config or RAGPipelineConfig()
        self._embedding_provider = embedding_provider or LocalEmbeddingProvider()
        self._chunks: list[DocumentChunk] = []
        self._collections: dict[str, KnowledgeCollection] = {}

    async def ingest(self, source: str, collection_name: str = "default", **kwargs: Any) -> int:
        """
        Load documents, split into chunks, generate embeddings, and store.
        Returns the number of chunks created.
        """
        # Step 1: Load documents
        loader = self._get_loader(source)
        documents = await loader.load(source, **kwargs)

        # Step 2: Split into chunks
        splitter = self._get_splitter()
        all_chunks: list[DocumentChunk] = []
        for doc in documents:
            chunks = splitter.split(doc, self._config.chunk_size, self._config.chunk_overlap)
            all_chunks.extend(chunks)

        # Step 3: Generate embeddings (graceful degradation if unavailable)
        texts = [c.content for c in all_chunks if c.content.strip()]
        if texts:
            try:
                embeddings = await self._embedding_provider.embed_batch(texts)
                for chunk, embedding in zip(all_chunks, embeddings):
                    chunk.embedding = embedding
            except Exception:
                pass  # Chunks stored without embeddings; BM25 retrieval still works

        # Step 4: Store
        self._chunks.extend(all_chunks)

        # Update collection metadata
        if collection_name not in self._collections:
            self._collections[collection_name] = KnowledgeCollection(
                id=collection_name,
                name=collection_name,
                document_count=len(documents),
                chunk_count=len(all_chunks),
            )
        else:
            coll = self._collections[collection_name]
            coll.document_count += len(documents)
            coll.chunk_count += len(all_chunks)

        return len(all_chunks)

    async def query(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Query the knowledge base using the configured strategy."""
        k = top_k or self._config.retrieval_top_k

        if not self._chunks:
            return []

        strategy = self._create_strategy()
        return await strategy.run(query, top_k=k)

    async def query_context(self, query: str, top_k: int = 5) -> str:
        """Query and return formatted context string for LLM injection."""
        results = await self.query(query, top_k)
        if not results:
            return ""

        parts = []
        for i, result in enumerate(results, 1):
            source = result.chunk.source or "unknown"
            parts.append(f"[{i}] (source: {source}, score: {result.score:.2f})\n{result.chunk.content}")

        return "\n\n".join(parts)

    def _get_loader(self, source: str) -> BaseDocumentLoader:
        from pathlib import Path
        path = Path(source)

        if path.is_dir():
            return DirectoryLoader()
        suffix = path.suffix.lower()
        loaders: dict[str, type[BaseDocumentLoader]] = {
            ".txt": TextLoader,
            ".md": MarkdownLoader,
            ".pdf": PDFLoader,
            ".docx": DocxLoader,
        }
        loader_cls = loaders.get(suffix, TextLoader)
        return loader_cls()

    def _get_splitter(self) -> BaseTextSplitter:
        st = self._config.splitter_type
        if st == "code":
            return CodeSplitter()
        if st == "semantic":
            return SemanticSplitter()
        return RecursiveTextSplitter()

    def _create_strategy(self):
        from synapse_core.rag.retrieval import VectorRetriever
        retriever = VectorRetriever(self._chunks, self._embedding_provider)

        s = self._config.strategy
        if s == "self-rag":
            return SelfRAG(retriever)
        if s == "corrective-rag":
            return CorrectiveRAG(retriever)
        if s == "adaptive-rag":
            return AdaptiveRAG(self._chunks, self._embedding_provider)
        return BasicRAG(retriever)

    def list_collections(self) -> list[KnowledgeCollection]:
        return list(self._collections.values())

    def get_chunk_count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._collections.clear()
