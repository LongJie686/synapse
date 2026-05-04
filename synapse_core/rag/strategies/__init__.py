"""RAG strategies: basic, self-RAG, corrective-RAG, adaptive-RAG."""

from __future__ import annotations

import json
from typing import Any

from synapse_core.rag import BaseRetriever, BaseEmbeddingProvider, DocumentChunk, RetrievalResult
from synapse_core.rag.retrieval import VectorRetriever, HybridRetriever, MMRRetriever


class BasicRAG:
    """Standard RAG: retrieve chunks and return them as context."""

    def __init__(self, retriever: BaseRetriever) -> None:
        self.retriever = retriever

    async def run(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return await self.retriever.retrieve(query, top_k=top_k)


class SelfRAG:
    """
    Self-RAG: retrieves, then self-assesses retrieval quality.
    If quality is low, tries again with a reformulated query.
    """

    def __init__(self, retriever: BaseRetriever, assess_fn: Any | None = None) -> None:
        self.retriever = retriever
        self._assess_fn = assess_fn

    async def run(self, query: str, top_k: int = 5, max_retries: int = 2) -> list[RetrievalResult]:
        current_query = query
        for attempt in range(max_retries + 1):
            results = await self.retriever.retrieve(current_query, top_k=top_k)

            if not results:
                break

            # Self-assessment
            if self._assess_fn:
                assessment = await self._assess_fn(query, [r.chunk.content for r in results])
                if assessment.get("is_relevant", True):
                    return results
                current_query = assessment.get("reformulated_query", query)
            else:
                # Heuristic: check if top result score is above threshold
                if results[0].score >= 0.6:
                    return results
                # Reformulate by making query more specific
                current_query = f"detailed information about {current_query}"

        return results


class CorrectiveRAG:
    """
    Corrective RAG: if retrieval quality is low, falls back to web search.
    Corrects the retrieval by augmenting with external results.
    """

    def __init__(self, retriever: BaseRetriever, web_search_fn: Any | None = None) -> None:
        self.retriever = retriever
        self._web_search_fn = web_search_fn

    async def run(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        results = await self.retriever.retrieve(query, top_k=top_k)

        # Assess: if top score is low, trigger correction
        if not results or results[0].score < 0.4:
            if self._web_search_fn:
                web_results = await self._web_search_fn(query)
                if web_results:
                    # Create synthetic chunks from web results
                    for i, text in enumerate(web_results[:3]):
                        results.append(RetrievalResult(
                            chunk=DocumentChunk(
                                content=text,
                                source="web-search",
                                chunk_index=1000 + i,
                                metadata={"source": "web_correction"},
                            ),
                            score=0.5,
                            strategy="corrective-web",
                        ))

        return results


class AdaptiveRAG:
    """
    Adaptive RAG: classifies query type first, then selects
    the best retrieval strategy for that query type.
    """

    QUERY_TYPES = ["factual", "analytical", "navigational", "conversational"]

    def __init__(
        self,
        chunks: list[DocumentChunk],
        embedding_provider: BaseEmbeddingProvider,
        classify_fn: Any | None = None,
    ) -> None:
        self._chunks = chunks
        self._embedding_provider = embedding_provider
        self._classify_fn = classify_fn

    async def run(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        # Step 1: Classify query
        query_type = await self._classify_query(query)

        # Step 2: Select strategy based on query type
        retriever = self._select_retriever(query_type)
        results = await retriever.retrieve(query, top_k=top_k)

        # Tag results with the strategy used
        for r in results:
            r.strategy = f"adaptive-{query_type}"

        return results

    async def _classify_query(self, query: str) -> str:
        if self._classify_fn:
            return await self._classify_fn(query)

        # Heuristic classification
        q = query.lower()
        if any(w in q for w in ["what is", "define", "definition", "是什么", "定义"]):
            return "factual"
        if any(w in q for w in ["compare", "analyze", "difference", "对比", "分析"]):
            return "analytical"
        if any(w in q for w in ["where", "find", "在哪", "查找"]):
            return "navigational"
        return "conversational"

    def _select_retriever(self, query_type: str) -> BaseRetriever:
        if query_type == "factual":
            return VectorRetriever(self._chunks, self._embedding_provider)
        elif query_type == "analytical":
            return HybridRetriever(self._chunks, self._embedding_provider)
        elif query_type == "navigational":
            return MMRRetriever(self._chunks, self._embedding_provider, lambda_param=0.3)
        else:
            return VectorRetriever(self._chunks, self._embedding_provider)
