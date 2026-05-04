"""Retrieval strategies: vector, hybrid, MMR, reranking."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from synapse_core.rag import BaseRetriever, BaseEmbeddingProvider, DocumentChunk, RetrievalResult


class VectorRetriever(BaseRetriever):
    """Standard vector similarity search using cosine similarity."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        embedding_provider: BaseEmbeddingProvider,
    ) -> None:
        self._chunks = chunks
        self._embedding_provider = embedding_provider

    async def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> list[RetrievalResult]:
        query_embedding = await self._embedding_provider.embed(query)

        scored: list[tuple[float, DocumentChunk]] = []
        for chunk in self._chunks:
            if not chunk.embedding:
                continue
            similarity = self._cosine_similarity(query_embedding, chunk.embedding)
            scored.append((similarity, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        threshold = kwargs.get("score_threshold", 0.5)

        return [
            RetrievalResult(chunk=chunk, score=score, strategy="vector")
            for score, chunk in scored[:top_k]
            if score >= threshold
        ]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval: combines BM25 (keyword) + vector (semantic) search.
    Results are merged using reciprocal rank fusion.
    """

    def __init__(
        self,
        chunks: list[DocumentChunk],
        embedding_provider: BaseEmbeddingProvider,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> None:
        self._chunks = chunks
        self._embedding_provider = embedding_provider
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight
        self._bm25_index: dict[str, list[tuple[int, float]]] = {}
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Build a simple BM25-like inverted index."""
        k1 = 1.5
        b = 0.75
        avg_dl = sum(len(c.content.split()) for c in self._chunks) / max(len(self._chunks), 1)
        doc_freq: dict[str, int] = defaultdict(int)

        for chunk in self._chunks:
            seen = set(chunk.content.lower().split())
            for word in seen:
                doc_freq[word] += 1

        n_docs = len(self._chunks)
        for i, chunk in enumerate(self._chunks):
            words = chunk.content.lower().split()
            dl = len(words)
            tf_map: dict[str, int] = defaultdict(int)
            for w in words:
                tf_map[w] += 1

            for word, tf in tf_map.items():
                idf = math.log((n_docs - doc_freq[word] + 0.5) / (doc_freq[word] + 0.5) + 1)
                score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
                if word not in self._bm25_index:
                    self._bm25_index[word] = []
                self._bm25_index[word].append((i, score))

    async def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> list[RetrievalResult]:
        # BM25 scores
        bm25_scores: dict[int, float] = defaultdict(float)
        for word in query.lower().split():
            for idx, score in self._bm25_index.get(word, []):
                bm25_scores[idx] += score

        # Vector scores
        vector_retriever = VectorRetriever(self._chunks, self._embedding_provider)
        vector_results = await vector_retriever.retrieve(query, top_k=len(self._chunks))

        vector_scores: dict[int, float] = {}
        for vr in vector_results:
            idx = vr.chunk.chunk_index
            vector_scores[idx] = vr.score

        # Reciprocal rank fusion
        all_indices = set(bm25_scores.keys()) | set(vector_scores.keys())
        fused: list[tuple[float, int]] = []
        for idx in all_indices:
            score = (
                self._keyword_weight * bm25_scores.get(idx, 0)
                + self._vector_weight * vector_scores.get(idx, 0)
            )
            fused.append((score, idx))

        fused.sort(key=lambda x: x[0], reverse=True)
        threshold = kwargs.get("score_threshold", 0.3)

        results = []
        for score, idx in fused[:top_k]:
            if score >= threshold:
                results.append(RetrievalResult(
                    chunk=self._chunks[idx],
                    score=score,
                    strategy="hybrid",
                ))
        return results


class MMRRetriever(BaseRetriever):
    """
    Maximal Marginal Relevance retrieval.
    Balances relevance with diversity to avoid redundant results.
    """

    def __init__(
        self,
        chunks: list[DocumentChunk],
        embedding_provider: BaseEmbeddingProvider,
        lambda_param: float = 0.5,
    ) -> None:
        self._chunks = chunks
        self._embedding_provider = embedding_provider
        self._lambda = lambda_param

    async def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> list[RetrievalResult]:
        query_embedding = await self._embedding_provider.embed(query)

        # Score all chunks against query
        scored: list[tuple[float, int]] = []
        for i, chunk in enumerate(chunk for chunk in self._chunks if chunk.embedding):
            sim = VectorRetriever._cosine_similarity(query_embedding, chunk.embedding)
            scored.append((sim, i))

        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            return []

        # MMR selection
        selected_indices: list[int] = []
        selected_embeddings: list[list[float]] = []
        results: list[RetrievalResult] = []
        remaining = list(range(len(scored)))

        for _ in range(min(top_k, len(scored))):
            best_mmr = -float("inf")
            best_idx = -1

            for pos in remaining:
                relevance = scored[pos][0]
                chunk = self._chunks[scored[pos][1]]
                if not chunk.embedding:
                    continue

                # Max similarity to already selected
                diversity = 0.0
                if selected_embeddings:
                    diversity = max(
                        VectorRetriever._cosine_similarity(chunk.embedding, se)
                        for se in selected_embeddings
                    )

                mmr = self._lambda * relevance - (1 - self._lambda) * diversity
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = pos

            if best_idx == -1:
                break

            chunk = self._chunks[scored[best_idx][1]]
            selected_indices.append(best_idx)
            if chunk.embedding:
                selected_embeddings.append(chunk.embedding)
            remaining.remove(best_idx)

            results.append(RetrievalResult(
                chunk=chunk,
                score=best_mmr,
                strategy="mmr",
            ))

        return results


class CrossEncoderReranker:
    """
    Cross-Encoder reranker for two-stage retrieval.
    Stage 1: fast vector/BM25 retrieval (recall top-K)
    Stage 2: precise Cross-Encoder scoring (select top-N)
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results using Cross-Encoder scoring."""
        if not results:
            return results

        model = self._get_model()
        pairs = [(query, r.chunk.content) for r in results]
        scores = model.predict(pairs)

        scored = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for result, score in scored[:top_k]:
            reranked.append(RetrievalResult(
                chunk=result.chunk,
                score=float(score),
                strategy=f"reranked-{result.strategy}",
            ))
        return reranked


class LLMBasedReranker:
    """
    LLM-based reranker for environments without sentence-transformers.
    Uses keyword overlap heuristic as lightweight reranking.
    """

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Rerank using keyword overlap + original score."""
        if not results:
            return results

        query_words = set(query.lower().split())
        scored: list[tuple[float, RetrievalResult]] = []

        for result in results:
            content_words = set(result.chunk.content.lower().split())
            overlap = len(query_words & content_words) / max(len(query_words), 1)
            combined = result.score * 0.6 + overlap * 0.4
            scored.append((combined, result))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievalResult(chunk=r.chunk, score=s, strategy=f"reranked-{r.strategy}")
            for s, r in scored[:top_k]
        ]
