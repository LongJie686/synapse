"""ChromaDB-based long-term memory store."""

from __future__ import annotations

import time
import uuid
from typing import Any

from synapse_core.memory import (
    BaseMemoryStore,
    MemoryEntry,
    MemorySearchOptions,
    MemoryType,
)


class ChromaMemoryStore(BaseMemoryStore):
    """
    Long-term memory store backed by ChromaDB.

    Supports episodic, semantic, and procedural memory types.
    Auto-deduplicates by merging entries with similarity > 0.95.
    """

    SIMILARITY_MERGE_THRESHOLD = 0.95

    def __init__(self, collection_name: str = "synapse_memory", chroma_path: str | None = None) -> None:
        self._collection_name = collection_name
        self._chroma_path = chroma_path
        self._client: Any = None
        self._collection: Any = None
        self._entries: dict[str, MemoryEntry] = {}

    def _get_collection(self) -> Any:
        if self._collection is None:
            import chromadb

            if self._chroma_path:
                self._client = chromadb.PersistentClient(path=self._chroma_path)
            else:
                self._client = chromadb.Client()

            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def add(self, entry: MemoryEntry) -> str:
        if not entry.id:
            entry.id = str(uuid.uuid4())
        if not entry.created_at:
            entry.created_at = time.time()
        entry.updated_at = time.time()

        collection = self._get_collection()

        # Auto-deduplicate: check for very similar existing entries
        if entry.embedding:
            similar = collection.query(
                query_embeddings=[entry.embedding],
                n_results=1,
                include=["documents", "metadatas", "distances"],
            )
            if similar["distances"] and similar["distances"][0]:
                distance = similar["distances"][0][0]
                similarity = 1 - distance
                if similarity > self.SIMILARITY_MERGE_THRESHOLD:
                    # Merge with existing entry instead of creating duplicate
                    existing_id = similar["ids"][0][0]
                    await self._merge_entry(existing_id, entry)
                    return existing_id

        # Add new entry
        metadata = {
            "memory_type": entry.memory_type.value,
            "importance": entry.importance,
            "session_id": entry.session_id or "",
            "agent_id": entry.agent_id or "",
            "created_at": entry.created_at,
        }
        metadata.update({k: str(v) for k, v in entry.metadata.items()})

        doc_args: dict[str, Any] = {
            "ids": [entry.id],
            "documents": [entry.content],
            "metadatas": [metadata],
        }
        if entry.embedding:
            doc_args["embeddings"] = [entry.embedding]

        collection.upsert(**doc_args)
        self._entries[entry.id] = entry
        return entry.id

    async def search(self, query: str, options: MemorySearchOptions) -> list[MemoryEntry]:
        collection = self._get_collection()

        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": options.top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if options.memory_type:
            kwargs["where"] = {"memory_type": options.memory_type.value}

        results = collection.query(**kwargs)

        entries = []
        if not results["ids"] or not results["ids"][0]:
            return entries

        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i] if results["distances"] else 0
            similarity = 1 - distance

            if similarity < options.score_threshold:
                continue

            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            memory_type = MemoryType(metadata.get("memory_type", "semantic"))

            entries.append(MemoryEntry(
                id=doc_id,
                content=results["documents"][0][i],
                memory_type=memory_type,
                importance=metadata.get("importance", 0.0),
                session_id=metadata.get("session_id"),
                agent_id=metadata.get("agent_id"),
                created_at=metadata.get("created_at", 0),
                updated_at=metadata.get("updated_at", 0),
                metadata={k: v for k, v in metadata.items()
                          if k not in ("memory_type", "importance", "session_id", "agent_id", "created_at", "updated_at")},
            ))

        return entries

    async def get(self, entry_id: str) -> MemoryEntry | None:
        collection = self._get_collection()
        results = collection.get(ids=[entry_id], include=["documents", "metadatas"])
        if not results["ids"]:
            return None

        metadata = results["metadatas"][0] if results["metadatas"] else {}
        return MemoryEntry(
            id=entry_id,
            content=results["documents"][0],
            memory_type=MemoryType(metadata.get("memory_type", "semantic")),
            importance=metadata.get("importance", 0.0),
            metadata=metadata,
        )

    async def delete(self, entry_id: str) -> bool:
        collection = self._get_collection()
        collection.delete(ids=[entry_id])
        self._entries.pop(entry_id, None)
        return True

    async def update(self, entry_id: str, content: str, metadata: dict[str, Any] | None = None) -> bool:
        collection = self._get_collection()
        update_kwargs: dict[str, Any] = {
            "ids": [entry_id],
            "documents": [content],
        }
        if metadata:
            update_kwargs["metadatas"] = [metadata]
        collection.update(**update_kwargs)
        return True

    async def _merge_entry(self, existing_id: str, new_entry: MemoryEntry) -> None:
        """Merge a new entry into an existing one, combining content and taking higher importance."""
        existing = await self.get(existing_id)
        if not existing:
            return

        merged_content = f"{existing.content}\n\nAdditional context: {new_entry.content}"
        merged_importance = max(existing.importance, new_entry.importance)
        merged_metadata = {**existing.metadata, **new_entry.metadata}
        merged_metadata["updated_at"] = time.time()
        merged_metadata["importance"] = merged_importance

        await self.update(existing_id, merged_content, merged_metadata)
