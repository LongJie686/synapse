"""Unified memory manager - coordinates all memory subsystems."""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Awaitable

from synapse_core.graph.state import Message
from synapse_core.memory import (
    BaseMemoryStore,
    CompactionResult,
    MemoryEntry,
    MemorySearchOptions,
    MemoryType,
    UserProfile,
)
from synapse_core.memory.short_term import ConversationBuffer, SlidingWindow, SummaryBuffer
from synapse_core.memory.long_term import ChromaMemoryStore
from synapse_core.memory.long_term.sqlite_store import SqliteMemoryStore
from synapse_core.memory.user_profile import ProfileExtractor


SummarizeFn = Callable[[str, str], Awaitable[str]]


class MemoryManager:
    """
    Unified memory manager that coordinates:
    - Short-term memory (conversation context)
    - Long-term memory (persistent knowledge)
    - User profile (auto-updating traits and preferences)

    Auto-features:
    - Auto compaction: compresses short-term memory when token limit approached
    - Auto save: persists high-value interactions to long-term memory
    - Auto profile update: extracts user traits after each conversation
    """

    def __init__(
        self,
        store: BaseMemoryStore | None = None,
        *,
        short_term_type: str = "summary-buffer",
        max_context_tokens: int = 8000,
        compaction_threshold: float = 0.8,
        auto_save_importance: float = 0.6,
        summarize_fn: SummarizeFn | None = None,
    ) -> None:
        self._store = store or SqliteMemoryStore()
        self._summarize_fn = summarize_fn
        self._auto_save_importance = auto_save_importance

        # Short-term memory
        if short_term_type == "conversation-buffer":
            self._short_term: ConversationBuffer | SlidingWindow | SummaryBuffer = ConversationBuffer()
        elif short_term_type == "sliding-window":
            self._short_term = SlidingWindow(window_size=20)
        else:
            self._short_term = SummaryBuffer(
                max_tokens=max_context_tokens,
                compaction_threshold=compaction_threshold,
                summarize_fn=summarize_fn,
            )

        # User profile extractor
        self._profile_extractor = ProfileExtractor(summarize_fn=summarize_fn)
        self._profiles: dict[str, UserProfile] = {}

    # --- Short-term Memory ---

    async def get_conversation_context(self, session_id: str) -> list[Message]:
        """Get current conversation context for a session."""
        return self._short_term.get_all()

    async def add_message(self, session_id: str, message: Message) -> None:
        """Add a message to short-term memory."""
        self._short_term.add(message)
        await self.compact_if_needed(session_id)

        # Auto-save: check if this interaction is worth persisting
        if message.role == "assistant":
            importance = self._estimate_importance(message.content)
            if importance >= self._auto_save_importance:
                await self.auto_save(
                    MemoryEntry(
                        id="",
                        content=message.content,
                        memory_type=MemoryType.EPISODIC,
                        importance=importance,
                        session_id=session_id,
                        created_at=time.time(),
                    ),
                    importance,
                )

    async def compact_if_needed(self, session_id: str) -> CompactionResult | None:
        """Auto-compact short-term memory if threshold exceeded."""
        if isinstance(self._short_term, SummaryBuffer):
            result = await self._short_term.compact_if_needed()
            if result and result.summary:
                # Save compacted summary as semantic memory
                await self._store.add(MemoryEntry(
                    id="",
                    content=f"Session {session_id} summary: {result.summary}",
                    memory_type=MemoryType.SEMANTIC,
                    importance=0.5,
                    session_id=session_id,
                    created_at=time.time(),
                ))
            return result
        return None

    # --- Long-term Memory ---

    async def search_long_term(self, query: str, options: MemorySearchOptions | None = None) -> list[MemoryEntry]:
        """Search long-term memory for relevant entries."""
        opts = options or MemorySearchOptions()
        return await self._store.search(query, opts)

    async def auto_save(self, memory: MemoryEntry, importance: float) -> None:
        """Auto-save a memory entry if importance exceeds threshold."""
        if importance < self._auto_save_importance:
            return
        memory.importance = importance
        await self._store.add(memory)

    async def retrieve(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        """Retrieve memories using the specified strategy."""
        opts = MemorySearchOptions(
            top_k=top_k,
            strategy=strategy,
        )
        return await self._store.search(query, opts)

    # --- User Profile ---

    async def get_user_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile."""
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id=user_id)
        return self._profiles[user_id]

    async def auto_update_profile(self, user_id: str, conversation_summary: str) -> None:
        """Auto-update user profile from conversation content."""
        profile = await self.get_user_profile(user_id)
        updated = await self._profile_extractor.extract_and_update(profile, conversation_summary)
        self._profiles[user_id] = updated

        # Persist profile as semantic memory
        profile_text = f"User profile: interests={updated.interests}, style={updated.communication_style}"
        for trait in updated.traits:
            profile_text += f", {trait.key}={trait.value}"

        await self._store.add(MemoryEntry(
            id="",
            content=profile_text,
            memory_type=MemoryType.SEMANTIC,
            importance=0.8,
            metadata={"user_id": user_id, "profile_version": updated.version},
            created_at=time.time(),
        ))

    # --- Internal Helpers ---

    def _estimate_importance(self, content: str) -> float:
        """
        Estimate the importance of a message without LLM.
        Returns 0-1 score based on heuristics.
        """
        score = 0.3  # Base score
        text = content.lower()

        # High-value signals
        high_signals = ["solved", "resolved", "找到了", "解决了", "确认", "correct", "perfect"]
        for signal in high_signals:
            if signal in text:
                score += 0.2
                break

        # Knowledge signals
        knowledge_signals = ["learn", "discovered", "important", "key point", "发现", "关键", "重要"]
        for signal in knowledge_signals:
            if signal in text:
                score += 0.15
                break

        # Length heuristic: substantial responses are more important
        if len(content) > 500:
            score += 0.1

        # Tool usage signals
        if "tool" in text or "工具" in text:
            score += 0.1

        return min(score, 1.0)
