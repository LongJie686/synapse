"""Memory system types and interfaces."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryEntry(BaseModel):
    id: str
    content: str
    memory_type: MemoryType
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float = 0.0
    session_id: str | None = None
    agent_id: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0


class CompactionResult(BaseModel):
    original_token_count: int
    compacted_token_count: int
    messages_compacted: int
    summary: str


class UserTrait(BaseModel):
    key: str
    value: str
    confidence: float = 0.5


class UserProfile(BaseModel):
    user_id: str
    traits: list[UserTrait] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    communication_style: str = "balanced"
    expertise_level: dict[str, str] = Field(default_factory=dict)
    frequent_tools: list[str] = Field(default_factory=list)
    last_updated: float = 0.0
    version: int = 1


class MemorySearchOptions(BaseModel):
    top_k: int = 5
    score_threshold: float = 0.5
    memory_type: MemoryType | None = None
    strategy: str = "hybrid"


class BaseMemoryStore:
    """Abstract base for memory storage backends."""

    async def add(self, entry: MemoryEntry) -> str:
        raise NotImplementedError

    async def search(self, query: str, options: MemorySearchOptions) -> list[MemoryEntry]:
        raise NotImplementedError

    async def get(self, entry_id: str) -> MemoryEntry | None:
        raise NotImplementedError

    async def delete(self, entry_id: str) -> bool:
        raise NotImplementedError

    async def update(self, entry_id: str, content: str, metadata: dict[str, Any] | None = None) -> bool:
        raise NotImplementedError
