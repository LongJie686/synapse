"""Conversation buffer - stores full conversation history."""

from __future__ import annotations

from synapse_core.graph.state import Message


class ConversationBuffer:
    """Stores the complete conversation history without any truncation."""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def add_many(self, messages: list[Message]) -> None:
        self._messages.extend(messages)

    def get_all(self) -> list[Message]:
        return list(self._messages)

    def get_recent(self, n: int) -> list[Message]:
        return list(self._messages[-n:])

    def clear(self) -> None:
        self._messages.clear()

    @property
    def count(self) -> int:
        return len(self._messages)

    def estimate_tokens(self) -> int:
        """Rough token estimate: ~4 chars per token for English, ~2 for Chinese."""
        total_chars = sum(len(m.content) for m in self._messages)
        return total_chars // 3
