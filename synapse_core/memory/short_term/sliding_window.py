"""Sliding window buffer - keeps only the most recent N messages."""

from __future__ import annotations

from synapse_core.graph.state import Message


class SlidingWindow:
    """Keeps only the most recent N messages, discarding older ones."""

    def __init__(self, window_size: int = 20) -> None:
        self._window_size = window_size
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)
        if len(self._messages) > self._window_size:
            self._messages = self._messages[-self._window_size:]

    def add_many(self, messages: list[Message]) -> None:
        self._messages.extend(messages)
        if len(self._messages) > self._window_size:
            self._messages = self._messages[-self._window_size:]

    def get_all(self) -> list[Message]:
        return list(self._messages)

    def get_recent(self, n: int) -> list[Message]:
        return list(self._messages[-n:])

    def clear(self) -> None:
        self._messages.clear()

    @property
    def count(self) -> int:
        return len(self._messages)

    @property
    def window_size(self) -> int:
        return self._window_size

    def estimate_tokens(self) -> int:
        total_chars = sum(len(m.content) for m in self._messages)
        return total_chars // 3
