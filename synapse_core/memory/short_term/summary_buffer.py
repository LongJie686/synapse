"""Summary buffer - auto-compacts old messages via LLM summarization."""

from __future__ import annotations

from typing import Any

from synapse_core.graph.state import Message
from synapse_core.memory import CompactionResult


class SummaryBuffer:
    """
    Auto-compacting buffer that summarizes old messages when token limit is exceeded.

    Priority for retention:
    1. User explicit instructions
    2. Tool call results
    3. Agent reasoning chains
    4. Greetings and repetitive content (compress first)
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        compaction_threshold: float = 0.8,
        summarize_fn: Any | None = None,
    ) -> None:
        self._max_tokens = max_tokens
        self._compaction_threshold = compaction_threshold
        self._summarize_fn = summarize_fn
        self._messages: list[Message] = []
        self._summary: str = ""
        self._compaction_count: int = 0

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def add_many(self, messages: list[Message]) -> None:
        self._messages.extend(messages)

    def get_all(self) -> list[Message]:
        result: list[Message] = []
        if self._summary:
            result.append(Message(
                role="system",
                content=f"[Conversation Summary]\n{self._summary}",
                metadata={"type": "auto_summary", "version": self._compaction_count},
            ))
        result.extend(self._messages)
        return result

    def get_recent(self, n: int) -> list[Message]:
        return list(self._messages[-n:])

    def clear(self) -> None:
        self._messages.clear()
        self._summary = ""

    @property
    def count(self) -> int:
        return len(self._messages)

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def compaction_count(self) -> int:
        return self._compaction_count

    def estimate_tokens(self) -> int:
        summary_tokens = len(self._summary) // 3 if self._summary else 0
        msg_tokens = sum(len(m.content) for m in self._messages) // 3
        return summary_tokens + msg_tokens

    def needs_compaction(self) -> bool:
        return self.estimate_tokens() >= int(self._max_tokens * self._compaction_threshold)

    async def compact_if_needed(self) -> CompactionResult | None:
        """Auto-compact if threshold exceeded. Returns None if no compaction needed."""
        if not self.needs_compaction():
            return None
        return await self._do_compact()

    async def _do_compact(self) -> CompactionResult:
        original_tokens = self.estimate_tokens()
        messages_to_compact = self._messages[:-4]  # Keep last 4 messages intact
        kept_messages = self._messages[-4:]

        if not messages_to_compact:
            return CompactionResult(
                original_token_count=original_tokens,
                compacted_token_count=original_tokens,
                messages_compacted=0,
                summary="",
            )

        if self._summarize_fn:
            conversation_text = self._format_messages(messages_to_compact)
            new_summary = await self._summarize_fn(self._summary, conversation_text)
        else:
            new_summary = self._build_simple_summary(messages_to_compact)

        self._summary = new_summary
        self._messages = kept_messages
        self._compaction_count += 1

        return CompactionResult(
            original_token_count=original_tokens,
            compacted_token_count=self.estimate_tokens(),
            messages_compacted=len(messages_to_compact),
            summary=new_summary,
        )

    def _format_messages(self, messages: list[Message]) -> str:
        lines = []
        for m in messages:
            prefix = m.role.upper()
            if m.name:
                prefix += f" ({m.name})"
            lines.append(f"{prefix}: {m.content}")
        return "\n".join(lines)

    def _build_simple_summary(self, messages: list[Message]) -> str:
        """Fallback summary without LLM - extract key points."""
        existing = f"Previous context: {self._summary}\n\n" if self._summary else ""
        user_msgs = [m for m in messages if m.role == "user"]
        tool_msgs = [m for m in messages if m.role == "tool"]

        parts = []
        if user_msgs:
            parts.append("User asked: " + "; ".join(m.content[:100] for m in user_msgs[-3:]))
        if tool_msgs:
            parts.append(f"Used tools: {', '.join(set(m.name or 'unknown' for m in tool_msgs))}")

        return existing + " | ".join(parts) if parts else existing + "Conversation continued."
