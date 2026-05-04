"""Knowledge extraction from conversations.

Extracts key facts, decisions, and patterns from session conversations
and stores them in long-term memory for cross-session RAG retrieval.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from synapse_core.memory import MemoryEntry, MemoryType


class KnowledgeExtractor:
    """Extract structured knowledge from conversations and store in memory."""

    def extract_from_messages(
        self,
        messages: List[Dict[str, str]],
        session_id: str = "",
        agent_id: str = "",
    ) -> List[MemoryEntry]:
        """Extract knowledge entries from a list of {role, content} messages."""
        pairs = self._build_pairs(messages)
        entries: List[MemoryEntry] = []

        for user_msg, assistant_msg in pairs:
            for content, mem_type, importance in self._extract(user_msg, assistant_msg):
                entries.append(MemoryEntry(
                    id=str(uuid.uuid4()),
                    content=content,
                    memory_type=mem_type,
                    importance=importance,
                    session_id=session_id,
                    agent_id=agent_id,
                    metadata={"source": "knowledge_extraction"},
                ))

        return entries

    async def extract_and_store(
        self,
        messages: List[Dict[str, str]],
        memory_manager: Any,
        session_id: str = "",
        agent_id: str = "",
    ) -> int:
        """Extract knowledge and persist to long-term memory. Returns count."""
        entries = self.extract_from_messages(messages, session_id, agent_id)
        for entry in entries:
            await memory_manager.auto_save(entry, entry.importance)
        return len(entries)

    # ── Internal ─────────────────────────────────────────────────────────

    def _build_pairs(self, messages: List[Dict[str, str]]) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        user_msg = ""
        for msg in messages:
            if msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant" and user_msg:
                pairs.append((user_msg, msg["content"]))
                user_msg = ""
        return pairs

    def _extract(self, user_msg: str, assistant_msg: str) -> List[Tuple[str, MemoryType, float]]:
        """Return list of (content, memory_type, importance)."""
        results: List[Tuple[str, MemoryType, float]] = []

        # Facts from assistant responses
        for fact in self._extract_facts(assistant_msg):
            results.append((fact, MemoryType.SEMANTIC, 0.7))

        # Decisions and recommendations
        for decision in self._extract_decisions(assistant_msg):
            results.append((decision, MemoryType.EPISODIC, 0.8))

        # Code patterns
        for pattern in self._extract_code_patterns(assistant_msg):
            results.append((pattern, MemoryType.PROCEDURAL, 0.7))

        # Problem-solution pairs
        for solution in self._extract_solutions(user_msg, assistant_msg):
            results.append((solution, MemoryType.SEMANTIC, 0.85))

        return results

    def _extract_facts(self, text: str) -> List[str]:
        facts: List[str] = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for s in sentences:
            s = s.strip()
            if len(s) < 15 or len(s) > 300:
                continue
            if self._is_factual(s) and self._is_important(s):
                facts.append(s)
        return facts

    def _extract_decisions(self, text: str) -> List[str]:
        decisions: List[str] = []
        patterns = [
            r'(?:I\s+)?(?:recommend|suggest|should|best\s+to)\s+(.+?)(?:\.|because)',
            r'(?:the\s+)?(?:best|optimal|preferred)\s+(?:approach|way|method)\s+(?:is|would\s+be)\s+(.+?)(?:\.|$)',
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                decision = match.group(0).strip()
                if len(decision) > 10:
                    decisions.append(decision[:250])
        return decisions

    def _extract_code_patterns(self, text: str) -> List[str]:
        patterns: List[str] = []
        blocks = re.findall(r'```(\w*)\n(.+?)```', text, re.DOTALL)
        for lang, code in blocks:
            code = code.strip()
            if len(code) < 10:
                continue
            idx = text.find("```" + lang)
            context = ""
            if idx > 0:
                prev_lines = [l.strip() for l in text[:idx].split("\n") if l.strip()]
                if prev_lines:
                    context = prev_lines[-1]
            summary = f"{context}: {lang or 'code'} snippet ({len(code.splitlines())} lines)" if context else f"{lang or 'code'} snippet ({len(code.splitlines())} lines)"
            patterns.append(summary[:250])
        return patterns

    def _extract_solutions(self, user_msg: str, assistant_msg: str) -> List[str]:
        solutions: List[str] = []
        problem_kw = ["error", "bug", "issue", "problem", "not working", "fix",
                       "how to", "how do", "how can", "why does", "why is", "help"]
        user_lower = user_msg.lower()
        if not any(kw in user_lower for kw in problem_kw):
            return solutions

        steps = re.findall(r'(?:\d+\.\s*)(.+?)(?:\n|$)', assistant_msg)
        if steps:
            summary = " | ".join(s.strip()[:60] for s in steps[:4])
            solutions.append(f"Q: {user_msg[:80]} -> {summary}"[:300])
        return solutions

    def _is_factual(self, text: str) -> bool:
        factual = [
            r'.+?(?:is|are|was|were)\s+(?:a|an|the)\s+',
            r'.+?(?:uses?|supports?|provides?|contains?|includes?)\s+',
            r'.+?(?:can|should|must|needs?)\s+(?:be|to)\s+',
        ]
        for p in factual:
            if re.match(p, text, re.IGNORECASE):
                return True
        return False

    def _is_important(self, text: str) -> bool:
        important = [
            "config", "setup", "install", "deploy", "api", "key", "token",
            "database", "schema", "migration", "architecture", "pattern",
            "security", "auth", "performance", "critical", "important",
            "note", "warning", "error", "solution", "best practice",
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in important)
