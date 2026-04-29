"""Safety guardrails: input/output filtering, PII detection, prompt injection defense."""

from __future__ import annotations

import re
import hashlib
from typing import Any

from pydantic import BaseModel


class GuardrailResult(BaseModel):
    blocked: bool = False
    filter_name: str = ""
    reason: str = ""
    sanitized_content: str = ""
    confidence: float = 0.0


class InputFilter:
    """Filters user input for safety threats before LLM processing."""

    # Prompt injection patterns
    _INJECTION_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"ignore\s+(all\s+)?previous\s+(instructions|prompts)", re.IGNORECASE),
        re.compile(r"forget\s+(all\s+)?(your|previous|above)\s+(instructions|rules)", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+(?:a\s+)?(?:DAN|jailbreak|unlocked)", re.IGNORECASE),
        re.compile(r"system\s*:\s*", re.IGNORECASE),
        re.compile(r"<\|(?:im_start|im_end)\|>", re.IGNORECASE),
        re.compile(r"###\s*(?:instruction|system|human)", re.IGNORECASE),
        re.compile(r"(?:pretend|act|roleplay)\s+(?:you\s+are|to\s+be)\s+(?!.*(?:user|customer))", re.IGNORECASE),
        re.compile(r"override\s+(?:your|the)\s+(?:safety|security|content)\s+(?:policy|filter|guideline)", re.IGNORECASE),
        re.compile(r"(?:reveal|show|tell|output)\s+(?:your|the)\s+(?:system|initial|original)\s+(?:prompt|instruction)", re.IGNORECASE),
    ]

    # Suspicious length ratio (very long input might be an injection attempt)
    _MAX_INPUT_LENGTH = 10000

    async def check(self, content: str) -> GuardrailResult:
        """Check input for safety threats. Returns result with blocked=True if threat detected."""
        # Length check
        if len(content) > self._MAX_INPUT_LENGTH:
            return GuardrailResult(
                blocked=True,
                filter_name="input_length",
                reason=f"Input exceeds maximum length ({len(content)} > {self._MAX_INPUT_LENGTH})",
                confidence=1.0,
            )

        # Prompt injection detection
        for pattern in self._INJECTION_PATTERNS:
            match = pattern.search(content)
            if match:
                return GuardrailResult(
                    blocked=True,
                    filter_name="prompt_injection",
                    reason=f"Potential prompt injection detected: pattern matched at position {match.start()}",
                    confidence=0.85,
                )

        return GuardrailResult(blocked=False, sanitized_content=content)


class OutputFilter:
    """Filters LLM output before returning to user."""

    _SENSITIVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"(?:api[_-]?key|secret|token|password)\s*[=:]\s*['\"]?[\w\-]{20,}", re.IGNORECASE), "[REDACTED_CREDENTIAL]"),
        (re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE), "[REDACTED_OPENAI_KEY]"),
        (re.compile(r"ghp_[a-zA-Z0-9]{36}", re.IGNORECASE), "[REDACTED_GITHUB_TOKEN]"),
        (re.compile(r"AKIA[0-9A-Z]{16}", re.IGNORECASE), "[REDACTED_AWS_KEY]"),
    ]

    _HARMFUL_CONTENT_KEYWORDS = {
        "en": ["bomb recipe", "how to hack", "exploit tutorial", "phishing template"],
        "zh": ["制造炸弹", "黑客教程", "钓鱼模板"],
    }

    async def check(self, content: str) -> GuardrailResult:
        """Filter output for sensitive information and harmful content."""
        sanitized = content

        # Redact sensitive patterns
        for pattern, replacement in self._SENSITIVE_PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)

        # Check for harmful content
        content_lower = content.lower()
        for lang_keywords in self._HARMFUL_CONTENT_KEYWORDS.values():
            for keyword in lang_keywords:
                if keyword in content_lower:
                    return GuardrailResult(
                        blocked=True,
                        filter_name="content_moderation",
                        reason=f"Potentially harmful content detected",
                        sanitized_content=sanitized,
                        confidence=0.8,
                    )

        if sanitized != content:
            return GuardrailResult(
                blocked=False,
                filter_name="sensitive_redaction",
                reason="Sensitive information redacted",
                sanitized_content=sanitized,
                confidence=1.0,
            )

        return GuardrailResult(blocked=False, sanitized_content=content)


class PIIDetector:
    """Detects and redacts Personally Identifiable Information."""

    _PATTERNS: list[tuple[str, re.Pattern[str]]] = [
        ("email", re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')),
        ("phone_cn", re.compile(r'\b1[3-9]\d{9}\b')),
        ("phone_intl", re.compile(r'\b\+\d{1,3}[-.\s]?\d{1,14}\b')),
        ("id_card_cn", re.compile(r'\b\d{17}[\dXx]\b')),
        ("credit_card", re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')),
        ("ip_address", re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')),
        ("bank_account", re.compile(r'\b\d{16,19}\b')),
    ]

    async def detect(self, content: str) -> list[tuple[str, str, int, int]]:
        """Detect PII in content. Returns list of (type, matched_text, start, end)."""
        findings = []
        for pii_type, pattern in self._PATTERNS:
            for match in pattern.finditer(content):
                findings.append((pii_type, match.group(), match.start(), match.end()))
        return findings

    async def redact(self, content: str) -> tuple[str, list[tuple[str, str]]]:
        """Redact PII from content. Returns (redacted_content, list of (type, original))."""
        redacted = content
        detected: list[tuple[str, str]] = []

        for pii_type, pattern in self._PATTERNS:
            for match in pattern.finditer(content):
                detected.append((pii_type, match.group()))
                redacted = redacted.replace(match.group(), f"[REDACTED_{pii_type.upper()}]")

        return redacted, detected


class ContentModerator:
    """Content safety classification for both input and output."""

    _UNSAFE_CATEGORIES = {
        "violence": [r"\b(kill|murder|assault|attack|weapon|bomb|shoot|stab)\b"],
        "self_harm": [r"\b(suicide|self.harm|cut myself|end my life)\b"],
        "hate": [r"\b(hate\s+speech|racial\s+slur|discrimination)\b"],
        "sexual": [r"\b(explicit|sexual\s+content|nsfw)\b"],
    }

    _compiled_patterns: dict[str, list[re.Pattern[str]]] = {}

    def __init__(self) -> None:
        for category, patterns in self._UNSAFE_CATEGORIES.items():
            self._compiled_patterns[category] = [re.compile(p, re.IGNORECASE) for p in patterns]

    async def classify(self, content: str) -> dict[str, float]:
        """Classify content across safety categories. Returns {category: score}."""
        scores: dict[str, float] = {}
        for category, patterns in self._compiled_patterns.items():
            max_score = 0.0
            for pattern in patterns:
                matches = pattern.findall(content.lower())
                if matches:
                    # Score based on number of matches relative to content length
                    score = min(len(matches) * 0.3, 1.0)
                    max_score = max(max_score, score)
            scores[category] = max_score
        return scores

    async def is_safe(self, content: str, threshold: float = 0.5) -> tuple[bool, str]:
        """Check if content is safe. Returns (is_safe, reason)."""
        scores = await self.classify(content)
        for category, score in scores.items():
            if score >= threshold:
                return False, f"Unsafe content detected in category '{category}' (score: {score:.2f})"
        return True, ""


class GuardrailPipeline:
    """
    Orchestrates all guardrails in sequence:
    input_filter -> PII_detection -> content_moderation -> [LLM] -> output_filter
    """

    def __init__(self) -> None:
        self.input_filter = InputFilter()
        self.output_filter = OutputFilter()
        self.pii_detector = PIIDetector()
        self.content_moderator = ContentModerator()

    async def check_input(self, content: str) -> GuardrailResult:
        """Run all input-side guardrails."""
        # Step 1: Input filter (prompt injection, length)
        result = await self.input_filter.check(content)
        if result.blocked:
            return result

        # Step 2: PII detection
        _, pii_found = await self.pii_detector.redact(content)
        if pii_found:
            redacted, _ = await self.pii_detector.redact(content)
            return GuardrailResult(
                blocked=False,
                filter_name="pii_detection",
                reason=f"PII detected and redacted: {[p[0] for p in pii_found]}",
                sanitized_content=redacted,
                confidence=1.0,
            )

        # Step 3: Content moderation
        is_safe, reason = await self.content_moderator.is_safe(content)
        if not is_safe:
            return GuardrailResult(
                blocked=True,
                filter_name="content_moderation",
                reason=reason,
                confidence=0.7,
            )

        return GuardrailResult(blocked=False, sanitized_content=content)

    async def check_output(self, content: str) -> GuardrailResult:
        """Run all output-side guardrails."""
        return await self.output_filter.check(content)
