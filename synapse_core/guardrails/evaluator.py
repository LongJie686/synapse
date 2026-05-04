"""Output quality evaluation: hallucination detection, relevance scoring.

Lightweight heuristic-based evaluation that doesn't require additional LLM calls.
For production, replace with LLM-based evaluators or use frameworks like Ragas.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel


class EvaluationResult(BaseModel):
    score: float  # 0.0 - 1.0
    category: str  # hallucination_risk, relevance, completeness, safety
    details: str = ""
    passed: bool = True


class OutputEvaluator:
    """Heuristic-based output quality evaluator."""

    def __init__(
        self,
        hallucination_threshold: float = 0.5,
        min_relevance_score: float = 0.3,
    ) -> None:
        self._hallucination_threshold = hallucination_threshold
        self._min_relevance_score = min_relevance_score

    # Phrases that indicate the model is guessing or fabricating
    _HALLUCINATION_SIGNALS = [
        r"i('m| am) not (sure|certain)",
        r"i (think|believe|guess|assume)",
        r"(probably|likely|possibly|maybe|perhaps)",
        r"as far as i know",
        r"to the best of my knowledge",
        r"i don't have (access|information|data)",
        r"i cannot (verify|confirm|guarantee)",
    ]

    # Phrases indicating grounded, factual response
    _GROUNDED_SIGNALS = [
        r"according to",
        r"based on (the|provided|given)",
        r"the (data|document|source|reference|file) (shows|states|indicates)",
        r"as (stated|mentioned|shown) in",
    ]

    _compiled_hallucination = None
    _compiled_grounded = None

    def _get_patterns(self) -> tuple[list[re.Pattern], list[re.Pattern]]:
        if self._compiled_hallucination is None:
            self._compiled_hallucination = [
                re.compile(p, re.IGNORECASE) for p in self._HALLUCINATION_SIGNALS
            ]
            self._compiled_grounded = [
                re.compile(p, re.IGNORECASE) for p in self._GROUNDED_SIGNALS
            ]
        return self._compiled_hallucination, self._compiled_grounded

    def evaluate_hallucination_risk(
        self, response: str, context: str = ""
    ) -> EvaluationResult:
        """Estimate hallucination risk based on hedging language patterns."""
        if not response:
            return EvaluationResult(
                score=0.0, category="hallucination_risk", details="Empty response", passed=True
            )

        hallucination_patterns, grounded_patterns = self._get_patterns()
        hedging_count = sum(1 for p in hallucination_patterns if p.search(response))
        grounded_count = sum(1 for p in grounded_patterns if p.search(response))

        # If context is provided, check if key facts from context appear in response
        context_overlap = 0.0
        if context:
            context_words = set(context.lower().split())
            response_words = set(response.lower().split())
            if context_words:
                context_overlap = len(context_words & response_words) / min(len(context_words), 50)

        # Score: more hedging = higher risk, more grounding = lower risk
        risk = max(0.0, min(1.0, hedging_count * 0.15 - grounded_count * 0.2 - context_overlap * 0.3))
        passed = risk < self._hallucination_threshold

        return EvaluationResult(
            score=round(1.0 - risk, 2),
            category="hallucination_risk",
            details=f"hedging={hedging_count}, grounded={grounded_count}, context_overlap={context_overlap:.2f}",
            passed=passed,
        )

    def evaluate_relevance(
        self, query: str, response: str
    ) -> EvaluationResult:
        """Estimate relevance based on keyword overlap between query and response."""
        if not query or not response:
            return EvaluationResult(
                score=0.5, category="relevance", details="Missing input", passed=True
            )

        query_words = set(re.findall(r"\w+", query.lower()))
        response_words = set(re.findall(r"\w+", response.lower()))

        if not query_words:
            return EvaluationResult(score=0.5, category="relevance", passed=True)

        overlap = len(query_words & response_words) / len(query_words)
        passed = overlap >= self._min_relevance_score

        return EvaluationResult(
            score=round(overlap, 2),
            category="relevance",
            details=f"keyword_overlap={len(query_words & response_words)}/{len(query_words)}",
            passed=passed,
        )

    def evaluate_completeness(self, response: str, min_length: int = 20) -> EvaluationResult:
        """Check if response is substantive enough."""
        if not response:
            return EvaluationResult(
                score=0.0, category="completeness", details="Empty response", passed=False
            )

        stripped = response.strip()
        word_count = len(stripped.split())
        score = min(1.0, word_count / max(min_length, 1))

        return EvaluationResult(
            score=round(score, 2),
            category="completeness",
            details=f"word_count={word_count}",
            passed=word_count >= min_length // 2,
        )

    def evaluate_all(
        self, query: str, response: str, context: str = ""
    ) -> list[EvaluationResult]:
        """Run all evaluations and return results."""
        return [
            self.evaluate_hallucination_risk(response, context),
            self.evaluate_relevance(query, response),
            self.evaluate_completeness(response),
        ]
