"""Feedback collection system for data flywheel and continuous improvement."""

from __future__ import annotations

import hashlib
import json
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FeedbackType(str, Enum):
    RATING = "rating"
    THUMBS = "thumbs"
    COMMENT = "comment"
    IMPLICIT = "implicit"


class FeedbackEntry(BaseModel):
    """Single feedback record."""
    feedback_id: str = ""
    run_id: str = ""
    agent_id: str = ""
    user_id: str = ""
    feedback_type: FeedbackType
    score: float = 0.0  # 1-5 for rating, 1/-1 for thumbs, 0 for comment-only
    comment: str = ""
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = 0.0


class FeedbackStore:
    """In-memory feedback store with aggregation capabilities."""

    def __init__(self) -> None:
        self._entries: list[FeedbackEntry] = []

    def add(self, entry: FeedbackEntry) -> str:
        if not entry.feedback_id:
            entry.feedback_id = hashlib.md5(
                f"{entry.run_id}:{entry.agent_id}:{time.time()}".encode()
            ).hexdigest()[:12]
        if not entry.timestamp:
            entry.timestamp = time.time()
        self._entries.append(entry)
        return entry.feedback_id

    def get_by_run(self, run_id: str) -> list[FeedbackEntry]:
        return [e for e in self._entries if e.run_id == run_id]

    def get_by_agent(self, agent_id: str) -> list[FeedbackEntry]:
        return [e for e in self._entries if e.agent_id == agent_id]

    def get_by_user(self, user_id: str) -> list[FeedbackEntry]:
        return [e for e in self._entries if e.user_id == user_id]

    def get_all(self, limit: int = 100, offset: int = 0) -> list[FeedbackEntry]:
        return self._entries[offset:offset + limit]

    def get_stats(self, agent_id: str | None = None) -> dict[str, Any]:
        """Aggregate feedback statistics."""
        entries = self._entries
        if agent_id:
            entries = [e for e in entries if e.agent_id == agent_id]

        if not entries:
            return {"count": 0, "avg_score": 0.0, "positive_ratio": 0.0}

        scored = [e for e in entries if e.score != 0]
        positive = sum(1 for e in scored if e.score > 0)

        return {
            "count": len(entries),
            "scored_count": len(scored),
            "avg_score": sum(e.score for e in scored) / len(scored) if scored else 0.0,
            "positive_ratio": positive / len(scored) if scored else 0.0,
            "by_type": {
                t.value: sum(1 for e in entries if e.feedback_type == t)
                for t in FeedbackType
            },
        }

    def get_low_score_entries(self, threshold: float = 2.0) -> list[FeedbackEntry]:
        """Get entries that need review (low scores)."""
        return [e for e in self._entries if 0 < e.score <= threshold]

    def clear(self) -> None:
        self._entries.clear()


class ABTestExperiment(BaseModel):
    """A/B testing experiment definition."""
    name: str
    control_prompt: str = ""
    treatment_prompt: str = ""
    traffic_percentage: int = 50  # % of users seeing treatment
    status: str = "active"  # active | completed | paused
    results: dict[str, list[float]] = Field(default_factory=lambda: {"control": [], "treatment": []})


class ABTestManager:
    """A/B testing framework with deterministic user assignment."""

    def __init__(self) -> None:
        self._experiments: dict[str, ABTestExperiment] = {}

    def create(self, experiment: ABTestExperiment) -> None:
        self._experiments[experiment.name] = experiment

    def get_variant(self, experiment_name: str, user_id: str) -> str:
        """Deterministically assign user to control or treatment."""
        exp = self._experiments.get(experiment_name)
        if not exp or exp.status != "active":
            return "control"

        hash_val = int(hashlib.md5(f"{experiment_name}:{user_id}".encode()).hexdigest(), 16) % 100
        return "treatment" if hash_val < exp.traffic_percentage else "control"

    def record_result(self, experiment_name: str, variant: str, score: float) -> None:
        exp = self._experiments.get(experiment_name)
        if not exp:
            return
        if variant in exp.results:
            exp.results[variant].append(score)

    def get_results(self, experiment_name: str) -> dict[str, Any] | None:
        exp = self._experiments.get(experiment_name)
        if not exp:
            return None

        control = exp.results.get("control", [])
        treatment = exp.results.get("treatment", [])

        control_avg = sum(control) / len(control) if control else 0.0
        treatment_avg = sum(treatment) / len(treatment) if treatment else 0.0

        improvement = 0.0
        if control_avg > 0:
            improvement = ((treatment_avg - control_avg) / control_avg) * 100

        return {
            "experiment": experiment_name,
            "status": exp.status,
            "control_count": len(control),
            "treatment_count": len(treatment),
            "control_avg": round(control_avg, 3),
            "treatment_avg": round(treatment_avg, 3),
            "improvement_pct": round(improvement, 1),
            "recommendation": self._recommend(improvement, len(control), len(treatment)),
        }

    def _recommend(self, improvement: float, n_control: int, n_per_group: int) -> str:
        min_samples = 5
        if n_control < min_samples or n_per_group < min_samples:
            return "insufficient_data"
        if improvement > 5.0:
            return "promote_treatment"
        elif improvement < -5.0:
            return "keep_control"
        return "no_significant_difference"

    def list_experiments(self) -> list[str]:
        return list(self._experiments.keys())

    def stop(self, experiment_name: str) -> None:
        exp = self._experiments.get(experiment_name)
        if exp:
            exp.status = "completed"
