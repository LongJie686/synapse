"""Cost-aware LLM model router with fallback."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel

from synapse_core.llm import LLMConfig, LLMMessage, LLMResponse
from synapse_core.llm.providers import create_provider


class TaskComplexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class ModelProfile(BaseModel):
    """Model capabilities and cost profile."""

    model: str
    provider: str
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0
    context_window: int = 128000
    max_output: int = 4096
    capability: TaskComplexity = TaskComplexity.MODERATE
    latency_tier: int = 1  # 1=fast, 2=medium, 3=slow


# Pre-configured model profiles
MODEL_PROFILES: dict[str, ModelProfile] = {
    "gpt-4o": ModelProfile(
        model="gpt-4o", provider="openai",
        input_cost_per_1k=0.0025, output_cost_per_1k=0.01,
        context_window=128000, capability=TaskComplexity.COMPLEX, latency_tier=2,
    ),
    "gpt-4o-mini": ModelProfile(
        model="gpt-4o-mini", provider="openai",
        input_cost_per_1k=0.00015, output_cost_per_1k=0.0006,
        context_window=128000, capability=TaskComplexity.MODERATE, latency_tier=1,
    ),
    "gpt-3.5-turbo": ModelProfile(
        model="gpt-3.5-turbo", provider="openai",
        input_cost_per_1k=0.0005, output_cost_per_1k=0.0015,
        context_window=16385, capability=TaskComplexity.SIMPLE, latency_tier=1,
    ),
    "claude-sonnet-4-20250514": ModelProfile(
        model="claude-sonnet-4-20250514", provider="anthropic",
        input_cost_per_1k=0.003, output_cost_per_1k=0.015,
        context_window=200000, capability=TaskComplexity.COMPLEX, latency_tier=2,
    ),
    "claude-haiku-4-5-20251001": ModelProfile(
        model="claude-haiku-4-5-20251001", provider="anthropic",
        input_cost_per_1k=0.0008, output_cost_per_1k=0.004,
        context_window=200000, capability=TaskComplexity.MODERATE, latency_tier=1,
    ),
    # Domestic models (Chinese LLM ecosystem)
    "glm-4-plus": ModelProfile(
        model="glm-4-plus", provider="openai",
        input_cost_per_1k=0.05, output_cost_per_1k=0.05,
        context_window=128000, capability=TaskComplexity.COMPLEX, latency_tier=2,
    ),
    "glm-4-flash": ModelProfile(
        model="glm-4-flash", provider="openai",
        input_cost_per_1k=0.0001, output_cost_per_1k=0.0001,
        context_window=128000, capability=TaskComplexity.SIMPLE, latency_tier=1,
    ),
    "qwen-max": ModelProfile(
        model="qwen-max", provider="openai",
        input_cost_per_1k=0.02, output_cost_per_1k=0.06,
        context_window=32000, capability=TaskComplexity.COMPLEX, latency_tier=2,
    ),
    "qwen-turbo": ModelProfile(
        model="qwen-turbo", provider="openai",
        input_cost_per_1k=0.002, output_cost_per_1k=0.006,
        context_window=128000, capability=TaskComplexity.MODERATE, latency_tier=1,
    ),
    "deepseek-chat": ModelProfile(
        model="deepseek-chat", provider="openai",
        input_cost_per_1k=0.001, output_cost_per_1k=0.002,
        context_window=64000, capability=TaskComplexity.COMPLEX, latency_tier=1,
    ),
    "deepseek-reasoner": ModelProfile(
        model="deepseek-reasoner", provider="openai",
        input_cost_per_1k=0.004, output_cost_per_1k=0.016,
        context_window=64000, capability=TaskComplexity.COMPLEX, latency_tier=3,
    ),
}


class ModelRouter:
    """Routes requests to optimal models based on task complexity and cost."""

    def __init__(
        self,
        profiles: dict[str, ModelProfile] | None = None,
        default_model: str = "gpt-4o-mini",
    ) -> None:
        self.profiles = profiles or MODEL_PROFILES
        self.default_model = default_model
        self._fallback_chains: dict[str, list[str]] = {
            "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            "anthropic": ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"],
            "glm": ["glm-4-plus", "glm-4-flash"],
            "qwen": ["qwen-max", "qwen-turbo"],
            "deepseek": ["deepseek-chat", "deepseek-reasoner"],
        }
        # Cross-family fallback: when all models in a family fail
        self._cross_family_fallback = ["deepseek-chat", "glm-4-flash", "gpt-4o-mini"]

    def classify_complexity(self, messages: list[LLMMessage], **kwargs: Any) -> TaskComplexity:
        """Classify task complexity from message content."""
        total_content = " ".join(m.content for m in messages if m.content)

        # Heuristics for complexity classification
        complex_signals = sum(1 for kw in [
            "analyze", "design", "architect", "compare", "evaluate",
            "reason", "explain why", "critique", "refactor", "optimize",
            "multi-step", "complex",
        ] if kw in total_content.lower())

        moderate_signals = sum(1 for kw in [
            "write", "create", "summarize", "translate", "convert",
            "extract", "list", "describe", "how to",
        ] if kw in total_content.lower())

        has_tools = kwargs.get("tools") is not None
        msg_count = len(messages)

        score = 0
        score += complex_signals * 2
        score += moderate_signals
        score += min(msg_count - 1, 5)
        if has_tools:
            score += 3

        if score >= 8:
            return TaskComplexity.COMPLEX
        elif score >= 3:
            return TaskComplexity.MODERATE
        return TaskComplexity.SIMPLE

    def select_model(self, complexity: TaskComplexity, preference: str = "balanced") -> str:
        """Select the best model for given complexity and preference.

        preference: 'cost' (cheapest), 'quality' (best), 'balanced' (default)
        """
        candidates = list(self.profiles.values())
        complexity_order = {
            TaskComplexity.SIMPLE: TaskComplexity.SIMPLE,
            TaskComplexity.MODERATE: TaskComplexity.MODERATE,
            TaskComplexity.COMPLEX: TaskComplexity.COMPLEX,
        }
        min_capability = complexity_order[complexity]

        # Filter by minimum capability
        capable = [p for p in candidates if
                   list(TaskComplexity).index(p.capability) >= list(TaskComplexity).index(min_capability)]

        if not capable:
            return self.default_model

        if preference == "cost":
            return min(capable, key=lambda p: p.input_cost_per_1k).model
        elif preference == "quality":
            return max(capable, key=lambda p: list(TaskComplexity).index(p.capability)).model
        else:  # balanced
            balanced = sorted(capable, key=lambda p: (
                list(TaskComplexity).index(p.capability) * 0.6 +
                p.latency_tier * 0.3 +
                p.input_cost_per_1k * 100 * 0.1
            ), reverse=True)
            return balanced[0].model

    async def route(
        self,
        messages: list[LLMMessage],
        preference: str = "balanced",
        **kwargs: Any,
    ) -> LLMResponse:
        """Route request to optimal model and execute with fallback."""
        complexity = self.classify_complexity(messages, **kwargs)
        model_name = self.select_model(complexity, preference)
        profile = self.profiles.get(model_name)

        if not profile:
            profile = self.profiles[self.default_model]

        config = LLMConfig(
            provider=profile.provider,
            model=profile.model,
            temperature=kwargs.pop("temperature", 0.7),
            max_tokens=kwargs.pop("max_tokens", profile.max_output),
        )

        provider = create_provider(config)

        try:
            return await provider.invoke(messages, **kwargs)
        except Exception:
            # Try same-family fallback chain first
            fallback_chain = self._fallback_chains.get(profile.provider, [])
            for fallback_model in fallback_chain:
                if fallback_model == model_name:
                    continue
                fb_profile = self.profiles.get(fallback_model)
                if not fb_profile:
                    continue
                fb_config = LLMConfig(
                    provider=fb_profile.provider,
                    model=fb_profile.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                try:
                    fb_provider = create_provider(fb_config)
                    return await fb_provider.invoke(messages, **kwargs)
                except Exception:
                    continue

            # Cross-family fallback: try models from other families
            tried = set(fallback_chain) | {model_name}
            for cross_model in self._cross_family_fallback:
                if cross_model in tried:
                    continue
                fb_profile = self.profiles.get(cross_model)
                if not fb_profile:
                    continue
                fb_config = LLMConfig(
                    provider=fb_profile.provider,
                    model=fb_profile.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                try:
                    fb_provider = create_provider(fb_config)
                    return await fb_provider.invoke(messages, **kwargs)
                except Exception:
                    continue
            raise
