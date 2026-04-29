"""Configuration management with Pydantic settings."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    provider: Literal["openai", "anthropic", "ollama", "azure-openai"] = "openai"
    model: str = "gpt-4o"
    base_url: str | None = None
    api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    temperature: float = 0.7
    max_tokens: int = 4096

    model_config = {"env_prefix": "SYNAPSE_LLM_", "populate_by_name": True}


class MemorySettings(BaseSettings):
    short_term_type: Literal["conversation-buffer", "sliding-window", "summary-buffer"] = "summary-buffer"
    long_term_provider: Literal["pgvector", "chroma", "qdrant"] = "chroma"
    max_context_tokens: int = 8000
    compaction_threshold: float = 0.8
    auto_save_importance: float = 0.6

    model_config = {"env_prefix": "SYNAPSE_MEMORY_"}


class SynapseSettings(BaseSettings):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    debug: bool = Field(default=False, alias="SYNAPSE_DEBUG")
    log_level: str = Field(default="INFO", alias="SYNAPSE_LOG_LEVEL")

    model_config = {"env_prefix": "SYNAPSE_", "populate_by_name": True}


def get_settings() -> SynapseSettings:
    return SynapseSettings()
