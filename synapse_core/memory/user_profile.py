"""User profile auto-extraction and update from conversations."""

from __future__ import annotations

import json
import time
from typing import Any

from synapse_core.memory import UserTrait, UserProfile


class ProfileExtractor:
    """
    Extracts user traits from conversation content and updates the user profile.

    Extracts: preferred languages, technical stack, interests, communication style,
    expertise level, frequently used tools.
    """

    def __init__(self, summarize_fn: Any | None = None) -> None:
        self._summarize_fn = summarize_fn

    async def extract_and_update(
        self,
        profile: UserProfile,
        conversation_text: str,
    ) -> UserProfile:
        """Extract traits from a conversation and merge into existing profile."""
        if self._summarize_fn:
            return await self._llm_extract(profile, conversation_text)
        return self._rule_based_extract(profile, conversation_text)

    async def _llm_extract(self, profile: UserProfile, conversation_text: str) -> UserProfile:
        """Use LLM to extract structured profile data."""
        prompt = f"""Analyze this conversation and extract user profile information.
Return a JSON object with these fields:
- traits: list of {{"key": string, "value": string, "confidence": number 0-1}}
- interests: list of interest topics
- communication_style: one of "concise-technical", "detailed-explanatory", "casual", "formal"
- expertise_areas: object mapping domain to "beginner"/"intermediate"/"expert"

Current profile: {profile.model_dump_json()}
Conversation: {conversation_text[:3000]}

Return ONLY the JSON, no other text."""

        result_text = await self._summarize_fn("", prompt)
        try:
            extracted = json.loads(result_text)
        except (json.JSONDecodeError, TypeError):
            return profile

        return self._merge_profile(profile, extracted)

    def _rule_based_extract(self, profile: UserProfile, conversation_text: str) -> UserProfile:
        """Rule-based trait extraction without LLM."""
        text_lower = conversation_text.lower()
        new_traits: list[UserTrait] = []
        new_interests: list[str] = list(profile.interests)
        expertise: dict[str, str] = dict(profile.expertise_level)

        # Detect programming language preferences
        lang_keywords = {
            "python": "preferred_language",
            "java": "preferred_language",
            "typescript": "preferred_language",
            "golang": "preferred_language",
            "rust": "preferred_language",
        }
        for keyword, trait_key in lang_keywords.items():
            if keyword in text_lower:
                new_traits.append(UserTrait(key=trait_key, value=keyword, confidence=0.7))

        # Detect technical interests
        interest_keywords = {
            "machine learning": "machine-learning",
            "deep learning": "deep-learning",
            "微服务": "microservice",
            "架构": "architecture",
            "数据库": "database",
            "agent": "ai-agent",
            "rag": "rag",
            "llm": "llm",
            "docker": "docker",
            "k8s": "kubernetes",
            "kubernetes": "kubernetes",
            "redis": "redis",
            "kafka": "kafka",
        }
        for keyword, interest in interest_keywords.items():
            if keyword in text_lower and interest not in new_interests:
                new_interests.append(interest)

        # Detect communication style
        style = profile.communication_style
        if any(w in text_lower for w in ["简单说", "简短", "briefly", "concise"]):
            style = "concise-technical"
        elif any(w in text_lower for w in ["详细", "解释一下", "explain", "detail"]):
            style = "detailed-explanatory"

        # Detect expertise signals
        if any(w in text_lower for w in ["生产环境", "线上", "production", "高并发"]):
            expertise["backend"] = "expert"
        if any(w in text_lower for w in ["基础", "入门", "beginner", "怎么用"]):
            for area in ["backend", "ai", "database"]:
                if area not in expertise:
                    expertise[area] = "beginner"

        return self._merge_profile(profile, {
            "traits": [t.model_dump() for t in new_traits],
            "interests": new_interests,
            "communication_style": style,
            "expertise_areas": expertise,
        })

    def _merge_profile(self, profile: UserProfile, extracted: dict[str, Any]) -> UserProfile:
        """Merge extracted data into existing profile, deduplicating traits."""
        # Merge traits - update existing, add new
        existing_trait_keys = {t.key for t in profile.traits}
        new_traits = list(profile.traits)

        for raw_trait in extracted.get("traits", []):
            if isinstance(raw_trait, dict):
                trait = UserTrait(
                    key=raw_trait.get("key", ""),
                    value=raw_trait.get("value", ""),
                    confidence=raw_trait.get("confidence", 0.5),
                )
                if trait.key in existing_trait_keys:
                    # Update: keep higher confidence
                    new_traits = [
                        t if not (t.key == trait.key and trait.confidence > t.confidence)
                        else trait
                        for t in new_traits
                    ]
                else:
                    new_traits.append(trait)
                    existing_trait_keys.add(trait.key)

        # Merge interests (deduplicate)
        interests = list(set(profile.interests + extracted.get("interests", [])))

        # Update expertise
        expertise = {**profile.expertise_level, **extracted.get("expertise_areas", {})}

        return UserProfile(
            user_id=profile.user_id,
            traits=new_traits,
            interests=interests,
            communication_style=extracted.get("communication_style", profile.communication_style),
            expertise_level=expertise,
            frequent_tools=profile.frequent_tools,
            last_updated=time.time(),
            version=profile.version + 1,
        )
