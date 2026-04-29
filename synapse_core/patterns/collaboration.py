"""Collaboration pattern - agents discuss to reach consensus."""

from __future__ import annotations

import time
import uuid
from typing import Any, AsyncIterator

from synapse_core.agent import AgentDefinition
from synapse_core.llm import LLMMessage
from synapse_core.llm.providers import create_provider
from synapse_core.streaming import StreamEvent, run_start, run_end, agent_think, agent_message, run_error
from synapse_core.graph.state import TokenUsage


class CollaborationPattern:
    """
    Collaboration pattern: multiple agents discuss a topic in rounds,
    building on each other's contributions to reach a refined answer.

    Consensus strategies: unanimous, majority, moderator-decides.
    """

    def __init__(
        self,
        agents: list[AgentDefinition],
        max_rounds: int = 3,
        consensus_strategy: str = "moderator-decides",
        moderator: AgentDefinition | None = None,
    ) -> None:
        self.agents = agents
        self.max_rounds = max_rounds
        self.consensus_strategy = consensus_strategy
        self.moderator = moderator
        self._providers = {a.id: create_provider(a.llm) for a in agents}
        if moderator:
            self._providers[moderator.id] = create_provider(moderator.llm)

    async def run(self, topic: str, run_id: str | None = None) -> AsyncIterator[StreamEvent]:
        run_id = run_id or str(uuid.uuid4())
        start_time = time.monotonic()

        yield run_start(run_id, "collaboration")

        discussion: list[str] = [f"Topic: {topic}"]
        final_responses: dict[str, str] = {}

        for round_num in range(self.max_rounds):
            yield agent_think("collaboration", f"Discussion round {round_num + 1}/{self.max_rounds}")

            for agent in self.agents:
                context = "\n".join(discussion[-10:])  # Last 10 entries for context
                prompt = self._build_agent_prompt(agent, topic, context, round_num)

                messages = [LLMMessage(role="system", content=prompt)]
                if context:
                    messages.append(LLMMessage(role="user", content=f"Discussion so far:\n{context}\n\nProvide your analysis:"))

                provider = self._providers[agent.id]
                response = await provider.invoke(messages)

                contribution = f"[{agent.name}]: {response.content}"
                discussion.append(contribution)
                final_responses[agent.id] = response.content

                yield agent_message(agent.id, response.content)

        # Moderator synthesis
        if self.moderator:
            yield agent_think("collaboration", "Moderator synthesizing final answer")
            synthesis_prompt = self._build_moderator_prompt(topic, discussion)

            messages = [LLMMessage(role="system", content=synthesis_prompt)]
            mod_provider = self._providers[self.moderator.id]
            synthesis = await mod_provider.invoke(messages)

            yield agent_message(self.moderator.id, synthesis.content)
        else:
            # Concatenate all responses
            combined = "\n\n".join(f"**{name}**:\n{resp}" for name, resp in final_responses.items())
            yield agent_message("collaboration", combined)

        duration = (time.monotonic() - start_time) * 1000
        yield run_end(run_id, {"total_tokens": 0}, duration)

    def _build_agent_prompt(self, agent: AgentDefinition, topic: str, context: str, round_num: int) -> str:
        return f"""You are {agent.name}, {agent.role}.
Goal: {agent.goal}
Backstory: {agent.backstory}

You are participating in a collaborative discussion about: {topic}
This is round {round_num + 1} of {self.max_rounds}.

{'Build on the previous contributions. Add your unique perspective.' if round_num > 0 else 'Provide your initial analysis.'}
Be concise and focused. Do not repeat what others have said."""

    def _build_moderator_prompt(self, topic: str, discussion: list[str]) -> str:
        full_discussion = "\n".join(discussion)
        return f"""You are a discussion moderator. Synthesize the following collaborative discussion into a clear, actionable conclusion.

Topic: {topic}

Discussion:
{full_discussion}

Provide a unified conclusion that incorporates the best insights from all participants."""
