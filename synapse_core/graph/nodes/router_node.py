"""Router node - uses LLM to decide which agent to route to next."""

from __future__ import annotations

import json

from synapse_core.graph.state import RunState, Message
from synapse_core.llm import LLMMessage
from synapse_core.llm.providers import create_provider
from synapse_core.llm import LLMConfig


async def router_node(
    state: RunState,
    agent_ids: list[str],
    agent_descriptions: dict[str, str],
    llm_config: LLMConfig,
) -> dict:
    """
    Graph node that uses LLM to route to the most appropriate next agent.
    Returns updated state with next_agents set.
    """
    if not agent_ids:
        return {"next_agents": []}

    if len(agent_ids) == 1:
        return {"next_agents": agent_ids}

    last_message = state.messages[-1].content if state.messages else ""
    agent_list = "\n".join(f"- {aid}: {agent_descriptions.get(aid, 'No description')}" for aid in agent_ids)

    prompt = f"""Based on the conversation, decide which agent should handle the next step.

Available agents:
{agent_list}

Last message: {last_message[:500]}

Respond with ONLY a JSON array of agent IDs, in priority order. Example: ["agent-1"]"""

    provider = create_provider(llm_config)
    messages = [LLMMessage(role="user", content=prompt)]

    response = await provider.invoke(messages)

    try:
        selected = json.loads(response.content)
        if isinstance(selected, list) and all(isinstance(s, str) for s in selected):
            return {"next_agents": [s for s in selected if s in agent_ids][:3]}
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: select first agent
    return {"next_agents": [agent_ids[0]]}
