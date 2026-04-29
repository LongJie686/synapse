"""Memory node - reads/writes memory within a graph execution."""

from __future__ import annotations

from synapse_core.graph.state import RunState, Message
from synapse_core.memory.manager import MemoryManager
from synapse_core.streaming import memory_retrieve, memory_store


async def memory_read_node(state: RunState, memory: MemoryManager) -> dict:
    """
    Graph node that retrieves relevant long-term memories
    and injects them into the graph state context.
    """
    query = state.messages[-1].content if state.messages else ""
    if not query:
        return {"context": {}}

    results = await memory.retrieve(query, top_k=3)

    memory_context = []
    for entry in results:
        memory_context.append(f"[{entry.memory_type.value}] {entry.content}")

    if memory_context:
        context_text = "\n".join(memory_context)
        return {
            "context": {**state.context, "relevant_memories": context_text},
            "metadata": {**state.metadata, "memory_count": len(results)},
        }

    return {"context": state.context}


async def memory_write_node(state: RunState, memory: MemoryManager) -> dict:
    """
    Graph node that saves important interactions to long-term memory
    and triggers user profile updates.
    """
    if state.user_id and state.messages:
        # Auto-update user profile from conversation
        conversation_text = "\n".join(m.content[:200] for m in state.messages[-6:])
        await memory.auto_update_profile(state.user_id, conversation_text)

    return {}
