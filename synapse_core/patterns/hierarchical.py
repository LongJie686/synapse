"""Hierarchical pattern - recursive agent delegation."""

from __future__ import annotations

import time
import uuid
from typing import Any, AsyncIterator

from synapse_core.agent import AgentDefinition
from synapse_core.streaming import StreamEvent, run_start, run_end, agent_think, agent_message, agent_delegate, run_error
from synapse_core.tools import ToolRegistry
from synapse_core.graph import GraphBuilder
from synapse_core.graph.state import TokenUsage


class HierarchicalPattern:
    """
    Hierarchical pattern: root agent can delegate to child agents,
    who can further delegate to their children. Max depth prevents infinite recursion.
    """

    def __init__(
        self,
        root: AgentDefinition,
        children: dict[str, list[AgentDefinition]],
        tool_registry: ToolRegistry,
        max_depth: int = 3,
        context_passing: str = "summary",
    ) -> None:
        self.root = root
        self.children = children
        self.tool_registry = tool_registry
        self.max_depth = max_depth
        self.context_passing = context_passing

    async def run(self, task: str, run_id: str | None = None) -> AsyncIterator[StreamEvent]:
        run_id = run_id or str(uuid.uuid4())
        start_time = time.monotonic()

        yield run_start(run_id, self.root.id)

        try:
            async for event in self._run_recursive(task, self.root, run_id, depth=0):
                yield event

            duration = (time.monotonic() - start_time) * 1000
            yield run_end(run_id, {"total_tokens": 0}, duration)
        except Exception as e:
            yield run_error(run_id, str(e), recoverable=True)

    async def _run_recursive(
        self,
        task: str,
        agent: AgentDefinition,
        run_id: str,
        depth: int,
    ) -> AsyncIterator[StreamEvent]:
        if depth >= self.max_depth:
            yield agent_think(agent.id, f"Max depth {self.max_depth} reached, answering directly")
            builder = GraphBuilder(agent, self.tool_registry)
            async for event in builder.run(task, f"{run_id}-d{depth}"):
                yield event
            return

        yield agent_think(agent.id, f"Processing at depth {depth}/{self.max_depth}")

        builder = GraphBuilder(agent, self.tool_registry)
        result_messages: list[str] = []

        async for event in builder.run(task, f"{run_id}-d{depth}"):
            yield event
            if event.type == "agent:message":
                result_messages.append(event.data.get("content", ""))

        # Check if this agent has children to delegate to
        child_agents = self.children.get(agent.id, [])
        if not child_agents:
            return

        # Delegate relevant subtasks to children
        for child in child_agents:
            yield agent_delegate(agent.id, child.id, f"Subtask from {agent.name}")
            async for event in self._run_recursive(task, child, run_id, depth + 1):
                yield event
