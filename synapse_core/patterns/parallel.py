"""Parallel pattern - multiple agents work simultaneously on subtasks."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncIterator

from synapse_core.agent import AgentDefinition
from synapse_core.streaming import StreamEvent, run_start, run_end, agent_think, agent_message, run_error
from synapse_core.tools import ToolRegistry
from synapse_core.graph import GraphBuilder
from synapse_core.graph.state import TokenUsage


class ParallelPattern:
    """
    Parallel pattern: multiple agents work on the same or different subtasks
    simultaneously, then results are merged.

    Merge strategies: concatenate, vote, llm-synthesis.
    """

    def __init__(
        self,
        agents: list[AgentDefinition],
        tool_registry: ToolRegistry,
        merge_strategy: str = "concatenate",
        timeout: float = 60.0,
    ) -> None:
        self.agents = agents
        self.tool_registry = tool_registry
        self.merge_strategy = merge_strategy
        self.timeout = timeout

    async def run(self, task: str, run_id: str | None = None) -> AsyncIterator[StreamEvent]:
        run_id = run_id or str(uuid.uuid4())
        start_time = time.monotonic()

        yield run_start(run_id, "parallel-coordinator")
        yield agent_think("parallel-coordinator", f"Launching {len(self.agents)} agents in parallel")

        # Run all agents concurrently
        builders = [GraphBuilder(agent, self.tool_registry) for agent in self.agents]
        tasks = [self._run_agent(builder, task, f"{run_id}-{i}") for i, builder in enumerate(builders)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results
        merged_content = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                merged_content.append(f"[Agent {self.agents[i].name}]: Error - {result}")
            elif isinstance(result, list):
                for event in result:
                    if event.type == "agent:message":
                        merged_content.append(f"[{self.agents[i].name}]: {event.data.get('content', '')}")

        final_output = "\n\n---\n\n".join(merged_content) if merged_content else "No results from agents."
        yield agent_message("parallel-coordinator", final_output)

        duration = (time.monotonic() - start_time) * 1000
        yield run_end(run_id, {"total_tokens": 0}, duration)

    async def _run_agent(self, builder: GraphBuilder, task: str, run_id: str) -> list[StreamEvent]:
        """Run a single agent and collect its events."""
        events = []
        async for event in builder.run(task, run_id):
            events.append(event)
        return events
