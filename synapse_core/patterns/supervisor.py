"""Supervisor pattern - central agent routes tasks to specialists."""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

from synapse_core.agent import AgentDefinition
from synapse_core.llm import LLMConfig, LLMMessage
from synapse_core.llm.providers import create_provider
from synapse_core.streaming import StreamEvent, run_start, run_end, agent_think, agent_message, agent_delegate, run_error
from synapse_core.tools import ToolRegistry
from synapse_core.graph import GraphBuilder
from synapse_core.graph.state import RunState, Message, TokenUsage


class SupervisorPattern:
    """
    Supervisor pattern: a central coordinator agent delegates tasks
    to specialist worker agents based on the task nature.

    The supervisor uses LLM to decide which agent(s) to invoke.
    """

    def __init__(
        self,
        supervisor: AgentDefinition,
        workers: list[AgentDefinition],
        tool_registry: ToolRegistry,
        max_delegations: int = 5,
    ) -> None:
        self.supervisor = supervisor
        self.workers = {w.id: w for w in workers}
        self.tool_registry = tool_registry
        self.max_delegations = max_delegations
        self._supervisor_provider = create_provider(supervisor.llm)

    def _build_routing_prompt(self) -> str:
        worker_descriptions = "\n".join(
            f"- {w.id}: {w.role} - {w.goal}" for w in self.workers.values()
        )
        return f"""You are a task routing supervisor named {self.supervisor.name}.
Your role: {self.supervisor.role}

Available specialist agents:
{worker_descriptions}

When you receive a task, decide which specialist agent(s) should handle it.
Respond with a JSON object:
{{"agent_id": "<id>", "task": "<rephrased task for the specialist>", "reasoning": "<why this agent>"}}

If the task can be answered directly without delegation, respond with:
{{"agent_id": "self", "answer": "<your direct answer>"}}

You can delegate to at most {self.max_delegations} agents per conversation."""

    async def run(self, user_input: str, run_id: str | None = None) -> AsyncIterator[StreamEvent]:
        run_id = run_id or str(uuid.uuid4())
        start_time = __import__("time").monotonic()
        total_usage = TokenUsage()
        all_messages: list[Message] = []

        yield run_start(run_id, self.supervisor.id)

        try:
            # Step 1: Supervisor analyzes the task
            yield agent_think(self.supervisor.id, "Analyzing task for delegation...")

            routing_messages = [
                LLMMessage(role="system", content=self._build_routing_prompt()),
                LLMMessage(role="user", content=user_input),
            ]

            routing_response = await self._supervisor_provider.invoke(routing_messages)
            total_usage = total_usage.add(TokenUsage(**routing_response.usage) if routing_response.usage else TokenUsage())

            yield agent_think(self.supervisor.id, f"Routing decision: {routing_response.content[:200]}")

            # Step 2: Parse routing decision
            try:
                decision = json.loads(routing_response.content)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract from text
                decision = {"agent_id": "self", "answer": routing_response.content}

            # Step 3: Execute based on decision
            if decision.get("agent_id") == "self" or decision.get("answer"):
                # Supervisor answers directly
                answer = decision.get("answer", routing_response.content)
                yield agent_message(self.supervisor.id, answer)
            else:
                # Delegate to worker
                target_id = decision.get("agent_id", "")
                task = decision.get("task", user_input)

                if target_id in self.workers:
                    worker = self.workers[target_id]
                    yield agent_delegate(self.supervisor.id, target_id, task)

                    # Run worker
                    worker_builder = GraphBuilder(worker, self.tool_registry)
                    async for event in worker_builder.run(task, run_id):
                        yield event
                        if event.type == "run:end":
                            duration = event.data.get("durationMs", 0)
                        # Collect worker messages
                        if event.type == "agent:message":
                            all_messages.append(Message(role="assistant", content=event.data.get("content", "")))
                else:
                    yield agent_message(
                        self.supervisor.id,
                        f"I couldn't find the right specialist for this task. Let me try to help directly.\n\n{routing_response.content}",
                    )

            duration = (__import__("time").monotonic() - start_time) * 1000
            yield run_end(run_id, total_usage.model_dump(), duration)

        except Exception as e:
            yield run_error(run_id, str(e), recoverable=True)
