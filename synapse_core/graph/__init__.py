"""Graph builder - core orchestration engine."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncIterator

from synapse_core.agent import AgentDefinition
from synapse_core.llm import LLMConfig, LLMMessage
from synapse_core.llm.providers import create_provider
from synapse_core.streaming import (
    StreamEvent,
    run_start,
    run_end,
    run_error,
    agent_think,
    agent_message,
    agent_call_tool,
    tool_result,
)
from synapse_core.tools import ToolRegistry
from synapse_core.graph.state import RunState, Message, TokenUsage


class GraphBuilder:
    """Builds and executes agent workflows."""

    def __init__(
        self,
        agent: AgentDefinition,
        tool_registry: ToolRegistry,
    ) -> None:
        self.agent = agent
        self.tool_registry = tool_registry
        self._provider = create_provider(agent.llm)

    def _build_system_prompt(self) -> str:
        parts = [
            f"You are {self.agent.name}.",
            f"Role: {self.agent.role}",
            f"Goal: {self.agent.goal}",
            f"Background: {self.agent.backstory}",
        ]
        if self.agent.tools:
            tool_descriptions = []
            for tool_name in self.agent.tools:
                try:
                    tool_def = self.tool_registry.get(tool_name)
                    params_desc = ", ".join(f"{p.name}: {p.description}" for p in tool_def.parameters)
                    tool_descriptions.append(f"- {tool_name}: {tool_def.description} (params: {params_desc})")
                except KeyError:
                    tool_descriptions.append(f"- {tool_name}: (description not available)")
            parts.append("\nAvailable tools:\n" + "\n".join(tool_descriptions))
            parts.append(
                "To use a tool, respond with: TOOL_CALL: <tool_name>(<json_arguments>)\n"
                "After receiving the tool result, continue your response."
            )
        return "\n\n".join(parts)

    def _parse_tool_calls(self, content: str) -> list[tuple[str, dict[str, Any]]]:
        """Parse TOOL_CALL: name(args) from response."""
        import json
        import re

        calls = []
        pattern = r'TOOL_CALL:\s*(\w+)\((.+?)\)'
        for match in re.finditer(pattern, content, re.DOTALL):
            tool_name = match.group(1)
            try:
                args = json.loads(match.group(2))
            except json.JSONDecodeError:
                args = {"expression": match.group(2)}
            calls.append((tool_name, args))
        return calls

    async def run(self, user_input: str, run_id: str | None = None) -> AsyncIterator[StreamEvent]:
        """Execute agent with streaming events."""
        run_id = run_id or str(uuid.uuid4())
        start_time = time.monotonic()
        total_usage = TokenUsage()

        yield run_start(run_id, self.agent.id)

        system_prompt = self._build_system_prompt()
        messages = [Message(role="system", content=system_prompt)]

        if user_input:
            messages.append(Message(role="user", content=user_input))

        try:
            for iteration in range(self.agent.max_iterations):
                self._check_guardrails(messages, total_usage, iteration)

                llm_messages = [
                    LLMMessage(role=m.role, content=m.content, name=m.name)
                    for m in messages
                ]

                yield agent_think(self.agent.id, f"Iteration {iteration + 1}/{self.agent.max_iterations}")

                response = await self._provider.invoke(llm_messages)
                total_usage = total_usage.add(
                    TokenUsage(**response.usage) if response.usage else TokenUsage()
                )

                messages.append(Message(role="assistant", content=response.content))

                tool_calls = self._parse_tool_calls(response.content)

                if not tool_calls:
                    yield agent_message(self.agent.id, response.content)
                    break

                for tool_name, tool_args in tool_calls:
                    yield agent_call_tool(self.agent.id, tool_name, tool_args)
                    result = await self.tool_registry.execute(tool_name, **tool_args)

                    yield tool_result(tool_name, result.output if result.output else result.error, result.execution_time_ms)

                    messages.append(Message(
                        role="tool",
                        content=str(result.output if result.output else result.error),
                        name=tool_name,
                    ))

                    if result.error:
                        messages.append(Message(
                            role="assistant",
                            content=f"Tool {tool_name} failed: {result.error}. Let me try another approach.",
                        ))

            duration = (time.monotonic() - start_time) * 1000
            yield run_end(run_id, total_usage.model_dump(), duration)

        except Exception as e:
            yield run_error(run_id, str(e), recoverable=True)

    def _check_guardrails(self, messages: list[Message], usage: TokenUsage, iteration: int) -> None:
        if iteration >= self.agent.max_iterations:
            raise RuntimeError(f"Max iterations reached: {self.agent.max_iterations}")
        if usage.total_tokens > self.agent.guardrails.max_tokens_per_run:
            raise RuntimeError(f"Token limit exceeded: {usage.total_tokens}")
