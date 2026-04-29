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
    memory_retrieve,
    memory_store,
)
from synapse_core.tools import ToolRegistry
from synapse_core.graph.state import RunState, Message, TokenUsage

# Lazy import to avoid circular dependency
_MEMORY_MANAGER = None


def _get_memory_manager():
    global _MEMORY_MANAGER
    if _MEMORY_MANAGER is None:
        from synapse_core.memory.manager import MemoryManager
        _MEMORY_MANAGER = MemoryManager(short_term_type="conversation-buffer")
    return _MEMORY_MANAGER


class GraphBuilder:
    """Builds and executes agent workflows."""

    def __init__(
        self,
        agent: AgentDefinition,
        tool_registry: ToolRegistry,
        memory_manager=None,
    ) -> None:
        self.agent = agent
        self.tool_registry = tool_registry
        self._provider = create_provider(agent.llm)
        self._memory = memory_manager or _get_memory_manager()

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
        """Parse TOOL_CALL patterns from response.

        Supports two formats:
          TOOL_CALL: tool_name({"key": "value"})
          TOOL_CALL: {"key": "value"}  (uses first available tool from agent config)
        """
        import json
        import re

        calls = []

        # Format 1: TOOL_CALL: name(args)
        pattern_named = r'TOOL_CALL:\s*(\w+)\((.+?)\)'
        for match in re.finditer(pattern_named, content, re.DOTALL):
            tool_name = match.group(1)
            try:
                args = json.loads(match.group(2))
            except json.JSONDecodeError:
                args = {"expression": match.group(2)}
            calls.append((tool_name, args))

        if calls:
            return calls

        # Format 2: TOOL_CALL: {json} (no function name -- infer from agent tools)
        pattern_json = r'TOOL_CALL:\s*(\{.+?\})'
        for match in re.finditer(pattern_json, content, re.DOTALL):
            try:
                args = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            tool_name = self._infer_tool(args)
            calls.append((tool_name, args))

        if calls:
            return calls

        # Format 3: tool_name\n{json} (GLM-style: name and JSON on separate lines)
        for tool_name in self.agent.tools:
            escaped = re.escape(tool_name)
            pattern_loose = rf'(?:^|\n){escaped}\s*\n\s*(\{{.+?\}})'
            for match in re.finditer(pattern_loose, content, re.DOTALL):
                try:
                    args = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
                calls.append((tool_name, args))

        if calls:
            return calls

        # Format 4: any standalone JSON with known arg keys
        for match in re.finditer(r'(\{[^{}]{5,200}?\})', content, re.DOTALL):
            try:
                args = json.loads(match.group(1))
                if isinstance(args, dict) and any(k in args for k in ("expression", "query", "url", "sql")):
                    tool_name = self._infer_tool(args)
                    calls.append((tool_name, args))
            except (json.JSONDecodeError, ValueError):
                continue

        return calls

    def _infer_tool(self, args: dict[str, Any]) -> str:
        """Infer which tool to use from the arguments."""
        if not self.agent.tools:
            return "calculator"
        if "expression" in args:
            return "calculator"
        if "query" in args:
            return "web_search"
        if "url" in args:
            return "http_request"
        if "sql" in args:
            return "sql_query"
        return self.agent.tools[0]

    async def run(self, user_input: str, run_id: str | None = None, session_id: str | None = None) -> AsyncIterator[StreamEvent]:
        """Execute agent with streaming events, memory integration."""
        run_id = run_id or str(uuid.uuid4())
        start_time = time.monotonic()
        total_usage = TokenUsage()

        yield run_start(run_id, self.agent.id)

        system_prompt = self._build_system_prompt()

        # Load conversation history from short-term memory
        history = await self._memory.get_conversation_context(session_id or "default")

        # Load relevant long-term memories
        long_term_memories = await self._memory.retrieve(user_input, top_k=3)
        if long_term_memories:
            memory_context = "\n\n[Relevant past memories]\n" + "\n".join(
                f"- ({m.memory_type.value}) {m.content}" for m in long_term_memories
            )
            yield memory_retrieve("long-term", user_input, len(long_term_memories))
        else:
            memory_context = ""

        # Build full system prompt with memory context
        full_system = system_prompt + memory_context

        messages = [Message(role="system", content=full_system)]

        # Add history messages
        for msg in history[-10:]:  # Last 10 messages for context window
            messages.append(msg)

        if user_input:
            messages.append(Message(role="user", content=user_input))
            # Save user message to short-term memory
            await self._memory.add_message(
                session_id or "default",
                Message(role="user", content=user_input),
            )

        final_content = ""

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
                    final_content = response.content
                    yield agent_message(self.agent.id, response.content)
                    break

                for tool_name, tool_args in tool_calls:
                    yield agent_call_tool(self.agent.id, tool_name, tool_args)
                    result = await self.tool_registry.execute(tool_name, **tool_args)

                    yield tool_result(tool_name, result.output if result.output else result.error, result.execution_time_ms)

                    tool_output = str(result.output if result.output else result.error)
                    messages.append(Message(
                        role="user",
                        content=f"[Tool Result: {tool_name}]\n{tool_output}",
                        name=tool_name,
                    ))

                    if result.error:
                        messages.append(Message(
                            role="assistant",
                            content=f"Tool {tool_name} failed: {result.error}. Let me try another approach.",
                        ))

            # Save assistant message to short-term memory
            if final_content:
                await self._memory.add_message(
                    session_id or "default",
                    Message(role="assistant", content=final_content),
                )
                yield memory_store("short-term", session_id or "default")

            duration = (time.monotonic() - start_time) * 1000
            yield run_end(run_id, total_usage.model_dump(), duration)

        except Exception as e:
            yield run_error(run_id, str(e), recoverable=True)

    def _check_guardrails(self, messages: list[Message], usage: TokenUsage, iteration: int) -> None:
        if iteration >= self.agent.max_iterations:
            raise RuntimeError(f"Max iterations reached: {self.agent.max_iterations}")
        if usage.total_tokens > self.agent.guardrails.max_tokens_per_run:
            raise RuntimeError(f"Token limit exceeded: {usage.total_tokens}")
