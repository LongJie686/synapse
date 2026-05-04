"""CrewAI-style sequential task orchestration with context passing."""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field

from synapse_core.agent import AgentDefinition
from synapse_core.llm import LLMMessage
from synapse_core.llm.providers import create_provider
from synapse_core.streaming import (
    StreamEvent,
    run_start,
    run_end,
    agent_think,
    agent_message,
    run_error,
)
from synapse_core.tools import ToolRegistry
from synapse_core.graph import GraphBuilder
from synapse_core.graph.state import TokenUsage
from synapse_core.observability import get_hub


class TaskDefinition(BaseModel):
    """CrewAI-style task definition with expected output validation."""

    id: str
    description: str
    expected_output: str = ""
    agent_id: str = ""
    context_task_ids: list[str] = Field(default_factory=list)
    output: str = ""
    status: str = "pending"  # pending | running | completed | failed


class CrewPattern:
    """
    CrewAI-style sequential task execution with automatic context passing.

    Key features:
    - Tasks define description, expected_output, and context dependencies
    - Sequential execution: each task gets outputs from its context dependencies
    - Output validation against expected_output format
    - Per-task agent assignment
    """

    def __init__(
        self,
        agents: dict[str, AgentDefinition],
        tasks: list[TaskDefinition],
        tool_registry: ToolRegistry | None = None,
        validate_output: bool = True,
    ) -> None:
        self.agents = agents
        self.tasks = tasks
        self.tool_registry = tool_registry
        self.validate_output = validate_output
        self._task_map = {t.id: t for t in tasks}

    async def run(self, goal: str, run_id: str | None = None) -> AsyncIterator[StreamEvent]:
        run_id = run_id or str(uuid.uuid4())
        start_time = time.monotonic()
        total_usage = TokenUsage()
        hub = get_hub()

        yield run_start(run_id, "crew")
        yield agent_think("crew", f"Starting crew execution: {len(self.tasks)} tasks")

        try:
            for task in self.tasks:
                task.status = "running"
                agent = self.agents.get(task.agent_id)

                if not agent:
                    yield agent_think("crew", f"No agent assigned for task {task.id}, skipping")
                    task.status = "failed"
                    continue

                # Build context from dependent tasks
                context = self._build_context(task)

                yield agent_think(
                    agent.id,
                    f"Task: {task.description[:100]}...",
                )

                # Execute the task
                task_prompt = self._build_task_prompt(task, context, goal)

                if self.tool_registry and agent.tools:
                    builder = GraphBuilder(agent, self.tool_registry)
                    result_text = ""
                    async for event in builder.run(task_prompt, f"{run_id}-{task.id}"):
                        yield event
                        if event.type == "agent:message":
                            result_text = event.data.get("content", "")
                        if event.type == "run:end":
                            wu = event.data.get("tokenUsage", event.data.get("token_usage", {}))
                            if wu:
                                total_usage = total_usage.add(TokenUsage(
                                    prompt_tokens=wu.get("prompt_tokens", 0),
                                    completion_tokens=wu.get("completion_tokens", 0),
                                    total_tokens=wu.get("total_tokens", 0),
                                ))
                else:
                    provider = create_provider(agent.llm)
                    messages = [
                        LLMMessage(role="system", content=self._build_system_prompt(agent)),
                        LLMMessage(role="user", content=task_prompt),
                    ]
                    t0 = time.monotonic()
                    response = await provider.invoke(messages)
                    call_dur = (time.monotonic() - t0) * 1000
                    if response.usage:
                        u = response.usage
                        total_usage = total_usage.add(TokenUsage(**u))
                        hub.track_llm_call(
                            provider=agent.llm.provider,
                            model=response.model or agent.llm.model,
                            tokens_in=u.get("prompt_tokens", 0),
                            tokens_out=u.get("completion_tokens", 0),
                            duration_ms=call_dur,
                        )
                    result_text = response.content
                    yield agent_message(agent.id, result_text)

                if result_text:
                    task.output = result_text
                    task.status = "completed"

                    # Validate output if enabled
                    if self.validate_output and task.expected_output:
                        validation = self._validate_output(result_text, task.expected_output)
                        if not validation["passed"]:
                            yield agent_think(
                                agent.id,
                                f"Output validation note: {validation['note']}",
                            )
                else:
                    task.status = "failed"

            # Final summary
            completed = sum(1 for t in self.tasks if t.status == "completed")
            failed = sum(1 for t in self.tasks if t.status == "failed")
            yield agent_think(
                "crew",
                f"Crew execution complete: {completed} completed, {failed} failed",
            )

            # Yield final combined output from the last task
            last_completed = None
            for t in reversed(self.tasks):
                if t.status == "completed":
                    last_completed = t
                    break

            if last_completed:
                yield agent_message("crew", last_completed.output)

            duration = (time.monotonic() - start_time) * 1000
            yield run_end(run_id, total_usage.model_dump(), duration)

        except Exception as e:
            yield run_error(run_id, str(e), recoverable=True)

    def _build_context(self, task: TaskDefinition) -> str:
        """Build context string from dependent task outputs."""
        if not task.context_task_ids:
            return ""

        parts = []
        for dep_id in task.context_task_ids:
            dep_task = self._task_map.get(dep_id)
            if dep_task and dep_task.output:
                agent = self.agents.get(dep_task.agent_id)
                agent_name = agent.name if agent else dep_task.agent_id
                parts.append(f"[{agent_name} - {dep_task.description[:60]}]:\n{dep_task.output}")

        if not parts:
            return ""
        return "Context from previous tasks:\n\n" + "\n\n---\n\n".join(parts)

    def _build_task_prompt(self, task: TaskDefinition, context: str, goal: str) -> str:
        """Build the full prompt for a task."""
        prompt = f"""Overall Goal: {goal}

Your Task: {task.description}"""

        if task.expected_output:
            prompt += f"\n\nExpected Output Format:\n{task.expected_output}"

        if context:
            prompt += f"\n\n{context}"

        prompt += "\n\nComplete this task thoroughly. Focus only on your assigned task."
        return prompt

    def _build_system_prompt(self, agent: AgentDefinition) -> str:
        return f"""You are {agent.name}, {agent.role}.
Goal: {agent.goal}
Backstory: {agent.backstory}

Follow the task description precisely. Produce output matching the expected format."""

    def _validate_output(self, output: str, expected_output: str) -> dict[str, Any]:
        """Lightweight output validation against expected format."""
        issues = []

        # Check if expected sections are present
        sections = re.findall(r"#{1,3}\s+(.+)", expected_output)
        if sections:
            for section in sections:
                if section.strip().lower() not in output.lower():
                    issues.append(f"Missing section: {section.strip()}")

        # Check minimum length (if expected_output specifies word count)
        word_match = re.search(r"(\d+)\s*[-~]\s*(\d+)\s*(?:字|words?|word)", expected_output)
        if word_match:
            min_words = int(word_match.group(1))
            actual_words = len(output.split())
            if actual_words < min_words * 0.5:
                issues.append(f"Output too short: {actual_words} words, expected ~{min_words}")

        # Check format hints
        if "json" in expected_output.lower() and "{" not in output:
            issues.append("Expected JSON format but no JSON found")

        if "markdown" in expected_output.lower() and "#" not in output:
            issues.append("Expected Markdown format but no headings found")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "note": "; ".join(issues) if issues else "Output matches expected format",
        }
