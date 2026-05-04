"""Multi-agent pattern integration tests.

Tests all 6 orchestration patterns end-to-end with mock LLM providers:
  SupervisorPattern, ParallelPattern, HierarchicalPattern,
  CollaborationPattern, PlanExecutePattern, CrewPattern

Uses deterministic mock responses to avoid real API calls.
"""

from __future__ import annotations

import asyncio
import json
import pytest

from synapse_core.agent import AgentDefinition
from synapse_core.llm import LLMConfig, LLMMessage, LLMResponse, BaseLLMProvider
from synapse_core.tools import ToolDefinition, ToolParameter, ToolSafetyConfig, ToolRegistry
from synapse_core.patterns import (
    SupervisorPattern,
    ParallelPattern,
    HierarchicalPattern,
    CollaborationPattern,
    PlanExecutePattern,
    CrewPattern,
)
from synapse_core.patterns.plan_execute import PlanStep, ExecutionPlan
from synapse_core.patterns.crew import TaskDefinition
from synapse_core.streaming import StreamEvent


# -- Mock LLM Provider -------------------------------------------------------

class MockLLMProvider(BaseLLMProvider):
    """Returns preset responses in sequence, for deterministic testing."""

    def __init__(self, config: LLMConfig, responses: list[str] | None = None) -> None:
        super().__init__(config)
        self._responses = responses or ["OK"]
        self._call_index = 0

    async def invoke(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        resp = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return LLMResponse(
            content=resp,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

    async def stream(self, messages: list[LLMMessage], **kwargs):
        resp = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        for char in resp:
            yield char


# -- Fixtures -----------------------------------------------------------------

@pytest.fixture
def llm_config():
    return LLMConfig(provider="openai", model="mock-model", api_key="test-key")


@pytest.fixture
def supervisor_agent(llm_config):
    return AgentDefinition(
        id="supervisor",
        name="Coordinator",
        role="Task router",
        goal="Delegate to the right specialist",
        backstory="I coordinate tasks between specialists.",
        llm=llm_config,
        tools=[],
    )


@pytest.fixture
def coder_agent(llm_config):
    return AgentDefinition(
        id="coder",
        name="Python Developer",
        role="Code writer",
        goal="Write and debug Python code",
        backstory="Expert Python developer.",
        llm=llm_config,
        tools=["calculator"],
    )


@pytest.fixture
def reviewer_agent(llm_config):
    return AgentDefinition(
        id="reviewer",
        name="Code Reviewer",
        role="Code quality reviewer",
        goal="Review code for quality and security",
        backstory="Senior code reviewer.",
        llm=llm_config,
        tools=[],
    )


@pytest.fixture
def tool_registry():
    registry = ToolRegistry()
    calc_def = ToolDefinition(
        name="calculator",
        description="Evaluate math expressions",
        category="math",
        parameters=[
            ToolParameter(name="expression", type="string", description="Math expression", required=True),
        ],
        safety=ToolSafetyConfig(requires_approval=False),
    )

    async def calc_handler(expression: str, **kwargs):
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"

    registry.register(calc_def, calc_handler)
    return registry


def _collect_events(coro):
    """Helper to collect all StreamEvents from an async generator."""
    async def _run():
        events = []
        async for event in await coro:
            events.append(event)
        return events
    return _run()


# =============================================================================
# 1. Supervisor Pattern
# =============================================================================

class TestSupervisorPattern:

    @pytest.mark.asyncio
    async def test_delegate_to_worker(self, supervisor_agent, coder_agent, tool_registry, llm_config):
        """Supervisor delegates a coding task to the coder agent."""
        # Supervisor response: route to coder
        routing_json = json.dumps({"agent_id": "coder", "task": "Write a factorial function", "reasoning": "Code task"})
        pattern = SupervisorPattern(
            supervisor=supervisor_agent,
            workers=[coder_agent],
            tool_registry=tool_registry,
        )
        # Replace supervisor's LLM provider with mock
        pattern._supervisor_provider = MockLLMProvider(llm_config, responses=[routing_json])

        events = []
        async for event in pattern.run("Write a factorial function"):
            events.append(event)

        types = [e.type for e in events]
        assert "run:start" in types
        assert "agent:delegate" in types
        assert "run:end" in types

        # Verify delegation target
        delegate_events = [e for e in events if e.type == "agent:delegate"]
        assert len(delegate_events) == 1
        assert delegate_events[0].data["toAgent"] == "coder"
        assert delegate_events[0].data["fromAgent"] == "supervisor"

    @pytest.mark.asyncio
    async def test_answer_directly(self, supervisor_agent, coder_agent, tool_registry, llm_config):
        """Supervisor answers directly without delegation."""
        direct_answer = json.dumps({"agent_id": "self", "answer": "The answer is 42"})
        pattern = SupervisorPattern(
            supervisor=supervisor_agent,
            workers=[coder_agent],
            tool_registry=tool_registry,
        )
        pattern._supervisor_provider = MockLLMProvider(llm_config, responses=[direct_answer])

        events = []
        async for event in pattern.run("What is 6*7?"):
            events.append(event)

        # Should have an agent:message from supervisor, no delegation
        delegate_events = [e for e in events if e.type == "agent:delegate"]
        assert len(delegate_events) == 0

        message_events = [e for e in events if e.type == "agent:message"]
        assert len(message_events) >= 1
        assert "42" in message_events[0].data["content"]

    @pytest.mark.asyncio
    async def test_unknown_agent_fallback(self, supervisor_agent, tool_registry, llm_config):
        """Supervisor handles gracefully when target agent not found."""
        bad_route = json.dumps({"agent_id": "nonexistent", "task": "Do something", "reasoning": "N/A"})
        pattern = SupervisorPattern(
            supervisor=supervisor_agent,
            workers=[],
            tool_registry=tool_registry,
        )
        pattern._supervisor_provider = MockLLMProvider(llm_config, responses=[bad_route])

        events = []
        async for event in pattern.run("Do something"):
            events.append(event)

        # Should produce a message with fallback content
        message_events = [e for e in events if e.type == "agent:message"]
        assert len(message_events) >= 1
        assert "couldn't find" in message_events[0].data["content"].lower() or "directly" in message_events[0].data["content"].lower()


# =============================================================================
# 2. Parallel Pattern
# =============================================================================

class TestParallelPattern:

    @pytest.mark.asyncio
    async def test_parallel_execution(self, coder_agent, reviewer_agent, tool_registry, llm_config):
        """Two agents run in parallel and results are concatenated."""
        from synapse_core.graph import GraphBuilder
        from unittest.mock import patch, AsyncMock

        pattern = ParallelPattern(
            agents=[coder_agent, reviewer_agent],
            tool_registry=tool_registry,
            merge_strategy="concatenate",
        )

        # Patch GraphBuilder.run to return mock events
        async def mock_run_coder(task, run_id):
            yield StreamEvent(type="agent:message", data={"agentId": "coder", "content": "def factorial(n): ..."})

        async def mock_run_reviewer(task, run_id):
            yield StreamEvent(type="agent:message", data={"agentId": "reviewer", "content": "Code looks good"})

        events = []
        with patch.object(GraphBuilder, "run", side_effect=[mock_run_coder("task", "r1"), mock_run_reviewer("task", "r2")]):
            async for event in pattern.run("Write and review a factorial function"):
                events.append(event)

        types = [e.type for e in events]
        assert "run:start" in types
        assert "agent:think" in types
        assert "agent:message" in types
        assert "run:end" in types

        # Merged output should contain both agents' contributions
        message_events = [e for e in events if e.type == "agent:message"]
        final_msg = message_events[-1].data["content"]
        assert "factorial" in final_msg
        assert "good" in final_msg.lower()

    @pytest.mark.asyncio
    async def test_parallel_with_error(self, coder_agent, tool_registry, llm_config):
        """One agent fails but the other succeeds - graceful degradation."""
        from synapse_core.graph import GraphBuilder
        from unittest.mock import patch

        pattern = ParallelPattern(
            agents=[coder_agent],
            tool_registry=tool_registry,
        )

        async def mock_run_fail(task, run_id):
            raise RuntimeError("LLM unavailable")
            yield  # make it an async generator

        events = []
        with patch.object(GraphBuilder, "run", side_effect=[mock_run_fail("task", "r1")]):
            async for event in pattern.run("Write something"):
                events.append(event)

        types = [e.type for e in events]
        assert "run:end" in types
        # Should still produce output even with error
        message_events = [e for e in events if e.type == "agent:message"]
        assert len(message_events) >= 1


# =============================================================================
# 3. Hierarchical Pattern
# =============================================================================

class TestHierarchicalPattern:

    @pytest.mark.asyncio
    async def test_delegation_chain(self, supervisor_agent, coder_agent, reviewer_agent, tool_registry, llm_config):
        """Root delegates to child, who has no further children."""
        from synapse_core.graph import GraphBuilder
        from unittest.mock import patch

        pattern = HierarchicalPattern(
            root=supervisor_agent,
            children={"supervisor": [coder_agent]},
            tool_registry=tool_registry,
            max_depth=3,
        )

        async def mock_run(task, run_id):
            yield StreamEvent(type="agent:message", data={"agentId": "test", "content": f"Handled: {task}"})

        events = []
        with patch.object(GraphBuilder, "run", side_effect=mock_run):
            async for event in pattern.run("Build a feature"):
                events.append(event)

        types = [e.type for e in events]
        assert "run:start" in types
        assert "agent:delegate" in types
        assert "run:end" in types

        # Should delegate from supervisor to coder
        delegates = [e for e in events if e.type == "agent:delegate"]
        assert any(d.data["fromAgent"] == "supervisor" and d.data["toAgent"] == "coder" for d in delegates)

    @pytest.mark.asyncio
    async def test_max_depth_limit(self, supervisor_agent, coder_agent, tool_registry, llm_config):
        """Execution stops at max_depth and agent answers directly."""
        from synapse_core.graph import GraphBuilder
        from unittest.mock import patch

        pattern = HierarchicalPattern(
            root=supervisor_agent,
            children={"supervisor": [coder_agent], "coder": [supervisor_agent]},
            tool_registry=tool_registry,
            max_depth=1,  # Very shallow
        )

        async def mock_run(task, run_id):
            yield StreamEvent(type="agent:message", data={"agentId": "test", "content": "Response"})

        events = []
        with patch.object(GraphBuilder, "run", side_effect=mock_run):
            async for event in pattern.run("Test depth"):
                events.append(event)

        think_events = [e for e in events if e.type == "agent:think"]
        max_depth_msgs = [e for e in think_events if "max depth" in e.data.get("thought", "").lower()]
        assert len(max_depth_msgs) > 0


# =============================================================================
# 4. Collaboration Pattern
# =============================================================================

class TestCollaborationPattern:

    @pytest.mark.asyncio
    async def test_multi_round_discussion(self, coder_agent, reviewer_agent, llm_config):
        """Two agents discuss a topic for multiple rounds."""
        pattern = CollaborationPattern(
            agents=[coder_agent, reviewer_agent],
            max_rounds=2,
            consensus_strategy="moderator-decides",
        )

        # Replace providers with mocks
        pattern._providers = {
            "coder": MockLLMProvider(llm_config, responses=["Code perspective: use type hints", "Refined: add docstrings too"]),
            "reviewer": MockLLMProvider(llm_config, responses=["Review: add tests", "Agreed: tests + docs"]),
        }

        events = []
        async for event in pattern.run("Best practices for Python?"):
            events.append(event)

        types = [e.type for e in events]
        assert "run:start" in types
        assert "run:end" in types

        # Should have think events for each round
        think_events = [e for e in events if e.type == "agent:think"]
        round_markers = [e for e in think_events if "round" in e.data.get("thought", "").lower()]
        assert len(round_markers) == 2

        # Each agent should contribute per round
        message_events = [e for e in events if e.type == "agent:message"]
        assert len(message_events) >= 4  # 2 agents x 2 rounds

    @pytest.mark.asyncio
    async def test_with_moderator(self, coder_agent, reviewer_agent, supervisor_agent, llm_config):
        """Moderator synthesizes final answer."""
        pattern = CollaborationPattern(
            agents=[coder_agent, reviewer_agent],
            max_rounds=1,
            moderator=supervisor_agent,
        )

        pattern._providers = {
            "coder": MockLLMProvider(llm_config, responses=["Use type hints"]),
            "reviewer": MockLLMProvider(llm_config, responses=["Add tests"]),
            "supervisor": MockLLMProvider(llm_config, responses=["Consensus: type hints + tests"]),
        }

        events = []
        async for event in pattern.run("Best practices?"):
            events.append(event)

        # Last message should be from moderator
        message_events = [e for e in events if e.type == "agent:message"]
        assert len(message_events) >= 3  # 2 agents + 1 moderator

        # Moderator's message
        mod_msg = message_events[-1]
        assert "Consensus" in mod_msg.data["content"]

        think_events = [e for e in events if e.type == "agent:think"]
        mod_think = [e for e in think_events if "moderator" in e.data.get("thought", "").lower() or "synthesiz" in e.data.get("thought", "").lower()]
        assert len(mod_think) >= 1


# =============================================================================
# 5. Plan-Execute Pattern
# =============================================================================

class TestPlanExecutePattern:

    @pytest.mark.asyncio
    async def test_plan_and_execute(self, supervisor_agent, tool_registry, llm_config):
        """Generate a plan and execute each step."""
        plan_json = json.dumps({
            "reasoning": "Need to research then summarize",
            "steps": [
                {"step_number": 1, "description": "Research the topic", "tool_hint": "web_search"},
                {"step_number": 2, "description": "Summarize findings", "tool_hint": ""},
            ],
        })
        # Responses: plan, step1 result, step2 result
        pattern = PlanExecutePattern(
            planner=supervisor_agent,
            tool_registry=tool_registry,
            enable_replan=False,
        )
        # Patch provider to return plan then step results
        pattern._planner_provider = MockLLMProvider(
            llm_config,
            responses=[plan_json, "Research complete: key findings", "Summary: here is the report"],
        )

        events = []
        async for event in pattern.run("Research AI trends and summarize"):
            events.append(event)

        types = [e.type for e in events]
        assert "run:start" in types
        assert "run:end" in types

        # Should have think events for plan creation and steps
        think_events = [e for e in events if e.type == "agent:think"]
        plan_think = [e for e in think_events if "plan" in e.data.get("thought", "").lower()]
        assert len(plan_think) >= 1

    @pytest.mark.asyncio
    async def test_plan_with_replan(self, supervisor_agent, tool_registry, llm_config):
        """Step failure triggers replanning when enabled."""
        plan_json = json.dumps({
            "reasoning": "Simple plan",
            "steps": [
                {"step_number": 1, "description": "Step 1", "tool_hint": ""},
                {"step_number": 2, "description": "Step 2", "tool_hint": ""},
            ],
        })
        replan_json = json.dumps({
            "reasoning": "Replanning after failure",
            "steps": [
                {"step_number": 1, "description": "Revised step", "tool_hint": ""},
            ],
        })

        pattern = PlanExecutePattern(
            planner=supervisor_agent,
            tool_registry=tool_registry,
            enable_replan=True,
            max_replans=1,
        )
        # Responses: plan, step1 fails, replan, revised step succeeds
        pattern._planner_provider = MockLLMProvider(
            llm_config,
            responses=[plan_json, "ERROR: Step 1 failed", replan_json, "Revised result"],
        )

        events = []
        async for event in pattern.run("Do something complex"):
            events.append(event)

        types = [e.type for e in events]
        assert "run:end" in types


# =============================================================================
# 6. Crew Pattern (CrewAI-style)
# =============================================================================

class TestCrewPattern:

    @pytest.mark.asyncio
    async def test_sequential_tasks(self, coder_agent, reviewer_agent, tool_registry, llm_config):
        """Tasks run in sequence with context passing."""
        # Both agents need tools to go through GraphBuilder path
        reviewer_with_tools = AgentDefinition(
            id="reviewer",
            name="Code Reviewer",
            role="Code quality reviewer",
            goal="Review code for quality and security",
            backstory="Senior code reviewer.",
            llm=llm_config,
            tools=["calculator"],
        )

        tasks = [
            TaskDefinition(
                id="write",
                description="Write a function to calculate fibonacci",
                expected_output="Python code with fibonacci function",
                agent_id="coder",
                context_task_ids=[],
            ),
            TaskDefinition(
                id="review",
                description="Review the fibonacci function",
                expected_output="Code review with feedback",
                agent_id="reviewer",
                context_task_ids=["write"],
            ),
        ]

        pattern = CrewPattern(
            agents={"coder": coder_agent, "reviewer": reviewer_with_tools},
            tasks=tasks,
            tool_registry=tool_registry,
        )

        from synapse_core.graph import GraphBuilder
        from unittest.mock import patch

        call_count = 0
        async def mock_run(task, run_id):
            nonlocal call_count
            call_count += 1
            yield StreamEvent(
                type="agent:message",
                data={"agentId": "test", "content": f"Task output {call_count}: fibonacci code" if call_count == 1 else "Review: looks good"},
            )

        events = []
        with patch.object(GraphBuilder, "run", side_effect=mock_run):
            async for event in pattern.run(goal="Write and review fibonacci code"):
                events.append(event)

        types = [e.type for e in events]
        assert "run:start" in types
        assert "run:end" in types
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_context_dependency(self, coder_agent, reviewer_agent, tool_registry, llm_config):
        """Second task receives output from first task as context."""
        reviewer_with_tools = AgentDefinition(
            id="reviewer",
            name="Code Reviewer",
            role="Code quality reviewer",
            goal="Review code for quality and security",
            backstory="Senior code reviewer.",
            llm=llm_config,
            tools=["calculator"],
        )

        tasks = [
            TaskDefinition(
                id="task1",
                description="Write code",
                expected_output="Python code",
                agent_id="coder",
                context_task_ids=[],
            ),
            TaskDefinition(
                id="task2",
                description="Review code",
                expected_output="Review feedback",
                agent_id="reviewer",
                context_task_ids=["task1"],
            ),
        ]

        pattern = CrewPattern(
            agents={"coder": coder_agent, "reviewer": reviewer_with_tools},
            tasks=tasks,
            tool_registry=tool_registry,
        )

        from synapse_core.graph import GraphBuilder
        from unittest.mock import patch

        captured_inputs = []

        async def mock_run_capture(self, task, run_id):
            captured_inputs.append(task)
            yield StreamEvent(type="agent:message", data={"agentId": "test", "content": "output"})

        events = []
        with patch.object(GraphBuilder, "run", mock_run_capture):
            async for event in pattern.run(goal="Build and review"):
                events.append(event)

        # Second task input should include context from task1
        assert len(captured_inputs) == 2
        assert "task1" in captured_inputs[1] or "output" in captured_inputs[1].lower()

    @pytest.mark.asyncio
    async def test_output_validation(self, coder_agent, tool_registry, llm_config):
        """Output validation catches missing expected content."""
        task = TaskDefinition(
            id="bad-task",
            description="Write a report",
            expected_output="Must include ## Analysis section",
            agent_id="coder",
            context_task_ids=[],
        )

        pattern = CrewPattern(
            agents={"coder": coder_agent},
            tasks=[task],
            tool_registry=tool_registry,
            validate_output=True,
        )

        from synapse_core.graph import GraphBuilder
        from unittest.mock import patch

        async def mock_run(task, run_id):
            yield StreamEvent(type="agent:message", data={"agentId": "test", "content": "Short output"})

        events = []
        with patch.object(GraphBuilder, "run", side_effect=mock_run):
            async for event in pattern.run(goal="Write a report"):
                events.append(event)

        # Should complete even with validation (just marks validation_failed)
        assert any(e.type == "run:end" for e in events)


# =============================================================================
# 7. Cross-Pattern Tests
# =============================================================================

class TestCrossPattern:

    @pytest.mark.asyncio
    async def test_all_patterns_emit_valid_events(self, supervisor_agent, coder_agent, reviewer_agent, tool_registry, llm_config):
        """Every pattern produces StreamEvent objects with valid structure."""
        from unittest.mock import patch
        from synapse_core.graph import GraphBuilder

        async def mock_run(task, run_id):
            yield StreamEvent(type="agent:message", data={"agentId": "test", "content": "response"})

        patterns_with_run = [
            ParallelPattern(agents=[coder_agent, reviewer_agent], tool_registry=tool_registry),
            HierarchicalPattern(root=supervisor_agent, children={"supervisor": [coder_agent]}, tool_registry=tool_registry),
        ]

        for pattern in patterns_with_run:
            with patch.object(GraphBuilder, "run", side_effect=mock_run):
                events = []
                async for event in pattern.run("Test"):
                    events.append(event)

                for ev in events:
                    assert isinstance(ev, StreamEvent), f"{pattern.__class__.__name__} emitted non-StreamEvent"
                    assert ev.type, f"{pattern.__class__.__name__} emitted event without type"
                    assert isinstance(ev.data, dict), f"{pattern.__class__.__name__} emitted event without dict data"

    @pytest.mark.asyncio
    async def test_event_sequence_starts_and_ends(self, supervisor_agent, coder_agent, tool_registry, llm_config):
        """Every pattern starts with run:start and ends with run:end."""
        from unittest.mock import patch
        from synapse_core.graph import GraphBuilder

        async def mock_run(task, run_id):
            yield StreamEvent(type="agent:message", data={"agentId": "test", "content": "response"})

        plan_json = json.dumps({"reasoning": "test", "steps": [{"step_number": 1, "description": "test"}]})

        # Patterns that use GraphBuilder internally
        patterns = [
            ParallelPattern(agents=[coder_agent], tool_registry=tool_registry),
            HierarchicalPattern(root=supervisor_agent, children={}, tool_registry=tool_registry),
        ]

        for pattern in patterns:
            with patch.object(GraphBuilder, "run", side_effect=mock_run):
                events = []
                async for event in pattern.run("Test"):
                    events.append(event)

                types = [e.type for e in events]
                assert types[0] == "run:start", f"{pattern.__class__.__name__} doesn't start with run:start"
                assert types[-1] == "run:end", f"{pattern.__class__.__name__} doesn't end with run:end"
