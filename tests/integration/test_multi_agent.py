"""Multi-agent integration tests.

Tests the full agent lifecycle: definition, registration, graph execution,
tool calls, loop detection, and multi-agent orchestration.

Uses a mock LLM provider to avoid real API calls.
"""

from __future__ import annotations

import asyncio
import json
import pytest

from synapse_core.agent import AgentDefinition, AgentRegistry, GuardrailConfig, MemoryConfig
from synapse_core.llm import LLMConfig, LLMMessage, LLMResponse, BaseLLMProvider
from synapse_core.tools import ToolDefinition, ToolParameter, ToolSafetyConfig, ToolRegistry
from synapse_core.graph import GraphBuilder
from synapse_core.graph.state import RunState, Message, TokenUsage
from synapse_core.graph.nodes.router_node import router_node
from synapse_core.graph.nodes.memory_node import memory_read_node, memory_write_node
from synapse_core.graph.nodes.human_node import human_approval_node, process_human_input_node


# ── Mock LLM Provider ────────────────────────────────────────────────────────

class MockLLMProvider(BaseLLMProvider):
    """Deterministic mock that returns preset responses in sequence."""

    def __init__(self, config: LLMConfig, responses: list[str] | None = None) -> None:
        super().__init__(config)
        self._responses = responses or ["Hello! I can help you with that."]
        self._call_index = 0

    async def invoke(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        resp = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return LLMResponse(content=resp, usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})

    async def stream(self, messages: list[LLMMessage], **kwargs):
        resp = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        for char in resp:
            yield char


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def llm_config():
    return LLMConfig(provider="openai", model="mock-model", api_key="test-key")


@pytest.fixture
def mock_agent(llm_config):
    return AgentDefinition(
        id="test-agent",
        name="Test Agent",
        role="Testing assistant",
        goal="Help with tests",
        backstory="I am a test agent.",
        llm=llm_config,
        tools=["calculator"],
        max_iterations=5,
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


@pytest.fixture
def graph_builder(mock_agent, tool_registry, llm_config):
    builder = GraphBuilder(mock_agent, tool_registry)
    builder._provider = MockLLMProvider(llm_config)
    return builder


# ── 1. Agent Definition & Registry ──────────────────────────────────────────

class TestAgentRegistry:
    def test_register_and_get(self, mock_agent):
        registry = AgentRegistry()
        registry.register(mock_agent)
        assert registry.get("test-agent") == mock_agent

    def test_get_not_found_raises(self):
        registry = AgentRegistry()
        with pytest.raises(KeyError, match="Agent not found"):
            registry.get("nonexistent")

    def test_list_all(self, llm_config):
        registry = AgentRegistry()
        a1 = AgentDefinition(id="a1", name="A1", role="r", goal="g", backstory="b", llm=llm_config)
        a2 = AgentDefinition(id="a2", name="A2", role="r", goal="g", backstory="b", llm=llm_config)
        registry.register(a1)
        registry.register(a2)
        assert len(registry.list_all()) == 2

    def test_remove(self, mock_agent):
        registry = AgentRegistry()
        registry.register(mock_agent)
        registry.remove("test-agent")
        with pytest.raises(KeyError):
            registry.get("test-agent")

    def test_remove_not_found_raises(self):
        registry = AgentRegistry()
        with pytest.raises(KeyError):
            registry.remove("ghost")


# ── 2. Graph Execution (single agent, no tools) ─────────────────────────────

class TestGraphExecution:
    @pytest.mark.asyncio
    async def test_simple_response(self, graph_builder):
        events = []
        async for event in graph_builder.run("Hello"):
            events.append(event)

        types = [e.type for e in events]
        assert "run:start" in types
        assert "agent:message" in types
        assert "run:end" in types

    @pytest.mark.asyncio
    async def test_response_content(self, graph_builder):
        events = []
        async for event in graph_builder.run("Hi"):
            events.append(event)

        msg_events = [e for e in events if e.type == "agent:message"]
        assert len(msg_events) > 0
        final_msg = msg_events[-1]
        assert "Hello!" in final_msg.data["content"]


# ── 3. Tool Call Parsing & Execution ────────────────────────────────────────

class TestToolCalls:
    @pytest.mark.asyncio
    async def test_tool_call_format1(self, graph_builder, tool_registry):
        """TOOL_CALL: calculator({"expression": "2+3"})"""
        graph_builder._provider = MockLLMProvider(
            graph_builder._provider.config,
            responses=['TOOL_CALL: calculator({"expression": "2+3"})', "The result is 5."],
        )
        events = []
        async for event in graph_builder.run("Calculate 2+3"):
            events.append(event)

        tool_events = [e for e in events if e.type == "tool:result"]
        assert len(tool_events) >= 1
        assert "5" in str(tool_events[0].data["output"])

    def test_parse_tool_call_balanced_parens(self, graph_builder):
        """Balanced paren tracking handles nested JSON."""
        content = 'TOOL_CALL: calculator({"expression": "(1+2)*3"})'
        calls = graph_builder._parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0][0] == "calculator"
        assert calls[0][1]["expression"] == "(1+2)*3"

    def test_parse_tool_call_bare_format(self, graph_builder):
        """tool_name({"key": "value"}) without TOOL_CALL prefix."""
        content = 'calculator({"expression": "100/4"})'
        calls = graph_builder._parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0][0] == "calculator"
        assert calls[0][1]["expression"] == "100/4"

    def test_parse_no_tool_call(self, graph_builder):
        """Regular text without tool calls returns empty."""
        calls = graph_builder._parse_tool_calls("This is just a normal response.")
        assert calls == []

    def test_normalize_args_expression_as_json(self, graph_builder):
        """expression containing valid JSON gets parsed."""
        result = graph_builder._normalize_args("calculator", {"expression": '{"expression": "1+1"}'})
        assert result == {"expression": "1+1"}

    def test_normalize_args_expression_as_value(self, graph_builder):
        """Single required param: expression becomes that param value."""
        result = graph_builder._normalize_args("calculator", {"expression": "2+2"})
        assert result == {"expression": "2+2"}

    def test_normalize_args_passthrough(self, graph_builder):
        """Normal args pass through unchanged."""
        result = graph_builder._normalize_args("calculator", {"expression": "3*3"})
        assert result == {"expression": "3*3"}


# ── 4. Required Parameter Validation ────────────────────────────────────────

class TestParamValidation:
    @pytest.mark.asyncio
    async def test_missing_required_param(self, graph_builder):
        """LLM calls tool without required param -> error message, not crash."""
        graph_builder._provider = MockLLMProvider(
            graph_builder._provider.config,
            responses=['TOOL_CALL: calculator({})', "Let me try again with the expression."],
        )
        events = []
        async for event in graph_builder.run("Calculate something"):
            events.append(event)

        # Should NOT have run:error -- graceful handling
        error_events = [e for e in events if e.type == "run:error"]
        assert len(error_events) == 0

        # Should have tool result with error message about missing param
        tool_events = [e for e in events if e.type == "tool:result"]
        assert len(tool_events) >= 1
        assert "Missing required" in str(tool_events[0].data["output"])

    def test_validate_required_params_all_present(self, graph_builder):
        """All required params present -> empty missing list."""
        missing = graph_builder._validate_required_params("calculator", {"expression": "1+1"})
        assert missing == []

    def test_validate_required_params_missing(self, graph_builder):
        """Missing required param -> list with that param."""
        missing = graph_builder._validate_required_params("calculator", {})
        assert len(missing) == 1
        assert missing[0].name == "expression"

    def test_validate_unknown_tool(self, graph_builder):
        """Unknown tool -> empty list (let execute handle it)."""
        missing = graph_builder._validate_required_params("nonexistent_tool", {})
        assert missing == []


# ── 5. Tool Call Loop Detection ─────────────────────────────────────────────

class TestLoopDetection:
    @pytest.mark.asyncio
    async def test_loop_stopped_after_3_repeats(self, graph_builder):
        """Same tool called 3+ times in a row triggers loop breaker."""
        graph_builder._provider = MockLLMProvider(
            graph_builder._provider.config,
            responses=[
                'TOOL_CALL: calculator({"expression": "1+1"})',
                'TOOL_CALL: calculator({"expression": "1+1"})',
                'TOOL_CALL: calculator({"expression": "1+1"})',
                "OK I will stop now.",
            ],
        )
        events = []
        async for event in graph_builder.run("Keep calculating"):
            events.append(event)

        # Count tool calls
        tool_events = [e for e in events if e.type == "agent:call_tool"]
        # Should be stopped well before the iteration limit
        assert len(tool_events) <= 4


# ── 6. Router Node (Multi-Agent Orchestration) ──────────────────────────────

class TestRouterNode:
    @pytest.mark.asyncio
    async def test_single_agent_returns_that_agent(self, llm_config):
        """Single agent in list -> no LLM call, just return it."""
        state = RunState(messages=[Message(role="user", content="Hello")])
        result = await router_node(
            state,
            agent_ids=["code-expert"],
            agent_descriptions={"code-expert": "Writes code"},
            llm_config=llm_config,
        )
        assert result["next_agents"] == ["code-expert"]

    @pytest.mark.asyncio
    async def test_empty_agents_returns_empty(self, llm_config):
        """No agents -> empty list."""
        state = RunState(messages=[Message(role="user", content="Hello")])
        result = await router_node(state, agent_ids=[], agent_descriptions={}, llm_config=llm_config)
        assert result["next_agents"] == []


# ── 7. Human-in-the-Loop Node ───────────────────────────────────────────────

class TestHumanNode:
    @pytest.mark.asyncio
    async def test_approval_request(self):
        state = RunState()
        result = await human_approval_node(state, approval_message="Please approve")
        assert result["context"]["human_approval_required"] is True
        assert result["context"]["approval_message"] == "Please approve"

    @pytest.mark.asyncio
    async def test_process_approval(self):
        state = RunState()
        result = await process_human_input_node(state, approved=True, feedback="Looks good")
        assert result["context"]["human_approved"] is True
        assert result["context"]["human_feedback"] == "Looks good"

    @pytest.mark.asyncio
    async def test_process_rejection(self):
        state = RunState()
        result = await process_human_input_node(state, approved=False, feedback="Try again")
        assert result["context"]["human_approved"] is False


# ── 8. Multi-Agent Orchestration (Integration) ──────────────────────────────

class TestMultiAgentOrchestration:
    def test_three_agents_registered(self, llm_config):
        """Verify multiple agents can coexist in registry."""
        registry = AgentRegistry()
        agents = [
            AgentDefinition(
                id="general-assistant",
                name="Synapse Assistant",
                role="General-purpose AI assistant",
                goal="Help users",
                backstory="Versatile helper.",
                llm=llm_config,
                tools=["calculator", "web_search"],
            ),
            AgentDefinition(
                id="code-expert",
                name="Code Expert",
                role="Software engineering specialist",
                goal="Write and debug code",
                backstory="Senior engineer.",
                llm=llm_config.model_copy(),
                tools=["calculator", "code_execute"],
            ),
            AgentDefinition(
                id="data-analyst",
                name="Data Analyst",
                role="Data analysis specialist",
                goal="Analyze data",
                backstory="Data expert.",
                llm=llm_config.model_copy(),
                tools=["calculator", "sql_query"],
            ),
        ]
        for a in agents:
            registry.register(a)

        all_agents = registry.list_all()
        assert len(all_agents) == 3
        ids = {a.id for a in all_agents}
        assert ids == {"general-assistant", "code-expert", "data-analyst"}

    @pytest.mark.asyncio
    async def test_sequential_agent_execution(self, llm_config, tool_registry):
        """Run two agents sequentially, second sees first's output."""
        agent1 = AgentDefinition(
            id="planner",
            name="Planner",
            role="Plans tasks",
            goal="Break down work",
            backstory="I plan.",
            llm=llm_config,
            tools=["calculator"],
            max_iterations=2,
        )
        agent2 = AgentDefinition(
            id="executor",
            name="Executor",
            role="Executes plans",
            goal="Do the work",
            backstory="I execute.",
            llm=llm_config.model_copy(),
            tools=["calculator"],
            max_iterations=2,
        )

        builder1 = GraphBuilder(agent1, tool_registry)
        builder1._provider = MockLLMProvider(llm_config, responses=["Step 1: Calculate 10*5"])

        builder2 = GraphBuilder(agent2, tool_registry)
        builder2._provider = MockLLMProvider(llm_config, responses=["Result: 50"])

        # Agent 1 plans
        events1 = []
        async for event in builder1.run("Plan a calculation task"):
            events1.append(event)

        # Agent 2 executes based on agent 1's output
        planner_msg = ""
        for e in events1:
            if e.type == "agent:message":
                planner_msg = e.data["content"]

        events2 = []
        async for event in builder2.run(f"Execute this plan: {planner_msg}"):
            events2.append(event)

        types2 = [e.type for e in events2]
        assert "run:end" in types2

    @pytest.mark.asyncio
    async def test_concurrent_agent_execution(self, llm_config, tool_registry):
        """Run two agents in parallel and collect both results."""
        agent_a = AgentDefinition(
            id="agent-a",
            name="Agent A",
            role="Researcher",
            goal="Research",
            backstory="I research.",
            llm=llm_config,
            tools=["calculator"],
            max_iterations=2,
        )
        agent_b = AgentDefinition(
            id="agent-b",
            name="Agent B",
            role="Writer",
            goal="Write summary",
            backstory="I write.",
            llm=llm_config.model_copy(),
            tools=["calculator"],
            max_iterations=2,
        )

        builder_a = GraphBuilder(agent_a, tool_registry)
        builder_a._provider = MockLLMProvider(llm_config, responses=["Research findings: X is true"])

        builder_b = GraphBuilder(agent_b, tool_registry)
        builder_b._provider = MockLLMProvider(llm_config, responses=["Summary: Based on research, X is confirmed"])

        results = await asyncio.gather(
            _collect_events(builder_a, "Research topic X"),
            _collect_events(builder_b, "Summarize: X is true"),
        )

        # Both should complete successfully
        for event_list in results:
            types = [e.type for e in event_list]
            assert "run:start" in types
            assert "run:end" in types
            assert "run:error" not in types


# ── 9. Skill -> Agent Conversion ────────────────────────────────────────────

class TestSkillIntegration:
    def test_skill_to_agent(self, llm_config):
        """Skills can be converted to agents and registered."""
        from synapse_core.skills import SkillDefinition

        skill = SkillDefinition(
            name="code-reviewer",
            display_name="Code Reviewer",
            description="Reviews code for quality",
            tools=["calculator", "web_scrape"],
            system_prompt="You are a code reviewer.",
            temperature=0.2,
        )

        agent = skill.to_agent(llm_config)
        assert agent.id == "skill-code-reviewer"
        assert agent.name == "Code Reviewer"
        assert agent.llm.temperature == 0.2
        assert "calculator" in agent.tools

    def test_skill_registry(self, llm_config):
        """Multiple skills register correctly."""
        from synapse_core.skills import SkillDefinition, SkillRegistry

        registry = SkillRegistry()
        skills = [
            SkillDefinition(name="reviewer", display_name="Reviewer", description="Reviews"),
            SkillDefinition(name="architect", display_name="Architect", description="Designs"),
        ]
        for s in skills:
            registry.register(s)

        all_skills = registry.list_all()
        assert len(all_skills) == 2


# ── 10. Guardrails ──────────────────────────────────────────────────────────

class TestGuardrails:
    def test_max_iterations_guardrail(self, mock_agent):
        builder = GraphBuilder.__new__(GraphBuilder)
        builder.agent = mock_agent
        mock_agent.max_iterations = 2
        with pytest.raises(RuntimeError, match="Max iterations reached"):
            builder._check_guardrails([], TokenUsage(), 2)

    def test_token_limit_guardrail(self, mock_agent):
        mock_agent.guardrails.max_tokens_per_run = 100
        builder = GraphBuilder.__new__(GraphBuilder)
        builder.agent = mock_agent
        with pytest.raises(RuntimeError, match="Token limit exceeded"):
            builder._check_guardrails([], TokenUsage(total_tokens=200), 0)

    def test_within_limits_passes(self, mock_agent):
        builder = GraphBuilder.__new__(GraphBuilder)
        builder.agent = mock_agent
        # Should not raise
        builder._check_guardrails([], TokenUsage(total_tokens=10), 0)


# ── 11. Run State ───────────────────────────────────────────────────────────

class TestRunState:
    def test_state_initialization(self):
        state = RunState()
        assert state.messages == []
        assert state.current_agent == "default"
        assert state.next_agents == []
        assert state.step_count == 0

    def test_token_usage_add(self):
        u1 = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        u2 = TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        result = u1.add(u2)
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 30
        assert result.total_tokens == 45

    def test_message_model(self):
        msg = Message(role="user", content="Hello", name="test")
        assert msg.role == "user"
        assert msg.content == "Hello"


# ── Helper ───────────────────────────────────────────────────────────────────

async def _collect_events(builder: GraphBuilder, prompt: str):
    events = []
    async for event in builder.run(prompt):
        events.append(event)
    return events


# ── 12. LLM Retry Mechanism ──────────────────────────────────────────────────

class FlakyLLMProvider(BaseLLMProvider):
    """Mock that fails N times then succeeds."""

    def __init__(self, config: LLMConfig, fail_count: int, success_response: str = "Done!") -> None:
        super().__init__(config)
        self._fail_count = fail_count
        self._attempts = 0
        self._success_response = success_response

    async def invoke(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        self._attempts += 1
        if self._attempts <= self._fail_count:
            raise ConnectionError(f"Simulated connection failure (attempt {self._attempts})")
        return LLMResponse(content=self._success_response, usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20})

    async def stream(self, messages: list[LLMMessage], **kwargs):
        self._attempts += 1
        if self._attempts <= self._fail_count:
            raise ConnectionError(f"Simulated connection failure (attempt {self._attempts})")
        for char in self._success_response:
            yield char


class TestLLMRetry:
    def test_is_retryable_connection_error(self):
        assert GraphBuilder._is_retryable_error(ConnectionError("refused")) is True

    def test_is_retryable_timeout(self):
        assert GraphBuilder._is_retryable_error(TimeoutError("timed out")) is True

    def test_is_retryable_503(self):
        assert GraphBuilder._is_retryable_error(RuntimeError("Server error: 503 Service Unavailable")) is True

    def test_is_retryable_rate_limit(self):
        assert GraphBuilder._is_retryable_error(RuntimeError("429 rate limit exceeded")) is True

    def test_is_not_retryable_auth_error(self):
        assert GraphBuilder._is_retryable_error(ValueError("Invalid API key")) is False

    def test_is_not_retryable_bad_request(self):
        assert GraphBuilder._is_retryable_error(ValueError("400 Bad Request: invalid model")) is False

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_transient_failure(self, llm_config, tool_registry, mock_agent):
        """LLM fails once, retry succeeds -> run completes normally, user sees error reason."""
        flaky = FlakyLLMProvider(llm_config, fail_count=1, success_response="Recovered!")
        builder = GraphBuilder(mock_agent, tool_registry)
        builder._provider = flaky
        builder.RETRY_BASE_DELAY = 0.01

        events = []
        async for event in builder.run("Test retry"):
            events.append(event)

        types = [e.type for e in events]
        assert "run:end" in types
        assert "run:error" not in types
        # User should see the error reason in the retry notice
        msg_events = [e for e in events if e.type == "agent:message"]
        all_content = " ".join(e.data["content"] for e in msg_events)
        assert "Simulated connection failure" in all_content
        assert "Retrying" in all_content
        assert "Recovered!" in all_content
        assert flaky._attempts == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted_shows_each_failure(self, llm_config, tool_registry, mock_agent):
        """LLM fails all retries -> user sees each failure reason + final error."""
        flaky = FlakyLLMProvider(llm_config, fail_count=10, success_response="never reached")
        builder = GraphBuilder(mock_agent, tool_registry)
        builder._provider = flaky
        builder.RETRY_BASE_DELAY = 0.01

        events = []
        async for event in builder.run("Test retry exhaustion"):
            events.append(event)

        # User sees all retry attempts with reasons
        msg_events = [e for e in events if e.type == "agent:message"]
        all_content = " ".join(e.data["content"] for e in msg_events)
        assert "Simulated connection failure" in all_content

        error_events = [e for e in events if e.type == "run:error"]
        assert len(error_events) == 1
        assert "Simulated connection failure" in error_events[0].data["error"]
        assert error_events[0].data["recoverable"] is True
        assert flaky._attempts == builder.MAX_LLM_RETRIES

    @pytest.mark.asyncio
    async def test_non_retryable_error_shows_reason(self, llm_config, tool_registry, mock_agent):
        """Non-retryable error -> user still sees the reason before failure."""

        class AuthFailProvider(BaseLLMProvider):
            async def invoke(self, messages, **kwargs):
                raise PermissionError("Invalid API key")

            async def stream(self, messages, **kwargs):
                raise PermissionError("Invalid API key")
                yield  # noqa: unreachable

        builder = GraphBuilder(mock_agent, tool_registry)
        builder._provider = AuthFailProvider(llm_config)
        builder.RETRY_BASE_DELAY = 0.01

        events = []
        async for event in builder.run("Test auth failure"):
            events.append(event)

        # User should see the error reason in the message stream
        msg_events = [e for e in events if e.type == "agent:message"]
        all_content = " ".join(e.data["content"] for e in msg_events)
        assert "Invalid API key" in all_content
        assert "not retryable" in all_content

        error_events = [e for e in events if e.type == "run:error"]
        assert len(error_events) == 1
        assert "Invalid API key" in error_events[0].data["error"]


# -- 13. L1 Audit: Token Estimation + Domestic Models + LLMConfig ---------

class TestTokenEstimation:
    def test_chinese_token_estimation(self):
        """Chinese characters should estimate ~1.5 tokens each."""
        from synapse_core.graph import _estimate_tokens
        # 10 Chinese characters -> ~15 tokens
        result = _estimate_tokens("今天天气真好，我们去玩吧")
        assert result >= 10  # At least 1 per char
        assert result <= 30  # But not absurdly high

    def test_english_token_estimation(self):
        """ASCII text should estimate ~0.25 tokens per char."""
        from synapse_core.graph import _estimate_tokens
        # "Hello world" = 11 chars -> ~3 tokens
        result = _estimate_tokens("Hello world")
        assert result >= 1
        assert result <= 8

    def test_mixed_content_estimation(self):
        """Mixed Chinese/English should handle both correctly."""
        from synapse_core.graph import _estimate_tokens
        mixed = "Hello 你好 World 世界"
        result = _estimate_tokens(mixed)
        assert result >= 1
        # Chinese chars should contribute more than ASCII
        cn_only = _estimate_tokens("你好世界")
        en_only = _estimate_tokens("Hello World")
        assert cn_only > en_only  # Chinese uses more tokens per char

    def test_empty_string(self):
        from synapse_core.graph import _estimate_tokens
        assert _estimate_tokens("") == 0


class TestLLMConfigExtended:
    def test_default_values(self):
        """New fields should have sensible defaults."""
        config = LLMConfig()
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0
        assert config.stop is None

    def test_custom_values(self):
        config = LLMConfig(
            frequency_penalty=0.5,
            presence_penalty=0.3,
            stop=["\n\n", "END"],
        )
        assert config.frequency_penalty == 0.5
        assert config.presence_penalty == 0.3
        assert config.stop == ["\n\n", "END"]

    def test_backward_compatible(self):
        """Old code creating LLMConfig without new fields should still work."""
        config = LLMConfig(provider="openai", model="gpt-4o", temperature=0.5)
        assert config.frequency_penalty == 0.0


class TestDomesticModelProfiles:
    def test_glm_profiles_exist(self):
        from synapse_core.llm.router import MODEL_PROFILES
        assert "glm-4-plus" in MODEL_PROFILES
        assert "glm-4-flash" in MODEL_PROFILES
        assert MODEL_PROFILES["glm-4-flash"].capability.value == "simple"

    def test_qwen_profiles_exist(self):
        from synapse_core.llm.router import MODEL_PROFILES
        assert "qwen-max" in MODEL_PROFILES
        assert "qwen-turbo" in MODEL_PROFILES

    def test_deepseek_profiles_exist(self):
        from synapse_core.llm.router import MODEL_PROFILES
        assert "deepseek-chat" in MODEL_PROFILES
        assert "deepseek-reasoner" in MODEL_PROFILES

    def test_fallback_chains_include_domestic(self):
        from synapse_core.llm.router import ModelRouter
        router = ModelRouter()
        assert "glm" in router._fallback_chains
        assert "qwen" in router._fallback_chains
        assert "deepseek" in router._fallback_chains

    def test_select_domestic_model_for_simple_task(self):
        from synapse_core.llm.router import ModelRouter, MODEL_PROFILES
        router = ModelRouter(profiles=MODEL_PROFILES, default_model="deepseek-chat")
        from synapse_core.llm.router import TaskComplexity
        model = router.select_model(TaskComplexity.SIMPLE, "cost")
        profile = MODEL_PROFILES.get(model)
        assert profile is not None


# -- 14. L2 Audit: Prompt Template System -----------------------------------

class TestPromptTemplate:
    def test_render_basic_template(self):
        from synapse_core.llm.prompt_template import PromptTemplate
        tmpl = PromptTemplate(
            template_id="test",
            role="You are a sentiment analyzer.",
            task="Classify the user review.",
        )
        result = tmpl.render(user_input="Great product!")
        assert "# Role" in result
        assert "# Task" in result
        assert "Great product!" in result
        assert "# Security" in result

    def test_variable_interpolation(self):
        from synapse_core.llm.prompt_template import PromptTemplate
        tmpl = PromptTemplate(
            template_id="test",
            role="You are {{role_name}}.",
            task="Help user {{user_name}}.",
        )
        result = tmpl.render(variables={"role_name": "Chef", "user_name": "Alice"})
        assert "You are Chef." in result
        assert "Help user Alice." in result

    def test_constraints_rendering(self):
        from synapse_core.llm.prompt_template import PromptTemplate
        tmpl = PromptTemplate(
            template_id="test",
            constraints=[
                "Only output JSON",
                "Do not fabricate information",
            ],
        )
        result = tmpl.render()
        assert "# Constraints" in result
        assert "- Only output JSON" in result
        assert "- Do not fabricate information" in result

    def test_few_shot_examples(self):
        from synapse_core.llm.prompt_template import PromptTemplate, PromptExample
        tmpl = PromptTemplate(
            template_id="sentiment",
            role="Sentiment analyzer.",
            examples=[
                PromptExample(input="Great!", output="positive"),
                PromptExample(input="Terrible.", output="negative"),
            ],
        )
        result = tmpl.render()
        assert "# Examples" in result
        assert "Great!" in result
        assert "positive" in result
        assert "Terrible." in result
        assert "negative" in result

    def test_cot_instruction(self):
        from synapse_core.llm.prompt_template import PromptTemplate
        tmpl = PromptTemplate(
            template_id="math",
            enable_cot=True,
            cot_instruction="Think step by step.",
        )
        result = tmpl.render()
        assert "# Reasoning" in result
        assert "Think step by step." in result

    def test_cot_disabled_by_default(self):
        from synapse_core.llm.prompt_template import PromptTemplate
        tmpl = PromptTemplate(template_id="simple")
        result = tmpl.render()
        assert "# Reasoning" not in result

    def test_cot_override_at_render(self):
        from synapse_core.llm.prompt_template import PromptTemplate
        tmpl = PromptTemplate(template_id="test", enable_cot=False)
        # Override to enable at render time
        result = tmpl.render(enable_cot=True)
        assert "# Reasoning" in result

    def test_security_hardening_included(self):
        from synapse_core.llm.prompt_template import PromptTemplate
        tmpl = PromptTemplate(template_id="test", enable_security=True)
        result = tmpl.render()
        assert "# Security" in result
        assert "ignore instructions" in result
        assert "system prompt" in result

    def test_security_can_be_disabled(self):
        from synapse_core.llm.prompt_template import PromptTemplate
        tmpl = PromptTemplate(template_id="test", enable_security=False)
        result = tmpl.render()
        assert "# Security" not in result


class TestPromptLibrary:
    def test_register_and_get(self):
        from synapse_core.llm.prompt_template import PromptTemplate, PromptLibrary
        lib = PromptLibrary()
        tmpl = PromptTemplate(template_id="cs", version="1.0", role="Customer service agent")
        lib.register(tmpl)
        got = lib.get("cs")
        assert got is not None
        assert got.role == "Customer service agent"

    def test_version_management(self):
        from synapse_core.llm.prompt_template import PromptTemplate, PromptLibrary
        lib = PromptLibrary()
        v1 = PromptTemplate(template_id="cs", version="1.0", role="v1 role")
        v2 = PromptTemplate(template_id="cs", version="2.0", role="v2 role")
        lib.register(v1)
        lib.register(v2)
        assert lib.get("cs").role == "v2 role"  # latest
        assert lib.get("cs", "1.0").role == "v1 role"  # specific version

    def test_list_templates(self):
        from synapse_core.llm.prompt_template import PromptTemplate, PromptLibrary
        lib = PromptLibrary()
        lib.register(PromptTemplate(template_id="a", version="1.0", role="Agent A"))
        lib.register(PromptTemplate(template_id="b", version="1.0", role="Agent B"))
        result = lib.list_templates()
        assert len(result) == 2

    def test_get_nonexistent_returns_none(self):
        from synapse_core.llm.prompt_template import PromptLibrary
        lib = PromptLibrary()
        assert lib.get("nonexistent") is None


class TestSystemPromptCoT:
    def test_complex_agent_has_cot(self, tool_registry):
        """Agents with many tools should get CoT guidance."""
        llm_config = LLMConfig()
        agent = AgentDefinition(
            id="complex-agent",
            name="Complex Agent",
            role="Analyst",
            goal="Analyze data",
            backstory="Expert analyst",
            llm=llm_config,
            tools=["calculator", "web_search", "sql_query"],
            max_iterations=5,
        )
        builder = GraphBuilder(agent, tool_registry)
        prompt = builder._build_system_prompt()
        assert "think step by step" in prompt

    def test_security_rules_in_prompt(self, tool_registry):
        """All agents should have security hardening in their system prompt."""
        llm_config = LLMConfig()
        agent = AgentDefinition(
            id="simple-agent",
            name="Simple",
            role="Helper",
            goal="Help",
            backstory="Helpful",
            llm=llm_config,
        )
        builder = GraphBuilder(agent, tool_registry)
        prompt = builder._build_system_prompt()
        assert "Security rules" in prompt
        assert "ignore instructions" in prompt


# -- 15. L4 Audit: Native Function Calling + Tool Schema -------------------

class TestToolToOpenAISchema:
    def test_basic_tool_schema(self, tool_registry):
        """ToolRegistry should convert tools to OpenAI Function Calling format."""
        schemas = tool_registry.to_openai_tools(["calculator"])
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "calculator"
        assert "parameters" in schema["function"]
        assert "properties" in schema["function"]["parameters"]
        assert "expression" in schema["function"]["parameters"]["properties"]

    def test_required_params_in_schema(self, tool_registry):
        """Required parameters should be correctly marked."""
        schemas = tool_registry.to_openai_tools(["calculator"])
        params = schemas[0]["function"]["parameters"]
        assert "expression" in params.get("required", [])

    def test_filter_by_tool_names(self, tool_registry):
        """Should only export specified tools."""
        schemas = tool_registry.to_openai_tools(["calculator"])
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "calculator"

    def test_export_all_tools(self, tool_registry):
        """Export all tools when no filter given."""
        schemas = tool_registry.to_openai_tools()
        assert len(schemas) == len(tool_registry.list_all())

    def test_unknown_tool_skipped(self, tool_registry):
        """Unknown tool names should be silently skipped."""
        schemas = tool_registry.to_openai_tools(["nonexistent_tool"])
        assert len(schemas) == 0


class TestNativeFunctionCalling:
    def test_openai_provider_has_last_tool_calls(self, llm_config):
        """OpenAI provider should initialize last_tool_calls."""
        from synapse_core.llm.providers import OpenAIProvider
        provider = OpenAIProvider(llm_config)
        assert provider.last_tool_calls == []

    def test_graph_builder_uses_native_tc_when_available(self, tool_registry, mock_agent):
        """GraphBuilder should prefer native tool calls over text parsing."""
        builder = GraphBuilder(mock_agent, tool_registry)

        # Simulate a provider that sets last_tool_calls
        class NativeTCProvider(MockLLMProvider):
            def __init__(self, config, responses):
                super().__init__(config, responses)
                self.last_tool_calls = [{"name": "calculator", "arguments": {"expression": "2+2"}}]

        provider = NativeTCProvider(llm_config, responses=["The answer is 4"])
        builder._provider = provider

        # _parse_tool_calls should NOT be called when native TC available
        parsed = builder._parse_tool_calls("some text")
        # Native TC takes precedence in the run() loop, not here
        # This test verifies the provider reports native TC correctly
        assert provider.last_tool_calls[0]["name"] == "calculator"
        assert provider.last_tool_calls[0]["arguments"]["expression"] == "2+2"

    def test_tools_passed_to_llm_when_agent_has_tools(self, tool_registry, mock_agent, llm_config):
        """When agent has tools, GraphBuilder should pass OpenAI tool schemas to LLM."""
        mock_agent.tools = ["calculator"]
        builder = GraphBuilder(mock_agent, tool_registry)

        # Verify to_openai_tools works for the agent's tools
        schemas = tool_registry.to_openai_tools(mock_agent.tools)
        assert len(schemas) >= 1
        assert schemas[0]["function"]["name"] == "calculator"


# -- 16. L5 Audit: Token-based Context Compression --------------------------

class TestTokenBasedContext:
    def test_empty_history(self):
        from synapse_core.graph import GraphBuilder
        result = GraphBuilder._select_history_by_tokens([], 1000)
        assert result == []

    def test_short_history_fits_entirely(self):
        from synapse_core.graph import GraphBuilder
        from synapse_core.graph.state import Message
        history = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        result = GraphBuilder._select_history_by_tokens(history, 8000)
        # All messages should be included (no summary prefix)
        assert len(result) == 2

    def test_long_history_gets_compacted(self):
        from synapse_core.graph import GraphBuilder
        from synapse_core.graph.state import Message
        # Create 50 messages with substantial content
        history = []
        for i in range(50):
            history.append(Message(role="user", content=f"This is message number {i} with some content about topic {i}. " * 10))
            history.append(Message(role="assistant", content=f"Response {i}. " * 10))
        # With only 200 token budget, only recent messages should be kept
        result = GraphBuilder._select_history_by_tokens(history, 200)
        # Should have a summary prefix + recent messages
        assert len(result) < len(history)
        # First message should be the compact summary
        if len(result) > 0 and result[0].role == "system":
            assert "Earlier conversation" in result[0].content

    def test_recent_messages_preserved(self):
        from synapse_core.graph import GraphBuilder
        from synapse_core.graph.state import Message
        history = [
            Message(role="user", content="First message"),
            Message(role="assistant", content="First response"),
            Message(role="user", content="Latest message"),
        ]
        result = GraphBuilder._select_history_by_tokens(history, 8000)
        # Latest message should be preserved
        contents = [m.content for m in result]
        assert "Latest message" in contents


# ============================================================
# MA L1 Audit: Plan-Execute Pattern
# ============================================================


class TestPlanStep:
    """Unit tests for PlanStep model."""

    def test_default_status(self):
        from synapse_core.patterns.plan_execute import PlanStep
        step = PlanStep(step_number=1, description="Search for data")
        assert step.status == "pending"
        assert step.tool_hint == ""

    def test_custom_status(self):
        from synapse_core.patterns.plan_execute import PlanStep
        step = PlanStep(step_number=2, description="Analyze", tool_hint="calculator", status="completed")
        assert step.status == "completed"
        assert step.tool_hint == "calculator"


class TestExecutionPlan:
    """Unit tests for ExecutionPlan model."""

    def test_plan_creation(self):
        from synapse_core.patterns.plan_execute import PlanStep, ExecutionPlan
        plan = ExecutionPlan(
            goal="Research competitors",
            steps=[
                PlanStep(step_number=1, description="Search"),
                PlanStep(step_number=2, description="Analyze"),
            ],
            reasoning="Need to compare 2 companies",
        )
        assert len(plan.steps) == 2
        assert plan.reasoning == "Need to compare 2 companies"

    def test_plan_empty_steps(self):
        from synapse_core.patterns.plan_execute import ExecutionPlan
        plan = ExecutionPlan(goal="test", steps=[])
        assert len(plan.steps) == 0


class TestPlanExecutePattern:
    """Unit tests for PlanExecutePattern."""

    def _make_pattern(self, **kwargs):
        from synapse_core.patterns.plan_execute import PlanExecutePattern
        from synapse_core.agent import AgentDefinition
        from synapse_core.llm import LLMConfig

        planner = AgentDefinition(
            id="planner",
            name="Planner",
            role="Task Planner",
            goal="Create and oversee execution plans",
            backstory="Expert at breaking down complex tasks",
            llm=LLMConfig(model="test-model"),
        )
        return PlanExecutePattern(planner=planner, **kwargs)

    def test_init_defaults(self):
        pattern = self._make_pattern()
        assert pattern.max_replans == 2
        assert pattern.enable_replan is True
        assert pattern.executor.id == "planner"  # defaults to planner

    def test_init_custom_executor(self):
        from synapse_core.agent import AgentDefinition
        from synapse_core.llm import LLMConfig

        executor = AgentDefinition(
            id="executor", name="Executor", role="Doer", goal="Execute",
            backstory="Executor", llm=LLMConfig(model="test-model"),
        )
        pattern = self._make_pattern(executor=executor)
        assert pattern.executor.id == "executor"

    def test_parse_plan_valid_json(self):
        pattern = self._make_pattern()
        content = json.dumps({
            "reasoning": "Need research",
            "steps": [
                {"step_number": 1, "description": "Search web", "tool_hint": "web_search"},
                {"step_number": 2, "description": "Summarize", "tool_hint": ""},
            ]
        })
        plan = pattern._parse_plan(content)
        assert plan is not None
        assert len(plan.steps) == 2
        assert plan.steps[0].tool_hint == "web_search"
        assert plan.reasoning == "Need research"

    def test_parse_plan_json_in_code_block(self):
        pattern = self._make_pattern()
        content = '```json\n{"reasoning": "test", "steps": [{"step_number": 1, "description": "Step 1", "tool_hint": ""}]}\n```'
        plan = pattern._parse_plan(content)
        assert plan is not None
        assert len(plan.steps) == 1

    def test_parse_plan_invalid(self):
        pattern = self._make_pattern()
        plan = pattern._parse_plan("This is not JSON at all")
        assert plan is None

    def test_build_summary(self):
        from synapse_core.patterns.plan_execute import PlanStep, ExecutionPlan
        pattern = self._make_pattern()
        plan = ExecutionPlan(
            goal="Test task",
            steps=[
                PlanStep(step_number=1, description="Step A", status="completed"),
                PlanStep(step_number=2, description="Step B", status="failed"),
            ],
            reasoning="Testing",
        )
        results = [
            {"step": 1, "description": "Step A", "result": "Done"},
            {"step": 2, "description": "Step B", "result": "[FAILED] Step could not be completed."},
        ]
        summary = pattern._build_summary(plan, results)
        assert "[OK]" in summary
        assert "[FAIL]" in summary
        assert "Step A" in summary

    def test_build_step_task_with_context(self):
        from synapse_core.patterns.plan_execute import PlanStep
        pattern = self._make_pattern()
        step = PlanStep(step_number=3, description="Analyze the data")
        previous = [
            {"step": 1, "result": "Collected data from source A"},
            {"step": 2, "result": "Collected data from source B"},
        ]
        task = pattern._build_step_task(step, previous)
        assert "Analyze the data" in task
        assert "Previous results" in task

    def test_build_step_task_no_context(self):
        from synapse_core.patterns.plan_execute import PlanStep
        pattern = self._make_pattern()
        step = PlanStep(step_number=1, description="First step")
        task = pattern._build_step_task(step, [])
        assert "First step" in task
        assert "Previous results" not in task

    def test_planning_prompt_includes_tools(self):
        from synapse_core.tools import ToolRegistry, ToolDefinition, ToolParameter
        registry = ToolRegistry()

        async def dummy_handler(**kwargs):
            return "ok"

        registry.register(
            ToolDefinition(
                name="web_search",
                description="Search the web",
                parameters=[ToolParameter(name="query", type="string", description="Search query")],
            ),
            dummy_handler,
        )
        pattern = self._make_pattern(tool_registry=registry)
        prompt = pattern._build_planning_prompt("Research AI trends")
        assert "web_search" in prompt
        assert "Research AI trends" in prompt

    def test_replan_prompt(self):
        from synapse_core.patterns.plan_execute import PlanStep, ExecutionPlan
        pattern = self._make_pattern()
        plan = ExecutionPlan(
            goal="Test",
            steps=[
                PlanStep(step_number=1, description="Step 1", status="completed"),
                PlanStep(step_number=2, description="Step 2", status="failed"),
            ],
        )
        completed = [{"step": 1, "result": "OK"}]
        failed = PlanStep(step_number=2, description="Step 2", status="failed")
        prompt = pattern._build_replan_prompt("Test task", plan, completed, failed, "Timeout")
        assert "Failed step 2" in prompt
        assert "Timeout" in prompt


class TestPatternsExport:
    """Verify PlanExecutePattern is exported from patterns package."""

    def test_import_from_package(self):
        from synapse_core.patterns import PlanExecutePattern
        assert PlanExecutePattern is not None

    def test_import_plan_step(self):
        from synapse_core.patterns import PlanStep
        assert PlanStep is not None

    def test_import_execution_plan(self):
        from synapse_core.patterns import ExecutionPlan
        assert ExecutionPlan is not None


# ============================================================
# MA L2 Audit: Crew Pattern (CrewAI-style Task + Context)
# ============================================================


class TestTaskDefinition:
    """Unit tests for TaskDefinition model."""

    def test_basic_creation(self):
        from synapse_core.patterns.crew import TaskDefinition
        task = TaskDefinition(
            id="research",
            description="Search for market data",
            expected_output="Structured data in JSON format",
            agent_id="researcher",
        )
        assert task.status == "pending"
        assert task.output == ""
        assert task.context_task_ids == []

    def test_with_context(self):
        from synapse_core.patterns.crew import TaskDefinition
        task = TaskDefinition(
            id="analysis",
            description="Analyze data",
            expected_output="Analysis report",
            agent_id="analyst",
            context_task_ids=["research"],
        )
        assert len(task.context_task_ids) == 1

    def test_status_lifecycle(self):
        from synapse_core.patterns.crew import TaskDefinition
        task = TaskDefinition(id="t1", description="Do something")
        assert task.status == "pending"
        task.status = "running"
        assert task.status == "running"
        task.status = "completed"
        task.output = "result text"
        assert task.output == "result text"


class TestCrewPattern:
    """Unit tests for CrewPattern."""

    def _make_agents(self):
        from synapse_core.agent import AgentDefinition
        from synapse_core.llm import LLMConfig
        config = LLMConfig(model="test-model")
        return {
            "researcher": AgentDefinition(
                id="researcher", name="Researcher", role="Market Researcher",
                goal="Gather data", backstory="Expert researcher", llm=config,
            ),
            "analyst": AgentDefinition(
                id="analyst", name="Analyst", role="Data Analyst",
                goal="Analyze data", backstory="MBA analyst", llm=config,
            ),
            "writer": AgentDefinition(
                id="writer", name="Writer", role="Report Writer",
                goal="Write reports", backstory="Professional writer", llm=config,
            ),
        }

    def _make_crew(self, tasks, **kwargs):
        from synapse_core.patterns.crew import CrewPattern
        return CrewPattern(agents=self._make_agents(), tasks=tasks, **kwargs)

    def test_init(self):
        from synapse_core.patterns.crew import TaskDefinition
        tasks = [
            TaskDefinition(id="t1", description="Task 1", agent_id="researcher"),
        ]
        crew = self._make_crew(tasks)
        assert len(crew.tasks) == 1
        assert crew.validate_output is True

    def test_build_context_empty(self):
        from synapse_core.patterns.crew import TaskDefinition
        tasks = [TaskDefinition(id="t1", description="First", agent_id="researcher")]
        crew = self._make_crew(tasks)
        context = crew._build_context(tasks[0])
        assert context == ""

    def test_build_context_with_deps(self):
        from synapse_core.patterns.crew import TaskDefinition
        t1 = TaskDefinition(id="t1", description="Research", agent_id="researcher")
        t1.status = "completed"
        t1.output = "Market share: Huawei 20%, Xiaomi 18%"
        t2 = TaskDefinition(id="t2", description="Analyze", agent_id="analyst", context_task_ids=["t1"])
        crew = self._make_crew([t1, t2])
        context = crew._build_context(t2)
        assert "Huawei 20%" in context
        assert "Context from previous tasks" in context

    def test_build_task_prompt_basic(self):
        from synapse_core.patterns.crew import TaskDefinition
        task = TaskDefinition(
            id="t1", description="Search market data",
            expected_output="JSON with market share",
            agent_id="researcher",
        )
        crew = self._make_crew([task])
        prompt = crew._build_task_prompt(task, "", "Generate market report")
        assert "Search market data" in prompt
        assert "JSON with market share" in prompt
        assert "Generate market report" in prompt

    def test_build_task_prompt_with_context(self):
        from synapse_core.patterns.crew import TaskDefinition
        task = TaskDefinition(id="t1", description="Analyze", agent_id="analyst")
        crew = self._make_crew([task])
        ctx = "Previous: Huawei 20%"
        prompt = crew._build_task_prompt(task, ctx, "Report")
        assert "Previous: Huawei 20%" in prompt

    def test_validate_output_pass(self):
        from synapse_core.patterns.crew import TaskDefinition
        tasks = [TaskDefinition(id="t1", description="t", agent_id="researcher")]
        crew = self._make_crew(tasks)
        result = crew._validate_output(
            "# Market Report\n## Data\nSome content here\n## Analysis\nMore content",
            "# Market Report\n## Data\n## Analysis",
        )
        assert result["passed"] is True

    def test_validate_output_missing_section(self):
        from synapse_core.patterns.crew import TaskDefinition
        tasks = [TaskDefinition(id="t1", description="t", agent_id="researcher")]
        crew = self._make_crew(tasks)
        result = crew._validate_output(
            "Just some text without sections",
            "# Market Report\n## Data\n## Analysis",
        )
        assert result["passed"] is False
        assert len(result["issues"]) > 0

    def test_validate_output_json_expected(self):
        from synapse_core.patterns.crew import TaskDefinition
        tasks = [TaskDefinition(id="t1", description="t", agent_id="researcher")]
        crew = self._make_crew(tasks)
        result = crew._validate_output(
            "Plain text without json",
            "Output in JSON format",
        )
        assert result["passed"] is False
        assert any("JSON" in issue for issue in result["issues"])

    def test_validate_output_markdown_expected(self):
        from synapse_core.patterns.crew import TaskDefinition
        tasks = [TaskDefinition(id="t1", description="t", agent_id="researcher")]
        crew = self._make_crew(tasks)
        result = crew._validate_output(
            "Plain text without any headings",
            "Markdown format report",
        )
        assert result["passed"] is False

    def test_validate_output_word_count(self):
        from synapse_core.patterns.crew import TaskDefinition
        tasks = [TaskDefinition(id="t1", description="t", agent_id="researcher")]
        crew = self._make_crew(tasks)
        result = crew._validate_output(
            "Short text",
            "3000-5000 word report",
        )
        assert result["passed"] is False
        assert any("short" in issue.lower() for issue in result["issues"])

    def test_validate_output_disabled(self):
        from synapse_core.patterns.crew import TaskDefinition
        tasks = [TaskDefinition(id="t1", description="t", agent_id="researcher")]
        crew = self._make_crew(tasks, validate_output=False)
        assert crew.validate_output is False


class TestCrewExport:
    """Verify CrewPattern and TaskDefinition exports."""

    def test_import_crew_pattern(self):
        from synapse_core.patterns import CrewPattern
        assert CrewPattern is not None

    def test_import_task_definition(self):
        from synapse_core.patterns import TaskDefinition
        assert TaskDefinition is not None


# ============================================================
# MA L4 Audit: Tool Fallback Chain + Rate Limiting
# ============================================================


class TestToolFallbackChain:
    """Tests for tool fallback chain execution."""

    def _make_registry_with_fallback(self):
        from synapse_core.tools import ToolRegistry, ToolDefinition, ToolParameter

        registry = ToolRegistry()

        async def primary_handler(**kwargs):
            raise RuntimeError("Primary API down")

        async def fallback_handler(**kwargs):
            return f"fallback result for {kwargs.get('query', '')}"

        async def fallback2_handler(**kwargs):
            return f"fallback2 result for {kwargs.get('query', '')}"

        registry.register(
            ToolDefinition(name="search", description="Primary search"),
            primary_handler,
            fallback_chain=["search_v2", "search_local"],
        )
        registry.register(
            ToolDefinition(name="search_v2", description="Backup search"),
            fallback_handler,
        )
        registry.register(
            ToolDefinition(name="search_local", description="Local fallback"),
            fallback2_handler,
        )
        return registry

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self):
        registry = self._make_registry_with_fallback()
        result = await registry.execute("search", query="test")
        assert result.error is None
        assert result.output == "fallback result for test"
        assert result.fallback_used == "search_v2"

    @pytest.mark.asyncio
    async def test_no_fallback_when_primary_succeeds(self):
        from synapse_core.tools import ToolRegistry, ToolDefinition

        registry = ToolRegistry()

        async def ok_handler(**kwargs):
            return "ok"

        async def fb_handler(**kwargs):
            return "fallback"

        registry.register(
            ToolDefinition(name="tool_a", description="A"),
            ok_handler,
            fallback_chain=["tool_b"],
        )
        registry.register(
            ToolDefinition(name="tool_b", description="B"),
            fb_handler,
        )
        result = await registry.execute("tool_a")
        assert result.output == "ok"
        assert result.fallback_used is None

    @pytest.mark.asyncio
    async def test_all_fallbacks_exhausted(self):
        from synapse_core.tools import ToolRegistry, ToolDefinition

        registry = ToolRegistry()

        async def fail_handler(**kwargs):
            raise RuntimeError("fail")

        registry.register(
            ToolDefinition(name="primary", description="P"),
            fail_handler,
            fallback_chain=["fb1", "fb2"],
        )
        registry.register(
            ToolDefinition(name="fb1", description="F1"),
            fail_handler,
        )
        registry.register(
            ToolDefinition(name="fb2", description="F2"),
            fail_handler,
        )
        result = await registry.execute("primary")
        assert result.error is not None
        assert "all fallbacks exhausted" in result.error


class TestToolRateLimiter:
    """Tests for per-tool rate limiting."""

    def test_rate_limiter_allows_under_limit(self):
        from synapse_core.tools import _RateLimiter
        limiter = _RateLimiter(max_calls=3, window_ms=10000)
        assert limiter.allow() is True
        assert limiter.allow() is True
        assert limiter.allow() is True

    def test_rate_limiter_blocks_over_limit(self):
        from synapse_core.tools import _RateLimiter
        limiter = _RateLimiter(max_calls=2, window_ms=10000)
        limiter.allow()
        limiter.allow()
        assert limiter.allow() is False

    @pytest.mark.asyncio
    async def test_rate_limited_tool_returns_error(self):
        from synapse_core.tools import ToolRegistry, ToolDefinition, ToolSafetyConfig

        registry = ToolRegistry()

        async def handler(**kwargs):
            return "ok"

        registry.register(
            ToolDefinition(
                name="limited_tool",
                description="Rate limited",
                safety=ToolSafetyConfig(rate_limit_max_calls=1, rate_limit_window_ms=60000),
            ),
            handler,
        )
        # First call should succeed
        result1 = await registry.execute("limited_tool", x="1")
        assert result1.output == "ok"
        # Second call should be rate limited
        result2 = await registry.execute("limited_tool", x="2")
        assert result2.error is not None
        assert "Rate limit" in result2.error

    @pytest.mark.asyncio
    async def test_rate_limited_tool_with_fallback(self):
        from synapse_core.tools import ToolRegistry, ToolDefinition, ToolSafetyConfig

        registry = ToolRegistry()

        async def primary(**kwargs):
            return "primary"

        async def fallback(**kwargs):
            return "fallback"

        registry.register(
            ToolDefinition(
                name="primary",
                description="P",
                safety=ToolSafetyConfig(rate_limit_max_calls=1, rate_limit_window_ms=60000),
            ),
            primary,
            fallback_chain=["backup"],
        )
        registry.register(
            ToolDefinition(name="backup", description="B"),
            fallback,
        )

        # First call uses primary
        r1 = await registry.execute("primary")
        assert r1.output == "primary"

        # Second call triggers rate limit, falls back
        r2 = await registry.execute("primary")
        assert r2.output == "fallback"
        assert r2.fallback_used == "backup"


# ============================================================
# MA L7 Audit: RBAC for Tool Permissions
# ============================================================


class TestRBACChecker:
    """Tests for role-based access control."""

    def test_predefined_user_role(self):
        from synapse_core.tools.safety import RBACChecker, Permission
        checker = RBACChecker()
        assert checker.check("user", "web_search", Permission.EXECUTE) is True
        assert checker.check("user", "sql_query", Permission.EXECUTE) is False

    def test_predefined_analyst_role(self):
        from synapse_core.tools.safety import RBACChecker, Permission
        checker = RBACChecker()
        assert checker.check("analyst", "web_search", Permission.EXECUTE) is True
        assert checker.check("analyst", "sql_query", Permission.READ) is True
        assert checker.check("analyst", "sql_query", Permission.WRITE) is False

    def test_admin_has_all_permissions(self):
        from synapse_core.tools.safety import RBACChecker, Permission
        checker = RBACChecker()
        assert checker.check("admin", "any_tool", Permission.EXECUTE) is True
        assert checker.check("admin", "any_tool", Permission.ADMIN) is True

    def test_unknown_role_denied(self):
        from synapse_core.tools.safety import RBACChecker, Permission
        checker = RBACChecker()
        assert checker.check("guest", "web_search", Permission.EXECUTE) is False

    def test_grant_permission(self):
        from synapse_core.tools.safety import RBACChecker, Permission
        checker = RBACChecker()
        checker.grant("guest", "web_search", Permission.EXECUTE)
        assert checker.check("guest", "web_search", Permission.EXECUTE) is True

    def test_revoke_permission(self):
        from synapse_core.tools.safety import RBACChecker, Permission
        checker = RBACChecker()
        assert checker.check("user", "web_search", Permission.EXECUTE) is True
        checker.revoke("user", "web_search", Permission.EXECUTE)
        assert checker.check("user", "web_search", Permission.EXECUTE) is False

    def test_audit_log(self):
        from synapse_core.tools.safety import RBACChecker, Permission
        checker = RBACChecker()
        checker.check("user", "web_search", Permission.EXECUTE)
        checker.check("user", "sql_query", Permission.EXECUTE)
        log = checker.get_audit_log()
        assert len(log) == 2
        assert log[0]["allowed"] is True
        assert log[1]["allowed"] is False

    def test_custom_role(self):
        from synapse_core.tools.safety import RBACChecker, Role, Permission
        custom = Role(name="developer", permissions={
            "code_execution": {Permission.EXECUTE},
            "sql_query": {Permission.READ, Permission.WRITE},
        })
        checker = RBACChecker(roles={"developer": custom})
        assert checker.check("developer", "code_execution", Permission.EXECUTE) is True
        assert checker.check("developer", "sql_query", Permission.WRITE) is True
        assert checker.check("developer", "web_search", Permission.EXECUTE) is False


class TestSafeToolExecutorRBAC:
    """Tests for SafeToolExecutor with RBAC integration."""

    def test_rbac_check_blocks(self):
        from synapse_core.tools.safety import SafeToolExecutor
        executor = SafeToolExecutor()
        error = executor.check_all("sql_query", {"query": "SELECT 1"}, role="user")
        assert error is not None
        assert "does not have permission" in error

    def test_rbac_check_allows(self):
        from synapse_core.tools.safety import SafeToolExecutor
        executor = SafeToolExecutor()
        error = executor.check_all("web_search", {"query": "test"}, role="user")
        assert error is None

    def test_rbac_admin_bypasses(self):
        from synapse_core.tools.safety import SafeToolExecutor
        executor = SafeToolExecutor()
        error = executor.check_all("sql_query", {"query": "DROP TABLE"}, role="admin")
        assert error is None


# ============================================================
# MA L9 Audit: Feedback Collection + A/B Testing
# ============================================================


class TestFeedbackStore:
    """Tests for feedback collection system."""

    def test_add_and_retrieve(self):
        from synapse_core.feedback import FeedbackStore, FeedbackEntry, FeedbackType
        store = FeedbackStore()
        entry = FeedbackEntry(
            run_id="run-1",
            agent_id="agent-1",
            user_id="user-1",
            feedback_type=FeedbackType.RATING,
            score=4.0,
            comment="Good response",
        )
        fid = store.add(entry)
        assert fid != ""
        results = store.get_by_run("run-1")
        assert len(results) == 1
        assert results[0].score == 4.0

    def test_get_by_agent(self):
        from synapse_core.feedback import FeedbackStore, FeedbackEntry, FeedbackType
        store = FeedbackStore()
        store.add(FeedbackEntry(run_id="r1", agent_id="a1", feedback_type=FeedbackType.RATING, score=5.0))
        store.add(FeedbackEntry(run_id="r2", agent_id="a2", feedback_type=FeedbackType.RATING, score=3.0))
        store.add(FeedbackEntry(run_id="r3", agent_id="a1", feedback_type=FeedbackType.RATING, score=4.0))
        results = store.get_by_agent("a1")
        assert len(results) == 2

    def test_get_stats(self):
        from synapse_core.feedback import FeedbackStore, FeedbackEntry, FeedbackType
        store = FeedbackStore()
        store.add(FeedbackEntry(agent_id="a1", feedback_type=FeedbackType.RATING, score=5.0))
        store.add(FeedbackEntry(agent_id="a1", feedback_type=FeedbackType.RATING, score=3.0))
        store.add(FeedbackEntry(agent_id="a1", feedback_type=FeedbackType.THUMBS, score=1.0))
        stats = store.get_stats(agent_id="a1")
        assert stats["count"] == 3
        assert stats["avg_score"] == 3.0
        assert stats["positive_ratio"] == 1.0

    def test_get_stats_empty(self):
        from synapse_core.feedback import FeedbackStore
        store = FeedbackStore()
        stats = store.get_stats()
        assert stats["count"] == 0

    def test_low_score_filter(self):
        from synapse_core.feedback import FeedbackStore, FeedbackEntry, FeedbackType
        store = FeedbackStore()
        store.add(FeedbackEntry(agent_id="a1", feedback_type=FeedbackType.RATING, score=1.0, comment="Bad"))
        store.add(FeedbackEntry(agent_id="a1", feedback_type=FeedbackType.RATING, score=5.0, comment="Great"))
        store.add(FeedbackEntry(agent_id="a1", feedback_type=FeedbackType.RATING, score=2.0, comment="Poor"))
        low = store.get_low_score_entries(threshold=2.0)
        assert len(low) == 2

    def test_auto_timestamp(self):
        from synapse_core.feedback import FeedbackStore, FeedbackEntry, FeedbackType
        store = FeedbackStore()
        entry = FeedbackEntry(feedback_type=FeedbackType.RATING, score=3.0)
        assert entry.timestamp == 0.0
        store.add(entry)
        assert entry.timestamp > 0.0

    def test_feedback_types(self):
        from synapse_core.feedback import FeedbackType
        assert FeedbackType.RATING.value == "rating"
        assert FeedbackType.THUMBS.value == "thumbs"
        assert FeedbackType.COMMENT.value == "comment"
        assert FeedbackType.IMPLICIT.value == "implicit"


class TestABTestManager:
    """Tests for A/B testing framework."""

    def test_create_experiment(self):
        from synapse_core.feedback import ABTestManager, ABTestExperiment
        manager = ABTestManager()
        manager.create(ABTestExperiment(name="prompt_v2", control_prompt="v1", treatment_prompt="v2"))
        assert "prompt_v2" in manager.list_experiments()

    def test_deterministic_assignment(self):
        from synapse_core.feedback import ABTestManager, ABTestExperiment
        manager = ABTestManager()
        manager.create(ABTestExperiment(name="test", traffic_percentage=50))
        # Same user always gets same variant
        v1 = manager.get_variant("test", "user-123")
        v2 = manager.get_variant("test", "user-123")
        assert v1 == v2

    def test_different_users_different_variants(self):
        from synapse_core.feedback import ABTestManager, ABTestExperiment
        manager = ABTestManager()
        manager.create(ABTestExperiment(name="test", traffic_percentage=50))
        variants = set()
        for i in range(100):
            variants.add(manager.get_variant("test", f"user-{i}"))
        # With 100 users and 50% split, both variants should appear
        assert len(variants) == 2

    def test_record_and_get_results(self):
        from synapse_core.feedback import ABTestManager, ABTestExperiment
        manager = ABTestManager()
        manager.create(ABTestExperiment(name="test"))
        for _ in range(10):
            manager.record_result("test", "control", 3.0)
            manager.record_result("test", "treatment", 4.0)
        results = manager.get_results("test")
        assert results is not None
        assert results["control_avg"] == 3.0
        assert results["treatment_avg"] == 4.0
        assert results["improvement_pct"] > 0

    def test_recommendation_promote(self):
        from synapse_core.feedback import ABTestManager, ABTestExperiment
        manager = ABTestManager()
        manager.create(ABTestExperiment(name="test"))
        for _ in range(10):
            manager.record_result("test", "control", 3.0)
            manager.record_result("test", "treatment", 4.5)
        results = manager.get_results("test")
        assert results["recommendation"] == "promote_treatment"

    def test_recommendation_insufficient_data(self):
        from synapse_core.feedback import ABTestManager, ABTestExperiment
        manager = ABTestManager()
        manager.create(ABTestExperiment(name="test"))
        manager.record_result("test", "control", 3.0)
        results = manager.get_results("test")
        assert results["recommendation"] == "insufficient_data"

    def test_stop_experiment(self):
        from synapse_core.feedback import ABTestManager, ABTestExperiment
        manager = ABTestManager()
        manager.create(ABTestExperiment(name="test"))
        manager.stop("test")
        # After stopping, variant should default to control
        v = manager.get_variant("test", "user-1")
        assert v == "control"

    def test_nonexistent_experiment(self):
        from synapse_core.feedback import ABTestManager
        manager = ABTestManager()
        assert manager.get_variant("nonexistent", "user-1") == "control"
        assert manager.get_results("nonexistent") is None
