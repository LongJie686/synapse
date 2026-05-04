# Synapse

A production-grade multi-agent collaboration framework built on LangGraph, with intelligent memory, RAG pipelines, safety guardrails, and full observability.

## Features

- **Multi-Agent Orchestration** -- 6 patterns: Supervisor, Parallel, Hierarchical, Collaboration, Plan-Execute, Crew
- **Intelligent Memory System** -- Three-layer architecture (working/short-term/long-term) with auto context compaction, user profile auto-update, and importance-based long-term memory save
- **RAG Knowledge Management** -- 4 strategies (Basic/Self-RAG/Corrective-RAG/Adaptive-RAG), 3 retrieval methods (Vector/Hybrid/MMR), local BGE embedding (free, offline), PDF/DOCX/TXT/MD loaders
- **Safety Guardrails** -- Input filtering (9 injection patterns), output filtering (credential redaction), PII detection (7 types), content moderation (4 categories)
- **Tool System** -- Built-in tools (calculator, web search, HTTP, SQL, code execution, file operations) with rate limiting, permission checks, and sandboxed execution
- **Cost-Aware LLM Router** -- Automatic model selection based on task complexity with fallback chains across GLM, OpenAI, Anthropic, and Ollama
- **Structured Output** -- Pydantic model to validated LLM JSON output with retry logic
- **Full Observability** -- Structured logging, span-based tracing, token usage tracking, metrics (counters/gauges/histograms), LangSmith integration, Prometheus endpoint
- **Plugin System** -- Extensible via ToolPlugin, AgentPlugin, GuardrailPlugin with directory-based auto-loading + MCP client + Skills YAML
- **CLI Interface** -- Full-featured command line with streaming output, session management, knowledge commands, cost tracking
- **API Server** -- FastAPI with SSE streaming, 22+ REST endpoints, rate limiting, CORS
- **Docker Deployment** -- One-command deployment with PostgreSQL (pgvector) + Redis

## Architecture

```
synapse/
  synapse_core/                  # Core engine
    agent/                       # Agent definitions and registry
    llm/                         # LLM providers, router, structured output
    graph/                       # StateGraph builder, nodes, state
    patterns/                    # supervisor, parallel, hierarchical, collaboration
    tools/                       # Tool registry, safety, built-in tools
    memory/                      # Short-term + long-term + user profiles
    rag/                         # Document loaders, splitters, embedding, retrieval
    guardrails/                  # Input/output filtering, PII, content moderation
    observability/               # Logging, tracing, metrics, LangSmith
    plugin/                      # Plugin host and interfaces
    streaming/                   # SSE event management
    checkpoint/                  # Memory and DB checkpoints
  synapse_server/                # FastAPI server
    routes/                      # REST endpoints (runs, agents, sessions, knowledge, metrics)
    middleware/                   # Rate limiting, error handling
    services/                    # Business logic layer
  docker/                        # Docker Compose + Dockerfile
```

## Quick Start

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager

### Install

```bash
git clone https://github.com/LongJie686/synapse.git
cd synapse
uv venv --python 3.9
uv sync
```

### Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Run Development Server

```bash
uvicorn synapse_server.app:create_app --factory --reload --port 8000
```

### Docker Compose

```bash
docker compose -f docker/docker-compose.dev.yml up
```

## Usage

### Create a Run (SSE Streaming)

```bash
curl -X POST http://localhost:8000/api/runs/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Calculate 15% tip on $85.50"}'
```

### Python SDK

```python
from synapse_core.agent import AgentDefinition
from synapse_core.llm import LLMConfig
from synapse_core.graph import GraphBuilder
from synapse_core.tools.builtin import register_calculator
from synapse_core.tools import ToolRegistry

# Define an agent
agent = AgentDefinition(
    id="math-tutor",
    name="Math Tutor",
    role="Mathematics expert",
    goal="Help users solve math problems",
    backstory="You are an experienced math tutor.",
    llm=LLMConfig(model="gpt-4o-mini"),
    tools=["calculator"],
)

# Set up tools
registry = ToolRegistry()
register_calculator(registry)

# Run
builder = GraphBuilder(agent, registry)
async for event in builder.run("What is sqrt(144) + 37?"):
    print(f"[{event.type}] {event.data}")
```

### Multi-Agent with Supervisor

```python
from synapse_core.patterns.supervisor import SupervisorPattern
from synapse_core.agent import AgentDefinition, AgentRegistry
from synapse_core.llm import LLMConfig

registry = AgentRegistry()
registry.register(AgentDefinition(
    id="coder", name="Coder", role="Code expert",
    goal="Write code", backstory="Senior developer.",
    llm=LLMConfig(), tools=["calculator"],
))
registry.register(AgentDefinition(
    id="analyst", name="Analyst", role="Data analyst",
    goal="Analyze data", backstory="Data science expert.",
    llm=LLMConfig(), tools=["calculator"],
))

supervisor = SupervisorPattern(agent_registry=registry, llm_config=LLMConfig())
result = await supervisor.run("Write a function to calculate fibonacci and analyze its complexity")
```

### Cost-Aware Model Routing

```python
from synapse_core.llm.router import ModelRouter
from synapse_core.llm import LLMMessage

router = ModelRouter()
messages = [LLMMessage(role="user", content="Design a microservice architecture for e-commerce")]

# Automatic: classifies complexity and selects optimal model
response = await router.route(messages, preference="balanced")
```

### Structured Output

```python
from pydantic import BaseModel, Field
from synapse_core.llm.structured_output import generate_structured_output

class SentimentResult(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="Confidence score 0-1")
    keywords: list[str] = Field(default_factory=list)

result = await generate_structured_output(
    model=SentimentResult,
    prompt="Analyze: 'This product exceeded all my expectations!'",
)
print(result.sentiment)  # "positive"
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/runs` | Create a new run |
| POST | `/api/runs/stream` | Stream run with SSE |
| GET | `/api/runs/{id}` | Get run status |
| GET | `/api/agents` | List agents |
| POST | `/api/agents` | Create agent |
| GET | `/api/agents/{id}` | Get agent details |
| DELETE | `/api/agents/{id}` | Delete agent |
| POST | `/api/sessions` | Create session |
| GET | `/api/sessions` | List sessions |
| GET | `/api/sessions/{id}` | Get session with messages |
| DELETE | `/api/sessions/{id}` | Delete session |
| POST | `/api/knowledge/collections` | Create knowledge collection |
| POST | `/api/knowledge/upload` | Upload document |
| POST | `/api/knowledge/query` | Query knowledge base |
| GET | `/api/metrics` | Get observability metrics |
| GET | `/api/health` | Health check |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Multi-Agent | Supervisor, Parallel, Hierarchical, Collaboration, Plan-Execute, Crew |
| Core Engine | LangGraph + LangChain |
| API Server | FastAPI + Uvicorn + SSE |
| Database | PostgreSQL + pgvector |
| Cache | Redis |
| LLM | Anthropic + OpenAI + Ollama |
| Embedding | BGE-small-zh-v1.5 (local, free) |
| Validation | Pydantic v2 |
| Observability | Structured Logging + LangSmith + Prometheus |
| Deployment | Docker Compose |

## License

Apache License 2.0
