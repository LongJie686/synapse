# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synapse is a production-grade multi-agent AI orchestration framework. It is a Python/TypeScript monorepo with three packages:

- `synapse_core/` — Core multi-agent engine (Python), LangGraph-based state machine, memory, RAG, tools, guardrails
- `synapse_server/` — FastAPI REST server with SSE streaming, database persistence, auth
- `synapse_web/` — Next.js 15 frontend (React 19, Tailwind CSS 4)

## Commands

### Backend (Python, managed by `uv`)

```bash
# Setup
uv venv --python 3.11 && uv sync

# Dev server (auto-reload)
uvicorn synapse_server.app:create_app --factory --reload --port 8000

# Or via full stack with Postgres + Redis
docker compose -f docker/docker-compose.dev.yml up

# Tests
pytest                                          # all tests
pytest tests/unit/                             # unit only
pytest tests/integration/test_multi_agent.py   # single file
pytest -m unit                                 # by marker
pytest --cov=synapse_core --cov=synapse_server --cov-report=term-missing

# Lint & format
ruff check synapse_core synapse_server
ruff check --fix synapse_core synapse_server
ruff format synapse_core synapse_server
mypy synapse_core synapse_server
```

### Frontend (`synapse_web/`)

```bash
npm install
npm run dev    # http://localhost:3000
npm run build
```

## Architecture

### Request Flow

```
Browser → Next.js (port 3000)
       → FastAPI (port 8000) via SSE stream (direct, bypasses Next.js proxy)
       → GraphBuilder (LangGraph StateGraph)
       → Router → Memory retrieval → LLM Provider → Tool Execution
       → SSE events streamed back (agent_think, tool_result, done)
```

**Critical**: The frontend connects to the FastAPI backend directly via SSE (`http://localhost:8000`), not through the Next.js proxy, to avoid buffering issues.

### Core Engine (`synapse_core/`)

| Module | Purpose |
|--------|---------|
| `graph/` | LangGraph StateGraph builder — orchestrates all node execution |
| `patterns/` | 6 multi-agent patterns: Supervisor, Parallel, Hierarchical, Collaboration, Plan-Execute, Crew |
| `llm/` | LLM provider router supporting OpenAI, Anthropic, Ollama |
| `memory/` | 3-layer memory: working (summary-buffer), short-term (Redis), long-term (ChromaDB/pgvector) |
| `rag/` | RAG pipelines with 4 strategies: Basic, Self-RAG, Corrective-RAG, Adaptive-RAG |
| `tools/` | Tool registry + built-ins: calculator, web_search, http, sql, code_execution, file_ops |
| `guardrails/` | Input/output filters, PII detection (7 types), content moderation (4 categories) |
| `observability/` | Structured JSON logging, span tracing, LangSmith integration, Prometheus metrics |
| `plugin/` | Auto-discovery plugin system for Tool/Agent/Guardrail extensions |
| `mcp/` | MCP (Model Context Protocol) client |
| `streaming/` | SSE event management |

### Server (`synapse_server/`)

FastAPI app created via factory function `create_app()` in `app.py`. Routes are in `routes/`, business logic in `services/`. Key routes: `/api/runs/stream` (SSE), `/api/agents`, `/api/sessions`, `/api/knowledge`, `/api/multi-agent/run`, `/metrics`, `/health`.

### Frontend (`synapse_web/`)

Next.js App Router. Components in `src/components/`, hooks in `src/hooks/`, utilities in `src/lib/`. Bilingual (zh/en) i18n built-in. Sidebar conversation list with SSE-driven chat.

## Key Technologies

- **LangGraph 0.4+** — State machine for agent orchestration
- **LangChain 0.3+** — LLM abstractions, vector stores, loaders
- **Pydantic v2** — All data models; use v2 syntax (`model_validator`, `field_validator`)
- **SQLAlchemy + Alembic** — Async ORM + migrations
- **ChromaDB + pgvector** — Vector storage (local BGE-small-zh-v1.5 embeddings, no API key needed)
- **SSE-Starlette** — Server-Sent Events for streaming

## Code Conventions

- Python 3.11+ with full type hints; Ruff line length 100; async/await for all I/O
- Pydantic v2 for all models throughout `synapse_core` and `synapse_server`
- Max file: 800 lines; max function: 50 lines
- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`
- Environment config via `.env` (copy from `.env.example`); required vars: `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, `DATABASE_URL`, `REDIS_URL`

## Infrastructure

Docker Compose runs three services: `postgres` (pgvector:pg16, port 5432), `redis` (7-alpine, port 6379), `server` (FastAPI, port 8000). Development compose adds volume mounts for hot-reload.
