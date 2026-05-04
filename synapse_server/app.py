"""FastAPI application factory with all routes and middleware."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from synapse_server.middleware import RateLimitMiddleware, ErrorHandlingMiddleware
from synapse_server.services import AgentService, ToolService, SessionService, RunService
from synapse_server.routes import runs, agents, sessions, knowledge, observability, scheduler, skills, mcp, files, multi_agent
from synapse_core.scheduler import TaskScheduler
from synapse_core.mcp import MCPClient


def create_app() -> FastAPI:
    app = FastAPI(
        title="Synapse",
        description="Multi-agent collaboration framework API",
        version="0.1.0",
    )

    # Middleware (order matters: outermost first)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize services
    agent_svc = AgentService()
    tool_svc = ToolService()
    session_svc = SessionService()
    run_svc = RunService(agent_svc, tool_svc)

    # Scheduler & MCP
    task_scheduler = TaskScheduler()
    mcp_client = MCPClient()
    mcp_client.load_config(Path(__file__).resolve().parent.parent / "config" / "mcp_servers.json")

    # Configure routes with service references
    runs.configure(agent_svc, tool_svc, session_svc)
    multi_agent.configure(agent_svc, tool_svc)
    agents.configure(agent_svc)
    sessions.configure(session_svc)
    scheduler.configure(task_scheduler)
    skills.configure(agent_svc.skill_registry)
    mcp.configure(mcp_client)

    # Store tool_service reference for MCP connect
    import synapse_server.services as _svc_mod
    _svc_mod._tool_service_instance = tool_svc

    # Register routes
    app.include_router(runs.router, prefix="/api", tags=["runs"])
    app.include_router(multi_agent.router, prefix="/api", tags=["multi-agent"])
    app.include_router(agents.router, prefix="/api", tags=["agents"])
    app.include_router(sessions.router, prefix="/api", tags=["sessions"])
    app.include_router(knowledge.router, prefix="/api", tags=["knowledge"])
    app.include_router(observability.router, prefix="/api", tags=["observability"])
    app.include_router(scheduler.router, prefix="/api", tags=["scheduler"])
    app.include_router(skills.router, prefix="/api", tags=["skills"])
    app.include_router(mcp.router, prefix="/api", tags=["mcp"])
    app.include_router(files.router, prefix="/api", tags=["files"])

    # Serve uploaded files as static assets (must come after API routes)
    from fastapi.staticfiles import StaticFiles
    uploads_dir = Path(__file__).resolve().parent.parent / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Detailed health check for readiness/liveness probes."""
        from synapse_core.observability import get_hub
        checks: dict[str, str] = {}
        try:
            hub = get_hub()
            checks["observability"] = "ok"
        except Exception:
            checks["observability"] = "degraded"
        overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
        return {
            "status": overall,
            "version": "0.1.0",
            "checks": checks,
        }

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        """Prometheus-compatible metrics in JSON format."""
        try:
            hub = get_hub()
            summary = hub.get_summary()
            return {"status": "ok", "data": summary}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    return app


app = create_app()
