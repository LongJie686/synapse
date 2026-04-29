"""FastAPI application factory with all routes and middleware."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from synapse_server.middleware import RateLimitMiddleware, ErrorHandlingMiddleware
from synapse_server.services import AgentService, ToolService, SessionService, RunService
from synapse_server.routes import runs, agents, sessions, knowledge, observability


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

    # Configure routes with service references
    runs.configure(agent_svc, tool_svc, session_svc)
    agents.configure(agent_svc)
    sessions.configure(session_svc)

    # Register routes
    app.include_router(runs.router, prefix="/api", tags=["runs"])
    app.include_router(agents.router, prefix="/api", tags=["agents"])
    app.include_router(sessions.router, prefix="/api", tags=["sessions"])
    app.include_router(knowledge.router, prefix="/api", tags=["knowledge"])
    app.include_router(observability.router, prefix="/api", tags=["observability"])

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.1.0"}

    return app
