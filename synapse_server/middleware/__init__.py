"""Middleware: JWT auth, rate limiting, error handling."""

from __future__ import annotations

import hmac
import logging
import time
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding window rate limiter per client IP."""

    # Sweep stale entries every 5 minutes regardless of traffic
    _CLEANUP_INTERVAL = 300

    def __init__(
        self,
        app: Any,
        max_requests: int = 60,
        window_seconds: int = 60,
        trust_proxy: bool = False,
    ) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.trust_proxy = trust_proxy
        self._clients: dict[str, list[float]] = {}
        self._last_cleanup: float = time.time()

    def _get_client_id(self, request: Request) -> str:
        if self.trust_proxy:
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _purge_stale(self, now: float) -> None:
        """Remove entries whose entire timestamp window has expired."""
        if now - self._last_cleanup < self._CLEANUP_INTERVAL:
            return
        cutoff = now - self.window_seconds
        stale = [cid for cid, ts in self._clients.items() if not ts or ts[-1] <= cutoff]
        for cid in stale:
            del self._clients[cid]
        self._last_cleanup = now

    def _is_rate_limited(self, client_id: str, now: float) -> bool:
        if client_id not in self._clients:
            self._clients[client_id] = [now]
            return False

        timestamps = self._clients[client_id]
        cutoff = now - self.window_seconds
        timestamps[:] = [t for t in timestamps if t > cutoff]

        if len(timestamps) >= self.max_requests:
            return True

        timestamps.append(now)
        return False

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        now = time.time()
        self._purge_stale(now)

        client_id = self._get_client_id(request)
        if self._is_rate_limited(client_id, now):
            return Response(
                content='{"detail":"Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
            )

        response = await call_next(request)
        remaining = self.max_requests - len(self._clients.get(client_id, []))
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject security-related HTTP response headers."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"] = "0"
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate X-API-Key header on all /api/* endpoints.

    Skips authentication when API_KEY env var is not set (dev mode — logs a warning).
    Paths in EXEMPT_PATHS are always accessible without a key.
    """

    EXEMPT_PATHS = {"/health", "/docs", "/redoc", "/openapi.json", "/"}
    PROTECTED_NON_API = {"/metrics"}

    def __init__(self, app: Any, api_key: str | None = None) -> None:
        super().__init__(app)
        self.api_key = api_key
        if not api_key:
            logger.warning(
                "API_KEY is not set — all /api/* endpoints are unprotected. "
                "Set API_KEY environment variable before exposing to a network."
            )

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self.api_key:
            return await call_next(request)

        path = request.url.path
        if path in self.EXEMPT_PATHS:
            return await call_next(request)
        # Protect /api/* and explicitly listed non-api paths (e.g. /metrics)
        if not path.startswith("/api/") and path not in self.PROTECTED_NON_API:
            return await call_next(request)

        # OPTIONS pre-flight must pass through for CORS to work
        if request.method == "OPTIONS":
            return await call_next(request)

        provided = request.headers.get("x-api-key", "")
        if not provided or not hmac.compare_digest(provided, self.api_key):
            return Response(
                content='{"detail":"Invalid or missing API key"}',
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Catch unhandled exceptions and return structured JSON errors."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            return await call_next(request)
        except Exception:
            logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
            return Response(
                content='{"detail":"Internal server error"}',
                status_code=500,
                media_type="application/json",
            )
