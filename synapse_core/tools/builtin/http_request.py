"""HTTP request tool."""

from __future__ import annotations

import json
from typing import Any

from synapse_core.tools import ToolDefinition, ToolParameter, ToolSafetyConfig
from synapse_core.tools.builtin._ssrf_guard import check_ssrf


HTTP_REQUEST_DEF = ToolDefinition(
    name="http_request",
    description="Make HTTP requests to external APIs. Supports GET, POST, PUT, DELETE.",
    category="http",
    parameters=[
        ToolParameter(name="url", type="string", description="Target URL"),
        ToolParameter(name="method", type="string", description="HTTP method (GET, POST, PUT, DELETE)", required=False, default="GET"),
        ToolParameter(name="headers", type="object", description="Request headers as JSON object", required=False),
        ToolParameter(name="body", type="object", description="Request body as JSON object (for POST/PUT)", required=False),
    ],
    safety=ToolSafetyConfig(
        requires_approval=True,
        timeout_ms=30000,
        input_validation=True,
    ),
)


async def http_request_handler(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    **kwargs: Any,
) -> str:
    """Execute an HTTP request."""
    try:
        import httpx
    except ImportError:
        return "Error: httpx not installed"

    method = method.upper()
    if method not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
        return f"Error: Unsupported method '{method}'"

    try:
        await check_ssrf(url)
    except ValueError as e:
        return f"Error: {e}"

    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        request_fn = getattr(client, method.lower())
        request_kwargs: dict[str, Any] = {}
        if headers:
            request_kwargs["headers"] = headers
        if body and method in ("POST", "PUT", "PATCH"):
            request_kwargs["json"] = body

        response = await request_fn(url, **request_kwargs)

        return json.dumps({
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response.text[:2000],
        }, ensure_ascii=False, indent=2)


def register_http_request(registry: "ToolRegistry") -> None:
    registry.register(HTTP_REQUEST_DEF, http_request_handler)
