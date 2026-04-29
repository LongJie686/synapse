"""Web search tool using Tavily API."""

from __future__ import annotations

import json
from typing import Any

from synapse_core.tools import ToolDefinition, ToolParameter, ToolSafetyConfig


WEB_SEARCH_DEF = ToolDefinition(
    name="web_search",
    description="Search the web for information. Returns top results with titles, URLs, and snippets.",
    category="search",
    parameters=[
        ToolParameter(name="query", type="string", description="Search query"),
        ToolParameter(name="max_results", type="integer", description="Max results (default 5)", required=False, default=5),
    ],
    safety=ToolSafetyConfig(requires_approval=False, timeout_ms=15000),
)


async def web_search_handler(query: str, max_results: int = 5, **kwargs: Any) -> str:
    """Search the web using Tavily API."""
    import os

    api_key = kwargs.get("api_key") or os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY not configured"

    try:
        import httpx
    except ImportError:
        return "Error: httpx not installed. Run: pip install httpx"

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": True,
            },
        )
        response.raise_for_status()
        data = response.json()

    results = []
    if data.get("answer"):
        results.append(f"Answer: {data['answer']}")

    for r in data.get("results", []):
        results.append(f"- {r.get('title', 'No title')}: {r.get('url', '')}\n  {r.get('content', '')}")

    return "\n\n".join(results)


def register_web_search(registry: "ToolRegistry") -> None:
    registry.register(WEB_SEARCH_DEF, web_search_handler)
