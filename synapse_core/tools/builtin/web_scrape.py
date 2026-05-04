"""Web page scraping tool - extract text content from any URL."""

from __future__ import annotations

import re
from typing import Any

from synapse_core.tools import ToolDefinition, ToolHandler, ToolSafetyConfig, ToolParameter


SCRAPE_DEF = ToolDefinition(
    name="web_scrape",
    description="Scrape a web page and extract its main text content, title, and metadata. Returns clean text without HTML tags.",
    category="web",
    parameters=[
        ToolParameter(
            name="url",
            type="string",
            description="The URL of the web page to scrape",
            required=True,
        ),
        ToolParameter(
            name="selector",
            type="string",
            description="Optional CSS selector to extract specific content (e.g. 'article', 'main', '.content')",
            required=False,
        ),
        ToolParameter(
            name="max_length",
            type="integer",
            description="Maximum characters to return (default: 5000)",
            required=False,
        ),
    ],
    safety=ToolSafetyConfig(
        requires_approval=False,
        timeout_ms=15000,
        rate_limit_max_calls=20,
        rate_limit_window_ms=60000,
    ),
)


async def web_scrape_handler(
    url: str,
    selector: str = "",
    max_length: int = 5000,
    **kwargs: Any,
) -> str:
    """Fetch a web page and extract its main text content."""
    import httpx

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=15.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
    except httpx.HTTPError as e:
        return f"Error fetching {url}: {e}"

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
    except ImportError:
        # Fallback to html.parser if lxml not available
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

    # Extract title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Remove unwanted elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    # Use selector if provided, otherwise find main content
    if selector:
        target = soup.select_one(selector)
        if not target:
            return f"Selector '{selector}' not found on page. Title: {title}"
        content = target.get_text(separator="\n", strip=True)
    else:
        # Try common content containers
        for sel in ["article", "main", '[role="main"]', ".post-content", ".article-content", ".entry-content", "#content", ".content"]:
            target = soup.select_one(sel)
            if target:
                content = target.get_text(separator="\n", strip=True)
                break
        else:
            content = soup.get_text(separator="\n", strip=True)

    # Clean up whitespace
    content = re.sub(r"\n{3,}", "\n\n", content)

    # Truncate if needed
    if len(content) > max_length:
        content = content[:max_length] + f"\n\n... (truncated, {len(content)} total characters)"

    # Extract meta description
    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_desc = meta_tag["content"].strip()

    parts = []
    if title:
        parts.append(f"# {title}")
    if meta_desc:
        parts.append(f"Description: {meta_desc}")
    parts.append(f"URL: {url}")
    parts.append("")
    parts.append(content)

    return "\n".join(parts)


def register_web_scrape(registry: Any) -> None:
    registry.register(SCRAPE_DEF, web_scrape_handler)
