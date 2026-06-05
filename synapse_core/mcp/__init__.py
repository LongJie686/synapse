"""MCP Client - connect to external MCP servers and use their tools."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from synapse_core.tools import ToolDefinition, ToolHandler, ToolSafetyConfig, ToolParameter


class MCPServerConfig:
    """Configuration for a single MCP server connection."""

    def __init__(
        self,
        name: str,
        command: str = "",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str = "",
    ) -> None:
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.url = url

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> MCPServerConfig:
        return cls(
            name=name,
            command=data.get("command", ""),
            args=data.get("args", []),
            env=data.get("env", {}),
            url=data.get("url", ""),
        )


class MCPToolWrapper:
    """Wraps an MCP server tool as a Synapse ToolHandler."""

    def __init__(self, server_name: str, tool_name: str, session: Any) -> None:
        self.server_name = server_name
        self.tool_name = tool_name
        self.session = session

    async def __call__(self, **kwargs: Any) -> str:
        """Execute the MCP tool."""
        try:
            result = await self.session.call_tool(self.tool_name, arguments=kwargs)
            if result.content:
                texts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        texts.append(item.text)
                return "\n".join(texts) if texts else str(result)
            return str(result)
        except Exception as e:
            return f"MCP tool error ({self.server_name}/{self.tool_name}): {e}"


class MCPClient:
    """Client that connects to MCP servers and registers their tools."""

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerConfig] = {}
        self._sessions: dict[str, Any] = {}

    def load_config(self, config_path: str | Path) -> None:
        """Load MCP server configurations from a JSON file."""
        path = Path(config_path)
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        servers = data.get("mcpServers", data.get("servers", {}))
        for name, config in servers.items():
            self._servers[name] = MCPServerConfig.from_dict(name, config)

    def add_server(self, config: MCPServerConfig) -> None:
        self._servers[config.name] = config

    async def connect_all(self, tool_registry: Any) -> int:
        """Connect to all configured servers and register their tools.

        Returns the number of tools registered.
        """
        total_tools = 0
        for name, config in self._servers.items():
            try:
                tools = await self._connect_server(name, config, tool_registry)
                total_tools += tools
            except Exception as e:
                print(f"MCP: Failed to connect to '{name}': {e}")
        return total_tools

    async def _connect_server(
        self,
        name: str,
        config: MCPServerConfig,
        tool_registry: Any,
    ) -> int:
        """Connect to a single MCP server and register its tools."""
        from mcp.client.stdio import stdio_client
        from mcp import ClientSession, StdioServerParameters

        # Merge user-supplied env on top of the current process environment.
        # config.env has already been validated by MCPServerCreateRequest to exclude
        # dangerous keys (LD_PRELOAD, PATH, PYTHONPATH, etc.).
        merged_env = {**os.environ, **config.env} if config.env else None

        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=merged_env,
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # List available tools
                result = await session.list_tools()

                for tool in result.tools:
                    # Convert MCP tool params to Synapse params
                    params = []
                    if tool.inputSchema:
                        properties = tool.inputSchema.get("properties", {})
                        required = tool.inputSchema.get("required", [])
                        for prop_name, prop_schema in properties.items():
                            params.append(ToolParameter(
                                name=prop_name,
                                type=prop_schema.get("type", "string"),
                                description=prop_schema.get("description", ""),
                                required=prop_name in required,
                            ))

                    tool_name = f"mcp_{name}_{tool.name}"
                    definition = ToolDefinition(
                        name=tool_name,
                        description=f"[MCP:{name}] {tool.description or tool.name}",
                        category="mcp",
                        parameters=params,
                        safety=ToolSafetyConfig(
                            requires_approval=False,
                            timeout_ms=30000,
                        ),
                    )

                    wrapper = MCPToolWrapper(name, tool.name, session)
                    tool_registry.register(definition, wrapper)

                return len(result.tools)

    async def shutdown(self) -> None:
        """Clean up all MCP connections."""
        self._sessions.clear()


def load_mcp_config() -> dict[str, Any]:
    """Load MCP server config from default locations."""
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "config" / "mcp_servers.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
