"""MCP server management endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

_mcp_client = None


def configure(mcp_client) -> None:
    global _mcp_client
    _mcp_client = mcp_client


class MCPServerCreateRequest(BaseModel):
    name: str
    command: str = ""
    args: list[str] = []
    env: dict[str, str] = {}
    url: str = ""


@router.get("/mcp/servers")
async def list_servers() -> list[dict]:
    """List configured MCP servers and their status."""
    if not _mcp_client:
        return []
    servers = []
    for name, config in _mcp_client._servers.items():
        servers.append({
            "name": name,
            "command": config.command,
            "args": config.args,
            "url": config.url,
            "has_env": bool(config.env),
        })
    return servers


@router.post("/mcp/servers")
async def add_server(req: MCPServerCreateRequest) -> dict:
    """Add a new MCP server configuration."""
    from synapse_core.mcp import MCPServerConfig

    if not _mcp_client:
        raise HTTPException(status_code=500, detail="MCP client not available")

    config = MCPServerConfig(
        name=req.name,
        command=req.command,
        args=req.args,
        env=req.env,
        url=req.url,
    )
    _mcp_client.add_server(config)
    _save_mcp_config(_mcp_client)
    return {"name": req.name, "status": "added"}


@router.delete("/mcp/servers/{server_name}")
async def remove_server(server_name: str) -> dict:
    """Remove an MCP server configuration."""
    if not _mcp_client:
        raise HTTPException(status_code=500, detail="MCP client not available")
    if server_name not in _mcp_client._servers:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")
    del _mcp_client._servers[server_name]
    _save_mcp_config(_mcp_client)
    return {"name": server_name, "status": "removed"}


@router.post("/mcp/connect")
async def connect_servers() -> dict:
    """Connect to all configured MCP servers and register tools."""
    if not _mcp_client:
        raise HTTPException(status_code=500, detail="MCP client not available")

    from synapse_server.services import _get_tool_service
    tool_svc = _get_tool_service()
    if not tool_svc:
        raise HTTPException(status_code=500, detail="Tool service not available")

    try:
        count = await _mcp_client.connect_all(tool_svc.registry)
        return {"status": "connected", "tools_registered": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _save_mcp_config(client) -> None:
    """Persist MCP server config to disk."""
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "config" / "mcp_servers.json"
    config_path.parent.mkdir(exist_ok=True)

    servers = {}
    for name, cfg in client._servers.items():
        entry: dict = {}
        if cfg.command:
            entry["command"] = cfg.command
        if cfg.args:
            entry["args"] = cfg.args
        if cfg.env:
            entry["env"] = cfg.env
        if cfg.url:
            entry["url"] = cfg.url
        servers[name] = entry

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"mcpServers": servers}, f, indent=2, ensure_ascii=False)
