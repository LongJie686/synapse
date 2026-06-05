"""MCP server management endpoints."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

router = APIRouter()

_mcp_client = None

# Allowed MCP launcher executables — no absolute paths, no shell builtins
_ALLOWED_COMMANDS = {"uvx", "npx", "node", "python", "python3", "deno"}

_SAFE_SERVER_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Args: deny shell metacharacters and path traversal
_SAFE_ARG = re.compile(r'^[^;&|`$<>"\'\\\n\r/]+$')

# Env var names that can alter dynamic linker / interpreter search paths — deny override
_BLOCKED_ENV_KEYS = {
    "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "LD_DEBUG",
    "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
    "PATH", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE",
    "HOME", "USER", "LOGNAME", "SHELL",
    "NODE_OPTIONS", "NODE_PATH",
    "DENO_DIR",
}

_SAFE_ENV_KEY = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


def configure(mcp_client) -> None:
    global _mcp_client
    _mcp_client = mcp_client


class MCPServerCreateRequest(BaseModel):
    name: str
    command: str = ""
    args: list[str] = []
    env: dict[str, str] = {}
    url: str = ""

    @field_validator("env")
    @classmethod
    def validate_env(cls, v: dict) -> dict:
        for key in v:
            if not _SAFE_ENV_KEY.match(key):
                raise ValueError(f"env key '{key}' is not a valid environment variable name")
            if key in _BLOCKED_ENV_KEYS:
                raise ValueError(f"env key '{key}' is not allowed (reserved/dangerous variable)")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not _SAFE_SERVER_NAME.match(v):
            raise ValueError("name must be 1-64 alphanumeric/underscore/hyphen characters")
        return v

    @field_validator("command")
    @classmethod
    def validate_command(cls, v: str) -> str:
        if v and v not in _ALLOWED_COMMANDS:
            raise ValueError(
                f"command must be one of: {', '.join(sorted(_ALLOWED_COMMANDS))}"
            )
        return v

    @field_validator("args", mode="before")
    @classmethod
    def validate_args(cls, v: list) -> list:
        for arg in v:
            if not isinstance(arg, str):
                raise ValueError("each arg must be a string")
            if not arg:
                raise ValueError("args must not contain empty strings")
            if not _SAFE_ARG.match(arg):
                raise ValueError(f"arg contains disallowed characters: {arg!r}")
            # Extra check: deny path traversal patterns regardless of regex
            if ".." in arg or arg.startswith("/") or arg.startswith("\\"):
                raise ValueError(f"arg must not be an absolute path or contain '..': {arg!r}")
        return v

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if v and not re.match(r"^https?://", v):
            raise ValueError("url must start with http:// or https://")
        return v


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
    if not _SAFE_SERVER_NAME.match(server_name):
        raise HTTPException(status_code=400, detail="Invalid server name format")
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
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to connect to MCP servers")


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
