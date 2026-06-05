"""Code execution tool - run Python code in an isolated Docker container."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from synapse_core.tools import ToolDefinition, ToolHandler, ToolSafetyConfig, ToolParameter

logger = logging.getLogger(__name__)

CODE_EXEC_DEF = ToolDefinition(
    name="code_execute",
    description="Execute Python code and return the output. Code runs in an isolated Docker container with no network, read-only filesystem, and resource limits.",
    category="compute",
    parameters=[
        ToolParameter(
            name="code",
            type="string",
            description="Python code to execute",
            required=True,
        ),
        ToolParameter(
            name="timeout",
            type="integer",
            description="Execution timeout in seconds (default: 10, max: 30)",
            required=False,
        ),
    ],
    safety=ToolSafetyConfig(
        requires_approval=False,
        timeout_ms=35000,
        rate_limit_max_calls=10,
        rate_limit_window_ms=60000,
        sandboxed=True,
    ),
)

# Docker image used for execution — python:3.11-slim is small and has no extras
_DOCKER_IMAGE = os.environ.get("CODE_EXEC_DOCKER_IMAGE", "python:3.11-slim")

_SAFE_HEADER = """\
import sys, math, json, re, datetime, collections, itertools
import functools, statistics, string, random, fractions, decimal, typing
"""


async def _docker_available() -> bool:
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "info",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
        return proc.returncode == 0
    except (FileNotFoundError, asyncio.TimeoutError):
        return False


async def _run_in_docker(tmp_path: str, timeout: int) -> tuple[str, str, int]:
    """Execute code file inside an isolated Docker container."""
    cmd = [
        "docker", "run",
        "--rm",
        "--network", "none",
        "--read-only",
        "--no-new-privileges",
        "--memory", "128m",
        "--memory-swap", "128m",
        "--cpus", "0.5",
        "--user", "65534:65534",   # nobody:nogroup
        "--cap-drop", "ALL",
        "-v", f"{tmp_path}:/sandbox/code.py:ro",
        _DOCKER_IMAGE,
        "python", "/sandbox/code.py",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout + 5)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return "", "", -1
    return (
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
        proc.returncode or 0,
    )


async def _run_in_subprocess(tmp_path: str, timeout: int) -> tuple[str, str, int]:
    """Fallback execution in a subprocess (reduced isolation)."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, tmp_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(Path.home()),
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return "", "", -1
    return (
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
        proc.returncode or 0,
    )


def _format_result(stdout: str, stderr: str, returncode: int, timeout: int, timed_out: bool) -> str:
    if timed_out:
        return f"Execution timed out after {timeout} seconds."
    parts = []
    if stdout.strip():
        parts.append(f"Output:\n{stdout.strip()}")
    if stderr.strip():
        parts.append(f"Stderr:\n{stderr.strip()}")
    if returncode != 0:
        parts.append(f"Exit code: {returncode}")
    return "\n\n".join(parts) if parts else "(no output)"


async def code_execute_handler(
    code: str,
    timeout: int = 10,
    **kwargs: Any,
) -> str:
    """Execute Python code in an isolated container and return output."""
    timeout = min(max(timeout, 1), 30)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="synapse_exec_",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(_SAFE_HEADER)
        f.write("\n")
        f.write(code)
        f.write("\n")
        tmp_path = f.name

    try:
        if await _docker_available():
            stdout, stderr, rc = await _run_in_docker(tmp_path, timeout)
        else:
            logger.warning(
                "Docker not available — code_execute falling back to subprocess. "
                "This provides reduced isolation. Install Docker for full sandboxing."
            )
            stdout, stderr, rc = await _run_in_subprocess(tmp_path, timeout)

        timed_out = rc == -1
        return _format_result(stdout, stderr, rc, timeout, timed_out)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def register_code_execution(registry: Any) -> None:
    registry.register(CODE_EXEC_DEF, code_execute_handler)
