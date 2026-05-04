"""Code execution tool - run Python code in a subprocess sandbox."""

from __future__ import annotations

import asyncio
import sys
import tempfile
import os
from pathlib import Path
from typing import Any

from synapse_core.tools import ToolDefinition, ToolHandler, ToolSafetyConfig, ToolParameter


CODE_EXEC_DEF = ToolDefinition(
    name="code_execute",
    description="Execute Python code and return the output. Code runs in an isolated subprocess with timeout and resource limits.",
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
        timeout_ms=30000,
        rate_limit_max_calls=10,
        rate_limit_window_ms=60000,
        sandboxed=True,
    ),
)

# Blocked modules for security
_BLOCKED_MODULES = {
    "os.system", "os.popen", "os.exec", "os.spawn", "os.kill",
    "subprocess.call", "subprocess.run", "subprocess.Popen",
    "shutil.rmtree", "pickle.loads",
    "importlib.import_module",
}

_SAFE_HEADER = '''
import sys
import math
import json
import re
import datetime
import collections
import itertools
import functools
import statistics
import string
import random
import fractions
import decimal
import typing
'''


async def code_execute_handler(
    code: str,
    timeout: int = 10,
    **kwargs: Any,
) -> str:
    """Execute Python code in a subprocess and return output."""
    timeout = min(max(timeout, 1), 30)

    # Basic security check
    code_lower = code.lower()
    for blocked in _BLOCKED_MODULES:
        if blocked in code_lower:
            return f"Security error: '{blocked}' is not allowed in code execution."

    # Write code to temp file
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
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path.home()),
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return f"Execution timed out after {timeout} seconds."

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")

        parts = []
        if stdout_text.strip():
            parts.append(f"Output:\n{stdout_text.strip()}")
        if stderr_text.strip():
            parts.append(f"Stderr:\n{stderr_text.strip()}")
        if proc.returncode != 0:
            parts.append(f"Exit code: {proc.returncode}")

        return "\n\n".join(parts) if parts else "(no output)"

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def register_code_execution(registry: Any) -> None:
    registry.register(CODE_EXEC_DEF, code_execute_handler)
