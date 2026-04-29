"""Built-in calculator tool."""

from __future__ import annotations

import math
from typing import Any

from synapse_core.tools import ToolDefinition, ToolParameter, ToolSafetyConfig


CALCULATOR_DEF = ToolDefinition(
    name="calculator",
    description="Evaluate a mathematical expression. Supports basic arithmetic, trigonometry, and math functions.",
    category="compute",
    parameters=[
        ToolParameter(
            name="expression",
            type="string",
            description="Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sin(pi/2)')",
        ),
    ],
    safety=ToolSafetyConfig(timeout_ms=5000),
)

_SAFE_NAMES = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
_SAFE_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "pow": pow})


async def calculator_handler(expression: str, **kwargs: Any) -> str:
    """Safely evaluate a math expression."""
    compiled = compile(expression, "<calculator>", "eval")
    for name in compiled.co_names:
        if name not in _SAFE_NAMES:
            raise ValueError(f"Unsupported function or variable: {name}")
    result = eval(compiled, {"__builtins__": {}}, _SAFE_NAMES)  # noqa: S307
    return str(result)


def register_calculator(registry: "ToolRegistry") -> None:
    """Register calculator tool with a ToolRegistry."""
    registry.register(CALCULATOR_DEF, calculator_handler)
