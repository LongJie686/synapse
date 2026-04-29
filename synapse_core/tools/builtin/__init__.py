"""Built-in tools."""

from synapse_core.tools.builtin.calculator import register_calculator
from synapse_core.tools.builtin.web_search import register_web_search
from synapse_core.tools.builtin.http_request import register_http_request
from synapse_core.tools.builtin.sql_query import register_sql_query


def register_all_builtin_tools(registry: "ToolRegistry") -> None:
    """Register all built-in tools with a ToolRegistry."""
    register_calculator(registry)
    register_web_search(registry)
    register_http_request(registry)
    register_sql_query(registry)


__all__ = [
    "register_calculator",
    "register_web_search",
    "register_http_request",
    "register_sql_query",
    "register_all_builtin_tools",
]
