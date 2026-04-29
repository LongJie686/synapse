"""SQL query tool with safety checks."""

from __future__ import annotations

from typing import Any

from synapse_core.tools import ToolDefinition, ToolParameter, ToolSafetyConfig


SQL_QUERY_DEF = ToolDefinition(
    name="sql_query",
    description="Execute read-only SQL queries against a database. Only SELECT statements are allowed.",
    category="database",
    parameters=[
        ToolParameter(name="sql", type="string", description="SQL query to execute (SELECT only)"),
        ToolParameter(name="database", type="string", description="Database connection name or alias", required=False),
    ],
    safety=ToolSafetyConfig(
        requires_approval=True,
        timeout_ms=10000,
        sandboxed=True,
    ),
)

# Dangerous SQL keywords that should never appear in a read query
_BLOCKED_KEYWORDS = {"DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE"}


async def sql_query_handler(sql: str, database: str | None = None, **kwargs: Any) -> str:
    """Execute a read-only SQL query."""
    import json as json_mod

    # Safety: only allow SELECT
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return "Error: Only SELECT queries are allowed"

    for keyword in _BLOCKED_KEYWORDS:
        if keyword in sql_upper:
            return f"Error: Forbidden keyword '{keyword}' in query"

    connection_string = kwargs.get("connection_string")
    if not connection_string:
        return "Error: No database connection configured"

    try:
        import sqlalchemy
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine
    except ImportError:
        return "Error: sqlalchemy not installed"

    engine = create_async_engine(connection_string)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text(sql))
            rows = result.fetchall()
            columns = list(result.keys())

            data = [dict(zip(columns, row)) for row in rows[:100]]
            return json_mod.dumps({
                "columns": columns,
                "row_count": len(rows),
                "data": data,
            }, ensure_ascii=False, default=str, indent=2)
    except Exception as e:
        return f"Query error: {e}"
    finally:
        await engine.dispose()


def register_sql_query(registry: "ToolRegistry") -> None:
    registry.register(SQL_QUERY_DEF, sql_query_handler)
