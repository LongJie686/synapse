"""Streaming event types for real-time communication."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class StreamEvent(BaseModel):
    type: str
    data: dict[str, Any] = {}


# Event constructors
def run_start(run_id: str, agent_id: str) -> StreamEvent:
    return StreamEvent(type="run:start", data={"runId": run_id, "agentId": agent_id})


def agent_think(agent_id: str, thought: str) -> StreamEvent:
    return StreamEvent(type="agent:think", data={"agentId": agent_id, "thought": thought})


def agent_call_tool(agent_id: str, tool_name: str, tool_input: Any) -> StreamEvent:
    return StreamEvent(type="agent:call_tool", data={"agentId": agent_id, "toolName": tool_name, "input": tool_input})


def tool_result(tool_name: str, output: Any, duration_ms: float) -> StreamEvent:
    return StreamEvent(type="tool:result", data={"toolName": tool_name, "output": output, "durationMs": duration_ms})


def agent_message(agent_id: str, content: str, chunk: str | None = None) -> StreamEvent:
    return StreamEvent(type="agent:message", data={"agentId": agent_id, "content": content, "chunk": chunk})


def agent_delegate(from_agent: str, to_agent: str, task: str) -> StreamEvent:
    return StreamEvent(type="agent:delegate", data={"fromAgent": from_agent, "toAgent": to_agent, "task": task})


def memory_store(memory_type: str, key: str) -> StreamEvent:
    return StreamEvent(type="memory:store", data={"type": memory_type, "key": key})


def memory_retrieve(memory_type: str, query: str, count: int) -> StreamEvent:
    return StreamEvent(type="memory:retrieve", data={"type": memory_type, "query": query, "count": count})


def guardrail_block(filter_name: str, reason: str) -> StreamEvent:
    return StreamEvent(type="guardrail:block", data={"filter": filter_name, "reason": reason})


def run_end(run_id: str, token_usage: dict[str, int], duration_ms: float) -> StreamEvent:
    return StreamEvent(type="run:end", data={"runId": run_id, "tokenUsage": token_usage, "durationMs": duration_ms})


def run_error(run_id: str, error: str, recoverable: bool = True) -> StreamEvent:
    return StreamEvent(type="run:error", data={"runId": run_id, "error": error, "recoverable": recoverable})
