"""Hooks system - event-based extensibility for Synapse."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable


class HookEvent(str, Enum):
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    BEFORE_AGENT_RUN = "before_agent_run"
    AFTER_AGENT_RUN = "after_agent_run"
    ON_MEMORY_UPDATE = "on_memory_update"
    ON_ERROR = "on_error"


HookCallback = Callable[..., Awaitable[None]]


@dataclass
class HookContext:
    """Context passed to hook callbacks."""
    event: HookEvent
    agent_id: str = ""
    tool_name: str = ""
    session_id: str = ""
    input_data: Any = None
    output_data: Any = None
    error: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class HookManager:
    """Manages event hooks for the Synapse engine."""

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[HookCallback]] = {}

    def register(self, event: HookEvent, callback: HookCallback) -> None:
        """Register a callback for a specific event."""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

    def unregister(self, event: HookEvent, callback: HookCallback) -> None:
        """Remove a callback for a specific event."""
        if event in self._hooks:
            self._hooks[event] = [cb for cb in self._hooks[event] if cb is not callback]

    async def fire(self, event: HookEvent, **kwargs: Any) -> None:
        """Fire all callbacks for an event."""
        callbacks = self._hooks.get(event, [])
        context = HookContext(event=event, **kwargs)
        for callback in callbacks:
            try:
                await callback(context)
            except Exception as e:
                print(f"Hook error ({event.value}): {e}")

    def list_hooks(self) -> dict[str, int]:
        """Return count of registered hooks per event."""
        return {event.value: len(cbs) for event, cbs in self._hooks.items()}


# Global hook manager singleton
_hook_manager: HookManager | None = None


def get_hook_manager() -> HookManager:
    global _hook_manager
    if _hook_manager is None:
        _hook_manager = HookManager()
    return _hook_manager
