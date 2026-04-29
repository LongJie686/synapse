"""Tool safety: validation, rate limiting, permission checks."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from pydantic import BaseModel


class RateLimitEntry(BaseModel):
    calls: list[float] = []
    max_calls: int
    window_ms: int


class RateLimiter:
    """Per-tool rate limiting with sliding window."""

    def __init__(self) -> None:
        self._limits: dict[str, RateLimitEntry] = {}

    def configure(self, tool_name: str, max_calls: int, window_ms: int) -> None:
        self._limits[tool_name] = RateLimitEntry(calls=[], max_calls=max_calls, window_ms=window_ms)

    def check(self, tool_name: str) -> bool:
        """Returns True if the call is allowed, False if rate limited."""
        if tool_name not in self._limits:
            return True

        limit = self._limits[tool_name]
        now = time.monotonic() * 1000  # Convert to ms
        window_start = now - limit.window_ms

        # Remove expired calls
        limit.calls = [t for t in limit.calls if t > window_start]

        if len(limit.calls) >= limit.max_calls:
            return False

        limit.calls.append(now)
        return True

    def get_remaining(self, tool_name: str) -> int | None:
        if tool_name not in self._limits:
            return None
        limit = self._limits[tool_name]
        return max(0, limit.max_calls - len(limit.calls))


class PermissionChecker:
    """Whitelist/blacklist based tool permission checking."""

    def __init__(
        self,
        allowed_tools: list[str] | None = None,
        blocked_tools: list[str] | None = None,
    ) -> None:
        self._allowed = set(allowed_tools) if allowed_tools else None
        self._blocked = set(blocked_tools) if blocked_tools else set()

    def is_allowed(self, tool_name: str) -> bool:
        if tool_name in self._blocked:
            return False
        if self._allowed is not None and tool_name not in self._allowed:
            return False
        return True

    def allow(self, tool_name: str) -> None:
        self._blocked.discard(tool_name)
        if self._allowed is not None:
            self._allowed.add(tool_name)

    def block(self, tool_name: str) -> None:
        self._blocked.add(tool_name)


class InputValidator:
    """Validates tool input against JSON Schema."""

    def validate(self, tool_name: str, input_data: dict[str, Any], schema: dict[str, Any]) -> list[str]:
        """
        Validate input against schema. Returns list of errors (empty = valid).
        """
        errors: list[str] = []
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required:
            if field not in input_data:
                errors.append(f"Missing required field: {field}")

        for key, value in input_data.items():
            if key not in properties:
                errors.append(f"Unknown field: {key}")
                continue

            prop = properties[key]
            expected_type = prop.get("type")
            if expected_type and not self._check_type(value, expected_type):
                errors.append(f"Field '{key}' expected type '{expected_type}', got '{type(value).__name__}'")

        return errors

    @staticmethod
    def _check_type(value: Any, expected: str) -> bool:
        match expected:
            case "string":
                return isinstance(value, str)
            case "number":
                return isinstance(value, (int, float))
            case "integer":
                return isinstance(value, int) and not isinstance(value, bool)
            case "boolean":
                return isinstance(value, bool)
            case "array":
                return isinstance(value, list)
            case "object":
                return isinstance(value, dict)
            case _:
                return True


class SafeToolExecutor:
    """Wraps tool execution with safety checks."""

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        permission_checker: PermissionChecker | None = None,
        input_validator: InputValidator | None = None,
    ) -> None:
        self.rate_limiter = rate_limiter or RateLimiter()
        self.permission_checker = permission_checker or PermissionChecker()
        self.input_validator = input_validator or InputValidator()

    def check_all(self, tool_name: str, input_data: dict[str, Any], schema: dict[str, Any] | None = None) -> str | None:
        """
        Run all safety checks. Returns error message if blocked, None if allowed.
        """
        if not self.permission_checker.is_allowed(tool_name):
            return f"Tool '{tool_name}' is not permitted"

        if not self.rate_limiter.check(tool_name):
            return f"Rate limit exceeded for tool '{tool_name}'"

        if schema:
            errors = self.input_validator.validate(tool_name, input_data, schema)
            if errors:
                return f"Input validation failed: {'; '.join(errors)}"

        return None
