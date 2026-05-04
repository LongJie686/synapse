"""Tool safety: validation, rate limiting, RBAC permission checks."""

from __future__ import annotations

import time
from collections import defaultdict
from enum import Enum
from typing import Any

from pydantic import BaseModel


class Permission(str, Enum):
    """Tool permission types."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


class Role(BaseModel):
    """RBAC role definition."""
    name: str
    permissions: dict[str, set[Permission]] = {}  # tool_name -> set of permissions

    def has_permission(self, tool_name: str, perm: Permission) -> bool:
        tool_perms = self.permissions.get(tool_name, set())
        if Permission.ADMIN in tool_perms:
            return True
        return perm in tool_perms


# Predefined roles
ROLE_USER = Role(
    name="user",
    permissions={
        "web_search": {Permission.READ, Permission.EXECUTE},
        "calculator": {Permission.EXECUTE},
    },
)

ROLE_ANALYST = Role(
    name="analyst",
    permissions={
        "web_search": {Permission.READ, Permission.EXECUTE},
        "web_scrape": {Permission.READ, Permission.EXECUTE},
        "calculator": {Permission.EXECUTE},
        "sql_query": {Permission.READ},
        "http_request": {Permission.READ},
    },
)

ROLE_ADMIN = Role(
    name="admin",
    permissions={
        "__all__": {Permission.ADMIN},
    },
)

PREDEFINED_ROLES: dict[str, Role] = {
    "user": ROLE_USER,
    "analyst": ROLE_ANALYST,
    "admin": ROLE_ADMIN,
}


class RBACChecker:
    """Role-based access control for tool invocation."""

    def __init__(self, roles: dict[str, Role] | None = None) -> None:
        # Deep copy to avoid mutating predefined roles
        if roles is not None:
            self._roles = roles
        else:
            self._roles = {
                name: Role(name=r.name, permissions={k: set(v) for k, v in r.permissions.items()})
                for name, r in PREDEFINED_ROLES.items()
            }
        self._audit_log: list[dict[str, Any]] = []

    def check(self, role_name: str, tool_name: str, perm: Permission = Permission.EXECUTE) -> bool:
        """Check if a role has permission to use a tool."""
        role = self._roles.get(role_name)
        if not role:
            self._log_audit(role_name, tool_name, perm, False, "Unknown role")
            return False

        allowed = role.has_permission(tool_name, perm)
        # Check global admin
        if not allowed:
            global_perms = role.permissions.get("__all__", set())
            if Permission.ADMIN in global_perms:
                allowed = True

        self._log_audit(role_name, tool_name, perm, allowed, "" if allowed else "Permission denied")
        return allowed

    def grant(self, role_name: str, tool_name: str, *perms: Permission) -> None:
        """Grant permissions to a role for a specific tool."""
        if role_name not in self._roles:
            self._roles[role_name] = Role(name=role_name, permissions={})
        tool_perms = self._roles[role_name].permissions.get(tool_name, set())
        tool_perms.update(perms)
        self._roles[role_name].permissions[tool_name] = tool_perms

    def revoke(self, role_name: str, tool_name: str, *perms: Permission) -> None:
        """Revoke permissions from a role for a specific tool."""
        role = self._roles.get(role_name)
        if not role:
            return
        tool_perms = role.permissions.get(tool_name, set())
        tool_perms -= set(perms)
        if not tool_perms:
            role.permissions.pop(tool_name, None)

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent audit log entries."""
        return self._audit_log[-limit:]

    def _log_audit(self, role: str, tool: str, perm: Permission, allowed: bool, reason: str) -> None:
        self._audit_log.append({
            "timestamp": time.time(),
            "role": role,
            "tool": tool,
            "permission": perm.value,
            "allowed": allowed,
            "reason": reason,
        })


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
        if expected == "string":
            return isinstance(value, str)
        elif expected == "number":
            return isinstance(value, (int, float))
        elif expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        elif expected == "boolean":
            return isinstance(value, bool)
        elif expected == "array":
            return isinstance(value, list)
        elif expected == "object":
            return isinstance(value, dict)
        else:
            return True


class SafeToolExecutor:
    """Wraps tool execution with safety checks."""

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        permission_checker: PermissionChecker | None = None,
        input_validator: InputValidator | None = None,
        rbac_checker: RBACChecker | None = None,
    ) -> None:
        self.rate_limiter = rate_limiter or RateLimiter()
        self.permission_checker = permission_checker or PermissionChecker()
        self.input_validator = input_validator or InputValidator()
        self.rbac_checker = rbac_checker or RBACChecker()

    def check_all(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        schema: dict[str, Any] | None = None,
        role: str | None = None,
    ) -> str | None:
        """
        Run all safety checks. Returns error message if blocked, None if allowed.
        """
        # RBAC check
        if role:
            if not self.rbac_checker.check(role, tool_name, Permission.EXECUTE):
                return f"Role '{role}' does not have permission to execute tool '{tool_name}'"

        if not self.permission_checker.is_allowed(tool_name):
            return f"Tool '{tool_name}' is not permitted"

        if not self.rate_limiter.check(tool_name):
            return f"Rate limit exceeded for tool '{tool_name}'"

        if schema:
            errors = self.input_validator.validate(tool_name, input_data, schema)
            if errors:
                return f"Input validation failed: {'; '.join(errors)}"

        return None
