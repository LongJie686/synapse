"""Local file operations tool - read, write, list, search files.

Supports an authorized-directory model inspired by Claude Code:
- The workspace root is always authorized
- Users can authorize additional directories via CLI /allow <path>
- File write/delete require explicit approval regardless of directory
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from synapse_core.tools import ToolDefinition, ToolHandler, ToolSafetyConfig, ToolParameter


# ── Authorized directories ────────────────────────────────────────────────

_authorized_dirs: list[Path] = []


def init_file_ops() -> None:
    """Initialize with workspace root as the first authorized directory."""
    global _authorized_dirs
    root = Path(os.getenv("SYNAPSE_WORKSPACE", str(Path.cwd()))).resolve()
    if not _authorized_dirs:
        _authorized_dirs = [root]


def get_authorized_dirs() -> list[Path]:
    """Return a copy of the authorized directories list."""
    if not _authorized_dirs:
        init_file_ops()
    return list(_authorized_dirs)


def authorize_dir(path: str) -> str:
    """Add a directory to the authorized list. Returns status message."""
    target = Path(path).resolve()
    if not target.exists():
        return f"Error: path does not exist: {target}"
    if not target.is_dir():
        return f"Error: not a directory: {target}"
    if target in _authorized_dirs:
        return f"Already authorized: {target}"
    # Check if already covered by an existing authorized dir
    for d in _authorized_dirs:
        try:
            target.relative_to(d)
            return f"Already covered by: {d}"
        except ValueError:
            pass
    _authorized_dirs.append(target)
    return f"Authorized: {target}"


def deny_dir(path: str) -> str:
    """Remove a directory from the authorized list (cannot remove workspace root)."""
    target = Path(path).resolve()
    if not _authorized_dirs:
        return "No authorized directories"
    # Protect workspace root
    root = Path(os.getenv("SYNAPSE_WORKSPACE", str(Path.cwd()))).resolve()
    if target == root:
        return f"Cannot revoke workspace root: {target}"
    if target in _authorized_dirs:
        _authorized_dirs.remove(target)
        return f"Revoked: {target}"
    # Check if it was a subpath
    for d in _authorized_dirs:
        if d == target:
            _authorized_dirs.remove(d)
            return f"Revoked: {d}"
    return f"Not in authorized list: {target}"


def _is_authorized(target: Path) -> bool:
    """Check if a resolved path falls under any authorized directory."""
    if not _authorized_dirs:
        init_file_ops()
    for d in _authorized_dirs:
        try:
            target.relative_to(d)
            return True
        except ValueError:
            pass
    return False


def _resolve(path: str) -> Path:
    """Resolve a path (absolute or relative to workspace root).

    Returns the resolved Path if authorized, raises ValueError otherwise.
    """
    root = Path(os.getenv("SYNAPSE_WORKSPACE", str(Path.cwd()))).resolve()
    p = Path(path)

    # Absolute path: use as-is
    if p.is_absolute():
        target = p.resolve()
    else:
        target = (root / path).resolve()

    if not _is_authorized(target):
        authorized = ", ".join(str(d) for d in _authorized_dirs)
        raise ValueError(
            f"Access denied: '{target}' is outside authorized directories.\n"
            f"Authorized: {authorized}\n"
            f"Use /allow <path> to authorize a new directory."
        )
    return target


# ── file_read ─────────────────────────────────────────────────────────────

FILE_READ_DEF = ToolDefinition(
    name="file_read",
    description="Read the contents of a file. Supports absolute paths or relative to workspace.",
    category="file",
    parameters=[
        ToolParameter(name="path", type="string", description="File path (absolute or relative to workspace)", required=True),
        ToolParameter(name="encoding", type="string", description="File encoding (default: utf-8)", required=False),
        ToolParameter(name="lines", type="integer", description="Max number of lines to read (0 = all)", required=False),
        ToolParameter(name="offset", type="integer", description="Line number to start reading from (1-based, default: 1)", required=False),
    ],
    safety=ToolSafetyConfig(requires_approval=False, timeout_ms=10000, rate_limit_max_calls=30, rate_limit_window_ms=60000),
)


async def file_read_handler(path: str, encoding: str = "utf-8", lines: int = 0, offset: int = 0, **kwargs: Any) -> str:
    try:
        target = _resolve(path)
        if not target.exists():
            return f"Error: file not found: {path}"
        if not target.is_file():
            return f"Error: not a file: {path}"
        if target.stat().st_size > 5 * 1024 * 1024:
            return f"Error: file too large ({target.stat().st_size} bytes, max 5MB)"

        text = target.read_text(encoding=encoding)
        all_lines = text.splitlines()

        start = max(0, offset - 1) if offset > 0 else 0
        selected = all_lines[start:]
        if lines > 0:
            selected = selected[:lines]

        # Add line numbers
        numbered = []
        for i, line in enumerate(selected, start=start + 1):
            numbered.append(f"{i:6d}\t{line}")

        result = "\n".join(numbered)
        if start + len(selected) < len(all_lines):
            result += f"\n... ({len(all_lines) - start - len(selected)} more lines)"
        return result
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error reading file: {e}"


# ── file_write ────────────────────────────────────────────────────────────

FILE_WRITE_DEF = ToolDefinition(
    name="file_write",
    description="Write content to a file. Creates parent directories if needed.",
    category="file",
    parameters=[
        ToolParameter(name="path", type="string", description="File path (absolute or relative to workspace)", required=True),
        ToolParameter(name="content", type="string", description="Content to write", required=True),
        ToolParameter(name="append", type="boolean", description="Append to file instead of overwriting (default: false)", required=False),
    ],
    safety=ToolSafetyConfig(requires_approval=True, timeout_ms=10000, rate_limit_max_calls=20, rate_limit_window_ms=60000),
)


async def file_write_handler(path: str, content: str, append: bool = False, **kwargs: Any) -> str:
    try:
        target = _resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        target.write_text(content, encoding="utf-8")
        action = "Appended to" if append else "Wrote"
        return f"{action} {target} ({len(content)} chars)"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error writing file: {e}"


# ── file_list ─────────────────────────────────────────────────────────────

FILE_LIST_DEF = ToolDefinition(
    name="file_list",
    description="List files and directories.",
    category="file",
    parameters=[
        ToolParameter(name="path", type="string", description="Directory path (default: workspace root)", required=False),
        ToolParameter(name="pattern", type="string", description="Glob pattern to filter (e.g. '*.py')", required=False),
        ToolParameter(name="recursive", type="boolean", description="List recursively (default: false)", required=False),
    ],
    safety=ToolSafetyConfig(requires_approval=False, timeout_ms=10000, rate_limit_max_calls=30, rate_limit_window_ms=60000),
)


async def file_list_handler(path: str = ".", pattern: str = "", recursive: bool = False, **kwargs: Any) -> str:
    try:
        target = _resolve(path)
        if not target.exists():
            return f"Error: directory not found: {path}"
        if not target.is_dir():
            return f"Error: not a directory: {path}"

        if pattern:
            matches = target.rglob(pattern) if recursive else target.glob(pattern)
        else:
            matches = target.rglob("*") if recursive else target.iterdir()

        entries = []
        for p in sorted(matches):
            try:
                rel = str(p.relative_to(target))
            except ValueError:
                rel = str(p)
            if p.is_dir():
                entries.append(f"[DIR]  {rel}/")
            else:
                size = p.stat().st_size
                entries.append(f"[FILE] {rel}  ({size} bytes)")

            if len(entries) >= 200:
                entries.append("... (truncated at 200 entries)")
                break

        if not entries:
            return "(empty directory)"
        return "\n".join(entries)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error listing directory: {e}"


# ── file_search ───────────────────────────────────────────────────────────

FILE_SEARCH_DEF = ToolDefinition(
    name="file_search",
    description="Search for text content within files. Returns matching lines with context.",
    category="file",
    parameters=[
        ToolParameter(name="query", type="string", description="Text or regex pattern to search for", required=True),
        ToolParameter(name="path", type="string", description="Directory to search in (default: workspace root)", required=False),
        ToolParameter(name="file_pattern", type="string", description="File glob pattern (e.g. '*.py', default: '*')", required=False),
    ],
    safety=ToolSafetyConfig(requires_approval=False, timeout_ms=15000, rate_limit_max_calls=15, rate_limit_window_ms=60000),
)


async def file_search_handler(query: str, path: str = ".", file_pattern: str = "*", **kwargs: Any) -> str:
    import re
    try:
        target = _resolve(path)
        if not target.exists():
            return f"Error: directory not found: {path}"

        try:
            regex = re.compile(query, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(query), re.IGNORECASE)

        results = []

        for fp in target.rglob(file_pattern):
            if not fp.is_file():
                continue
            if fp.stat().st_size > 2 * 1024 * 1024:
                continue
            suffix = fp.suffix.lower()
            if suffix in (".png", ".jpg", ".jpeg", ".gif", ".zip", ".tar", ".gz", ".exe", ".dll", ".pyc", ".db", ".sqlite"):
                continue

            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            try:
                rel = str(fp.relative_to(target))
            except ValueError:
                rel = str(fp)

            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    results.append(f"{rel}:{i}: {line.strip()[:200]}")
                    if len(results) >= 50:
                        results.append("... (truncated at 50 matches)")
                        return "\n".join(results)

        if not results:
            return "No matches found."
        return "\n".join(results)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error searching files: {e}"


# ── Registration ──────────────────────────────────────────────────────────

def register_file_ops(registry: Any) -> None:
    init_file_ops()
    registry.register(FILE_READ_DEF, file_read_handler)
    registry.register(FILE_WRITE_DEF, file_write_handler)
    registry.register(FILE_LIST_DEF, file_list_handler)
    registry.register(FILE_SEARCH_DEF, file_search_handler)
