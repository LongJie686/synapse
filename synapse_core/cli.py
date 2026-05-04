"""Synapse CLI - Claude Code style interactive interface.

Usage:
  synapse                  Start interactive chat
  synapse "message"        One-shot message (print & exit)
  synapse -a code-expert   Chat with specific agent
  synapse -w ./project     Set workspace for file tools
  synapse --resume         Resume last session
  synapse --last           Continue last conversation
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.text import Text

console = Console()

API_BASE = os.getenv("SYNAPSE_API_URL", "http://localhost:8000/api")

# ── Session state ────────────────────────────────────────────────────────

_state = {
    "agent": "general-assistant",
    "session_id": "",
    "workspace": str(Path.cwd()),
    "model": "",
    "history_turns": 0,
    "last_message": "",
    "last_response": "",
    "verbose": False,
    "multiline": False,
}

# Cost tracking
_cost_tracker = {
    "total_tokens": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_calls": 0,
    "total_duration_ms": 0,
    "estimated_cost_usd": 0.0,
}

# Pricing per 1M tokens (USD) - adjustable
_MODEL_PRICING = {
    "glm-5-turbo": {"prompt": 0.5, "completion": 1.0},
    "glm-4-plus": {"prompt": 1.0, "completion": 2.0},
    "gpt-4o": {"prompt": 2.5, "completion": 10.0},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
    "claude-sonnet-4-6": {"prompt": 3.0, "completion": 15.0},
    "default": {"prompt": 1.0, "completion": 2.0},
}

# Conversation log for current session
_conversation: List[dict] = []

# ── API helpers ──────────────────────────────────────────────────────────

def _api() -> httpx.Client:
    return httpx.Client(timeout=10, base_url=API_BASE)


def _safe_print(text: str, style: str = "") -> None:
    try:
        if style:
            console.print(f"[{style}]{text}[/]")
        else:
            console.print(text)
    except UnicodeEncodeError:
        print(text.encode("utf-8", errors="replace").decode("utf-8"))


# ── Slash commands ───────────────────────────────────────────────────────

SLASH_COMMANDS: Dict[str, str] = {
    # Session
    "/new":        "Start a new conversation",
    "/resume":     "Resume a previous session (interactive picker)",
    "/sessions":   "List all sessions",
    "/session":    "Switch to session: /session <id>",
    "/history":    "Show current session message history",
    "/last":       "Show last exchange",
    # Agent
    "/agents":     "List all agents",
    "/agent":      "Switch agent: /agent <id>",
    "/skills":     "List custom skills",
    "/tools":      "List available tools for current agent",
    # Input
    "/multiline":  "Toggle multiline input mode (Ctrl+D to finish)",
    "/image":      "Attach an image: /image <path>",
    "/file":       "Attach a file: /file <path>",
    "/retry":      "Retry last message",
    "/edit":       "Edit last message and resend",
    # Config
    "/model":      "Show or set model: /model glm-5-turbo",
    "/workspace":  "Show or set workspace: /workspace ./dir",
    "/config":     "Show current configuration",
    "/verbose":    "Toggle verbose mode (show full tool outputs)",
    "/tasks":      "List scheduled tasks",
    "/status":     "Check backend connection status",
    # Knowledge
    "/knowledge":  "Extract knowledge from current session to RAG",
    "/recall":     "Search knowledge base: /recall <query>",
    "/cost":       "Show token usage and estimated cost",
    "/compact":    "Summarize and compress conversation context",
    # Output
    "/save":       "Save conversation to markdown file",
    "/copy":       "Copy last response to clipboard",
    "/clear":      "Clear the terminal screen",
    "/quit":       "Exit Synapse (Ctrl+C, Ctrl+D)",
    "/help":       "Show this help",
    # Git
    "/commit":     "Git commit current changes",
    "/pr":         "Create a pull request",
    "/git":        "Show git status and diff summary",
    # Plan
    "/plan":       "Enter plan mode: plan before execution",
    # Background
    "/bg":         "Run a task in background: /bg <message>",
    "/bglist":     "List background tasks",
    "/bgview":     "View background task result: /bgview <id>",
    # File access
    "/allow":      "Authorize a directory for file tools: /allow <path>",
    "/allow-list": "List all authorized directories",
    "/deny":       "Revoke directory access: /deny <path>",
}

# ── Session commands ─────────────────────────────────────────────────────

def _cmd_new() -> None:
    global _conversation
    _state["session_id"] = ""
    _conversation = []
    _state["history_turns"] = 0
    console.print("[green]New conversation started[/]")


def _cmd_resume() -> None:
    """Interactive session picker."""
    try:
        with _api() as c:
            resp = c.get("/sessions")
            if resp.status_code != 200:
                console.print(f"[red]Error {resp.status_code}[/]")
                return
            data = resp.json()
            if not data:
                console.print("[dim]No sessions found[/]")
                return

            console.print("\n[bold]Recent sessions:[/]")
            for i, s in enumerate(data[:15], 1):
                sid = s["session_id"]
                title = s.get("title", "Untitled")
                count = s.get("message_count", 0)
                ago = _time_ago(s.get("updated_at", 0))
                marker = " <-- current" if sid == _state["session_id"] else ""
                console.print(f"  [cyan]{i:>2}[/] {sid[:8]}  {title[:40]}  ({count} msgs, {ago}){marker}")

            console.print()
            try:
                choice = IntPrompt.ask("Select session number (0 to cancel)", default=0)
            except (EOFError, KeyboardInterrupt):
                return

            if 0 < choice <= len(data):
                selected = data[choice - 1]
                _state["session_id"] = selected["session_id"]
                _load_session_messages(selected["session_id"])
                title = selected.get("title", "Untitled")
                console.print(f"[green]Resumed: {title}[/]")
            else:
                console.print("[dim]Cancelled[/]")
    except httpx.ConnectError:
        console.print("[red]Backend not reachable[/]")


def _load_session_messages(session_id: str) -> None:
    """Load messages from a session into the conversation log."""
    global _conversation
    try:
        with _api() as c:
            resp = c.get(f"/sessions/{session_id}")
            if resp.status_code == 200:
                data = resp.json()
                msgs = data.get("messages", [])
                _conversation = [
                    {"role": m["role"], "content": m["content"]}
                    for m in msgs
                ]
    except Exception:
        _conversation = []


def _cmd_sessions() -> None:
    try:
        with _api() as c:
            resp = c.get("/sessions")
            if resp.status_code != 200:
                console.print(f"[red]Error {resp.status_code}[/]")
                return
            data = resp.json()
            if not data:
                console.print("[dim]No sessions[/]")
                return
            table = Table(border_style="dim", show_header=True)
            table.add_column("#", width=3)
            table.add_column("ID", style="cyan", width=10)
            table.add_column("Title")
            table.add_column("Msgs", width=5)
            table.add_column("Updated", width=12)
            for i, s in enumerate(data[:20], 1):
                sid = s["session_id"]
                title = s.get("title", "Untitled")[:35]
                count = s.get("message_count", 0)
                ago = _time_ago(s.get("updated_at", 0))
                marker = " *" if sid == _state["session_id"] else ""
                table.add_row(str(i), sid[:8], title + marker, str(count), ago)
            console.print(table)
            console.print("[dim]* = current session[/]")
    except httpx.ConnectError:
        console.print("[red]Backend not reachable[/]")


def _cmd_history() -> None:
    if not _conversation:
        console.print("[dim]No messages in current session[/]")
        return
    for msg in _conversation:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            console.print(f"[bold green]>[/] {content[:200]}")
        elif role == "assistant":
            console.print(f"[bold cyan]<[/] {content[:200]}")
        console.print()


def _cmd_last() -> None:
    if not _state["last_message"] and not _state["last_response"]:
        console.print("[dim]No previous exchange[/]")
        return
    if _state["last_message"]:
        console.print(f"[bold green]>[/] {_state['last_message']}")
    if _state["last_response"]:
        console.print(f"[bold cyan]<[/] {_state['last_response'][:300]}")


# ── Agent commands ───────────────────────────────────────────────────────

def _cmd_agents() -> None:
    try:
        with _api() as c:
            resp = c.get("/agents")
            if resp.status_code != 200:
                console.print(f"[red]Error {resp.status_code}[/]")
                return
            table = Table(border_style="dim", show_header=True)
            table.add_column("ID", style="cyan")
            table.add_column("Name")
            table.add_column("Role", style="dim")
            table.add_column("Tools", style="dim", max_width=45)
            for a in resp.json():
                tag = " (skill)" if a.get("is_skill") else ""
                active = " <--" if a["id"] == _state["agent"] else ""
                table.add_row(
                    a["id"], a["name"] + tag + active,
                    a["role"], ", ".join(a.get("tools", [])),
                )
            console.print(table)
    except httpx.ConnectError:
        console.print("[red]Backend not reachable[/]")


def _cmd_skills() -> None:
    try:
        with _api() as c:
            resp = c.get("/skills")
            if resp.status_code != 200:
                console.print(f"[red]Error {resp.status_code}[/]")
                return
            data = resp.json()
            if not data:
                console.print("[dim]No skills configured[/]")
                return
            for s in data:
                console.print(f"  [cyan]{s['name']}[/] ({s['display_name']})")
                if s.get("description"):
                    console.print(f"    {s['description']}")
                console.print(f"    [dim]Tools: {', '.join(s.get('tools', []))} | T={s.get('temperature', 0.7)}[/]")
    except httpx.ConnectError:
        console.print("[red]Backend not reachable[/]")


def _cmd_tools() -> None:
    try:
        with _api() as c:
            resp = c.get("/agents/" + _state["agent"])
            if resp.status_code != 200:
                console.print(f"[red]Agent not found[/]")
                return
            tools = resp.json().get("tools", [])
            if not tools:
                console.print("[dim]No tools[/]")
                return
            console.print(f"[dim]Tools for [cyan]{_state['agent']}[/]:[/]")
            for t in tools:
                console.print(f"  [cyan]{t}[/]")
    except httpx.ConnectError:
        console.print("[red]Backend not reachable[/]")


# ── Input commands ───────────────────────────────────────────────────────

def _cmd_multiline() -> None:
    _state["multiline"] = not _state["multiline"]
    status = "[green]ON[/]" if _state["multiline"] else "[dim]OFF[/]"
    console.print(f"Multiline mode: {status}  [dim](end input with empty line)[/]")


def _read_multiline() -> str:
    """Read multiple lines until empty line."""
    lines = []
    console.print("[dim]Multiline input (empty line to finish, Ctrl+C to cancel):[/]")
    while True:
        try:
            line = input("... ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            return "\n".join(lines) if lines else ""
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


def _cmd_image(path: str) -> str | None:
    """Attach an image and return base64."""
    if not path:
        console.print("[dim]Usage: /image <path-to-image>[/]")
        return None
    p = Path(path).expanduser()
    if not p.exists():
        console.print(f"[red]File not found: {path}[/]")
        return None
    import base64
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
    mime = mime_map.get(p.suffix.lower(), "image/png")
    data = base64.b64encode(p.read_bytes()).decode("utf-8")
    console.print(f"[green]Attached: {p.name} ({len(data)} chars)[/]")
    return data


def _cmd_retry() -> Tuple[str, bool]:
    """Retry last message. Returns (message, should_send)."""
    if not _state["last_message"]:
        console.print("[dim]No previous message to retry[/]")
        return "", False
    console.print(f"[dim]Retrying: {_state['last_message'][:80]}...[/]")
    return _state["last_message"], True


def _cmd_edit() -> Tuple[str, bool]:
    """Edit last message."""
    if not _state["last_message"]:
        console.print("[dim]No previous message to edit[/]")
        return "", False
    console.print(f"[dim]Previous: {_state['last_message']}[/]")
    try:
        new_msg = Prompt.ask("[bold green]Edit>", default=_state["last_message"])
    except (EOFError, KeyboardInterrupt):
        return "", False
    return new_msg, True


# ── Config commands ──────────────────────────────────────────────────────

def _cmd_config() -> None:
    table = Table(show_header=False, border_style="dim")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_row("Agent", _state["agent"])
    table.add_row("Session", _state["session_id"][:8] + "..." if _state["session_id"] else "(new)")
    table.add_row("Workspace", _state["workspace"])
    table.add_row("Model", _state["model"] or "(from .env)")
    table.add_row("API", API_BASE)
    table.add_row("Verbose", str(_state["verbose"]))
    table.add_row("Multiline", str(_state["multiline"]))
    table.add_row("Turns", str(_state["history_turns"]))
    console.print(table)


def _cmd_verbose() -> None:
    _state["verbose"] = not _state["verbose"]
    status = "[green]ON[/]" if _state["verbose"] else "[dim]OFF[/]"
    console.print(f"Verbose: {status}")


def _cmd_tasks() -> None:
    try:
        with _api() as c:
            resp = c.get("/scheduler/tasks")
            if resp.status_code != 200:
                console.print(f"[red]Error {resp.status_code}[/]")
                return
            data = resp.json()
            if not data:
                console.print("[dim]No scheduled tasks[/]")
                return
            for t in data:
                status = "[green]ON[/]" if t["enabled"] else "[red]OFF[/]"
                cron = t["cron"]
                name = t["name"]
                agent = t.get("agent_id", "?")
                count = t["run_count"]
                console.print(f"  {status} [cyan]{name}[/] [{cron}] agent={agent} runs={count}")
    except httpx.ConnectError:
        console.print("[red]Backend not reachable[/]")


def _cmd_status() -> None:
    try:
        # /health is on root, not under /api
        base = API_BASE.rstrip("/api")
        with httpx.Client(timeout=3) as c:
            resp = c.get(f"{base}/health")
            if resp.status_code == 200:
                data = resp.json()
                console.print(f"[green]Backend: {data.get('status', 'ok')} | v{data.get('version', '?')}[/]")
            else:
                console.print(f"[yellow]Backend responded with {resp.status_code}[/]")
    except httpx.ConnectError:
        console.print("[red]Backend not reachable[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


# ── Knowledge commands ────────────────────────────────────────────────────

def _cmd_knowledge() -> None:
    """Extract knowledge from current session conversation into RAG."""
    if not _conversation:
        console.print("[dim]No conversation to extract knowledge from[/]")
        return
    try:
        from synapse_core.knowledge import KnowledgeExtractor
        extractor = KnowledgeExtractor()
        entries = extractor.extract_from_messages(
            _conversation,
            session_id=_state["session_id"],
            agent_id=_state["agent"],
        )
        if not entries:
            console.print("[dim]No significant knowledge extracted from this conversation[/]")
            return
        # Store via API (memory endpoint)
        console.print(f"[green]Extracted {len(entries)} knowledge entries:[/]")
        table = Table(border_style="dim", show_header=True)
        table.add_column("Type", style="cyan", width=12)
        table.add_column("Importance", width=10)
        table.add_column("Content", max_width=60)
        for entry in entries[:20]:
            table.add_row(
                entry.memory_type.value,
                f"{entry.importance:.1f}",
                entry.content[:60] + "..." if len(entry.content) > 60 else entry.content,
            )
        console.print(table)
        console.print(f"[dim]Knowledge auto-saved to long-term memory for future sessions[/]")
    except ImportError:
        console.print("[yellow]Knowledge module not available[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


def _cmd_recall(query: str) -> None:
    """Search knowledge base for relevant memories."""
    if not query:
        console.print("[dim]Usage: /recall <search query>[/]")
        return
    try:
        with _api() as c:
            resp = c.get("/memory/search", params={"q": query, "limit": 10})
            if resp.status_code == 200:
                results = resp.json()
                if not results:
                    console.print("[dim]No matching memories found[/]")
                    return
                console.print(f"[green]Found {len(results)} memories:[/]")
                for m in results:
                    mem_type = m.get("memory_type", "?")
                    importance = m.get("importance", 0)
                    content = m.get("content", "")
                    console.print(f"  [{mem_type}] {content[:100]}")
                    console.print(f"  [dim]importance={importance:.1f}[/]")
                    console.print()
            elif resp.status_code == 404:
                # Fallback: search via sessions
                console.print("[dim]Memory search API not available, searching conversations...[/]")
                _cmd_recall_local(query)
            else:
                console.print(f"[red]Error {resp.status_code}[/]")
    except httpx.ConnectError:
        console.print("[red]Backend not reachable[/]")


def _cmd_recall_local(query: str) -> None:
    """Local fallback: search conversation history."""
    query_lower = query.lower()
    found = 0
    for msg in _conversation:
        if query_lower in msg["content"].lower():
            role = "user" if msg["role"] == "user" else "assistant"
            console.print(f"  [{role}] {msg['content'][:120]}")
            found += 1
            if found >= 10:
                break
    if found == 0:
        console.print("[dim]No matches in current conversation[/]")


# ── Cost tracking ─────────────────────────────────────────────────────────

def _cmd_cost() -> None:
    """Display token usage and estimated cost for this session."""
    t = _cost_tracker
    pricing = _MODEL_PRICING.get(_state["model"] or "default", _MODEL_PRICING["default"])
    model_label = _state["model"] or "(default)"

    table = Table(title="Session Cost", border_style="dim", show_header=True)
    table.add_column("Metric", style="cyan", width=22)
    table.add_column("Value", justify="right")
    table.add_row("API Calls", str(t["total_calls"]))
    table.add_row("Total Tokens", f"{t['total_tokens']:,}")
    table.add_row("Completion Tokens", f"{t['completion_tokens']:,}")
    table.add_row("Total Duration", f"{t['total_duration_ms'] / 1000:.1f}s")
    table.add_row("Model", model_label)
    table.add_row("Pricing ($/1M tokens)", f"${pricing['prompt']:.2f} / ${pricing['completion']:.2f}")
    table.add_row("[bold]Estimated Cost[/]", f"[bold green]${t['estimated_cost_usd']:.4f}[/]")
    console.print(table)


# ── Context compaction ────────────────────────────────────────────────────

def _cmd_compact() -> None:
    """Compact conversation: keep last few turns, summarize the rest."""
    global _conversation
    if len(_conversation) <= 4:
        console.print("[dim]Context is already small, nothing to compact[/]")
        return

    # Keep last 2 user+assistant pairs (4 messages)
    keep_count = 4
    old_msgs = _conversation[:-keep_count]
    recent_msgs = _conversation[-keep_count:]

    # Build a brief summary of older messages
    summary_parts = []
    for msg in old_msgs:
        role = "User" if msg["role"] == "user" else "Asst"
        summary_parts.append(f"{role}: {msg['content'][:100]}")
    summary = "[Context Summary]\n" + "\n".join(summary_parts)

    _conversation = [{"role": "system", "content": summary}] + recent_msgs
    _state["history_turns"] = len(recent_msgs) // 2

    console.print(f"[green]Compacted: {len(old_msgs)} old messages summarized[/]")
    console.print(f"[dim]Context: {len(_conversation)} messages (was {len(old_msgs) + len(recent_msgs)})[/]")


# ── Git workflow ──────────────────────────────────────────────────────────

def _run_git(*args: str, check: bool = True) -> str:
    """Run a git command and return stdout."""
    import subprocess
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True, encoding="utf-8",
        cwd=_state["workspace"],
    )
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _cmd_git() -> None:
    """Show git status and recent diff summary."""
    try:
        status = _run_git("status", "--short")
        if not status:
            console.print("[green]Working tree clean[/]")
            return
        lines = status.split("\n")
        console.print(f"[bold]Changed files ({len(lines)}):[/]")
        for line in lines[:20]:
            console.print(f"  {line}")
        if len(lines) > 20:
            console.print(f"  ... and {len(lines) - 20} more")

        # Brief diff stat
        diff_stat = _run_git("diff", "--stat", check=False)
        if diff_stat:
            console.print(f"\n[bold]Diff summary:[/]")
            console.print(f"  {diff_stat}")
    except FileNotFoundError:
        console.print("[red]git not found. Install git first.[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


def _cmd_commit() -> None:
    """Interactive git commit workflow."""
    try:
        status = _run_git("status", "--short")
        if not status:
            console.print("[green]Nothing to commit[/]")
            return

        # Show diff stat
        diff_stat = _run_git("diff", "--stat", check=False)
        if diff_stat:
            console.print(f"[bold]Changes:[/]\n  {diff_stat}\n")

        # Generate suggestion from diff
        diff = _run_git("diff", check=False)
        staged = _run_git("diff", "--cached", "--stat", check=False)
        if staged:
            console.print(f"[dim]Staged:\n  {staged}[/]\n")

        try:
            message = Prompt.ask("[bold green]Commit message[/]")
        except (EOFError, KeyboardInterrupt):
            console.print("[dim]Cancelled[/]")
            return

        if not message.strip():
            console.print("[dim]Empty message, cancelled[/]")
            return

        # Stage all changes
        _run_git("add", "-A")
        _run_git("commit", "-m", message)
        console.print(f"[green]Committed: {message}[/]")

        # Show log
        log = _run_git("log", "-1", "--oneline")
        if log:
            console.print(f"[dim]{log}[/]")
    except FileNotFoundError:
        console.print("[red]git not found[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


def _cmd_pr() -> None:
    """Create a pull request using gh CLI."""
    try:
        # Check gh is available
        import subprocess
        subprocess.run(["gh", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        console.print("[red]gh CLI not found. Install: https://cli.github.com[/]")
        return

    try:
        # Check current branch
        branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
        if branch == "main" or branch == "master":
            console.print("[yellow]You are on the default branch. Create a feature branch first:[/]")
            console.print("[dim]  git checkout -b feature/my-feature[/]")
            return

        # Check for uncommitted changes
        status = _run_git("status", "--short")
        if status:
            console.print(f"[yellow]Uncommitted changes ({len(status.split(chr(10)))} files). Commit first with /commit[/]")
            return

        # Get commits ahead of main
        base = "main"
        log = _run_git("log", f"{base}..HEAD", "--oneline", check=False)
        if not log:
            console.print("[dim]No commits ahead of main[/]")
            return

        console.print(f"[bold]Branch:[/] {branch}")
        console.print(f"[bold]Commits:[/]")
        for line in log.split("\n"):
            console.print(f"  {line}")

        try:
            title = Prompt.ask("[bold green]PR title[/]", default=log.split("\n")[0].split(" ", 1)[-1][:60])
        except (EOFError, KeyboardInterrupt):
            return

        try:
            body = Prompt.ask("[bold green]PR body[/] (optional)", default="")
        except (EOFError, KeyboardInterrupt):
            body = ""

        # Push branch if needed
        import subprocess
        tracking = _run_git("rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}", check=False)
        if not tracking:
            console.print("[dim]Pushing branch to remote...[/]")
            subprocess.run(
                ["git", "push", "-u", "origin", branch],
                cwd=_state["workspace"], capture_output=True, encoding="utf-8",
            )

        # Create PR
        cmd = ["gh", "pr", "create", "--title", title, "--base", base]
        if body:
            cmd.extend(["--body", body])
        result = subprocess.run(
            cmd, cwd=_state["workspace"],
            capture_output=True, text=True, encoding="utf-8",
        )
        if result.returncode == 0:
            console.print(f"[green]PR created: {result.stdout.strip()}[/]")
        else:
            console.print(f"[red]Error: {result.stderr.strip()}[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


# ── Plan mode ────────────────────────────────────────────────────────────

_plan_buffer: List[str] = []
_plan_active = False


def _cmd_plan(arg: str) -> Tuple[str, bool]:
    """Enter plan mode: collect thoughts, then execute."""
    global _plan_active, _plan_buffer

    if not arg:
        if _plan_active:
            # Show current plan
            console.print("[bold]Current plan:[/]")
            for i, step in enumerate(_plan_buffer, 1):
                console.print(f"  {i}. {step}")
            console.print("\n[dim]/plan <step> to add | /plan go to execute | /plan off to cancel[/]")
        else:
            console.print("[dim]Usage: /plan <description>  -- start planning[/]")
            console.print("[dim]       /plan go             -- execute the plan[/]")
            console.print("[dim]       /plan off            -- cancel plan mode[/]")
        return "", False

    if arg == "off":
        _plan_active = False
        _plan_buffer = []
        console.print("[dim]Plan mode cancelled[/]")
        return "", False

    if arg == "go":
        if not _plan_buffer:
            console.print("[dim]No plan steps. Use /plan <step> to add[/]")
            return "", False
        _plan_active = False
        # Build a planning prompt from the steps
        plan_text = "\n".join(f"{i}. {s}" for i, s in enumerate(_plan_buffer, 1))
        message = f"Please follow this plan step by step:\n\n{plan_text}"
        _plan_buffer = []
        console.print("[green]Executing plan...[/]")
        return message, True

    # Add step to plan
    _plan_active = True
    _plan_buffer.append(arg)
    console.print(f"[green]Plan step {len(_plan_buffer)}: {arg}[/]")
    console.print(f"[dim]{len(_plan_buffer)} steps total | /plan go to execute | /plan off to cancel[/]")
    return "", False


# ── Background tasks ─────────────────────────────────────────────────────

_bg_tasks: Dict[str, dict] = {}
_bg_counter = 0


def _cmd_bg(arg: str) -> None:
    """Run a task in background."""
    global _bg_counter
    if not arg:
        console.print("[dim]Usage: /bg <message>  -- run in background[/]")
        console.print("[dim]       /bglist       -- list background tasks[/]")
        console.print("[dim]       /bgview <id>  -- view task result[/]")
        return

    _bg_counter += 1
    task_id = f"bg-{_bg_counter}"
    _bg_tasks[task_id] = {"status": "running", "message": arg, "result": "", "started": time.time()}

    console.print(f"[green]Background task {task_id} started[/]")

    # Run in a separate thread
    import threading
    def _run():
        try:
            result = asyncio.run(_stream(arg))
            _bg_tasks[task_id]["status"] = "done"
            _bg_tasks[task_id]["result"] = result
            _bg_tasks[task_id]["finished"] = time.time()
        except Exception as e:
            _bg_tasks[task_id]["status"] = "error"
            _bg_tasks[task_id]["result"] = str(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def _cmd_bglist() -> None:
    """List background tasks."""
    if not _bg_tasks:
        console.print("[dim]No background tasks[/]")
        return
    for tid, info in _bg_tasks.items():
        status_style = "green" if info["status"] == "done" else "yellow" if info["status"] == "running" else "red"
        elapsed = ""
        if info["status"] == "done" and "finished" in info:
            elapsed = f" ({info['finished'] - info['started']:.1f}s)"
        console.print(
            f"  [{status_style}]{info['status']}[/] [cyan]{tid}[/] "
            f"{info['message'][:50]}{elapsed}"
        )


def _cmd_bgview(arg: str) -> None:
    """View background task result."""
    if not arg:
        console.print("[dim]Usage: /bgview <task-id>[/]")
        return
    task = _bg_tasks.get(arg)
    if not task:
        console.print(f"[dim]Task {arg} not found. Use /bglist[/]")
        return
    console.print(f"[bold]Task {arg}[/] ({task['status']})")
    console.print(f"[dim]Message: {task['message']}[/]")
    if task["result"]:
        try:
            console.print(Markdown(task["result"]))
        except Exception:
            console.print(task["result"])
    elif task["status"] == "running":
        console.print("[yellow]Still running...[/]")


# ── File access commands ──────────────────────────────────────────────────

def _cmd_allow(arg: str) -> None:
    """Authorize a directory for file tool access."""
    if not arg:
        console.print("[dim]Usage: /allow <path>  (e.g. /allow C:\\Users\\20597\\Desktop)[/]")
        return
    from synapse_core.tools.builtin.file_ops import authorize_dir
    result = authorize_dir(arg)
    if result.startswith("Error"):
        console.print(f"[red]{result}[/]")
    elif result.startswith("Already"):
        console.print(f"[yellow]{result}[/]")
    else:
        console.print(f"[green]{result}[/]")


def _cmd_allow_list() -> None:
    """List all authorized directories."""
    from synapse_core.tools.builtin.file_ops import get_authorized_dirs
    dirs = get_authorized_dirs()
    console.print("[bold]Authorized directories:[/]")
    root = Path(os.getenv("SYNAPSE_WORKSPACE", str(Path.cwd()))).resolve()
    for d in dirs:
        marker = " (workspace)" if d == root else ""
        console.print(f"  [green]{d}[/]{marker}")


def _cmd_deny(arg: str) -> None:
    """Revoke a directory from file tool access."""
    if not arg:
        console.print("[dim]Usage: /deny <path>[/]")
        return
    from synapse_core.tools.builtin.file_ops import deny_dir
    result = deny_dir(arg)
    if result.startswith("Error") or result.startswith("Cannot"):
        console.print(f"[red]{result}[/]")
    elif result.startswith("Not"):
        console.print(f"[yellow]{result}[/]")
    else:
        console.print(f"[green]{result}[/]")


# ── Output commands ──────────────────────────────────────────────────────

def _cmd_save() -> None:
    if not _conversation:
        console.print("[dim]No conversation to save[/]")
        return
    filename = f"synapse_{_state['session_id'][:8]}_{int(time.time())}.md"
    filepath = Path(_state["workspace"]) / filename
    lines = [f"# Synapse Conversation\n"]
    for msg in _conversation:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"## {role}\n\n{msg['content']}\n")
    filepath.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Saved to {filepath}[/]")


def _cmd_copy() -> None:
    if not _state["last_response"]:
        console.print("[dim]No response to copy[/]")
        return
    try:
        import subprocess
        if sys.platform == "win32":
            subprocess.run(["clip"], input=_state["last_response"].encode("utf-8"), check=True)
        elif sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=_state["last_response"].encode("utf-8"), check=True)
        else:
            subprocess.run(["xclip", "-selection", "clipboard"], input=_state["last_response"].encode("utf-8"), check=True)
        console.print("[green]Copied to clipboard[/]")
    except Exception as e:
        console.print(f"[yellow]Copy failed: {e}[/]")
        console.print("[dim]Tip: select the text above to copy manually[/]")


# ── Dispatch ─────────────────────────────────────────────────────────────

def _handle_slash(raw: str) -> Tuple[str, bool]:
    """Handle slash command. Returns (message_to_send, should_send)."""
    stripped = raw.strip()
    if not stripped.startswith("/"):
        return raw, True

    parts = stripped.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    # Help & exit
    if cmd in ("/help", "/h", "/?"):
        _print_help()
    elif cmd in ("/quit", "/exit", "/q"):
        console.print("[dim]Goodbye![/]")
        raise SystemExit(0)
    elif cmd == "/clear":
        console.clear()
        _print_banner()

    # Session
    elif cmd == "/new":
        _cmd_new()
    elif cmd in ("/resume", "/r"):
        _cmd_resume()
    elif cmd in ("/sessions", "/ss"):
        _cmd_sessions()
    elif cmd == "/session":
        if not arg:
            console.print(f"[dim]Current: {_state['session_id'][:8] or '(new)'}[/]")
            console.print("[dim]Usage: /session <id>  or  /resume for picker[/]")
        else:
            _state["session_id"] = arg
            _load_session_messages(arg)
            console.print(f"[green]Session -> {arg[:8]}...[/]")
    elif cmd in ("/history", "/hi"):
        _cmd_history()
    elif cmd in ("/last", "/l"):
        _cmd_last()

    # Agent
    elif cmd == "/agents":
        _cmd_agents()
    elif cmd == "/agent":
        if not arg:
            console.print(f"[dim]Current: {_state['agent']}[/]")
            console.print("[dim]Usage: /agent <agent-id>  (tab-complete with /agents)[/]")
        else:
            _state["agent"] = arg
            console.print(f"[green]Agent -> {arg}[/]")
    elif cmd == "/skills":
        _cmd_skills()
    elif cmd in ("/tools", "/t"):
        _cmd_tools()

    # Input
    elif cmd in ("/multiline", "/ml"):
        _cmd_multiline()
    elif cmd in ("/image", "/img"):
        _cmd_image(arg)
    elif cmd == "/retry":
        return _cmd_retry()
    elif cmd in ("/edit", "/e"):
        return _cmd_edit()

    # Config
    elif cmd == "/model":
        if not arg:
            console.print(f"[dim]Current: {_state['model'] or '(from .env)'}[/]")
        else:
            _state["model"] = arg
            console.print(f"[green]Model -> {arg}[/]")
    elif cmd in ("/workspace", "/ws"):
        if not arg:
            console.print(f"[dim]Current: {_state['workspace']}[/]")
        else:
            _state["workspace"] = str(Path(arg).resolve())
            os.environ["SYNAPSE_WORKSPACE"] = _state["workspace"]
            console.print(f"[green]Workspace -> {_state['workspace']}[/]")
    elif cmd == "/config":
        _cmd_config()
    elif cmd in ("/verbose", "/v"):
        _cmd_verbose()
    elif cmd == "/tasks":
        _cmd_tasks()
    elif cmd == "/status":
        _cmd_status()

    # Knowledge
    elif cmd in ("/knowledge", "/k"):
        _cmd_knowledge()
    elif cmd in ("/recall", "/rc"):
        _cmd_recall(arg)
    elif cmd == "/cost":
        _cmd_cost()
    elif cmd == "/compact":
        _cmd_compact()

    # Git
    elif cmd == "/git":
        _cmd_git()
    elif cmd == "/commit":
        _cmd_commit()
    elif cmd == "/pr":
        _cmd_pr()

    # Plan
    elif cmd == "/plan":
        return _cmd_plan(arg)

    # Background
    elif cmd == "/bg":
        _cmd_bg(arg)
    elif cmd in ("/bglist", "/bgl"):
        _cmd_bglist()
    elif cmd in ("/bgview", "/bgv"):
        _cmd_bgview(arg)

    # File access
    elif cmd == "/allow":
        _cmd_allow(arg)
    elif cmd in ("/allow-list", "/allowl", "/al"):
        _cmd_allow_list()
    elif cmd == "/deny":
        _cmd_deny(arg)

    # Output
    elif cmd == "/save":
        _cmd_save()
    elif cmd in ("/copy", "/cp"):
        _cmd_copy()

    else:
        # Fuzzy match
        close = [k for k in SLASH_COMMANDS if k.startswith(cmd[:3])]
        hint = f"  [dim]Did you mean: {', '.join(close[:3])}?[/]" if close else ""
        console.print(f"[yellow]Unknown: {cmd}[/]{hint}")
        console.print("[dim]Type /help for all commands[/]")

    return "", False


# ── Banner & help ────────────────────────────────────────────────────────

def _print_banner() -> None:
    console.print(Panel(
        "[bold cyan]Synapse[/] [dim]v0.1.0[/]  |  Multi-Agent CLI\n"
        "[dim]/help for commands | /resume to continue a chat | /quit to exit[/]",
        border_style="cyan",
    ))


def _print_help() -> None:
    sections = {
        "Session":   ["/new", "/resume", "/sessions", "/session", "/history", "/last"],
        "Agent":     ["/agents", "/agent", "/skills", "/tools"],
        "Input":     ["/multiline", "/image", "/retry", "/edit"],
        "Config":    ["/model", "/workspace", "/config", "/verbose", "/tasks", "/status"],
        "Knowledge": ["/knowledge", "/recall"],
        "Cost":      ["/cost", "/compact"],
        "Git":       ["/git", "/commit", "/pr"],
        "Planning":  ["/plan"],
        "Background": ["/bg", "/bglist", "/bgview"],
        "File Access": ["/allow", "/allow-list", "/deny"],
        "Output":    ["/save", "/copy"],
        "Other":     ["/clear", "/quit", "/help"],
    }
    for section, cmds in sections.items():
        console.print(f"\n[bold]{section}[/]")
        for c in cmds:
            desc = SLASH_COMMANDS.get(c, "")
            console.print(f"  [cyan]{c:14}[/] {desc}")
    console.print()


# ── Streaming ────────────────────────────────────────────────────────────

async def _stream(message: str, images: List[dict] = None) -> str:
    """Send a message and stream the response token by token."""
    body: dict = {"message": message, "agent_id": _state["agent"]}
    if _state["session_id"]:
        body["session_id"] = _state["session_id"]
    if images:
        body["images"] = images

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
            async with client.stream("POST", f"{API_BASE}/runs/stream", json=body) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    console.print(f"[red]Error {resp.status_code}: {text.decode()[:200]}[/]")
                    return ""

                full_content = ""
                displayed = 0
                tool_buf = ""
                usage = {}
                duration_ms = 0

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    evt_type = event.get("type", "")
                    data = event.get("data", {})

                    if evt_type == "agent:call_tool":
                        tool_name = data.get("toolName", "?")
                        tool_input = json.dumps(data.get("input", {}), ensure_ascii=False)
                        if len(tool_input) > 80:
                            tool_input = tool_input[:77] + "..."
                        tool_buf = f"  [dim]|[cyan]{tool_name}|({tool_input})[/]"

                    elif evt_type == "tool:result":
                        ms = data.get("durationMs", 0) or data.get("executionTimeMs", 0)
                        output = str(data.get("output", ""))
                        if _state["verbose"]:
                            console.print(f"{tool_buf}\n    [dim]-> {output[:300]} ({ms:.0f}ms)[/]")
                        else:
                            console.print(f"{tool_buf} [dim]-> {output[:60]}... ({ms:.0f}ms)[/]")

                    elif evt_type == "agent:message" and data.get("content"):
                        content = data["content"]
                        new_text = content[displayed:]
                        if new_text:
                            _print_stream_chunk(new_text)
                            displayed = len(content)
                        full_content = content

                    elif evt_type == "run:end":
                        usage = data.get("token_usage", {})
                        duration_ms = data.get("duration_ms", 0)
                        # Update cost tracker
                        _cost_tracker["total_tokens"] += usage.get("total_tokens", 0)
                        _cost_tracker["completion_tokens"] += usage.get("completion_tokens", 0)
                        _cost_tracker["total_calls"] += 1
                        _cost_tracker["total_duration_ms"] += duration_ms
                        pricing = _MODEL_PRICING.get(
                            _state["model"] or "default",
                            _MODEL_PRICING["default"],
                        )
                        prompt_t = usage.get("prompt_tokens", usage.get("total_tokens", 0) // 3)
                        comp_t = usage.get("completion_tokens", usage.get("total_tokens", 0) * 2 // 3)
                        cost = (prompt_t / 1_000_000 * pricing["prompt"]) + (comp_t / 1_000_000 * pricing["completion"])
                        _cost_tracker["estimated_cost_usd"] += cost

                    elif evt_type == "session:title_update":
                        sid = data.get("session_id", "")
                        if sid:
                            _state["session_id"] = sid

                    elif evt_type == "run:error":
                        console.print(f"[red]Error: {data.get('error', 'Unknown')}[/]")
                        return ""

                if full_content:
                    console.print()
                    if usage:
                        total = usage.get("total_tokens", 0)
                        console.print(f"[dim]--- {total} tokens | {duration_ms:.0f}ms ---[/]")

                return full_content

    except httpx.ConnectError:
        console.print("[red]Backend not reachable. Start with: python -m synapse_server[/]")
        return ""
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        return ""


def _print_stream_chunk(text: str) -> None:
    """Print a chunk of streaming text directly to stdout."""
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except UnicodeEncodeError:
        sys.stdout.write(text.encode("utf-8", errors="replace").decode("utf-8"))
        sys.stdout.flush()


# ── Interactive loop ─────────────────────────────────────────────────────

def _interactive() -> None:
    """Main interactive chat loop."""
    _print_banner()

    # Auto-create session
    try:
        with _api() as c:
            resp = c.post("/sessions", json={"agent_id": _state["agent"]})
            if resp.status_code == 200:
                _state["session_id"] = resp.json().get("session_id", "")
    except httpx.ConnectError:
        console.print("[yellow]Backend offline -- start with: python -m synapse_server[/]\n")

    while True:
        try:
            if _state["multiline"]:
                raw = _read_multiline()
                if not raw:
                    continue
            else:
                prompt_text = f"[bold green][{_state['agent'][:12]}]>[/]"
                raw = Prompt.ask(prompt_text)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/]")
            break

        if not raw.strip():
            continue

        # Handle slash commands
        msg, should_send = _handle_slash(raw)
        if not should_send:
            continue
        if not msg:
            continue

        # Send message
        _state["last_message"] = msg
        _conversation.append({"role": "user", "content": msg})

        # Stream response in real-time
        result = asyncio.run(_stream(msg))

        if result:
            _state["last_response"] = result
            _conversation.append({"role": "assistant", "content": result})
            _state["history_turns"] += 1
            console.print()
        else:
            console.print("[dim](no response)[/]\n")


# ── Entry point ──────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        prog="synapse",
        description="Synapse - Multi-Agent CLI",
    )
    parser.add_argument("message", nargs="?", default="", help="One-shot message (interactive mode if omitted)")
    parser.add_argument("-a", "--agent", default="general-assistant", help="Agent ID")
    parser.add_argument("-s", "--session", default="", help="Session ID to resume")
    parser.add_argument("-w", "--workspace", default="", help="Workspace directory")
    parser.add_argument("--api", default="", help="API base URL")
    parser.add_argument("--resume", action="store_true", help="Resume last session interactively")
    parser.add_argument("--last", action="store_true", help="Continue last conversation")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full tool outputs")

    args = parser.parse_args()

    if args.api:
        global API_BASE
        API_BASE = args.api
    if args.agent:
        _state["agent"] = args.agent
    if args.session:
        _state["session_id"] = args.session
    if args.workspace:
        _state["workspace"] = str(Path(args.workspace).resolve())
        os.environ["SYNAPSE_WORKSPACE"] = _state["workspace"]
    if args.verbose:
        _state["verbose"] = True

    # --resume: auto enter interactive + resume picker
    if args.resume:
        _interactive()
        return

    # --last: continue last session
    if args.last:
        try:
            with _api() as c:
                resp = c.get("/sessions")
                if resp.status_code == 200 and resp.json():
                    last = resp.json()[0]
                    _state["session_id"] = last["session_id"]
                    _load_session_messages(last["session_id"])
        except Exception:
            pass
        _interactive()
        return

    # One-shot mode
    if args.message:
        result = asyncio.run(_stream(args.message))
        if result:
            try:
                console.print(Markdown(result))
            except UnicodeEncodeError:
                print(result.encode("utf-8", errors="replace").decode("utf-8"))
        return

    # Interactive mode (default)
    _interactive()


def _time_ago(ts: float) -> str:
    """Human-readable time ago."""
    if not ts:
        return "-"
    diff = time.time() - ts
    if diff < 60:
        return "just now"
    if diff < 3600:
        return f"{int(diff / 60)}m ago"
    if diff < 86400:
        return f"{int(diff / 3600)}h ago"
    return f"{int(diff / 86400)}d ago"


if __name__ == "__main__":
    main()
