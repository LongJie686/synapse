"""Plugin system: extensible tools, agents, loaders, and guardrails."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from synapse_core.tools import ToolDefinition, ToolRegistry
from synapse_core.agent import AgentDefinition, AgentRegistry
from synapse_core.guardrails import GuardrailPipeline


# ── Plugin Interfaces ───────────────────────────────────────────────────────

@runtime_checkable
class SynapsePlugin(Protocol):
    """Base protocol all plugins must implement."""

    name: str
    version: str

    def setup(self, host: PluginHost) -> None:
        """Register plugin components with the host."""
        ...

    def teardown(self) -> None:
        """Clean up plugin resources."""
        ...


class PluginMeta(BaseModel):
    """Plugin metadata."""

    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    tags: list[str] = []


class ToolPlugin:
    """Convenience base class for tool plugins."""

    name: str = ""
    version: str = "0.1.0"

    def get_definition(self) -> ToolDefinition:
        raise NotImplementedError

    def get_handler(self) -> Any:
        raise NotImplementedError

    def setup(self, host: PluginHost) -> None:
        host.register_tool(self.get_definition(), self.get_handler())

    def teardown(self) -> None:
        pass


class AgentPlugin:
    """Convenience base class for agent plugins."""

    name: str = ""
    version: str = "0.1.0"

    def get_definition(self) -> AgentDefinition:
        raise NotImplementedError

    def setup(self, host: PluginHost) -> None:
        host.register_agent(self.get_definition())

    def teardown(self) -> None:
        pass


class GuardrailPlugin:
    """Convenience base class for guardrail plugins."""

    name: str = ""
    version: str = "0.1.0"

    def check_input(self, content: str) -> dict[str, Any] | None:
        return None

    def check_output(self, content: str) -> dict[str, Any] | None:
        return None

    def setup(self, host: PluginHost) -> None:
        host.register_guardrail(self)

    def teardown(self) -> None:
        pass


# ── Plugin Host ─────────────────────────────────────────────────────────────

class PluginHost:
    """Central plugin manager that loads, registers, and manages plugins."""

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        agent_registry: AgentRegistry | None = None,
    ) -> None:
        self.tool_registry = tool_registry or ToolRegistry()
        self.agent_registry = agent_registry or AgentRegistry()
        self._plugins: dict[str, SynapsePlugin] = {}
        self._guardrails: list[GuardrailPlugin] = []
        self._plugin_dirs: list[Path] = []

    @property
    def plugins(self) -> dict[str, SynapsePlugin]:
        return dict(self._plugins)

    def register_tool(self, definition: ToolDefinition, handler: Any) -> None:
        self.tool_registry.register(definition, handler)

    def register_agent(self, definition: AgentDefinition) -> None:
        self.agent_registry.register(definition)

    def register_guardrail(self, guardrail: GuardrailPlugin) -> None:
        self._guardrails.append(guardrail)

    def load_plugin(self, plugin: SynapsePlugin) -> None:
        """Load a single plugin instance."""
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin '{plugin.name}' is already loaded")
        plugin.setup(self)
        self._plugins[plugin.name] = plugin

    def unload_plugin(self, name: str) -> None:
        """Unload a plugin by name."""
        if name not in self._plugins:
            raise KeyError(f"Plugin '{name}' is not loaded")
        plugin = self._plugins.pop(name)
        plugin.teardown()
        self._guardrails = [g for g in self._guardrails if g is not plugin]

    def load_from_module(self, module_path: str) -> None:
        """Load a plugin from a Python module path (e.g. 'my_package.my_plugin')."""
        module = importlib.import_module(module_path)

        plugin_class = None
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if isinstance(obj, type) and hasattr(obj, "setup") and hasattr(obj, "name"):
                if not inspect.isabstract(obj):
                    plugin_class = obj
                    break

        if plugin_class is None:
            raise ValueError(f"No valid plugin class found in module '{module_path}'")

        plugin = plugin_class()
        self.load_plugin(plugin)

    def load_from_directory(self, directory: Path | str) -> int:
        """Load all plugins from a directory. Returns count of loaded plugins."""
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Plugin directory not found: {directory}")

        loaded = 0
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            module_name = py_file.stem
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        hasattr(obj, "setup")
                        and hasattr(obj, "name")
                        and not inspect.isabstract(obj)
                        and obj is not ToolPlugin
                        and obj is not AgentPlugin
                        and obj is not GuardrailPlugin
                    ):
                        try:
                            plugin = obj()
                            if plugin.name not in self._plugins:
                                self.load_plugin(plugin)
                                loaded += 1
                        except Exception:
                            pass
        return loaded

    def get_all_guardrails(self) -> list[GuardrailPlugin]:
        return list(self._guardrails)

    def list_plugins(self) -> list[PluginMeta]:
        metas = []
        for plugin in self._plugins.values():
            metas.append(PluginMeta(
                name=plugin.name,
                version=getattr(plugin, "version", "0.0.0"),
                description=getattr(plugin, "description", ""),
            ))
        return metas
