# Contributing to Synapse

Thank you for your interest in contributing to Synapse!

## Development Setup

```bash
# Clone and setup
git clone https://github.com/LongJie686/synapse.git
cd synapse
uv venv --python 3.11
uv sync

# Run tests
pytest

# Start dev server
uvicorn synapse_server.app:create_app --factory --reload --port 8000
```

## Code Style

- Python 3.11+ with type hints
- Pydantic v2 for all data models
- Ruff for formatting and linting
- Max file length: 800 lines
- Functions: < 50 lines

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new feature
fix: resolve bug
docs: update documentation
refactor: restructure code
test: add tests
chore: maintenance tasks
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`feat/my-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest`)
5. Submit a PR with a clear description

## Adding a Plugin

Create a Python file in your plugin directory:

```python
from synapse_core.plugin import ToolPlugin
from synapse_core.tools import ToolDefinition, ToolParameter

class MyToolPlugin(ToolPlugin):
    name = "my-tool"
    version = "1.0.0"

    def get_definition(self):
        return ToolDefinition(
            name="my_tool",
            description="Does something useful",
            category="custom",
            parameters=[
                ToolParameter(name="input", type="string", description="Input value"),
            ],
        )

    def get_handler(self):
        async def handler(input: str, **kwargs):
            return f"Processed: {input}"
        return handler
```

## Reporting Issues

- Use GitHub Issues
- Include Python version, OS, and steps to reproduce
- Check existing issues before opening a new one

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
