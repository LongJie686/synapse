"""Structured output: Pydantic model to LLM structured output with validation."""

from __future__ import annotations

import json
from typing import Any, Type, get_type_hints

from pydantic import BaseModel, ValidationError

from synapse_core.llm import LLMConfig, LLMMessage, LLMResponse
from synapse_core.llm.providers import create_provider


def _python_type_to_json_schema(tp: Any) -> dict[str, Any]:
    """Convert Python type annotation to JSON Schema."""
    origin = getattr(tp, "__origin__", None)

    if tp is str:
        return {"type": "string"}
    elif tp is int:
        return {"type": "integer"}
    elif tp is float:
        return {"type": "number"}
    elif tp is bool:
        return {"type": "boolean"}
    elif origin is list:
        args = getattr(tp, "__args__", (Any,))
        if args[0] is Any:
            return {"type": "array"}
        return {"type": "array", "items": _python_type_to_json_schema(args[0])}
    elif origin is dict:
        return {"type": "object"}
    elif tp is Any:
        return {}
    return {"type": "string"}


def pydantic_to_schema(model: Type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to a JSON Schema for LLM consumption."""
    schema = model.model_json_schema()
    return schema


def schema_to_prompt(schema: dict[str, Any]) -> str:
    """Convert JSON Schema to a clear instruction prompt for the LLM."""
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    lines = ["Respond with a JSON object matching this schema:", ""]
    lines.append("{")

    for field_name, field_schema in properties.items():
        field_type = field_schema.get("type", "any")
        description = field_schema.get("description", "")
        is_required = field_name in required
        req_marker = " (required)" if is_required else " (optional)"

        if field_type == "array":
            items = field_schema.get("items", {})
            lines.append(f'  "{field_name}": [{items.get("type", "...")}],  // {description}{req_marker}')
        elif field_type == "object":
            lines.append(f'  "{field_name}": {{...}},  // {description}{req_marker}')
        else:
            lines.append(f'  "{field_name}": "{field_type}",  // {description}{req_marker}')

    lines.append("}")
    lines.append("")
    lines.append("Return ONLY valid JSON, no markdown fences, no explanation.")

    return "\n".join(lines)


def extract_json_from_response(content: str) -> str:
    """Extract JSON from LLM response, handling markdown fences and extra text."""
    content = content.strip()

    # Try direct parse first
    if content.startswith("{") or content.startswith("["):
        return content

    # Extract from markdown code fences
    if "```" in content:
        parts = content.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i].strip()
            # Remove language identifier
            if block.startswith(("json", "JSON")):
                block = block[4:].strip()
            if block.startswith(("{", "[")):
                return block

    # Find first JSON object
    start = content.find("{")
    if start == -1:
        start = content.find("[")
    if start != -1:
        brace_count = 0
        for i in range(start, len(content)):
            if content[i] == "{":
                brace_count += 1
            elif content[i] == "}":
                brace_count -= 1
            elif content[i] == "[":
                brace_count += 1
            elif content[i] == "]":
                brace_count -= 1
            if brace_count == 0:
                return content[start:i + 1]

    return content


async def generate_structured_output(
    model: Type[BaseModel],
    prompt: str,
    config: LLMConfig | None = None,
    system_prompt: str = "",
) -> BaseModel:
    """Generate structured output validated against a Pydantic model.

    Args:
        model: Pydantic model class to validate against
        prompt: User prompt for the LLM
        config: LLM configuration (uses default if not provided)
        system_prompt: Optional system prompt

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: If LLM response cannot be parsed into the model
    """
    if config is None:
        config = LLMConfig()

    schema = pydantic_to_schema(model)
    schema_prompt = schema_to_prompt(schema)

    messages = []
    if system_prompt:
        messages.append(LLMMessage(role="system", content=system_prompt))

    messages.append(LLMMessage(
        role="system",
        content=schema_prompt,
    ))
    messages.append(LLMMessage(role="user", content=prompt))

    provider = create_provider(config)
    response = await provider.invoke(messages)

    json_str = extract_json_from_response(response.content)

    try:
        data = json.loads(json_str)
        return model.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Failed to parse structured output: {e}\nRaw response: {response.content[:500]}") from e


async def generate_structured_output_with_retry(
    model: Type[BaseModel],
    prompt: str,
    config: LLMConfig | None = None,
    system_prompt: str = "",
    max_retries: int = 2,
) -> BaseModel:
    """Generate structured output with automatic retry on validation failure."""
    last_error: Exception | None = None

    current_prompt = prompt
    for attempt in range(max_retries + 1):
        try:
            return await generate_structured_output(model, current_prompt, config, system_prompt)
        except ValueError as e:
            last_error = e
            if attempt < max_retries:
                current_prompt = (
                    f"{prompt}\n\n"
                    f"[Previous attempt failed with error: {e}]\n"
                    f"Please ensure your response is valid JSON matching the schema exactly."
                )

    raise last_error or ValueError("Failed to generate structured output after retries")
