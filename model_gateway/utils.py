"""Tool introspection and conversion utilities."""

from __future__ import annotations

import inspect
from functools import partial
from typing import Any, Callable, Sequence, get_type_hints

from model_gateway.core import Tool, ToolSpec


def get_underlying_func(func: Callable | partial) -> Callable:
    """Unwrap all ``partial`` layers to return the innermost function."""
    while isinstance(func, partial):
        func = func.func
    return func


_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def function_to_tool_spec(func: Callable[..., Any]) -> ToolSpec:
    """Convert a Python function to an OpenAI-style tool spec."""
    underlying = get_underlying_func(func)

    sig = inspect.signature(func)
    hints = get_type_hints(underlying)
    doc = inspect.getdoc(underlying) or ""
    func_name = getattr(func, "__name__", underlying.__name__)

    param_docs: dict[str, str] = {}
    if "Args:" in doc:
        args_section = doc.split("Args:")[1].split("Returns:")[0]
        for line in args_section.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                param_name, param_desc = line.split(":", 1)
                param_docs[param_name.strip()] = param_desc.strip()

    description = doc.split("Args:")[0].strip() if "Args:" in doc else doc

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        param_type = hints.get(name, str)
        json_type = _TYPE_MAP.get(param_type, "string")

        properties[name] = {"type": json_type}
        if name in param_docs:
            properties[name]["description"] = param_docs[name]

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def normalize_tools(tools: Sequence[Tool] | None) -> list[ToolSpec] | None:
    """Convert a list of tools (functions or dicts) to tool specs."""
    if tools is None:
        return None
    return [function_to_tool_spec(t) if callable(t) else t for t in tools]


def tool_name(t: Tool) -> str:
    if callable(t):
        underlying = get_underlying_func(t)
        return getattr(t, "__name__", underlying.__name__)
    return t.get("function", {}).get("name", "unknown")


def build_tool_map(tools: Sequence[Tool] | None) -> dict[str, Callable[..., Any]]:
    """Build mapping from tool names to callable functions."""
    tool_map: dict[str, Callable[..., Any]] = {}
    if tools:
        for tool in tools:
            if callable(tool):
                underlying = get_underlying_func(tool)
                name = getattr(tool, "__name__", underlying.__name__)
                tool_map[name] = tool
    return tool_map
