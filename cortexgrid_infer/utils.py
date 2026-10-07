"""Tool introspection and conversion utilities."""

from __future__ import annotations

import inspect
from functools import partial
from typing import Annotated, Any, Callable, get_type_hints

from pydantic import PlainSerializer, PlainValidator

ToolSpec = dict[str, Any]


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


def tool_spec(tool: Callable[..., Any] | ToolSpec) -> ToolSpec:
    if callable(tool):
        return function_to_tool_spec(tool)
    return tool


Tool = Annotated[
    Callable[..., Any] | ToolSpec,
    PlainValidator(dict),
    PlainSerializer(tool_spec),
]
