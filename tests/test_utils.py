"""Tests for public interfaces in cortexgrid_infer.utils."""

from __future__ import annotations

import unittest

from pydantic import TypeAdapter

from cortexgrid_infer.utils import Tool, function_to_tool_spec


class TestFunctionToToolSpec(unittest.TestCase):
    def test_basic(self):
        def greet(name: str, excited: bool = False) -> str:
            """Say hello.

            Args:
                name: Who to greet.
                excited: Add exclamation mark.
            """
            return f"hi {name}{'!' if excited else ''}"

        spec = function_to_tool_spec(greet)
        self.assertEqual(spec["type"], "function")

        func_spec = spec["function"]
        self.assertEqual(func_spec["name"], "greet")
        self.assertIn("Say hello", func_spec["description"])

        params = func_spec["parameters"]
        self.assertIn("name", params["properties"])
        self.assertEqual(params["properties"]["name"]["type"], "string")
        self.assertIn("name", params["required"])
        self.assertNotIn("excited", params["required"])

    def test_no_docstring(self):
        def bare(x: int) -> int:
            return x

        spec = function_to_tool_spec(bare)
        self.assertEqual(spec["function"]["name"], "bare")
        self.assertEqual(spec["function"]["description"], "")


class TestTool(unittest.TestCase):
    def test_a_function_goes_to_json_as_its_spec(self) -> None:
        def now(timezone: str) -> str:
            return "noon"

        tool_on_the_wire = TypeAdapter(Tool)

        answered = tool_on_the_wire.dump_python(now, mode="json")

        self.assertEqual(
            answered,
            {
                "type": "function",
                "function": {
                    "name": "now",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                        "required": ["timezone"],
                    },
                },
            },
        )

    def test_a_spec_goes_to_json_as_it_is(self) -> None:
        tool_on_the_wire = TypeAdapter(Tool)

        answered = tool_on_the_wire.dump_python(
            {"type": "function", "function": {"name": "other"}}, mode="json"
        )

        self.assertEqual(answered, {"type": "function", "function": {"name": "other"}})

    def test_comes_from_json_as_its_spec(self) -> None:
        tool_on_the_wire = TypeAdapter(Tool)

        tool = tool_on_the_wire.validate_python({"type": "function", "function": {"name": "other"}})

        self.assertEqual(tool, {"type": "function", "function": {"name": "other"}})
