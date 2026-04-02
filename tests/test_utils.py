"""Tests for public interfaces in model_gateway.utils."""

from __future__ import annotations

import unittest
from functools import partial

from model_gateway.utils import (
    build_tool_map,
    function_to_tool_spec,
    normalize_tools,
    tool_name,
)


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


class TestNormalizeTools(unittest.TestCase):
    def test_none(self):
        self.assertIsNone(normalize_tools(None))

    def test_mixed(self):
        def my_func(a: str) -> str:
            """Do something."""
            return a

        dict_spec = {
            "type": "function",
            "function": {"name": "other", "description": "", "parameters": {}},
        }
        result = normalize_tools([my_func, dict_spec])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["function"]["name"], "my_func")
        self.assertIs(result[1], dict_spec)


class TestToolName(unittest.TestCase):
    def test_callable(self):
        def foo():
            pass

        self.assertEqual(tool_name(foo), "foo")

    def test_partial(self):
        def bar(x: int) -> int:
            return x

        p = partial(bar, x=1)
        self.assertEqual(tool_name(p), "bar")

    def test_dict(self):
        spec = {"function": {"name": "baz"}}
        self.assertEqual(tool_name(spec), "baz")


class TestBuildToolMap(unittest.TestCase):
    def test_none(self):
        self.assertEqual(build_tool_map(None), {})

    def test_callables(self):
        def alpha():
            pass

        def beta():
            pass

        m = build_tool_map([alpha, beta])
        self.assertIs(m["alpha"], alpha)
        self.assertIs(m["beta"], beta)

    def test_ignores_dicts(self):
        spec = {"function": {"name": "x"}}
        m = build_tool_map([spec])
        self.assertEqual(m, {})
