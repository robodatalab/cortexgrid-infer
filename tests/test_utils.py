"""Tests for public interfaces in cortexgrid_infer.utils."""

from __future__ import annotations

import tempfile
import unittest
from functools import partial
from pathlib import Path
from typing import Any
from unittest import mock

from cortexgrid_infer.utils import (
    build_tool_map,
    download_hf_snapshot,
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


def _fake_snapshot_download(*, local_dir: str, **_kwargs: Any) -> None:
    """Lay out what `snapshot_download(local_dir=...)` writes: the model files
    plus HuggingFace's download bookkeeping under `.cache/huggingface/`."""
    root = Path(local_dir)
    (root / "config.json").write_text("{}")
    metadata = root / ".cache" / "huggingface" / "download" / "config.json.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("etag")


def _files_under(local_dir: str) -> set[str]:
    root = Path(local_dir)
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


class TestDownloadHFSnapshot(unittest.TestCase):
    @mock.patch(
        "cortexgrid_infer.utils.snapshot_download", side_effect=_fake_snapshot_download
    )
    def test_downloads_model_files_without_hf_metadata(self, mock_snapshot: mock.Mock):
        with tempfile.TemporaryDirectory() as d:
            result = download_hf_snapshot("org/Model-4B", d, "tok", ["*.md"])

            self.assertEqual(result, d)
            self.assertEqual(_files_under(d), {"config.json"})
        mock_snapshot.assert_called_once_with(
            repo_id="org/Model-4B", local_dir=d, ignore_patterns=["*.md"], token="tok"
        )
