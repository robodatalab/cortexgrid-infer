"""Tests for public interfaces in cortexgrid_infer.core."""

from __future__ import annotations

import unittest
from collections.abc import AsyncIterator
from functools import partial
from typing import Any, Sequence

from cortexgrid_infer.core import (
    CompletingModel,
    CompletionChunk,
    DeployedModel,
    Message,
    Tool,
    ToolCall,
    _providers,
    complete,
    deploy_model,
    register_provider,
)


class _StubModel(CompletingModel):
    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    @property
    def name(self) -> str:
        return self._model_id

    async def complete(
        self,
        messages: list[Message],
        tools: Sequence[Tool] | None = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionChunk]:
        yield CompletionChunk(content="stub", finish_reason="stop")


class TestToolCall(unittest.TestCase):
    def test_executes_function(self):
        def add(a: int, b: int) -> int:
            return a + b

        tc = ToolCall(
            id="call_1",
            name="add",
            arguments={"a": 2, "b": 3},
            _func=partial(add),
        )
        self.assertEqual(tc(), 5)

    def test_fields(self):
        tc = ToolCall(
            id="call_x",
            name="noop",
            arguments={"k": "v"},
            _func=partial(lambda: None),
        )
        self.assertEqual(tc.id, "call_x")
        self.assertEqual(tc.name, "noop")
        self.assertEqual(tc.arguments, {"k": "v"})


class TestCompletionChunk(unittest.TestCase):
    def test_defaults(self):
        chunk = CompletionChunk()
        self.assertEqual(chunk.content, "")
        self.assertEqual(chunk.tool_calls, [])
        self.assertIsNone(chunk.finish_reason)
        self.assertFalse(chunk.has_text)
        self.assertFalse(chunk.has_tool_calls)

    def test_has_text(self):
        chunk = CompletionChunk(content="hello")
        self.assertTrue(chunk.has_text)

    def test_has_tool_calls(self):
        tc = ToolCall(id="1", name="f", arguments={}, _func=partial(lambda: None))
        chunk = CompletionChunk(tool_calls=[tc])
        self.assertTrue(chunk.has_tool_calls)

    def test_finish_reason(self):
        chunk = CompletionChunk(finish_reason="stop")
        self.assertEqual(chunk.finish_reason, "stop")


class TestProviderRegistry(unittest.TestCase):
    def setUp(self):
        self._snapshot = _providers.copy()
        _providers.clear()

    def tearDown(self):
        _providers.clear()
        _providers.extend(self._snapshot)

    def test_register_and_deploy(self):
        register_provider("stub/", _StubModel)
        model = deploy_model("stub/my-model")
        self.assertIsInstance(model, _StubModel)
        self.assertEqual(model.name, "stub/my-model")

    def test_deploy_unknown_prefix_raises(self):
        with self.assertRaises(ValueError):
            deploy_model("nonexistent/model")


class _FakeDeployedModel(DeployedModel):
    """Fake model that derives from DeployedModel but not CompletingModel."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    @property
    def name(self) -> str:
        return self._model_id


class _FakeCompletingModel(CompletingModel):
    """Fake model that derives from CompletingModel."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    @property
    def name(self) -> str:
        return self._model_id

    async def complete(
        self,
        messages: list[Message],
        tools: Sequence[Tool] | None = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionChunk]:
        yield CompletionChunk(content="fake", finish_reason="stop")


class TestDeployModel(unittest.TestCase):
    def setUp(self):
        self._snapshot = _providers.copy()
        _providers.clear()

    def tearDown(self):
        _providers.clear()
        _providers.extend(self._snapshot)

    def test_no_providers_registered_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            deploy_model("anything/model")
        self.assertIn("No provider registered", str(ctx.exception))

    def test_no_matching_prefix_raises_value_error(self):
        register_provider("foo/", _FakeCompletingModel)
        register_provider("bar/", _FakeCompletingModel)
        with self.assertRaises(ValueError) as ctx:
            deploy_model("baz/model")
        message = str(ctx.exception)
        self.assertIn("baz/model", message)
        self.assertIn("foo/", message)
        self.assertIn("bar/", message)

    def test_default_expected_type_returns_completing_model(self):
        register_provider("fake/", _FakeCompletingModel)
        model = deploy_model("fake/my-model")
        self.assertIsInstance(model, _FakeCompletingModel)
        self.assertIsInstance(model, CompletingModel)
        self.assertEqual(model.name, "fake/my-model")

    def test_factory_returning_none_skips_to_next_provider(self):
        register_provider("fake/", lambda _id: None)
        register_provider("fake/", _FakeCompletingModel)
        model = deploy_model("fake/my-model")
        self.assertIsInstance(model, _FakeCompletingModel)

    def test_all_matching_factories_return_none_raises_value_error(self):
        register_provider("fake/", lambda _id: None)
        register_provider("fake/", lambda _id: None)
        with self.assertRaises(ValueError):
            deploy_model("fake/my-model")

    def test_first_matching_provider_wins(self):
        class _FirstModel(_FakeCompletingModel):
            pass

        class _SecondModel(_FakeCompletingModel):
            pass

        register_provider("fake/", _FirstModel)
        register_provider("fake/", _SecondModel)
        model = deploy_model("fake/my-model")
        self.assertIsInstance(model, _FirstModel)
        self.assertNotIsInstance(model, _SecondModel)

    def test_non_matching_provider_skipped_before_matching_provider(self):
        register_provider("other/", _FakeCompletingModel)
        register_provider("fake/", _FakeCompletingModel)
        model = deploy_model("fake/my-model")
        self.assertEqual(model.name, "fake/my-model")

    def test_factory_receives_full_model_id(self):
        captured: list[str] = []

        def factory(model_id: str) -> _FakeCompletingModel:
            captured.append(model_id)
            return _FakeCompletingModel(model_id)

        register_provider("fake/", factory)
        deploy_model("fake/provider/some-model-name")
        self.assertEqual(captured, ["fake/provider/some-model-name"])

    def test_prefix_matches_exactly_at_start(self):
        register_provider("fake/", _FakeCompletingModel)
        with self.assertRaises(ValueError):
            deploy_model("prefix-fake/my-model")

    def test_empty_prefix_matches_any_model_id(self):
        register_provider("", _FakeCompletingModel)
        model = deploy_model("anything-goes")
        self.assertEqual(model.name, "anything-goes")


class TestComplete(unittest.IsolatedAsyncioTestCase):
    async def test_yields_chunks(self):
        model = _StubModel("test")
        chunks = [c async for c in complete(model, [{"role": "user", "content": "hi"}])]
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, "stub")
        self.assertEqual(chunks[0].finish_reason, "stop")
