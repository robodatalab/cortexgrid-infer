"""Tests for cortexgrid_infer.protocols.completion."""

from __future__ import annotations

import unittest
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

from cortexgrid_infer.protocols.completion import (
    ServedCompletingModel,
    encode_thinking,
    encode_tool_call,
    parse_tool_calls,
)


class TestEncodeToolCall(unittest.TestCase):
    def test_round_trips_through_the_parser(self):
        # A serve app whose upstream reports structured tool calls encodes them
        # with this; the client parses them back out. If the two ever disagree,
        # such an app silently streams its tool calls as prose.
        encoded = encode_tool_call("add", {"a": 2, "b": 3})

        text, calls = parse_tool_calls(encoded, {"add": lambda a, b: a + b})

        self.assertEqual(text, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "add")
        self.assertEqual(calls[0].arguments, {"a": 2, "b": 3})
        self.assertEqual(calls[0](), 5)

    def test_unknown_tools_are_dropped_not_bound(self):
        _text, calls = parse_tool_calls(encode_tool_call("add", {"a": 1}), {})
        self.assertEqual(calls, [])


# --- HTTP streaming fakes for ServedCompletingModel.complete ---


class _FakeResponse:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    def raise_for_status(self) -> None:
        pass

    async def aiter_text(self) -> AsyncIterator[str]:
        for c in self._chunks:
            yield c


class _FakeStreamCtx:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def __aenter__(self) -> _FakeResponse:
        return _FakeResponse(self._chunks)

    async def __aexit__(self, *args: Any) -> None:
        pass


class _FakeAsyncClient:
    captured: dict[str, Any] = {}

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    def stream(self, method: str, url: str, json: dict[str, Any]) -> _FakeStreamCtx:
        type(self).captured = {"method": method, "url": url, "json": json}
        return _FakeStreamCtx(self._chunks)


def _patch_httpx_client(chunks: list[str]):
    def factory(*_a: Any, **_kw: Any) -> _FakeAsyncClient:
        return _FakeAsyncClient(chunks)

    return mock.patch("cortexgrid_infer.protocols.completion.httpx.AsyncClient", factory)


class TestServedCompletingModelComplete(unittest.IsolatedAsyncioTestCase):
    async def test_streams_plain_text(self):
        model = ServedCompletingModel(url="http://x/r/f/s/r", model_id="Qwen/Qwen2-2.5B-Instruct")
        with _patch_httpx_client(["hello", " ", "world"]):
            out = [
                c
                async for c in model.complete(
                    messages=[{"role": "user", "content": "hi"}]
                )
            ]
        self.assertEqual("".join(c.content for c in out), "hello world")
        self.assertEqual(out[-1].finish_reason, "stop")

    async def test_posts_to_complete_endpoint_with_params(self):
        model = ServedCompletingModel(
            url="http://h:30000/r/F/S/R", model_id="Qwen/Qwen2-2.5B-Instruct"
        )
        with _patch_httpx_client(["ok"]):
            _ = [
                c
                async for c in model.complete(
                    messages=[{"role": "user", "content": "hi"}],
                    temperature=0.5,
                    max_new_tokens=42,
                )
            ]
        captured = _FakeAsyncClient.captured
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["url"], "http://h:30000/r/F/S/R/complete")
        self.assertEqual(captured["json"]["temperature"], 0.5)
        self.assertEqual(captured["json"]["max_new_tokens"], 42)
        self.assertEqual(
            captured["json"]["messages"], [{"role": "user", "content": "hi"}]
        )

    async def test_parses_tool_calls_from_stream(self):
        def add(a: int, b: int) -> int:
            return a + b

        chunks = [
            "Hello ",
            "<tool_call>",
            '{"name": "add", "arguments": {"a": 2, "b": 3}}',
            "</tool_call>",
        ]
        model = ServedCompletingModel(url="http://x/r/f/s/r", model_id="Qwen/Qwen2-2.5B-Instruct")
        with _patch_httpx_client(chunks):
            out = [
                c
                async for c in model.complete(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=[add],
                )
            ]
        text = "".join(c.content for c in out).strip()
        self.assertEqual(text, "Hello")
        tool_chunks = [c for c in out if c.has_tool_calls]
        self.assertEqual(len(tool_chunks), 1)
        tc = tool_chunks[0].tool_calls[0]
        self.assertEqual(tc.name, "add")
        self.assertEqual(tc.arguments, {"a": 2, "b": 3})
        self.assertEqual(tc(), 5)

    async def streamed(self, chunks: list[str]) -> list[Any]:
        model = ServedCompletingModel(url="http://x/r/f/s/r", model_id="Qwen/Qwen3-8B")
        with _patch_httpx_client(chunks):
            return [
                c
                async for c in model.complete(
                    messages=[{"role": "user", "content": "hi"}]
                )
            ]

    async def test_returns_the_thinking_apart_from_the_answer(self):
        out = await self.streamed(
            ["<think>", "Let me", " see.", "</think>", "\n\nThe answer."]
        )

        self.assertEqual("".join(c.thinking for c in out), "Let me see.")
        self.assertEqual("".join(c.content for c in out), "\n\nThe answer.")

    async def test_thinking_encoded_by_a_serve_app_comes_back_as_thinking(self):
        out = await self.streamed([encode_thinking("pondering"), "4"])

        self.assertEqual("".join(c.thinking for c in out), "pondering")
        self.assertEqual("".join(c.content for c in out), "4")

    async def test_finds_the_thinking_markers_split_across_chunks(self):
        out = await self.streamed(["<thi", "nk>reason", "ing</th", "ink>answer"])

        self.assertEqual("".join(c.thinking for c in out), "reasoning")
        self.assertEqual("".join(c.content for c in out), "answer")

    async def test_a_reply_cut_off_while_thinking_is_all_thinking(self):
        out = await self.streamed(["<think>still ", "going"])

        self.assertEqual("".join(c.thinking for c in out), "still going")
        self.assertEqual("".join(c.content for c in out), "")
        self.assertEqual(out[-1].finish_reason, "stop")

    async def test_parses_a_tool_call_made_after_thinking(self):
        def add(a: int, b: int) -> int:
            return a + b

        model = ServedCompletingModel(url="http://x/r/f/s/r", model_id="Qwen/Qwen3-8B")
        chunks = [
            "<think>I should add.</think>",
            '<tool_call>{"name": "add", "arguments": {"a": 2, "b": 3}}</tool_call>',
        ]
        with _patch_httpx_client(chunks):
            out = [
                c
                async for c in model.complete(
                    messages=[{"role": "user", "content": "hi"}], tools=[add]
                )
            ]

        self.assertEqual("".join(c.thinking for c in out), "I should add.")
        self.assertEqual([c.tool_calls[0]() for c in out if c.has_tool_calls], [5])
