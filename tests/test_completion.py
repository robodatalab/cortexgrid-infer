"""Tests for cortexgrid_infer.protocols.completion."""

from __future__ import annotations

import functools
import json
import unittest
from typing import Any
from unittest import mock

import httpx

from cortexgrid_infer.protocols import completion
from cortexgrid_infer.protocols.completion import ServedCompletingModel
from cortexgrid_infer.serve_apps.streamed_completion import completion_chunk


def _served(lines: list[bytes]) -> mock.Mock:
    content = b"".join(lines)
    return mock.Mock(return_value=httpx.Response(200, content=content))


class TestServedCompletingModelComplete(unittest.IsolatedAsyncioTestCase):
    async def test_hands_back_each_chunk_the_deployment_sent(self):
        def add(a: int, b: int) -> int:
            return a + b

        cases = [
            (
                "text_then_stop",
                [completion_chunk(content="hello"), completion_chunk(content=" world"), completion_chunk(finish_reason="stop")],
                None,
                [("hello", "", [], None), (" world", "", [], None), ("", "", [], "stop")],
            ),
            (
                "thinking_then_text",
                [completion_chunk(thinking="pondering"), completion_chunk(content="4"), completion_chunk(finish_reason="stop")],
                None,
                [("", "pondering", [], None), ("4", "", [], None), ("", "", [], "stop")],
            ),
            (
                "a_tool_call_bound_to_the_caller_s_function",
                [completion_chunk(tool_calls=[{"id": "call_7", "name": "add", "arguments": {"a": 2, "b": 3}}])],
                [add],
                [("", "", [("call_7", "add", {"a": 2, "b": 3}, 5)], None)],
            ),
            (
                "a_tool_call_the_caller_did_not_offer_is_dropped",
                [completion_chunk(tool_calls=[{"id": "call_7", "name": "delete_all", "arguments": {}}])],
                [add],
                [("", "", [], None)],
            ),
        ]
        for name, lines, tools, expected in cases:
            with self.subTest(name):
                server = _served(lines)
                model = ServedCompletingModel(url="http://h/r/F/S/R", model_id="Qwen/Qwen3-8B")

                with mock.patch.object(
                    completion.httpx,
                    "AsyncClient",
                    functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(server)),
                ):
                    chunks = [
                        chunk
                        async for chunk in model.complete([{"role": "user", "content": "hi"}], tools=tools)
                    ]

                received = [
                    (
                        chunk.content,
                        chunk.thinking,
                        [(call.id, call.name, call.arguments, call()) for call in chunk.tool_calls],
                        chunk.finish_reason,
                    )
                    for chunk in chunks
                ]
                self.assertEqual(received, expected)

    async def test_posts_the_request_to_the_complete_route(self):
        def now() -> str:
            return "noon"

        server = _served([completion_chunk(finish_reason="stop")])
        model = ServedCompletingModel(url="http://h:30000/r/F/S/R", model_id="Qwen/Qwen3-8B")

        with mock.patch.object(
            completion.httpx,
            "AsyncClient",
            functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(server)),
        ):
            [
                chunk
                async for chunk in model.complete(
                    [{"role": "user", "content": "hi"}],
                    tools=[now],
                    temperature=0.5,
                    max_new_tokens=42,
                    choices=["yes"],
                )
            ]

        request = server.call_args.args[0]
        sent = json.loads(request.content)
        self.assertEqual(str(request.url), "http://h:30000/r/F/S/R/complete")
        self.assertEqual(sent["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(sent["temperature"], 0.5)
        self.assertEqual(sent["max_new_tokens"], 42)
        self.assertEqual(sent["choices"], ["yes"])
        self.assertEqual([tool["function"]["name"] for tool in sent["tools"]], ["now"])


class TestServedCompletingModelReadsTheModel(unittest.IsolatedAsyncioTestCase):
    async def test_asks_for_the_loglikelihood_of_every_continuation(self):
        cases = [
            (
                "two_answers",
                [{"role": "user", "content": "Yes or no?"}],
                ["yes", "no"],
                {"loglikelihoods": [-0.1, -2.3]},
                [-0.1, -2.3],
            ),
            (
                "one_long_continuation",
                [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Name a colour."}],
                ["dark blue"],
                {"loglikelihoods": [-4.75]},
                [-4.75],
            ),
            (
                "no_continuations",
                [{"role": "user", "content": "Anything?"}],
                [],
                {"loglikelihoods": []},
                [],
            ),
        ]
        for name, messages, continuations, answered, expected in cases:
            with self.subTest(name):
                server = mock.Mock(return_value=httpx.Response(200, json=answered))
                with mock.patch.object(
                    completion.httpx,
                    "AsyncClient",
                    functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(server)),
                ):
                    result = await ServedCompletingModel("http://h/r/F/S/R", "m").loglikelihoods(
                        messages, continuations
                    )

                self.assertEqual(result, expected)
                self.assertEqual(str(server.call_args.args[0].url), "http://h/r/F/S/R/loglikelihoods")
                self.assertEqual(
                    json.loads(server.call_args.args[0].content),
                    {"messages": messages, "continuations": continuations},
                )

    async def test_hands_back_the_last_hidden_states_after_each_part(self):
        cases = [
            (
                "two_parts",
                ["It rained.\n", "Ann left.\n"],
                [["Ann | is | wet", "Ann | is | home"], ["Ann | is | home"]],
                [
                    b'{"last_hidden_states": [[0.5, -1.0], [0.25, 2.0]]}\n',
                    b'{"last_hidden_states": [[-0.75, 1.5]]}\n',
                ],
                [[[0.5, -1.0], [0.25, 2.0]], [[-0.75, 1.5]]],
            ),
            (
                "a_part_without_continuations",
                ["It rained.\n"],
                [[]],
                [b'{"last_hidden_states": []}\n'],
                [[]],
            ),
            (
                "no_parts",
                [],
                [],
                [],
                [],
            ),
        ]
        for name, parts, continuations_of_each_part, lines, expected in cases:
            with self.subTest(name):
                server = _served(lines)
                model = ServedCompletingModel("http://h/r/F/S/R", "m")
                with mock.patch.object(
                    completion.httpx,
                    "AsyncClient",
                    functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(server)),
                ):
                    result = [
                        last_hidden_states
                        async for last_hidden_states in model.last_hidden_states(
                            parts, continuations_of_each_part
                        )
                    ]

                self.assertEqual(result, expected)
                self.assertEqual(
                    str(server.call_args.args[0].url), "http://h/r/F/S/R/last_hidden_states"
                )
                self.assertEqual(
                    json.loads(server.call_args.args[0].content),
                    {"parts": parts, "continuations_of_each_part": continuations_of_each_part},
                )

    async def test_a_model_that_hides_its_hidden_states_refuses(self):
        server = mock.Mock(return_value=httpx.Response(501))
        model = ServedCompletingModel("http://h/r/F/S/R", "claude-sonnet-5")

        with mock.patch.object(
            completion.httpx,
            "AsyncClient",
            functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(server)),
        ):
            with self.assertRaises(httpx.HTTPStatusError):
                [
                    last_hidden_states
                    async for last_hidden_states in model.last_hidden_states(
                        ["It rained.\n"], [["Ann | is | wet"]]
                    )
                ]
