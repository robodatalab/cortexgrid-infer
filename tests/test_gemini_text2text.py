"""Tests for cortexgrid_infer.models.gemini.text2text, registered as a Hosted entry."""

from __future__ import annotations

import asyncio
import unittest
from unittest import mock

import cortexgrid
from google.genai import types

from cortexgrid_infer.completion import ServedCompletingModel, encode_tool_call
from cortexgrid_infer.models.gemini.text2text import (
    SKIP_THOUGHT_SIGNATURE,
    GeminiText2Text,
    to_gemini_contents,
    to_gemini_tools,
)
from cortexgrid_infer.registry import Hosted

BASE = "cortexgrid_infer.models.gemini.base"


class TestHostedGeminiText2Text(unittest.TestCase):
    def test_names_the_model_for_the_registry(self):
        entry = Hosted("gemini-2.5-pro", GeminiText2Text)
        self.assertEqual((entry.family, entry.suffix), ("gemini-2.5", "pro"))

    def test_client_is_the_shared_completion_client(self):
        # The serve app speaks the same /complete protocol as Text2Text, so
        # there is nothing Gemini-specific left on the client side.
        model = Hosted("gemini-2.5-pro", GeminiText2Text).client("http://h/r/F/S/R")
        self.assertIsInstance(model, ServedCompletingModel)
        self.assertEqual(model.name, "gemini-2.5-pro")

    def test_asks_for_no_hardware(self):
        self.assertEqual(
            Hosted("gemini-2.5-pro", GeminiText2Text).requirements(),
            cortexgrid.ModelRequirements(),
        )


def _chunk(*parts: types.Part) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))]
    )


async def _aiter(items):
    for item in items:
        yield item


class TestComplete(unittest.TestCase):
    def _complete(self, chunks, body):
        with (
            mock.patch(f"{BASE}.cortexgrid") as mock_cortexgrid,
            mock.patch(f"{BASE}.genai") as mock_genai,
        ):
            mock_cortexgrid.model_config.return_value = {
                "model": "gemini-2.5-pro",
                "api_key_secret": "GEMINI_API_KEY",
            }
            generate = mock.AsyncMock(return_value=_aiter(chunks))
            mock_genai.Client.return_value.aio.models.generate_content_stream = generate
            deployment = GeminiText2Text("gemini-2.5", "pro", "imported")

            async def run() -> bytes:
                response = await GeminiText2Text.complete(deployment, body)
                return b"".join([piece async for piece in response.body_iterator])

            return asyncio.run(run()).decode("utf-8"), generate

    def test_streams_text_and_re_encodes_function_calls(self):
        call = types.FunctionCall(name="add", args={"a": 1, "b": 2})

        text, _generate = self._complete(
            [
                _chunk(types.Part(text="let me ")),
                _chunk(types.Part(text="check")),
                _chunk(types.Part(function_call=call)),
            ],
            {"messages": [{"role": "user", "content": "1+2?"}]},
        )

        self.assertEqual(text, "let me check" + encode_tool_call("add", {"a": 1, "b": 2}))

    def test_leaves_out_thoughts_and_empty_chunks(self):
        text, _generate = self._complete(
            [
                _chunk(types.Part(text="pondering", thought=True)),
                types.GenerateContentResponse(candidates=[]),
                _chunk(types.Part(text="4")),
            ],
            {"messages": [{"role": "user", "content": "2+2?"}]},
        )

        self.assertEqual(text, "4")

    def test_sends_the_configured_model_and_request_settings(self):
        _text, generate = self._complete(
            [],
            {
                "messages": [
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "hi"},
                ],
                "tools": [{"function": {"name": "now"}}],
                "max_new_tokens": 64,
                "temperature": 0.2,
            },
        )

        kwargs = generate.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-2.5-pro")
        self.assertEqual(kwargs["contents"], [{"role": "user", "parts": [{"text": "hi"}]}])
        self.assertEqual(
            kwargs["config"],
            {
                "max_output_tokens": 64,
                "temperature": 0.2,
                "system_instruction": "be terse",
                "tools": [{"function_declarations": [{"name": "now", "description": ""}]}],
            },
        )


class TestMessageConversion(unittest.TestCase):
    def test_system_message_is_lifted_out(self):
        system, contents = to_gemini_contents(
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ]
        )

        self.assertEqual(system, "be terse")
        self.assertEqual(contents, [{"role": "user", "parts": [{"text": "hi"}]}])

    def test_assistant_tool_calls_become_function_call_parts(self):
        _system, contents = to_gemini_contents(
            [
                {
                    "role": "assistant",
                    "content": "let me check",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "add", "arguments": {"a": 1}},
                        }
                    ],
                }
            ]
        )

        self.assertEqual(
            contents,
            [
                {
                    "role": "model",
                    "parts": [
                        {"text": "let me check"},
                        {
                            "function_call": {"id": "call_1", "name": "add", "args": {"a": 1}},
                            "thought_signature": SKIP_THOUGHT_SIGNATURE,
                        },
                    ],
                }
            ],
        )

    def test_replayed_calls_skip_the_thought_signature_check(self):
        # Gemini 3 rejects a replayed call without its signature, and the real
        # one is lost on the way through the client. Like Gemini's own, the
        # signature goes on the first call of a parallel batch only.
        _system, contents = to_gemini_contents(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "add", "arguments": {}}},
                        {"id": "call_2", "function": {"name": "mul", "arguments": {}}},
                    ],
                }
            ]
        )

        first, second = (types.Part.model_validate(p) for p in contents[0]["parts"])
        self.assertEqual(
            first.model_dump(mode="json")["thought_signature"],
            "skip_thought_signature_validator",
        )
        self.assertIsNone(second.thought_signature)

    def test_tool_result_answers_its_call_by_name(self):
        # A tool message carries only the call's id; Gemini wants the name.
        _system, contents = to_gemini_contents(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "add", "arguments": {}}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "3"},
            ]
        )

        self.assertEqual(
            contents[1],
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "id": "call_1",
                            "name": "add",
                            "response": {"output": "3"},
                        }
                    }
                ],
            },
        )

    def test_consecutive_tool_results_share_one_user_turn(self):
        # Gemini expects the results of a parallel call in a single user turn,
        # not one turn each.
        _system, contents = to_gemini_contents(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "add", "arguments": {}}},
                        {"id": "call_2", "function": {"name": "mul", "arguments": {}}},
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "3"},
                {"role": "tool", "tool_call_id": "call_2", "content": "4"},
            ]
        )

        self.assertEqual(len(contents), 2)
        self.assertEqual(len(contents[1]["parts"]), 2)

    def test_tool_result_after_text_starts_a_new_turn(self):
        _system, contents = to_gemini_contents(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "add", "arguments": {}}}
                    ],
                },
                {"role": "user", "content": "hi"},
                {"role": "tool", "tool_call_id": "call_1", "content": "3"},
            ]
        )

        self.assertEqual(len(contents), 3)

    def test_contents_are_what_the_sdk_accepts(self):
        _system, contents = to_gemini_contents(
            [
                {"role": "user", "content": "1+2?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "add", "arguments": {"a": 1}}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "3"},
            ]
        )

        for content in contents:
            types.Content.model_validate(content)


class TestToolConversion(unittest.TestCase):
    def test_no_tools_stays_none(self):
        self.assertIsNone(to_gemini_tools(None))
        self.assertIsNone(to_gemini_tools([]))

    def test_openai_specs_become_one_tool_of_declarations(self):
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}

        result = to_gemini_tools(
            [
                {"function": {"name": "add", "description": "sum", "parameters": schema}},
                {"function": {"name": "now"}},
            ]
        )

        self.assertEqual(
            result,
            [
                {
                    "function_declarations": [
                        {"name": "add", "description": "sum", "parameters_json_schema": schema},
                        {"name": "now", "description": ""},
                    ]
                }
            ],
        )
        types.Tool.model_validate(result[0])
