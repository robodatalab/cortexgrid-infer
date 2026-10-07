"""Tests for cortexgrid_infer.serve_apps.streamed_completion."""

from __future__ import annotations

import itertools
import unittest
from unittest import mock

from cortexgrid_infer.core import CompletionChunk, ToolCall
from cortexgrid_infer.serve_apps import streamed_completion
from cortexgrid_infer.serve_apps.streamed_completion import (
    StreamedCompletionWriter,
    tool_calls_in,
)


class TestToolCallsIn(unittest.TestCase):
    def test_takes_every_tool_call_out_of_the_text(self):
        cases = [
            (
                "one_call",
                '<tool_call>{"name": "add", "arguments": {"a": 2, "b": 3}}</tool_call>',
                ("", [ToolCall(id="call_0", name="add", arguments={"a": 2, "b": 3})]),
            ),
            (
                "a_call_among_text",
                'Sure. <tool_call>{"name": "now", "arguments": {}}</tool_call> Done.',
                ("Sure.  Done.", [ToolCall(id="call_0", name="now", arguments={})]),
            ),
            (
                "two_calls_in_different_notations",
                '<tool_call>{"name": "a"}</tool_call><|tool_call|>{"name": "b", "arguments": {"x": 1}}<|/tool_call|>',
                (
                    "",
                    [
                        ToolCall(id="call_0", name="a", arguments={}),
                        ToolCall(id="call_1", name="b", arguments={"x": 1}),
                    ],
                ),
            ),
            (
                "a_malformed_call_is_dropped",
                "<tool_call>{not json}</tool_call>",
                ("", []),
            ),
            (
                "a_call_without_a_name_is_dropped",
                '<tool_call>{"arguments": {}}</tool_call>',
                ("", []),
            ),
            (
                "no_calls",
                "just words",
                ("just words", []),
            ),
        ]
        for name, text, expected in cases:
            with self.subTest(name), mock.patch.object(
                streamed_completion, "_tool_call_id_counter", itertools.count()
            ):
                self.assertEqual(tool_calls_in(text), expected)


class TestStreamedCompletionWriter(unittest.TestCase):
    def test_turns_the_generated_text_into_finished_chunks(self):
        add_call = ToolCall(id="call_0", name="add", arguments={"a": 2, "b": 3})
        cases = [
            (
                "plain_text",
                ["hello", " ", "world"],
                [
                    CompletionChunk(content="hello"),
                    CompletionChunk(content=" "),
                    CompletionChunk(content="world"),
                    CompletionChunk(finish_reason="stop"),
                ],
            ),
            (
                "the_thinking_apart_from_the_answer",
                ["<think>", "Let me", " see.", "</think>", "\n\nThe answer."],
                [
                    CompletionChunk(thinking="Let me"),
                    CompletionChunk(thinking=" see."),
                    CompletionChunk(content="\n\nThe answer."),
                    CompletionChunk(finish_reason="stop"),
                ],
            ),
            (
                "thinking_markers_split_across_pieces",
                ["<thi", "nk>reason", "ing</th", "ink>answer"],
                [
                    CompletionChunk(thinking="reason"),
                    CompletionChunk(thinking="ing"),
                    CompletionChunk(content="answer"),
                    CompletionChunk(finish_reason="stop"),
                ],
            ),
            (
                "a_reply_cut_off_while_thinking",
                ["<think>still ", "going"],
                [
                    CompletionChunk(thinking="still "),
                    CompletionChunk(thinking="going"),
                    CompletionChunk(finish_reason="stop"),
                ],
            ),
            (
                "a_tool_call_after_text",
                ["Hello ", "<tool_call>", '{"name": "add", "arguments": {"a": 2, "b": 3}}', "</tool_call>"],
                [
                    CompletionChunk(content="Hello "),
                    CompletionChunk(tool_calls=[add_call]),
                ],
            ),
            (
                "a_tool_call_after_thinking",
                [
                    "<think>I should add.</think>",
                    '<tool_call>{"name": "add", "arguments": {"a": 2, "b": 3}}</tool_call>',
                ],
                [
                    CompletionChunk(thinking="I should add."),
                    CompletionChunk(tool_calls=[add_call]),
                ],
            ),
            (
                "nothing_generated",
                [],
                [CompletionChunk(finish_reason="stop")],
            ),
        ]
        for name, pieces, expected_chunks in cases:
            with self.subTest(name), mock.patch.object(
                streamed_completion, "_tool_call_id_counter", itertools.count()
            ):
                writer = StreamedCompletionWriter()

                chunks = [chunk for piece in pieces for chunk in writer.read(piece)]
                chunks += writer.finish()

                self.assertEqual(chunks, expected_chunks)


class TestStreamedCompletionWriterReadsSeparatedOutput(unittest.TestCase):
    def test_reads_content_as_it_is(self):
        cases = [
            ("a_word", "hello", CompletionChunk(content="hello")),
            ("thinking_markup", "<think>quoted</think>", CompletionChunk(content="<think>quoted</think>")),
            (
                "tool_call_markup",
                '<tool_call>{"name": "add"}</tool_call>',
                CompletionChunk(content='<tool_call>{"name": "add"}</tool_call>'),
            ),
        ]
        for name, text, expected_chunk in cases:
            with self.subTest(name):
                writer = StreamedCompletionWriter()

                chunk = writer.read_content(text)
                finishing_chunks = writer.finish()

                self.assertEqual(chunk, expected_chunk)
                self.assertEqual(finishing_chunks, [CompletionChunk(finish_reason="stop")])

    def test_reads_thinking_as_it_is(self):
        cases = [
            ("a_thought", "pondering", CompletionChunk(thinking="pondering")),
            ("thinking_markup", "<think>nested</think>", CompletionChunk(thinking="<think>nested</think>")),
        ]
        for name, text, expected_chunk in cases:
            with self.subTest(name):
                writer = StreamedCompletionWriter()

                chunk = writer.read_thinking(text)
                finishing_chunks = writer.finish()

                self.assertEqual(chunk, expected_chunk)
                self.assertEqual(finishing_chunks, [CompletionChunk(finish_reason="stop")])

    def test_reads_a_tool_call_and_does_not_stop_after_it(self):
        cases = [
            (
                "the_id_the_model_gave",
                "add",
                {"a": 2, "b": 3},
                "toolu_1",
                CompletionChunk(tool_calls=[ToolCall(id="toolu_1", name="add", arguments={"a": 2, "b": 3})]),
            ),
            (
                "a_new_id_when_the_model_gave_none",
                "now",
                {},
                None,
                CompletionChunk(tool_calls=[ToolCall(id="call_0", name="now", arguments={})]),
            ),
        ]
        for name, tool_name, arguments, call_id, expected_chunk in cases:
            with self.subTest(name), mock.patch.object(
                streamed_completion, "_tool_call_id_counter", itertools.count()
            ):
                writer = StreamedCompletionWriter()

                chunk = writer.read_tool_call(tool_name, arguments, call_id)
                finishing_chunks = writer.finish()

                self.assertEqual(chunk, expected_chunk)
                self.assertEqual(finishing_chunks, [])
