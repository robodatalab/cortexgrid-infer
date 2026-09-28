"""Tests for cortexgrid_infer.serve_apps.streamed_completion."""

from __future__ import annotations

import itertools
import unittest
from unittest import mock

from cortexgrid_infer.serve_apps import streamed_completion
from cortexgrid_infer.serve_apps.streamed_completion import (
    StreamedCompletionWriter,
    completion_chunk,
    tool_calls_in,
)


class TestToolCallsIn(unittest.TestCase):
    def test_takes_every_tool_call_out_of_the_text(self):
        cases = [
            (
                "one_call",
                '<tool_call>{"name": "add", "arguments": {"a": 2, "b": 3}}</tool_call>',
                ("", [{"id": "call_0", "name": "add", "arguments": {"a": 2, "b": 3}}]),
            ),
            (
                "a_call_among_text",
                'Sure. <tool_call>{"name": "now", "arguments": {}}</tool_call> Done.',
                ("Sure.  Done.", [{"id": "call_0", "name": "now", "arguments": {}}]),
            ),
            (
                "two_calls_in_different_notations",
                '<tool_call>{"name": "a"}</tool_call><|tool_call|>{"name": "b", "arguments": {"x": 1}}<|/tool_call|>',
                (
                    "",
                    [
                        {"id": "call_0", "name": "a", "arguments": {}},
                        {"id": "call_1", "name": "b", "arguments": {"x": 1}},
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
        add_call = {"id": "call_0", "name": "add", "arguments": {"a": 2, "b": 3}}
        cases = [
            (
                "plain_text",
                ["hello", " ", "world"],
                [
                    completion_chunk(content="hello"),
                    completion_chunk(content=" "),
                    completion_chunk(content="world"),
                    completion_chunk(finish_reason="stop"),
                ],
            ),
            (
                "the_thinking_apart_from_the_answer",
                ["<think>", "Let me", " see.", "</think>", "\n\nThe answer."],
                [
                    completion_chunk(thinking="Let me"),
                    completion_chunk(thinking=" see."),
                    completion_chunk(content="\n\nThe answer."),
                    completion_chunk(finish_reason="stop"),
                ],
            ),
            (
                "thinking_markers_split_across_pieces",
                ["<thi", "nk>reason", "ing</th", "ink>answer"],
                [
                    completion_chunk(thinking="reason"),
                    completion_chunk(thinking="ing"),
                    completion_chunk(content="answer"),
                    completion_chunk(finish_reason="stop"),
                ],
            ),
            (
                "a_reply_cut_off_while_thinking",
                ["<think>still ", "going"],
                [
                    completion_chunk(thinking="still "),
                    completion_chunk(thinking="going"),
                    completion_chunk(finish_reason="stop"),
                ],
            ),
            (
                "a_tool_call_after_text",
                ["Hello ", "<tool_call>", '{"name": "add", "arguments": {"a": 2, "b": 3}}', "</tool_call>"],
                [
                    completion_chunk(content="Hello "),
                    completion_chunk(tool_calls=[add_call]),
                ],
            ),
            (
                "a_tool_call_after_thinking",
                [
                    "<think>I should add.</think>",
                    '<tool_call>{"name": "add", "arguments": {"a": 2, "b": 3}}</tool_call>',
                ],
                [
                    completion_chunk(thinking="I should add."),
                    completion_chunk(tool_calls=[add_call]),
                ],
            ),
            (
                "nothing_generated",
                [],
                [completion_chunk(finish_reason="stop")],
            ),
        ]
        for name, pieces, expected_lines in cases:
            with self.subTest(name), mock.patch.object(
                streamed_completion, "_tool_call_id_counter", itertools.count()
            ):
                writer = StreamedCompletionWriter()

                lines = [line for piece in pieces for line in writer.read(piece)]
                lines += writer.finish()

                self.assertEqual(lines, expected_lines)


class TestStreamedCompletionWriterReadsSeparatedOutput(unittest.TestCase):
    def test_reads_content_as_it_is(self):
        cases = [
            ("a_word", "hello", completion_chunk(content="hello")),
            ("thinking_markup", "<think>quoted</think>", completion_chunk(content="<think>quoted</think>")),
            (
                "tool_call_markup",
                '<tool_call>{"name": "add"}</tool_call>',
                completion_chunk(content='<tool_call>{"name": "add"}</tool_call>'),
            ),
        ]
        for name, text, expected_line in cases:
            with self.subTest(name):
                writer = StreamedCompletionWriter()

                line = writer.read_content(text)
                finishing_lines = writer.finish()

                self.assertEqual(line, expected_line)
                self.assertEqual(finishing_lines, [completion_chunk(finish_reason="stop")])

    def test_reads_thinking_as_it_is(self):
        cases = [
            ("a_thought", "pondering", completion_chunk(thinking="pondering")),
            ("thinking_markup", "<think>nested</think>", completion_chunk(thinking="<think>nested</think>")),
        ]
        for name, text, expected_line in cases:
            with self.subTest(name):
                writer = StreamedCompletionWriter()

                line = writer.read_thinking(text)
                finishing_lines = writer.finish()

                self.assertEqual(line, expected_line)
                self.assertEqual(finishing_lines, [completion_chunk(finish_reason="stop")])

    def test_reads_a_tool_call_and_does_not_stop_after_it(self):
        cases = [
            (
                "the_id_the_model_gave",
                "add",
                {"a": 2, "b": 3},
                "toolu_1",
                completion_chunk(tool_calls=[{"id": "toolu_1", "name": "add", "arguments": {"a": 2, "b": 3}}]),
            ),
            (
                "a_new_id_when_the_model_gave_none",
                "now",
                {},
                None,
                completion_chunk(tool_calls=[{"id": "call_0", "name": "now", "arguments": {}}]),
            ),
        ]
        for name, tool_name, arguments, call_id, expected_line in cases:
            with self.subTest(name), mock.patch.object(
                streamed_completion, "_tool_call_id_counter", itertools.count()
            ):
                writer = StreamedCompletionWriter()

                line = writer.read_tool_call(tool_name, arguments, call_id)
                finishing_lines = writer.finish()

                self.assertEqual(line, expected_line)
                self.assertEqual(finishing_lines, [])
