from __future__ import annotations

import itertools
import json
import re
from typing import Any, Sequence

from cortexgrid_infer.core import CompletionChunk, ToolCall

_tool_call_id_counter = itertools.count()

TOOL_CALL_OPENERS = ["<tool_call>", "<|tool_call|>", "```tool_call"]

THINKING_OPENER = "<think>"
THINKING_CLOSER = "</think>"

TOOL_CALL_PATTERNS = [
    re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL),
    re.compile(r"<\|tool_call\|>\s*(\{.*?\})\s*<\|/tool_call\|>", re.DOTALL),
    re.compile(r"```tool_call\s*(\{.*?\})\s*```", re.DOTALL),
]


def new_tool_call_id() -> str:
    number = next(_tool_call_id_counter)
    return f"call_{number}"


def _find_opener(text: str, openers: Sequence[str]) -> int | None:
    """Find the earliest position of any of `openers` in text, or None."""
    earliest = None
    for opener in openers:
        idx = text.find(opener)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    return earliest


def _split_at_potential_prefix(text: str, openers: Sequence[str]) -> tuple[str, str]:
    """Split text into (safe_to_stream, potential_opener_prefix).

    The second part is a suffix that could be the beginning of one of
    `openers`, so it must be held back until more text arrives.
    """
    for opener in openers:
        for length in range(1, len(opener)):
            if text.endswith(opener[:length]):
                return text[:-length], text[-length:]
    return text, ""


def tool_calls_in(text: str) -> tuple[str, list[ToolCall]]:
    tool_calls = []
    clean_text = text
    for pattern in TOOL_CALL_PATTERNS:
        matches = pattern.findall(text)
        for match in matches:
            try:
                data = json.loads(match)
                name = data["name"]
            except (json.JSONDecodeError, KeyError):
                continue
            arguments = data.get("arguments", {})
            call_id = new_tool_call_id()
            tool_call = ToolCall(id=call_id, name=name, arguments=arguments)
            tool_calls.append(tool_call)
        clean_text = pattern.sub("", clean_text)
    stripped = clean_text.strip()
    return stripped, tool_calls


class StreamedCompletionWriter:
    def __init__(self) -> None:
        self._pending = ""
        self._in_thinking = False
        self._in_tool_call = False
        self._tool_call_text = ""
        self._made_a_tool_call = False

    def read_content(self, text: str) -> CompletionChunk:
        return CompletionChunk(content=text)

    def read_thinking(self, text: str) -> CompletionChunk:
        return CompletionChunk(thinking=text)

    def read_tool_call(
        self, name: str, arguments: dict[str, Any], call_id: str | None = None
    ) -> CompletionChunk:
        self._made_a_tool_call = True
        tool_call_id = call_id or new_tool_call_id()
        tool_call = ToolCall(id=tool_call_id, name=name, arguments=arguments)
        return CompletionChunk(tool_calls=[tool_call])

    def read(self, text: str) -> list[CompletionChunk]:
        if not text:
            return []
        if self._in_tool_call:
            self._tool_call_text += text
            return []
        self._pending += text
        chunks: list[CompletionChunk] = []
        openers = [THINKING_OPENER, *TOOL_CALL_OPENERS]
        while self._pending and not self._in_tool_call:
            if self._in_thinking:
                closer_pos = self._pending.find(THINKING_CLOSER)
                if closer_pos == -1:
                    thought, self._pending = _split_at_potential_prefix(
                        self._pending, [THINKING_CLOSER]
                    )
                    if thought:
                        chunks.append(CompletionChunk(thinking=thought))
                    break
                if closer_pos:
                    thought = self._pending[:closer_pos]
                    chunks.append(CompletionChunk(thinking=thought))
                self._pending = self._pending[closer_pos + len(THINKING_CLOSER) :]
                self._in_thinking = False
                continue

            opener_pos = _find_opener(self._pending, openers)
            if opener_pos is None:
                safe, self._pending = _split_at_potential_prefix(self._pending, openers)
                if safe:
                    chunks.append(CompletionChunk(content=safe))
                break

            before = self._pending[:opener_pos]
            if before:
                chunks.append(CompletionChunk(content=before))
            if self._pending.startswith(THINKING_OPENER, opener_pos):
                self._pending = self._pending[opener_pos + len(THINKING_OPENER) :]
                self._in_thinking = True
            else:
                self._tool_call_text = self._pending[opener_pos:]
                self._in_tool_call = True
                self._pending = ""
        return chunks

    def finish(self) -> list[CompletionChunk]:
        chunks: list[CompletionChunk] = []
        if self._pending and self._in_thinking:
            chunks.append(CompletionChunk(thinking=self._pending))
        elif self._pending:
            chunks.append(CompletionChunk(content=self._pending))
        clean_text, tool_calls = tool_calls_in(self._tool_call_text)
        if clean_text:
            chunks.append(CompletionChunk(content=clean_text))
        if tool_calls:
            chunks.append(CompletionChunk(tool_calls=tool_calls))
        elif not self._made_a_tool_call:
            chunks.append(CompletionChunk(finish_reason="stop"))
        return chunks
