"""The wire protocol between a cortexgrid-served completion app and its client.

A completion serve app streams plain text from `POST /complete`, with tool calls
inlined in the text as `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.
That is the format an instruct-tuned causal LM emits on its own, so a serve app
wrapping one forwards its tokens untouched; a serve app wrapping a provider with
structured tool calls encodes them into the same form with `encode_tool_call`.

Either way the client is the same `ServedCompletingModel`, which streams text
back as `CompletionChunk`s and reassembles the tool calls.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import partial
import itertools
import json
import re
from typing import Any, Callable, Sequence

import httpx

from cortexgrid_infer.core import (
    CompletingModel,
    CompletionChunk,
    Message,
    Tool,
    ToolCall,
)
from cortexgrid_infer.utils import build_tool_map, normalize_tools

_tool_call_id_counter = itertools.count()

TOOL_CALL_OPENERS = ["<tool_call>", "<|tool_call|>", "```tool_call"]

THINKING_OPENER = "<think>"
THINKING_CLOSER = "</think>"

TOOL_CALL_PATTERNS = [
    re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL),
    re.compile(r"<\|tool_call\|>\s*(\{.*?\})\s*<\|/tool_call\|>", re.DOTALL),
    re.compile(r"```tool_call\s*(\{.*?\})\s*```", re.DOTALL),
]


def encode_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Render a tool call in the form the client parses back out of the stream.

    For serve apps whose upstream reports tool calls as structured data rather
    than as text the model generated - they are re-encoded here so one client
    handles every completion app."""
    return f'<tool_call>{json.dumps({"name": name, "arguments": arguments})}</tool_call>'


def encode_thinking(text: str) -> str:
    return f"{THINKING_OPENER}{text}{THINKING_CLOSER}"


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


def parse_tool_calls(
    text: str, tool_map: dict[str, Callable[..., Any]]
) -> tuple[str, list[ToolCall]]:
    """Parse tool calls from generated text and bind them to their functions."""
    tool_calls: list[ToolCall] = []
    clean_text = text

    for pattern in TOOL_CALL_PATTERNS:
        matches = pattern.findall(text)
        for match in matches:
            try:
                data = json.loads(match)
                name = data["name"]
                arguments = data.get("arguments", {})
                if name in tool_map:
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{next(_tool_call_id_counter)}",
                            name=name,
                            arguments=arguments,
                            _func=partial(tool_map[name], **arguments),
                        )
                    )
            except (json.JSONDecodeError, KeyError):
                continue
        clean_text = pattern.sub("", clean_text)

    return clean_text.strip(), tool_calls


@dataclass
class ServedCompletingModel(CompletingModel):
    """Client for a completion app cortexgrid has deployed at `url`.

    Provider-agnostic: every completion serve app speaks the protocol above, so
    which model is behind it only shows up in `model_id`."""

    url: str
    model_id: str

    @property
    def name(self) -> str:
        return self.model_id

    async def complete(
        self,
        messages: list[Message],
        tools: Sequence[Tool] | None = None,
        max_new_tokens: int = 16 * 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionChunk]:
        tool_specs = normalize_tools(tools)
        tool_map = build_tool_map(tools)

        body: dict[str, Any] = {
            "messages": messages,
            "tools": tool_specs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            **kwargs,
        }

        pending = ""
        in_thinking = False
        in_tool_call = False
        tool_call_text = ""
        openers = [THINKING_OPENER, *TOOL_CALL_OPENERS]

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", f"{self.url}/complete", json=body
            ) as response:
                response.raise_for_status()
                async for text in response.aiter_text():
                    if not text:
                        continue

                    if in_tool_call:
                        tool_call_text += text
                        continue

                    pending += text

                    while pending and not in_tool_call:
                        if in_thinking:
                            closer_pos = pending.find(THINKING_CLOSER)
                            if closer_pos == -1:
                                thought, pending = _split_at_potential_prefix(
                                    pending, [THINKING_CLOSER]
                                )
                                if thought:
                                    yield CompletionChunk(thinking=thought)
                                break
                            if closer_pos:
                                yield CompletionChunk(thinking=pending[:closer_pos])
                            pending = pending[closer_pos + len(THINKING_CLOSER) :]
                            in_thinking = False
                            continue

                        opener_pos = _find_opener(pending, openers)
                        if opener_pos is None:
                            safe, pending = _split_at_potential_prefix(pending, openers)
                            if safe:
                                yield CompletionChunk(content=safe)
                            break

                        before = pending[:opener_pos]
                        if before:
                            yield CompletionChunk(content=before)
                        if pending.startswith(THINKING_OPENER, opener_pos):
                            pending = pending[opener_pos + len(THINKING_OPENER) :]
                            in_thinking = True
                        else:
                            tool_call_text = pending[opener_pos:]
                            in_tool_call = True
                            pending = ""

        if pending and in_thinking:
            yield CompletionChunk(thinking=pending)
        elif pending:
            yield CompletionChunk(content=pending)

        if tool_call_text:
            clean_text, tool_calls = parse_tool_calls(tool_call_text, tool_map)
            if clean_text:
                yield CompletionChunk(content=clean_text)
            if tool_calls:
                yield CompletionChunk(tool_calls=tool_calls)
            else:
                yield CompletionChunk(finish_reason="stop")
        else:
            yield CompletionChunk(finish_reason="stop")
