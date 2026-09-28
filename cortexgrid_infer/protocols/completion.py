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
import json
from typing import Any, Sequence

import httpx

from cortexgrid_infer.core import CompletingModel, CompletionChunk, Message, Tool
from cortexgrid_infer.utils import build_tool_map, normalize_tools

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

    async def loglikelihoods(
        self, messages: list[Message], continuations: list[str]
    ) -> list[float]:
        body = {"messages": messages, "continuations": continuations}
        loglikelihoods_url = f"{self.url}/loglikelihoods"
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(loglikelihoods_url, json=body)
        response.raise_for_status()
        answered = response.json()
        return answered["loglikelihoods"]

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
        tools_by_name = build_tool_map(tools)
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
        complete_url = f"{self.url}/complete"
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", complete_url, json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    answered = json.loads(line)
                    chunk = CompletionChunk.from_wire(answered, tools_by_name)
                    yield chunk

    async def last_hidden_states(
        self, parts: list[str], continuations_of_each_part: list[list[str]]
    ) -> AsyncIterator[list[list[float]]]:
        body = {"parts": parts, "continuations_of_each_part": continuations_of_each_part}
        last_hidden_states_url = f"{self.url}/last_hidden_states"
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", last_hidden_states_url, json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    answered = json.loads(line)
                    yield answered["last_hidden_states"]
