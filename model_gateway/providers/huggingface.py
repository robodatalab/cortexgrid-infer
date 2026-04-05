"""HuggingFace model deployment with streaming and tool call parsing."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import partial
import itertools
import json
import re
from typing import Any, Callable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_gateway.core import (
    CompletingModel,
    Message,
    Tool,
    ToolCall,
    CompletionChunk,
    register_provider,
)
from model_gateway.device import detect_device
from model_gateway.utils import build_tool_map, normalize_tools


_tool_call_id_counter = itertools.count()

TOOL_CALL_OPENERS = ["<tool_call>", "<|tool_call|>", "```tool_call"]


def _find_opener(text: str) -> int | None:
    """Find the earliest tool call opener position in text, or None."""
    earliest = None
    for opener in TOOL_CALL_OPENERS:
        idx = text.find(opener)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    return earliest


def _split_at_potential_prefix(text: str) -> tuple[str, str]:
    """Split text into (safe_to_stream, potential_opener_prefix).

    The second part is a suffix that could be the beginning of a tool call
    opening tag, so it must be held back until more text arrives.
    """
    for opener in TOOL_CALL_OPENERS:
        for length in range(1, len(opener)):
            if text.endswith(opener[:length]):
                return text[:-length], text[-length:]
    return text, ""


TOOL_CALL_PATTERNS = [
    re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL),
    re.compile(r"<\|tool_call\|>\s*(\{.*?\})\s*<\|/tool_call\|>", re.DOTALL),
    re.compile(r"```tool_call\s*(\{.*?\})\s*```", re.DOTALL),
]


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
class HuggingFaceModel(CompletingModel):
    model: Any
    tokenizer: Any
    device: str

    @property
    def name(self) -> str:
        return self.tokenizer.name_or_path

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
        from threading import Thread
        from transformers import TextIteratorStreamer

        model = self.model
        tokenizer = self.tokenizer
        tool_specs = normalize_tools(tools)
        tool_map = build_tool_map(tools)

        prompt = tokenizer.apply_chat_template(
            messages,
            tools=tool_specs,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        error: list[BaseException] = []
        loop = asyncio.get_running_loop()

        def generate() -> None:
            try:
                streamer = TextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )
                generation_config = {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    "do_sample": temperature > 0,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "streamer": streamer,
                    **kwargs,
                }

                thread = Thread(
                    target=lambda: model.generate(**inputs, **generation_config)
                )
                thread.start()

                for text in streamer:
                    loop.call_soon_threadsafe(queue.put_nowait, text)

                thread.join()
            except BaseException as e:
                error.append(e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, generate)

        pending = ""
        in_tool_call = False
        tool_call_text = ""

        while True:
            text = await queue.get()
            if text is None:
                break
            if not text:
                continue

            if in_tool_call:
                tool_call_text += text
                continue

            pending += text

            opener_pos = _find_opener(pending)
            if opener_pos is not None:
                before = pending[:opener_pos]
                if before:
                    yield CompletionChunk(content=before)
                tool_call_text = pending[opener_pos:]
                in_tool_call = True
                pending = ""
                continue

            safe, held = _split_at_potential_prefix(pending)
            if safe:
                yield CompletionChunk(content=safe)
            pending = held

        if error:
            raise error[0]

        if pending:
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


def deploy_huggingface(model_id: str) -> HuggingFaceModel:
    device = detect_device()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model.to(device)  # type: ignore

    return HuggingFaceModel(model=model, tokenizer=tokenizer, device=str(device))


register_provider("", deploy_huggingface)
