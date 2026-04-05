"""Anthropic model deployment with streaming support."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from collections.abc import AsyncIterator
from typing import Any, Sequence

from model_gateway.core import (
    CompletingModel,
    Message,
    Tool,
    ToolCall,
    ToolSpec,
    CompletionChunk,
    register_provider,
)
from model_gateway.utils import build_tool_map, normalize_tools

from anthropic import AsyncAnthropic, Anthropic, NotFoundError
from dotenv import load_dotenv


_MODEL_PROVIDER_PREFIX = "Anthropic/"


def _to_anthropic_messages(
    messages: list[Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert OpenAI-style messages to Anthropic format."""
    system_prompt: str | None = None
    anthropic_messages: list[dict[str, Any]] = []

    for msg in messages:
        role = msg["role"]

        if role == "system":
            system_prompt = msg["content"]

        elif role == "user":
            anthropic_messages.append({"role": "user", "content": msg["content"]})

        elif role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            if msg.get("content"):
                content_blocks.append({"type": "text", "text": msg["content"]})
            for tc in msg.get("tool_calls", []):
                func = tc["function"]
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": func["name"],
                        "input": func["arguments"],
                    }
                )
            anthropic_messages.append({"role": "assistant", "content": content_blocks})

        elif role == "tool":
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": msg["tool_call_id"],
                "content": msg["content"],
            }
            if (
                anthropic_messages
                and anthropic_messages[-1]["role"] == "user"
                and isinstance(anthropic_messages[-1]["content"], list)
                and anthropic_messages[-1]["content"]
                and anthropic_messages[-1]["content"][0].get("type") == "tool_result"
            ):
                anthropic_messages[-1]["content"].append(tool_result_block)
            else:
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [tool_result_block],
                    }
                )

    return system_prompt, anthropic_messages


def _to_anthropic_tools(
    tool_specs: list[ToolSpec] | None,
) -> list[dict[str, Any]] | None:
    """Convert OpenAI-style tool specs to Anthropic format."""
    if not tool_specs:
        return None
    result = []
    for spec in tool_specs:
        func = spec["function"]
        result.append(
            {
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
        )
    return result


@dataclass
class AnthropicModel(CompletingModel):
    client: Any
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
        **kwargs: Any,
    ) -> AsyncIterator[CompletionChunk]:
        tool_specs = normalize_tools(tools)
        tool_map = build_tool_map(tools)
        anthropic_tools = _to_anthropic_tools(tool_specs)

        system_prompt, anthropic_messages = _to_anthropic_messages(messages)

        create_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": anthropic_messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt
        if anthropic_tools:
            create_kwargs["tools"] = anthropic_tools

        async with self.client.messages.stream(**create_kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield CompletionChunk(content=event.delta.text)

                elif event.type == "content_block_stop":
                    block = stream.current_message_snapshot.content[event.index]
                    if block.type == "tool_use":
                        func = tool_map.get(block.name)
                        tc = ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=dict(block.input),
                            _func=(
                                partial(func, **block.input)
                                if func
                                else partial(lambda: None)
                            ),
                        )
                        yield CompletionChunk(tool_calls=[tc])

                elif event.type == "message_stop":
                    stop_reason = stream.current_message_snapshot.stop_reason
                    if stop_reason == "end_turn":
                        yield CompletionChunk(finish_reason="stop")


def _is_valid_anthropic_model(model_id: str) -> bool:
    try:
        Anthropic().models.retrieve(model_id)
        return True
    except NotFoundError:
        return False


def deploy_anthropic(model_id: str) -> AnthropicModel | None:
    load_dotenv()
    actual_model_id = model_id.removeprefix(_MODEL_PROVIDER_PREFIX)
    if not _is_valid_anthropic_model(actual_model_id):
        return None

    client = AsyncAnthropic()
    return AnthropicModel(client=client, model_id=actual_model_id)


register_provider(_MODEL_PROVIDER_PREFIX, deploy_anthropic)
