"""The text-to-text serve app that forwards to the Anthropic API.

It loads no weights and needs no GPU, so there is nothing to import: it is
registered with `registry.Hosted`. What it needs instead is which Anthropic
model to call and a key to call it with, and those come from the registry
entry's `config`, set when the model was registered and editable on its model
card afterwards.

It is a `CompletingModel` like `Text2Text`, so a caller holds either the same
way. Anthropic reports tool calls as structured blocks rather than as
generated text, so `StreamedCompletionWriter` takes them as they are.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import cortexgrid
from anthropic import AsyncAnthropic
from cortexgrid import serve

from cortexgrid_infer.core import CompletingModel, CompletionChunk, Message, ToolSpec
from cortexgrid_infer.serve_apps.base import HostedModel
from cortexgrid_infer.serve_apps.streamed_completion import StreamedCompletionWriter
from cortexgrid_infer.utils import Tool

# Keys the registry entry must carry for the deployment to reach the API.
MODEL_PARAM = "model"
API_KEY_SECRET_PARAM = "api_key_secret"

# The cortexgrid secret the deployment reads its API key from. It is the secret's
# name that is stored on the registry entry, never the key itself: entries are
# readable by anyone who can see the model.
DEFAULT_API_KEY_SECRET = "ANTHROPIC_API_KEY"


def to_anthropic_messages(
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


def to_anthropic_tools(
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


@serve.ingress
class AnthropicText2Text(HostedModel, CompletingModel):
    @classmethod
    def config(
        cls, model_id: str, api_key_secret: str = DEFAULT_API_KEY_SECRET
    ) -> dict[str, str]:
        """`model_id` is the name Anthropic knows the model by, and is what the
        deployment sends upstream; `api_key_secret` names the cortexgrid secret
        holding the key to send it with."""
        return {MODEL_PARAM: model_id, API_KEY_SECRET_PARAM: api_key_secret}

    def __init__(self, deployment: cortexgrid.DeploymentKey) -> None:
        config = cortexgrid.model_config(deployment)
        missing = {MODEL_PARAM, API_KEY_SECRET_PARAM} - config.keys()
        if missing:
            raise RuntimeError(
                f"{deployment.family}/{deployment.suffix}/{deployment.run_name} "
                f"is missing {sorted(missing)} from its config; set them on the "
                "model card"
            )
        self._model = config[MODEL_PARAM]
        self._client = AsyncAnthropic(
            api_key=cortexgrid.get_secret(config[API_KEY_SECRET_PARAM])
        )

    @serve.endpoint
    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[CompletionChunk]:
        system_prompt, anthropic_messages = to_anthropic_messages(messages)
        anthropic_tools = to_anthropic_tools(tools)

        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": anthropic_messages,
            "max_tokens": 16 * 1024 if max_new_tokens is None else max_new_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt
        if anthropic_tools:
            create_kwargs["tools"] = anthropic_tools

        writer = StreamedCompletionWriter()
        async with self._client.messages.stream(**create_kwargs) as events:
            async for event in events:
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    chunk = writer.read_content(event.delta.text)
                    yield chunk

                elif event.type == "content_block_stop":
                    block = events.current_message_snapshot.content[event.index]
                    if block.type == "tool_use":
                        arguments = dict(block.input)
                        chunk = writer.read_tool_call(block.name, arguments, block.id)
                        yield chunk
        for chunk in writer.finish():
            yield chunk
