"""Ray Serve deployment that forwards completions to the Anthropic API.

The odd one out among serve apps: it loads no weights and needs no GPU. What it
needs instead is which Anthropic model to call and a key to call it with, and
those come from the registry entry - `params` on the model's requirements, set
when the model was imported and editable on its model card afterwards.

It speaks the same `/complete` protocol as every other completion app, so the
same client talks to it. Anthropic reports tool calls as structured blocks
rather than as generated text, so they are re-encoded on the way out.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import cortexgrid
from anthropic import AsyncAnthropic
from cortexgrid import serve
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from cortexgrid_infer.completion import encode_tool_call
from cortexgrid_infer.core import Message, ToolSpec

_app = FastAPI()

# Keys the registry entry must carry for the deployment to reach the API.
MODEL_PARAM = "model"
API_KEY_SECRET_PARAM = "api_key_secret"


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


@serve.ingress(_app)
class AnthropicDeployment:
    def __init__(self, family: str, suffix: str, run_name: str) -> None:
        saved = cortexgrid.model_registry_status(family, suffix, run_name)
        if saved is None:
            raise RuntimeError(f"No registry entry for {family}/{suffix}/{run_name}")
        params = saved.requirements.params  # type: ignore[attr-defined]
        missing = {MODEL_PARAM, API_KEY_SECRET_PARAM} - params.keys()
        if missing:
            raise RuntimeError(
                f"{family}/{suffix}/{run_name} is missing {sorted(missing)} in its "
                "requirements' params; set them on the model card"
            )
        self._model = params[MODEL_PARAM]
        self._client = AsyncAnthropic(
            api_key=cortexgrid.get_secret(params[API_KEY_SECRET_PARAM])
        )

    @_app.post("/complete")
    async def complete(self, body: dict[str, Any]) -> StreamingResponse:
        system_prompt, messages = to_anthropic_messages(body["messages"])
        tools = to_anthropic_tools(body.get("tools"))

        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": body.get("max_new_tokens", 16 * 1024),
            "temperature": body.get("temperature", 0.7),
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt
        if tools:
            create_kwargs["tools"] = tools

        async def stream() -> AsyncIterator[bytes]:
            async with self._client.messages.stream(**create_kwargs) as events:
                async for event in events:
                    if (
                        event.type == "content_block_delta"
                        and event.delta.type == "text_delta"
                    ):
                        yield event.delta.text.encode("utf-8")

                    elif event.type == "content_block_stop":
                        block = events.current_message_snapshot.content[event.index]
                        if block.type == "tool_use":
                            yield encode_tool_call(
                                block.name, dict(block.input)
                            ).encode("utf-8")

        return StreamingResponse(stream(), media_type="text/plain")
