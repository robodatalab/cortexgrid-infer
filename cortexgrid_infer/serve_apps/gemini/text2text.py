"""The text-to-text serve app that forwards to the Gemini API.

It speaks the same `/complete` protocol as `Text2Text`, so the same client talks
to it. Gemini reports tool calls as structured parts rather than as generated
text, so they are re-encoded on the way out.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import Any

from cortexgrid import serve
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from cortexgrid_infer.completion import ServedCompletingModel, encode_tool_call
from cortexgrid_infer.core import Message, ToolSpec
from cortexgrid_infer.models.gemini.base import GeminiModel

_app = FastAPI()

# Gemini 3 rejects a function call replayed without the thought signature it was
# returned with, and signatures do not survive the trip through the client as
# text. Google documents this value as the one that skips the check. The SDK
# base64-encodes the bytes it is given, so they are decoded here for the value
# to go out verbatim.
SKIP_THOUGHT_SIGNATURE = base64.urlsafe_b64decode("skip_thought_signature_validator")


def to_gemini_contents(
    messages: list[Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert OpenAI-style messages to Gemini contents."""
    system_prompt: str | None = None
    contents: list[dict[str, Any]] = []
    # Gemini answers a function call by name, which a tool message does not
    # carry, so it is looked up from the call the message answers.
    tool_names: dict[str, str] = {}

    for msg in messages:
        role = msg["role"]

        if role == "system":
            system_prompt = msg["content"]

        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": msg["content"]}]})

        elif role == "assistant":
            parts: list[dict[str, Any]] = []
            if msg.get("content"):
                parts.append({"text": msg["content"]})
            for i, tc in enumerate(msg.get("tool_calls", [])):
                func = tc["function"]
                tool_names[tc["id"]] = func["name"]
                call_part: dict[str, Any] = {
                    "function_call": {
                        "id": tc["id"],
                        "name": func["name"],
                        "args": func["arguments"],
                    }
                }
                # Gemini signs only the first call of a parallel batch.
                if i == 0:
                    call_part["thought_signature"] = SKIP_THOUGHT_SIGNATURE
                parts.append(call_part)
            contents.append({"role": "model", "parts": parts})

        elif role == "tool":
            response_part = {
                "function_response": {
                    "id": msg["tool_call_id"],
                    "name": tool_names[msg["tool_call_id"]],
                    "response": {"output": msg["content"]},
                }
            }
            if (
                contents
                and contents[-1]["role"] == "user"
                and "function_response" in contents[-1]["parts"][0]
            ):
                contents[-1]["parts"].append(response_part)
            else:
                contents.append({"role": "user", "parts": [response_part]})

    return system_prompt, contents


def to_gemini_tools(
    tool_specs: list[ToolSpec] | None,
) -> list[dict[str, Any]] | None:
    """Convert OpenAI-style tool specs to one Gemini tool of function
    declarations."""
    if not tool_specs:
        return None
    declarations = []
    for spec in tool_specs:
        func = spec["function"]
        declaration = {"name": func["name"], "description": func.get("description", "")}
        # Plain JSON Schema, which `parameters` (an OpenAPI subset) is not. A
        # function that takes nothing declares no schema at all.
        if "parameters" in func:
            declaration["parameters_json_schema"] = func["parameters"]
        declarations.append(declaration)
    return [{"function_declarations": declarations}]


@serve.ingress(_app)
class GeminiText2Text(GeminiModel):
    @classmethod
    def client(cls, url: str, name: str) -> ServedCompletingModel:
        return ServedCompletingModel(url=url, model_id=name)

    @_app.post("/complete")
    async def complete(self, body: dict[str, Any]) -> StreamingResponse:
        system_prompt, contents = to_gemini_contents(body["messages"])
        tools = to_gemini_tools(body.get("tools"))

        generate_config: dict[str, Any] = {
            "max_output_tokens": body.get("max_new_tokens", 16 * 1024),
            "temperature": body.get("temperature", 0.7),
        }
        if system_prompt:
            generate_config["system_instruction"] = system_prompt
        if tools:
            generate_config["tools"] = tools

        async def stream() -> AsyncIterator[bytes]:
            chunks = await self._client.aio.models.generate_content_stream(
                model=self._model, contents=contents, config=generate_config
            )
            async for chunk in chunks:
                if not chunk.candidates or not chunk.candidates[0].content:
                    continue
                for part in chunk.candidates[0].content.parts or []:
                    if part.function_call:
                        yield encode_tool_call(
                            part.function_call.name, dict(part.function_call.args or {})
                        ).encode("utf-8")

                    elif part.text and not part.thought:
                        yield part.text.encode("utf-8")

        return StreamingResponse(stream(), media_type="text/plain")
