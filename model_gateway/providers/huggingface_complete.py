"""HuggingFace provider: deploys models via cortexflow and clients them over HTTP."""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import partial
import itertools
import json
import re
from typing import Any, Callable, Sequence

import cortexflow
import httpx
from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError

from model_gateway.core import (
    CompletingModel,
    Message,
    Tool,
    ToolCall,
    CompletionChunk,
    register_provider,
)
from model_gateway.providers.huggingface_complete_serve import (
    HuggingFaceCompletingDeployment,
)
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
class HuggingFaceCompletingModel(CompletingModel):
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
        in_tool_call = False
        tool_call_text = ""

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


def _parse_hf_id(hf_id: str) -> tuple[str, str]:
    """Map an HF model id to a cortexflow (family, suffix).

    Strips the org (anything before the first '/'). Splits the remainder on
    the last '-': the part before becomes family, the part after becomes
    suffix. If there is no '-', suffix defaults to 'base'.
    """
    name = hf_id.split("/", 1)[-1]
    if "-" not in name:
        return name, "base"
    family, _, suffix = name.rpartition("-")
    return family, suffix


def deploy_huggingface(model_id: str) -> HuggingFaceCompletingModel | None:
    if not model_id.startswith("hf:"):
        return None
    hf_id = model_id[len("hf:") :]
    family, suffix = _parse_hf_id(hf_id)
    run_name = cortexflow.Experiment.get_instance().run_name()

    already_saved = any(
        m.family == family and m.suffix == suffix and m.run_name == run_name
        for m in cortexflow.list_models()
    )
    if not already_saved:
        try:
            with tempfile.TemporaryDirectory() as d:
                snapshot_download(repo_id=hf_id, local_dir=d)
                cortexflow.save_model(
                    d, HuggingFaceCompletingDeployment, family=family, suffix=suffix
                )
        except (RepositoryNotFoundError, OSError):
            return None

    existing = next(
        (
            d
            for d in cortexflow.list_deployed_models()
            if d.family == family and d.suffix == suffix and d.run_name == run_name
        ),
        None,
    )
    if existing is not None:
        url = existing.url
    else:
        deployment = cortexflow.deploy_model(
            family=family,
            suffix=suffix,
            run_name=run_name,
            wait=True,
            timeout=None,
        )
        url = deployment.url

    return HuggingFaceCompletingModel(url=url, model_id=model_id)


register_provider("hf:", deploy_huggingface)
