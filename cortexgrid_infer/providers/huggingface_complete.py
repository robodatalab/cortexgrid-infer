"""HuggingFace provider: deploys models via cortexgrid and clients them over HTTP."""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import partial
import itertools
import json
import os
import re
from typing import Any, Callable, Sequence

import cortexgrid
import httpx
from huggingface_hub import snapshot_download

from cortexgrid_infer.core import (
    CompletingModel,
    Message,
    Tool,
    ToolCall,
    CompletionChunk,
    register_deleter,
    register_provider,
    register_status_provider,
    register_uploader,
)
from cortexgrid_infer.providers.huggingface_complete_serve import (
    HuggingFaceCompletingDeployment,
)
from cortexgrid_infer.utils import build_tool_map, normalize_tools, parse_hf_id, remove_hf_download_metadata


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
    # cortexgrid deployment identity, carried so the owner can tear it down
    # without re-deriving it from the active experiment at shutdown.
    family: str = ""
    suffix: str = ""
    run_name: str = ""

    @property
    def name(self) -> str:
        return self.model_id

    def undeploy(self) -> None:
        """Tear down the Ray Serve app backing this model (frees its GPU).

        The weights + bundle stay in the cortexgrid registry, so a later
        deploy re-schedules the app without re-uploading."""
        if self.family and self.suffix and self.run_name:
            cortexgrid.undeploy_model(self.family, self.suffix, self.run_name)

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


def _ingest_huggingface(
    hf_id: str, family: str, suffix: str, token: str | None
) -> None:
    """Download the HF weights and register them in the cortexgrid registry.

    Submitted to the cluster via ``cortexgrid.remote`` (see `upload_huggingface`),
    so the weights travel HuggingFace -> cluster node -> registry and never transit
    the client. ``save_model`` scopes the version to the ambient Experiment, which
    the remote job inherits from the submitter, so it lands under the same run the
    deploy later reads."""
    with tempfile.TemporaryDirectory() as d:
        snapshot_download(repo_id=hf_id, local_dir=d, token=token)
        remove_hf_download_metadata(d)
        cortexgrid.save_model(
            d, HuggingFaceCompletingDeployment, family=family, suffix=suffix
        )


def upload_huggingface(model_id: str) -> str | None:
    """Start ingesting *model_id*'s weights into the registry, on the cluster.

    Returns the id of the background ``cortexgrid.remote`` job, or None if the
    model is already registered (phase `uploading`/`ready`) so there is nothing to
    submit. Poll progress via ``deployment_status(model_id)``; deploy once `ready`."""
    if not model_id.startswith("hf:"):
        return None
    hf_id = model_id[len("hf:") :]
    family, suffix = parse_hf_id(hf_id)
    run_name = cortexgrid.Experiment.get_instance().run_name()

    status = cortexgrid.model_registry_status(family, suffix, run_name)
    if status is not None and status.phase in ("uploading", "ready"):
        return None

    return cortexgrid.remote(
        _ingest_huggingface,
        hf_id,
        family,
        suffix,
        os.environ.get("HF_TOKEN"),
        num_gpus=0,
        num_cpus=2,
    )


def deploy_huggingface(model_id: str) -> HuggingFaceCompletingModel | None:
    if not model_id.startswith("hf:"):
        return None
    hf_id = model_id[len("hf:") :]
    family, suffix = parse_hf_id(hf_id)
    run_name = cortexgrid.Experiment.get_instance().run_name()

    status = cortexgrid.model_registry_status(family, suffix, run_name)
    if status is None or status.phase != "ready":
        phase = None if status is None else status.phase
        raise RuntimeError(
            f"Model '{model_id}' is not registry-ready (phase={phase}); call "
            f"upload_model('{model_id}') and wait for phase 'ready' before deploying."
        )

    existing = next(
        (
            d
            for d in cortexgrid.list_deployed_models()
            if d.family == family and d.suffix == suffix and d.run_name == run_name
        ),
        None,
    )
    if existing is not None:
        url = existing.url
    else:
        deployment = cortexgrid.deploy_model(
            family=family,
            suffix=suffix,
            run_name=run_name,
            wait=True,
            timeout=None,
        )
        url = deployment.url

    return HuggingFaceCompletingModel(
        url=url,
        model_id=model_id,
        family=family,
        suffix=suffix,
        run_name=run_name,
    )


def hf_deployment_status(model_id: str) -> Any:
    """Live phase of the deployment for *model_id*, delegated to cortexgrid.

    Read-only - safe to poll from a status endpoint while a deploy is in flight.
    Resolves the same (family, suffix, run_name) identity `deploy` uses. Reports
    the serving lifecycle (`cortexgrid.model_serving_status`) once a Serve app
    exists; before that - while the weights are still uploading to the registry -
    it falls back to the registry lifecycle (`cortexgrid.model_registry_status`),
    so a poll stays meaningful during weight staging / scheduling too."""
    if not model_id.startswith("hf:"):
        return None
    family, suffix = parse_hf_id(model_id[len("hf:") :])
    run_name = cortexgrid.Experiment.get_instance().run_name()
    serving = cortexgrid.model_serving_status(family, suffix, run_name)
    if serving.phase != "not_deployed":
        return serving
    return cortexgrid.model_registry_status(family, suffix, run_name)


def delete_huggingface(model_id: str) -> None:
    """Undeploy (if running) and delete this model's weights + serve bundle.

    Idempotent: safe whether or not the model is deployed or registered, so it is
    the inverse of `upload_huggingface` + `deploy_huggingface` for cleanup."""
    if not model_id.startswith("hf:"):
        return
    family, suffix = parse_hf_id(model_id[len("hf:") :])
    run_name = cortexgrid.Experiment.get_instance().run_name()
    cortexgrid.undeploy_model(family, suffix, run_name)
    cortexgrid.delete_model(family, suffix, run_name)


register_provider("hf:", deploy_huggingface)
register_status_provider("hf:", hf_deployment_status)
register_uploader("hf:", upload_huggingface)
register_deleter("hf:", delete_huggingface)
