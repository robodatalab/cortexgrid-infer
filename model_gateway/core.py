"""DeployedModel ABC, provider registry, and completion API."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from functools import partial
from collections.abc import AsyncIterator
from typing import (
    Any,
    Callable,
    Sequence,
    TypeVar,
)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    _func: partial[Any] = field(repr=False)

    def __call__(self) -> Any:
        return self._func(**self.arguments)


@dataclass
class CompletionChunk:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None

    @property
    def has_text(self) -> bool:
        return bool(self.content)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


Message = dict[str, Any]
ToolSpec = dict[str, Any]
Tool = Callable[..., Any] | ToolSpec


class DeployedModel(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    def undeploy(self) -> None:
        """Release the underlying deployment, if this provider manages one.

        The caller owns the model's lifecycle and calls this to tear it down
        (e.g. on server shutdown). Hosted-API providers (e.g. Anthropic) have
        nothing to release, so the default is a no-op; cluster-backed providers
        override it to free their compute."""


class CompletingModel(DeployedModel):
    @abc.abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: Sequence[Tool] | None = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionChunk]: ...


@dataclass
class GeneratedImage:
    """One generated image: the encoded file bytes (PNG) plus the resolved
    parameters the deployment used to produce it."""

    image: bytes
    width: int
    height: int
    params: dict[str, Any] = field(default_factory=dict)


class GeneratingModel(DeployedModel):
    """A deployed image model. Unlike completion, generation is a single
    request/response (no token streaming), so `generate` returns one result."""

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        image: bytes | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        size: int = 1024,
        seed: int | None = None,
        **kwargs: Any,
    ) -> GeneratedImage: ...


_providers: list[tuple[str, Callable[[str], DeployedModel | None]]] = []

T = TypeVar("T", bound=DeployedModel)


def register_provider(
    prefix: str, factory: Callable[[str], DeployedModel | None]
) -> None:
    """Register a model provider that handles model IDs starting with *prefix*."""
    _providers.append((prefix, factory))


def deploy_model(model_id: str) -> DeployedModel:
    for prefix, factory in _providers:
        if not model_id.startswith(prefix):
            continue
        model = factory(model_id)
        if model is None:
            continue
        return model
    raise ValueError(
        f"No provider registered for model '{model_id}'. "
        f"Known prefixes: {[p for p, _ in _providers]}"
    )


# Status providers mirror deploy providers: a prefix -> a function that reports the
# live deployment status for that model id (delegating to the platform, e.g.
# cortexgrid.model_serving_status). Kept separate from _providers because a status
# query must NOT construct/deploy anything - it just reads current state.
_status_providers: list[tuple[str, Callable[[str], Any]]] = []


def register_status_provider(prefix: str, fn: Callable[[str], Any]) -> None:
    """Register a deployment-status reporter for model IDs starting with *prefix*."""
    _status_providers.append((prefix, fn))


def deployment_status(model_id: str) -> Any:
    """Report the current deployment status/phase for *model_id*, or None if no
    status provider handles it (e.g. hosted-API models with nothing to schedule).
    The returned object is whatever the platform reports - for cluster-backed
    providers, a ``cortexgrid.ServingStatus`` (``phase`` + ``message``) once a
    Serve app exists, or a ``cortexgrid.SavedModel`` while the weights are still
    uploading to the registry."""
    for prefix, fn in _status_providers:
        if not model_id.startswith(prefix):
            continue
        result = fn(model_id)
        if result is not None:
            return result
    return None


# Uploaders mirror deploy/status providers: a prefix -> a function that STARTS an
# ingest of the model's weights into the cortexgrid registry and returns the id of
# the background job doing it, or None when there is nothing to upload (already
# registered, or a hosted-API model with no weights). This is deliberately split
# from deploy: a model must be uploaded (registry phase `ready`) before
# `deploy_model` can schedule it, and the upload runs on the cluster - not the
# caller's machine - so large weights never round-trip through the client.
_uploaders: list[tuple[str, Callable[[str], str | None]]] = []


def register_uploader(prefix: str, fn: Callable[[str], str | None]) -> None:
    """Register a registry-upload starter for model IDs starting with *prefix*."""
    _uploaders.append((prefix, fn))


def upload_model(model_id: str) -> str | None:
    """Start uploading *model_id*'s weights into the cortexgrid registry, returning
    the id of the background job doing it, or None if nothing needs uploading
    (already registered, or a hosted-API model with no weights to stage).

    The upload runs on the cluster, not the caller's machine. Poll its progress
    with ``deployment_status(model_id)`` (registry phase `uploading -> ready`);
    once `ready`, call ``deploy_model(model_id)``."""
    for prefix, fn in _uploaders:
        if model_id.startswith(prefix):
            return fn(model_id)
    return None


# Deleters mirror the other registries: a prefix -> a function that removes a model
# from the platform (undeploy its Serve app if running, then delete its weights +
# serve bundle from the registry). The inverse of upload+deploy; kept separate so a
# caller can tear a model down by id without holding a deployed-model handle.
_deleters: list[tuple[str, Callable[[str], None]]] = []


def register_deleter(prefix: str, fn: Callable[[str], None]) -> None:
    """Register a registry-cleanup handler for model IDs starting with *prefix*."""
    _deleters.append((prefix, fn))


def delete_model(model_id: str) -> None:
    """Remove *model_id* from the platform: undeploy its Serve app if running, then
    delete its weights and serve bundle from the registry.

    Idempotent - safe whether or not the model is currently deployed or registered -
    and a no-op for model ids no deleter handles (e.g. hosted-API models, which have
    nothing on the cluster to clean up)."""
    for prefix, fn in _deleters:
        if model_id.startswith(prefix):
            fn(model_id)
            return


async def complete(
    deployed_model: CompletingModel,
    messages: list[Message],
    tools: Sequence[Tool] | None = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    **kwargs: Any,
) -> AsyncIterator[CompletionChunk]:
    """Async completion request to the deployed model."""
    async for chunk in deployed_model.complete(
        messages,
        tools,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        **kwargs,
    ):
        yield chunk


async def generate(
    deployed_model: GeneratingModel,
    prompt: str,
    *,
    image: bytes | None = None,
    **kwargs: Any,
) -> GeneratedImage:
    """Single image-generation request to the deployed model."""
    return await deployed_model.generate(prompt, image=image, **kwargs)
