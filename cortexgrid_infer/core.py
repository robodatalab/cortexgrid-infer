"""Provider-agnostic model vocabulary: the types a caller programs against."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from functools import partial
from collections.abc import AsyncIterator
from typing import Any, Callable, Sequence

import numpy as np


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


@dataclass
class GeneratedMesh:
    """One mesh: `vertices` (N, 3) float32, x to the right of the picture it was
    made from, y up, z toward whoever looked at it; `faces` (M, 3) int32 indices
    into them, counter-clockwise seen from outside; `colours` (N, 3) uint8, each
    vertex's colour. Plus the resolved parameters the deployment used."""

    vertices: np.ndarray
    faces: np.ndarray
    colours: np.ndarray
    params: dict[str, Any] = field(default_factory=dict)


class MeshingModel(DeployedModel):
    """A deployed model that makes a mesh of the object in one picture."""

    @abc.abstractmethod
    async def mesh(self, image: bytes, **kwargs: Any) -> GeneratedMesh: ...


class RewritingModel(DeployedModel):
    """A deployed model that rewrites one text into another - corrects,
    paraphrases, summarises or translates it. Like generation, a single
    request/response."""

    @abc.abstractmethod
    async def rewrite(self, text: str, max_new_tokens: int = 512, **kwargs: Any) -> str: ...


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


async def mesh(deployed_model: MeshingModel, image: bytes, **kwargs: Any) -> GeneratedMesh:
    """Single image-to-mesh request to the deployed model."""
    return await deployed_model.mesh(image, **kwargs)


async def rewrite(
    deployed_model: RewritingModel,
    text: str,
    max_new_tokens: int = 512,
    **kwargs: Any,
) -> str:
    """Single rewriting request to the deployed model."""
    return await deployed_model.rewrite(text, max_new_tokens=max_new_tokens, **kwargs)
