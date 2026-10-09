"""Provider-agnostic model vocabulary: the types a caller programs against."""

from __future__ import annotations

import abc
import base64
from dataclasses import dataclass, field
from collections.abc import AsyncIterator
from functools import partial
from typing import Annotated, Any

import numpy as np
import torch
from pydantic import Base64Bytes, PlainSerializer, PlainValidator

from cortexgrid_infer.utils import Tool, ToolSpec


Tensor = Annotated[
    torch.Tensor,
    PlainValidator(torch.tensor),
    PlainSerializer(torch.Tensor.tolist),
]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class CompletionChunk:
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None

    @property
    def has_text(self) -> bool:
        return bool(self.content)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


Message = dict[str, Any]


@dataclass
class CausalLMArchitecture:
    layer_count: int
    hidden_size: int


class CompletingModel(abc.ABC):
    @abc.abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[CompletionChunk]: ...


@dataclass
class GeneratedImage:
    """One generated image: the encoded file bytes (PNG) plus the resolved
    parameters the deployment used to produce it."""

    image: Base64Bytes
    width: int
    height: int
    params: dict[str, Any] = field(default_factory=dict)


class GeneratingModel(abc.ABC):
    """A deployed image model. Unlike completion, generation is a single
    request/response (no token streaming), so `generate` returns one result."""

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        image: Base64Bytes | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        size: int = 1024,
        seed: int | None = None,
    ) -> GeneratedImage: ...


def _encoded_array(array: np.ndarray, dtype: str) -> str:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    encoded = base64.b64encode(contiguous.tobytes())
    return encoded.decode("ascii")


def _decoded_array(data: str, dtype: str, columns: int) -> np.ndarray:
    raw = base64.b64decode(data)
    flat = np.frombuffer(raw, dtype=dtype)
    return flat.reshape(-1, columns)


Vertices = Annotated[
    np.ndarray,
    PlainValidator(partial(_decoded_array, dtype="<f4", columns=3)),
    PlainSerializer(partial(_encoded_array, dtype="<f4")),
]
Faces = Annotated[
    np.ndarray,
    PlainValidator(partial(_decoded_array, dtype="<i4", columns=3)),
    PlainSerializer(partial(_encoded_array, dtype="<i4")),
]
Colours = Annotated[
    np.ndarray,
    PlainValidator(partial(_decoded_array, dtype="u1", columns=3)),
    PlainSerializer(partial(_encoded_array, dtype="u1")),
]


@dataclass
class GeneratedMesh:
    """One mesh: `vertices` (N, 3) float32, x to the right of the picture it was
    made from, y up, z toward whoever looked at it; `faces` (M, 3) int32 indices
    into them, counter-clockwise seen from outside; `colours` (N, 3) uint8, each
    vertex's colour. Plus the resolved parameters the deployment used."""

    vertices: Vertices
    faces: Faces
    colours: Colours
    params: dict[str, Any] = field(default_factory=dict)


class MeshingModel(abc.ABC):
    """A deployed model that makes a mesh of the object in one picture."""

    @abc.abstractmethod
    async def mesh(
        self, image: Base64Bytes, options: dict[str, Any] | None = None
    ) -> GeneratedMesh: ...


class RewritingModel(abc.ABC):
    """A deployed model that rewrites one text into another - corrects,
    paraphrases, summarises or translates it. Like generation, a single
    request/response."""

    @abc.abstractmethod
    async def rewrite(self, text: str, max_new_tokens: int = 512) -> str: ...


async def complete(
    deployed_model: CompletingModel,
    messages: list[Message],
    tools: list[Tool] | None = None,
    max_new_tokens: int | None = None,
    temperature: float = 0.7,
) -> AsyncIterator[CompletionChunk]:
    """Async completion request to the deployed model."""
    async for chunk in deployed_model.complete(
        messages,
        tools,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    ):
        yield chunk


async def generate(
    deployed_model: GeneratingModel,
    prompt: str,
    *,
    image: bytes | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    size: int = 1024,
    seed: int | None = None,
) -> GeneratedImage:
    """Single image-generation request to the deployed model."""
    return await deployed_model.generate(
        prompt, image=image, steps=steps, guidance=guidance, size=size, seed=seed
    )


async def mesh(
    deployed_model: MeshingModel, image: bytes, options: dict[str, Any] | None = None
) -> GeneratedMesh:
    """Single image-to-mesh request to the deployed model."""
    return await deployed_model.mesh(image, options)


async def rewrite(
    deployed_model: RewritingModel,
    text: str,
    max_new_tokens: int = 512,
) -> str:
    """Single rewriting request to the deployed model."""
    return await deployed_model.rewrite(text, max_new_tokens=max_new_tokens)
