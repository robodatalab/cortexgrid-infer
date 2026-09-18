"""Provider-agnostic model vocabulary: the types a caller programs against."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from functools import partial
from collections.abc import AsyncIterator
from typing import Any, Callable, Sequence


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
