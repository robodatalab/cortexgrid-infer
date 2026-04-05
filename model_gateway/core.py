"""DeployedModel ABC, provider registry, and completion API."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from functools import partial
from collections.abc import AsyncIterator
from typing import Any, Callable, Sequence, TypeVar, overload


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


_providers: list[tuple[str, Callable[[str], DeployedModel | None]]] = []

T = TypeVar("T", bound=DeployedModel)


def register_provider(
    prefix: str, factory: Callable[[str], DeployedModel | None]
) -> None:
    """Register a model provider that handles model IDs starting with *prefix*."""
    _providers.append((prefix, factory))


@overload
def deploy_model(model_id: str) -> CompletingModel: ...
@overload
def deploy_model(model_id: str, expected_type: type[T]) -> T: ...
def deploy_model(
    model_id: str, expected_type: type[DeployedModel] = CompletingModel
) -> DeployedModel:
    for prefix, factory in _providers:
        if not model_id.startswith(prefix):
            continue
        model = factory(model_id)
        if model is None:
            continue
        if not isinstance(model, expected_type):
            raise TypeError(
                f"Model '{model_id}' deployed as {type(model).__name__}, "
                f"expected {expected_type.__name__}"
            )
        return model
    raise ValueError(
        f"No provider registered for model '{model_id}'. "
        f"Known prefixes: {[p for p, _ in _providers]}"
    )


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
