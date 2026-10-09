"""The text-to-text serve app: a causal LM loaded from the cortexgrid registry,
streaming from its `complete` endpoint.

Loads with `AutoModelForCausalLM`, so it runs any causal LM whose weights are
in the transformers layout, whichever importer staged them. An instruct-tuned
model emits tool calls as text, which `StreamedCompletionWriter` turns into the
chunks the endpoint streams.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import AsyncIterator, Callable
from threading import Thread
from typing import Any

import cortexgrid
from cortexgrid import serve
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Cache,
    CompileConfig,
    PreTrainedModel,
    TextIteratorStreamer,
)
import torch

from cortexgrid_infer import compiling
from cortexgrid_infer.core import CompletingModel, CompletionChunk, Message, Tensor
from cortexgrid_infer.device import detect_device
from cortexgrid_infer.serve_apps.base import (
    COMPILE_PARAM,
    ENABLE_THINKING_PARAM,
    LocalModel,
)
from cortexgrid_infer.serve_apps.streamed_completion import StreamedCompletionWriter
from cortexgrid_infer.utils import Tool


log = logging.getLogger(__name__)


# Prompt + reply a replica is sized for. The static cache is allocated at this
# many slots and every decoded token reads all of them, filled or not: at
# Qwen2.5-3B's 32k context that is 1.2 GB of keys and values re-read per token,
# which costs far more than the prompt it makes room for. So the default is what
# a caller plausibly sends rather than what the model could hold; a model that
# needs more takes `max_total_tokens` on its model card, up to its own context.
MAX_TOTAL_TOKENS_PARAM = "max_total_tokens"
DEFAULT_MAX_TOTAL_TOKENS = 4096

DTYPE_PARAM = "dtype"
DEFAULT_DTYPE = "float16"


def allowed_next_tokens(
    choices: list[list[int]], generated: list[int], end_of_text: int
) -> list[int]:
    matching = [choice for choice in choices if choice[: len(generated)] == generated]
    continuing = {choice[len(generated)] for choice in matching if len(choice) > len(generated)}
    finished = any(choice == generated for choice in matching)
    allowed = sorted(continuing | {end_of_text}) if finished else sorted(continuing)
    return allowed or [end_of_text]


def continuation_loglikelihood(
    model: PreTrainedModel, prompt_ids: list[int], continuation_ids: list[int]
) -> float:
    read = torch.tensor([prompt_ids + continuation_ids], device=model.device)
    with torch.inference_mode():
        outputs = model(input_ids=read, logits_to_keep=len(continuation_ids) + 1)
    predicting_the_continuation = outputs.logits[0, :-1].float()
    log_probabilities = torch.log_softmax(predicting_the_continuation, dim=-1)
    positions = torch.arange(len(continuation_ids), device=log_probabilities.device)
    continuation = torch.tensor(continuation_ids, device=log_probabilities.device)
    return float(log_probabilities[positions, continuation].sum())


def read_further(model: PreTrainedModel, read: Cache | None, ids: list[int]) -> Cache:
    unread = torch.tensor([ids], device=model.device)
    with torch.inference_mode():
        outputs = model(input_ids=unread, past_key_values=read, use_cache=True, logits_to_keep=1)
    return outputs.past_key_values


def last_hidden_state_after(model: PreTrainedModel, read: Cache, ids: list[int]) -> torch.Tensor:
    read_for_the_continuation = copy.deepcopy(read)
    continuation = torch.tensor([ids], device=model.device)
    with torch.inference_mode():
        outputs = model(
            input_ids=continuation,
            past_key_values=read_for_the_continuation,
            use_cache=True,
            output_hidden_states=True,
            logits_to_keep=1,
        )
    last_layer = outputs.hidden_states[-1]
    return last_layer[0, -1].float()


def mean_hidden_state_after(
    model: PreTrainedModel, read: Cache, ids: list[int], layer: int
) -> torch.Tensor:
    unread = torch.tensor([ids], device=model.device)
    with torch.inference_mode():
        outputs = model(
            input_ids=unread,
            past_key_values=read,
            use_cache=True,
            output_hidden_states=True,
            logits_to_keep=1,
        )
    hidden_states_of_the_text = outputs.hidden_states[layer][0].float()
    mean_hidden_state = hidden_states_of_the_text.mean(dim=0)
    return mean_hidden_state


@serve.ingress
class Text2Text(LocalModel, CompletingModel):
    @classmethod
    def config(cls) -> dict[str, str]:
        return {**super().config(), ENABLE_THINKING_PARAM: "true"}

    def __init__(self, deployment: cortexgrid.DeploymentKey) -> None:
        path = cortexgrid.load_model(
            deployment.family, deployment.suffix, deployment.run_name
        )
        settings = cortexgrid.model_config(deployment)
        self._device = detect_device()
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=getattr(torch, settings.get(DTYPE_PARAM, DEFAULT_DTYPE))
        )
        self._model.to(self._device)  # type: ignore
        context = getattr(
            self._model.config, "max_position_embeddings", DEFAULT_MAX_TOTAL_TOKENS
        )
        self._max_total_tokens = min(
            int(settings.get(MAX_TOTAL_TOKENS_PARAM, DEFAULT_MAX_TOTAL_TOKENS)), context
        )
        self._enable_thinking = settings.get(ENABLE_THINKING_PARAM, "true") == "true"
        # A static cache of fixed length is what makes decode compilable: it
        # pins the shape every forward sees, so the loop is captured once
        # instead of recompiled per token. transformers then compiles `generate`
        # itself, which is why the module is not compiled here as well.
        requested = settings.get(COMPILE_PARAM, "false") == "true"
        self._compiled = requested and compiling.supported(self._device)
        if self._compiled:
            config = self._model.generation_config
            config.cache_implementation = "static"
            config.max_cache_len = self._max_total_tokens
            config.compile_config = CompileConfig(mode=compiling.Mode.GRAPHED)

    def _room_for_the_reply(self, requested: int) -> int:
        """How many tokens a reply may take, leaving the prompt the rest.

        Half the budget at most, so a caller asking for more output than the
        replica is sized for cannot squeeze the prompt down to nothing."""
        return min(requested, self._max_total_tokens // 2)

    def _truncate(self, inputs: Any, limit: int) -> Any:
        """Cut an over-long prompt to ``limit`` tokens, keeping its end.

        A prompt and reply that together outgrew the cache would reallocate it
        and recompile the decode loop, which costs more than the generation."""
        length = inputs["input_ids"].shape[-1]
        if length <= limit:
            return inputs
        log.warning(
            "truncating prompt from %d to %d tokens, the room this replica has",
            length, limit,
        )
        for key, value in inputs.items():
            inputs[key] = value[..., -limit:]
        return inputs

    def _prompt_ids(self, messages: list[dict[str, Any]]) -> list[int]:
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self._enable_thinking,
        )
        return self._tokenizer(prompt)["input_ids"]

    def _continuation_ids(self, text: str) -> list[int]:
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    def _only_the_choices(
        self, choices: list[str], prompt_length: int
    ) -> Callable[[int, torch.Tensor], list[int]]:
        choice_ids = [self._continuation_ids(choice) for choice in choices]
        end_of_text = self._tokenizer.eos_token_id

        def allowed(_batch: int, sequence: torch.Tensor) -> list[int]:
            generated = sequence[prompt_length:].tolist()
            return allowed_next_tokens(choice_ids, generated, end_of_text)

        return allowed

    def _loglikelihoods(
        self, messages: list[dict[str, Any]], continuations: list[str]
    ) -> list[float]:
        prompt_ids = self._prompt_ids(messages)
        continuation_ids = [self._continuation_ids(continuation) for continuation in continuations]
        return [
            continuation_loglikelihood(self._model, prompt_ids, ids) for ids in continuation_ids
        ]

    @serve.endpoint
    async def loglikelihoods(
        self, messages: list[Message], continuations: list[str]
    ) -> list[float]:
        loop = asyncio.get_running_loop()
        loglikelihoods = await loop.run_in_executor(
            None, self._loglikelihoods, messages, continuations
        )
        return loglikelihoods

    def _read_part(
        self, read: Cache | None, part: str, continuations: list[str]
    ) -> tuple[Cache, list[list[float]]]:
        tokenized = self._tokenizer(part, add_special_tokens=read is None)
        part_ids = tokenized["input_ids"]
        read = read_further(self._model, read, part_ids)
        continuation_ids = [self._continuation_ids(continuation) for continuation in continuations]
        last_hidden_states = [
            last_hidden_state_after(self._model, read, ids) for ids in continuation_ids
        ]
        answered = [hidden_state.tolist() for hidden_state in last_hidden_states]
        return read, answered

    @serve.endpoint
    async def last_hidden_states(
        self, parts: list[str], continuations_of_each_part: list[list[str]]
    ) -> AsyncIterator[list[list[float]]]:
        loop = asyncio.get_running_loop()
        read: Cache | None = None
        for part, continuations in zip(parts, continuations_of_each_part):
            read, last_hidden_states = await loop.run_in_executor(
                None, self._read_part, read, part, continuations
            )
            yield last_hidden_states

    def _mean_hidden_state(self, context: str, text: str, layer: int) -> torch.Tensor:
        tokenized_context = self._tokenizer(context)
        context_ids = tokenized_context["input_ids"]
        read = read_further(self._model, None, context_ids)
        text_ids = self._continuation_ids(text)
        mean_hidden_state = mean_hidden_state_after(self._model, read, text_ids, layer)
        return mean_hidden_state

    @serve.endpoint
    async def mean_hidden_state(self, context: str, text: str, layer: int) -> Tensor:
        loop = asyncio.get_running_loop()
        mean_hidden_state = await loop.run_in_executor(
            None, self._mean_hidden_state, context, text, layer
        )
        return mean_hidden_state

    @serve.endpoint
    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        choices: list[str] | None = None,
    ) -> AsyncIterator[CompletionChunk]:
        requested_new_tokens = (
            self._max_total_tokens if max_new_tokens is None else max_new_tokens
        )
        room_for_the_reply = self._room_for_the_reply(requested_new_tokens)
        extra: dict[str, Any] = {}

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self._enable_thinking,
        )
        inputs = self._truncate(
            self._tokenizer(prompt, return_tensors="pt"),
            self._max_total_tokens - room_for_the_reply,
        ).to(self._model.device)
        if choices:
            prompt_length = inputs["input_ids"].shape[-1]
            extra["prefix_allowed_tokens_fn"] = self._only_the_choices(choices, prompt_length)

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        error: list[BaseException] = []
        loop = asyncio.get_running_loop()

        def generate() -> None:
            try:
                streamer = TextIteratorStreamer(
                    self._tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )
                generation_config = {
                    "max_new_tokens": room_for_the_reply,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    "do_sample": temperature > 0,
                    "pad_token_id": self._tokenizer.pad_token_id,
                    "eos_token_id": self._tokenizer.eos_token_id,
                    "streamer": streamer,
                    **extra,
                }
                thread = Thread(
                    target=lambda: self._model.generate(  # type: ignore
                        **inputs, **generation_config
                    )
                )
                thread.start()
                for text in streamer:
                    loop.call_soon_threadsafe(queue.put_nowait, text)
                thread.join()
            except BaseException as e:
                error.append(e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, generate)

        writer = StreamedCompletionWriter()
        while True:
            text = await queue.get()
            if text is None:
                break
            for chunk in writer.read(text):
                yield chunk
        if error:
            raise error[0]
        for chunk in writer.finish():
            yield chunk
