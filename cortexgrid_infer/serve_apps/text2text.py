"""The text-to-text serve app: a causal LM loaded from the cortexgrid registry,
streaming from `POST /complete`.

Loads with `AutoModelForCausalLM`, so it runs any causal LM whose weights are
in the transformers layout, whichever importer staged them. An instruct-tuned
model emits tool calls as text in the form `ServedCompletingModel` parses, so
its tokens are forwarded untouched.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from threading import Thread
from typing import Any

import cortexgrid
from cortexgrid import serve
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CompileConfig,
    PreTrainedModel,
    TextIteratorStreamer,
)
import torch

from cortexgrid_infer import compiling
from cortexgrid_infer.device import detect_device
from cortexgrid_infer.protocols.completion import ServedCompletingModel
from cortexgrid_infer.serve_apps.base import (
    COMPILE_PARAM,
    ENABLE_THINKING_PARAM,
    LocalModel,
)
from cortexgrid_infer.serve_apps.streamed_completion import StreamedCompletionWriter


log = logging.getLogger(__name__)

_app = FastAPI()


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

_RESERVED_BODY_KEYS = {
    "messages",
    "tools",
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "choices",
}


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


@serve.ingress(_app)
class Text2Text(LocalModel):
    @classmethod
    def config(cls) -> dict[str, str]:
        return {**super().config(), ENABLE_THINKING_PARAM: "true"}

    @classmethod
    def client(cls, url: str, name: str) -> ServedCompletingModel:
        return ServedCompletingModel(url=url, model_id=name)

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

    @_app.post("/loglikelihoods")
    async def loglikelihoods(self, body: dict[str, Any]) -> dict[str, list[float]]:
        loop = asyncio.get_running_loop()
        loglikelihoods = await loop.run_in_executor(
            None, self._loglikelihoods, body["messages"], body["continuations"]
        )
        return {"loglikelihoods": loglikelihoods}

    @_app.post("/complete")
    async def complete(self, body: dict[str, Any]) -> StreamingResponse:
        messages = body["messages"]
        tools = body.get("tools")
        max_new_tokens = self._room_for_the_reply(
            body.get("max_new_tokens", self._max_total_tokens)
        )
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        top_k = body.get("top_k", 50)
        repetition_penalty = body.get("repetition_penalty", 1.0)
        choices = body.get("choices")
        extra = {k: v for k, v in body.items() if k not in _RESERVED_BODY_KEYS}

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self._enable_thinking,
        )
        inputs = self._truncate(
            self._tokenizer(prompt, return_tensors="pt"),
            self._max_total_tokens - max_new_tokens,
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
                    "max_new_tokens": max_new_tokens,
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

        async def stream() -> AsyncIterator[bytes]:
            writer = StreamedCompletionWriter()
            while True:
                text = await queue.get()
                if text is None:
                    break
                for line in writer.read(text):
                    yield line
            if error:
                raise error[0]
            for line in writer.finish():
                yield line

        return StreamingResponse(stream(), media_type="application/x-ndjson")
