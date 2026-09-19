"""Ray Serve deployment that loads a HuggingFace model from the cortexgrid registry."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
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
    TextIteratorStreamer,
)
import torch

from cortexgrid_infer import compiling
from cortexgrid_infer.device import detect_device


log = logging.getLogger(__name__)

_app = FastAPI()


# The longest prompt this deployment accepts, and the model-card key that sets
# it. Decode is a loop of forwards whose shapes follow the KV cache, so a prompt
# longer than the cache was sized for resizes it and recompiles the loop - which
# costs far more than compiling saves. Prompts are truncated to this instead, so
# the shape holds. How long to allow is the deployment's call, not this
# library's: it is paid for in VRAM whether or not a request uses it.
MAX_INPUT_TOKENS_PARAM = "max_input_tokens"
DEFAULT_MAX_INPUT_TOKENS = 4096


def _max_input_tokens(family: str, suffix: str, run_name: str) -> int:
    """The prompt limit this deployment was given, or the default.

    Read from the model card so it can be changed in the dashboard and picked up
    on the next deploy. A card that cannot be read, or holds something that is
    not a positive integer, leaves the replica on the default rather than
    refusing to load: the setting is a tuning knob, not a correctness one."""
    try:
        raw = cortexgrid.model_config(family, suffix, run_name).get(
            MAX_INPUT_TOKENS_PARAM
        )
    except Exception as exc:  # noqa: BLE001 - optional setting, never fatal
        log.warning("could not read %s/%s/%s's config: %s", family, suffix, run_name, exc)
        return DEFAULT_MAX_INPUT_TOKENS
    if raw is None:
        return DEFAULT_MAX_INPUT_TOKENS
    try:
        tokens = int(raw)
    except (TypeError, ValueError):
        tokens = 0
    if tokens <= 0:
        log.warning(
            "%s=%r on %s/%s/%s is not a positive integer; using %d",
            MAX_INPUT_TOKENS_PARAM, raw, family, suffix, run_name,
            DEFAULT_MAX_INPUT_TOKENS,
        )
        return DEFAULT_MAX_INPUT_TOKENS
    return tokens

_RESERVED_BODY_KEYS = {
    "messages",
    "tools",
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
}


@serve.ingress(_app)
class HuggingFaceCompletingDeployment:
    def __init__(self, family: str, suffix: str, run_name: str) -> None:
        path = cortexgrid.load_model(family, suffix, run_name)
        self._device = detect_device()
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=torch.float16
        )
        self._model.to(self._device)  # type: ignore
        self._max_input_tokens = _max_input_tokens(family, suffix, run_name)
        # A static cache of fixed length is what makes decode compilable: it
        # pins the shape every forward sees, so the loop is captured once
        # instead of recompiled per token. transformers then compiles `generate`
        # itself, which is why the module is not compiled here as well.
        self._compiled = compiling.supported(self._device)
        if self._compiled:
            config = self._model.generation_config
            config.cache_implementation = "static"
            config.max_cache_len = self._max_input_tokens
            config.compile_config = CompileConfig(mode=compiling.Mode.GRAPHED)

    def _truncate(self, inputs: Any) -> Any:
        """Cut an over-long prompt down to the deployment's limit, loudly.

        Keeps the *end* of the prompt: a chat template puts the system message
        first and the turn to answer last, so dropping the head costs context
        while dropping the tail would cost the instruction to reply at all.

        Truncating rather than serving the long prompt is deliberate. The KV
        cache is sized once, at the limit; a longer prompt would resize it and
        recompile the decode loop, which is slower than the whole generation."""
        length = inputs["input_ids"].shape[-1]
        if length <= self._max_input_tokens:
            return inputs
        log.warning(
            "truncating prompt from %d to %d tokens: the deployment's "
            "%s. The dropped %d tokens are from the start of the prompt; raise "
            "%s on the model card if they matter.",
            length, self._max_input_tokens, MAX_INPUT_TOKENS_PARAM,
            length - self._max_input_tokens, MAX_INPUT_TOKENS_PARAM,
        )
        for key, value in inputs.items():
            inputs[key] = value[..., -self._max_input_tokens :]
        return inputs

    @_app.post("/complete")
    async def complete(self, body: dict[str, Any]) -> StreamingResponse:
        messages = body["messages"]
        tools = body.get("tools")
        max_new_tokens = body.get("max_new_tokens", 16 * 1024)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        top_k = body.get("top_k", 50)
        repetition_penalty = body.get("repetition_penalty", 1.0)
        extra = {k: v for k, v in body.items() if k not in _RESERVED_BODY_KEYS}

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._truncate(self._tokenizer(prompt, return_tensors="pt")).to(
            self._model.device
        )

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
            while True:
                text = await queue.get()
                if text is None:
                    break
                if text:
                    yield text.encode("utf-8")
            if error:
                raise error[0]

        return StreamingResponse(stream(), media_type="text/plain")
