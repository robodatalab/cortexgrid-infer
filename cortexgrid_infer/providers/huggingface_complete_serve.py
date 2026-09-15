"""Ray Serve deployment that loads a HuggingFace model from the cortexgrid registry."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from threading import Thread
from typing import Any

import cortexgrid
from cortexgrid import serve
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch

from cortexgrid_infer.device import detect_device


_app = FastAPI()


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
    num_gpus = 1
    num_replicas = 1

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
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

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
