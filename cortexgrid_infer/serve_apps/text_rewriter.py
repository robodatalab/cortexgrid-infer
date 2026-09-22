"""The text-rewriting serve app: an encoder-decoder LM loaded from the cortexgrid
registry, answering `POST /rewrite` (see `cortexgrid_infer.protocols.rewriting`).

Loads with `AutoModelForSeq2SeqLM`, so it runs any encoder-decoder model (T5,
BART, Marian, ...) whose weights are in the transformers layout, whichever
importer staged them. Any task prefix the model expects (`gec: ` for grammar
correction, say) is the caller's to put in the text.
"""

from __future__ import annotations

import logging
from typing import Any

import cortexgrid
from cortexgrid import serve
from fastapi import FastAPI
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, CompileConfig
import torch

from cortexgrid_infer import compiling
from cortexgrid_infer.device import detect_device
from cortexgrid_infer.protocols.rewriting import ServedRewritingModel
from cortexgrid_infer.serve_apps.base import COMPILE_PARAM, LocalModel


log = logging.getLogger(__name__)

_app = FastAPI()

# Output budget for a request that names none: T5's training context, and
# generation stops at the end-of-sequence token well before it for most text.
DEFAULT_MAX_NEW_TOKENS = 512

# Lengths a compiled replica pads a text up to. `generate` sizes the
# cross-attention cache to the encoded text, so every distinct length would
# capture the decode loop afresh; rounding up to a power of two bounds that at
# one capture per bucket, at the cost of a little masked-out padding.
_INPUT_BUCKETS = (32, 64, 128, 256, 512)


def _bucket(length: int) -> int:
    """The smallest of `_INPUT_BUCKETS` that holds `length` tokens, else the largest."""
    return next((b for b in _INPUT_BUCKETS if b >= length), _INPUT_BUCKETS[-1])


@serve.ingress(_app)
class TextRewriter(LocalModel):
    @classmethod
    def client(cls, url: str, name: str) -> ServedRewritingModel:
        return ServedRewritingModel(url=url, model_id=name)

    def __init__(self, family: str, suffix: str, run_name: str) -> None:
        path = cortexgrid.load_model(family, suffix, run_name)
        self._device = detect_device()
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            str(path), torch_dtype=torch.bfloat16
        )
        self._model.to(self._device)
        settings = cortexgrid.model_config(family, suffix, run_name)
        # As in Text2Text: a static cache pins the decoder's self-attention to
        # one shape, and transformers compiles the decode step itself. The
        # cross-attention cache is pinned by padding the text; see `_tokenize`.
        requested = settings.get(COMPILE_PARAM, "false") == "true"
        self._compiled = requested and compiling.supported(self._device)
        if self._compiled:
            config = self._model.generation_config
            config.cache_implementation = "static"
            config.max_cache_len = DEFAULT_MAX_NEW_TOKENS
            config.compile_config = CompileConfig(mode=compiling.Mode.GRAPHED)

    def _tokenize(self, text: str) -> Any:
        """The model's inputs for `text`, on the device.

        Compiled, the text is padded up to its bucket, and one longer than the
        last bucket is cut to it, keeping its start."""
        if not self._compiled:
            return self._tokenizer(text, return_tensors="pt").to(self._device)
        length = len(self._tokenizer(text)["input_ids"])
        bucket = _bucket(length)
        if length > bucket:
            log.warning(
                "truncating text from %d to %d tokens, the most this replica takes",
                length, bucket,
            )
        return self._tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=bucket,
            return_tensors="pt",
        ).to(self._device)

    @_app.post("/rewrite")
    async def rewrite(self, body: dict[str, Any]) -> dict[str, str]:
        generation_options = {"max_new_tokens": DEFAULT_MAX_NEW_TOKENS} | {
            key: value for key, value in body.items() if key != "text"
        }
        if self._compiled:
            # A reply longer than the cache would reallocate it and recompile.
            generation_options["max_new_tokens"] = min(
                generation_options["max_new_tokens"], DEFAULT_MAX_NEW_TOKENS
            )
        inputs = self._tokenize(body["text"])
        output = self._model.generate(**inputs, **generation_options)
        return {"text": self._tokenizer.decode(output[0], skip_special_tokens=True)}
