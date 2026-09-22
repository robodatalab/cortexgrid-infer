"""The text-rewriting serve app: an encoder-decoder LM loaded from the cortexgrid
registry, answering `POST /rewrite` (see `cortexgrid_infer.protocols.rewriting`).

Loads with `AutoModelForSeq2SeqLM`, so it runs any encoder-decoder model (T5,
BART, Marian, ...) whose weights are in the transformers layout, whichever
importer staged them. Any task prefix the model expects (`gec: ` for grammar
correction, say) is the caller's to put in the text.
"""

from __future__ import annotations

from typing import Any

import cortexgrid
from cortexgrid import serve
from fastapi import FastAPI
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

from cortexgrid_infer.device import detect_device
from cortexgrid_infer.protocols.rewriting import ServedRewritingModel
from cortexgrid_infer.serve_apps.base import LocalModel


_app = FastAPI()

# Output budget for a request that names none: T5's training context, and
# generation stops at the end-of-sequence token well before it for most text.
DEFAULT_MAX_NEW_TOKENS = 512


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

    @_app.post("/rewrite")
    async def rewrite(self, body: dict[str, Any]) -> dict[str, str]:
        generation_options = {"max_new_tokens": DEFAULT_MAX_NEW_TOKENS} | {
            key: value for key, value in body.items() if key != "text"
        }
        inputs = self._tokenizer(body["text"], return_tensors="pt").to(self._device)
        output = self._model.generate(**inputs, **generation_options)
        return {"text": self._tokenizer.decode(output[0], skip_special_tokens=True)}
