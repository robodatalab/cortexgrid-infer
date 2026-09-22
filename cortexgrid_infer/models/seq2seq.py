from __future__ import annotations

from typing import Any

import cortexgrid
from cortexgrid import serve
from fastapi import FastAPI
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

from cortexgrid_infer.device import detect_device
from cortexgrid_infer.models.base import LocalModel
from cortexgrid_infer.seq2seq import ServedSeq2SeqModel


_app = FastAPI()


@serve.ingress(_app)
class Seq2Seq(LocalModel):
    @classmethod
    def client(cls, url: str, name: str) -> ServedSeq2SeqModel:
        return ServedSeq2SeqModel(url=url, model_id=name)

    def __init__(self, family: str, suffix: str, run_name: str) -> None:
        path = cortexgrid.load_model(family, suffix, run_name)
        self._device = detect_device()
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            str(path), torch_dtype=torch.bfloat16
        )
        self._model.to(self._device)

    @_app.post("/generate")
    async def generate(self, body: dict[str, Any]) -> dict[str, str]:
        generation_options = {key: value for key, value in body.items() if key != "text"}
        inputs = self._tokenizer(body["text"], return_tensors="pt").to(self._device)
        output = self._model.generate(**inputs, **generation_options)
        return {"text": self._tokenizer.decode(output[0], skip_special_tokens=True)}
