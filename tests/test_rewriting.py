from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest import mock

import torch

from cortexgrid_infer.protocols.rewriting import ServedRewritingModel
from cortexgrid_infer.serve_apps.text_rewriter import TextRewriter

SERVE = "cortexgrid_infer.serve_apps.text_rewriter"


class _FakeTokenizedText(dict):
    def to(self, device: torch.device) -> "_FakeTokenizedText":
        self.moved_to = device
        return self


class _FakeTokenizer:
    def __call__(self, text: str, return_tensors: str) -> _FakeTokenizedText:
        self.tokenized = (text, return_tensors)
        return _FakeTokenizedText(input_ids=[[1, 2, 3]])

    def decode(self, ids: list[int], skip_special_tokens: bool) -> str:
        self.decoded = (ids, skip_special_tokens)
        return "She goes home."


class _FakeEncoderDecoder:
    def to(self, device: torch.device) -> "_FakeEncoderDecoder":
        self.device = device
        return self

    def generate(self, **generate_arguments: Any) -> list[list[int]]:
        self.generate_arguments = generate_arguments
        return [[7, 8, 9]]


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


def _deployed_rewriter() -> tuple[TextRewriter, _FakeTokenizer, _FakeEncoderDecoder]:
    tokenizer = _FakeTokenizer()
    model = _FakeEncoderDecoder()
    with mock.patch(f"{SERVE}.cortexgrid.load_model", return_value="/w"), \
         mock.patch(f"{SERVE}.AutoTokenizer.from_pretrained", return_value=tokenizer), \
         mock.patch(f"{SERVE}.AutoModelForSeq2SeqLM.from_pretrained", return_value=model), \
         mock.patch(f"{SERVE}.detect_device", return_value=torch.device("cpu")):
        deployment = TextRewriter("family", "suffix", "imported")
    return deployment, tokenizer, model


class TestTextRewriterLoads(unittest.TestCase):
    def test_loads_the_staged_encoder_decoder_in_bfloat16_on_the_detected_device(self):
        model = mock.Mock()
        with mock.patch(f"{SERVE}.cortexgrid.load_model", return_value="/w"), \
             mock.patch(f"{SERVE}.AutoTokenizer.from_pretrained") as tokenizer, \
             mock.patch(f"{SERVE}.AutoModelForSeq2SeqLM.from_pretrained",
                        return_value=model) as loads, \
             mock.patch(f"{SERVE}.detect_device", return_value=torch.device("cpu")):
            TextRewriter("family", "suffix", "imported")

        tokenizer.assert_called_once_with("/w")
        loads.assert_called_once_with("/w", torch_dtype=torch.bfloat16)
        model.to.assert_called_once_with(torch.device("cpu"))


class TestTextRewriterRewrites(unittest.TestCase):
    def test_reads_the_text_as_sent_and_answers_with_the_decoded_generation(self):
        deployment, tokenizer, _ = _deployed_rewriter()

        reply = asyncio.run(deployment.rewrite({"text": "gec: She go home."}))

        self.assertEqual(tokenizer.tokenized, ("gec: She go home.", "pt"))
        self.assertEqual(tokenizer.decoded, ([7, 8, 9], True))
        self.assertEqual(reply, {"text": "She goes home."})

    def test_hands_generate_every_option_the_request_carried(self):
        deployment, _, model = _deployed_rewriter()

        asyncio.run(deployment.rewrite(
            {"text": "gec: She go home.", "max_new_tokens": 128, "do_sample": False}
        ))

        self.assertEqual(
            model.generate_arguments,
            {"input_ids": [[1, 2, 3]], "max_new_tokens": 128, "do_sample": False},
        )


class TestServedRewritingModel(unittest.IsolatedAsyncioTestCase):
    async def test_round_trips_a_rewrite_through_the_route(self):
        deployment, _, model = _deployed_rewriter()
        posted: dict[str, Any] = {}

        class _FakeAsyncClientServedByTextRewriter:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> "_FakeAsyncClientServedByTextRewriter":
                return self

            async def __aexit__(self, *args: Any) -> None:
                pass

            async def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:
                posted.update(url=url, json=json)
                return _FakeResponse(await deployment.rewrite(dict(json)))

        client = TextRewriter.client("http://h/r/Unbabel/gec-t5_small/R", "Unbabel/gec-t5_small")
        with mock.patch("cortexgrid_infer.protocols.rewriting.httpx.AsyncClient",
                        _FakeAsyncClientServedByTextRewriter):
            rewritten = await client.rewrite("gec: She go home.", max_new_tokens=128)

        self.assertIsInstance(client, ServedRewritingModel)
        self.assertEqual(client.name, "Unbabel/gec-t5_small")
        self.assertEqual(rewritten, "She goes home.")
        self.assertEqual(posted["url"], "http://h/r/Unbabel/gec-t5_small/R/rewrite")
        self.assertEqual(model.generate_arguments["max_new_tokens"], 128)


if __name__ == "__main__":
    unittest.main()
