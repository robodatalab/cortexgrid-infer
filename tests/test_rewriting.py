from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest import mock

import cortexgrid
import torch

from cortexgrid_infer.protocols.rewriting import ServedRewritingModel
from cortexgrid_infer.serve_apps.text_rewriter import TextRewriter, _bucket

SERVE = "cortexgrid_infer.serve_apps.text_rewriter"


class _FakeTokenizedText(dict):
    def to(self, device: torch.device) -> "_FakeTokenizedText":
        self.moved_to = device
        return self


class _FakeTokenizer:
    def __init__(self, length: int = 3) -> None:
        self.length = length

    def __call__(
        self, text: str, return_tensors: str | None = None, **padding: Any
    ) -> _FakeTokenizedText:
        self.tokenized = (text, return_tensors)
        self.padding = padding
        ids = list(range(1, self.length + 1))
        return _FakeTokenizedText(input_ids=[ids] if return_tensors else ids)

    def decode(self, ids: list[int], skip_special_tokens: bool) -> str:
        self.decoded = (ids, skip_special_tokens)
        return "She goes home."


class _GenerationConfig:
    def __init__(self) -> None:
        self.cache_implementation = None
        self.max_cache_len = None
        self.compile_config = None


class _FakeEncoderDecoder:
    def __init__(self) -> None:
        self.generation_config = _GenerationConfig()

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


def _deployed_rewriter(
    card: dict[str, str] | None = None, device: str = "cpu", length: int = 3
) -> tuple[TextRewriter, _FakeTokenizer, _FakeEncoderDecoder]:
    tokenizer = _FakeTokenizer(length)
    model = _FakeEncoderDecoder()
    with mock.patch(f"{SERVE}.cortexgrid.load_model", return_value="/w"), \
         mock.patch(f"{SERVE}.cortexgrid.model_config", return_value=card or {}), \
         mock.patch(f"{SERVE}.AutoTokenizer.from_pretrained", return_value=tokenizer), \
         mock.patch(f"{SERVE}.AutoModelForSeq2SeqLM.from_pretrained", return_value=model), \
         mock.patch(f"{SERVE}.detect_device", return_value=torch.device(device)):
        deployment = TextRewriter(cortexgrid.DeploymentKey("family", "suffix", "imported"))
    return deployment, tokenizer, model


def _compiled_rewriter(length: int = 3) -> tuple[TextRewriter, _FakeTokenizer, _FakeEncoderDecoder]:
    return _deployed_rewriter({"compile": "true"}, device="cuda", length=length)


class TestTextRewriterLoads(unittest.TestCase):
    def test_the_model_card_serves_eager_unless_told_otherwise(self):
        self.assertEqual(TextRewriter.config(), {"compile": "false"})

    def test_loads_the_staged_encoder_decoder_in_bfloat16_on_the_detected_device(self):
        model = mock.Mock()
        with mock.patch(f"{SERVE}.cortexgrid.load_model", return_value="/w"), \
             mock.patch(f"{SERVE}.cortexgrid.model_config", return_value={}), \
             mock.patch(f"{SERVE}.AutoTokenizer.from_pretrained") as tokenizer, \
             mock.patch(f"{SERVE}.AutoModelForSeq2SeqLM.from_pretrained",
                        return_value=model) as loads, \
             mock.patch(f"{SERVE}.detect_device", return_value=torch.device("cpu")):
            TextRewriter(cortexgrid.DeploymentKey("family", "suffix", "imported"))

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

    def test_budgets_512_new_tokens_when_the_request_names_none(self):
        deployment, _, model = _deployed_rewriter()

        asyncio.run(deployment.rewrite({"text": "gec: She go home."}))

        self.assertEqual(model.generate_arguments["max_new_tokens"], 512)


class TestTextRewriterCompiles(unittest.TestCase):
    def test_asks_for_cuda_graphs_around_decode(self):
        _, _, model = _compiled_rewriter()

        self.assertEqual(model.generation_config.cache_implementation, "static")
        self.assertEqual(model.generation_config.max_cache_len, 512)
        self.assertEqual(model.generation_config.compile_config.mode, "reduce-overhead")

    def test_stays_eager_unless_the_model_card_asks(self):
        deployment, _, model = _deployed_rewriter({"compile": "false"}, device="cuda")

        self.assertFalse(deployment._compiled)
        self.assertIsNone(model.generation_config.cache_implementation)
        self.assertIsNone(model.generation_config.compile_config)

    def test_stays_eager_off_cuda(self):
        deployment, _, model = _deployed_rewriter({"compile": "true"}, device="cpu")

        self.assertFalse(deployment._compiled)
        self.assertIsNone(model.generation_config.cache_implementation)

    def test_eager_sends_the_text_unpadded(self):
        deployment, tokenizer, _ = _deployed_rewriter()

        asyncio.run(deployment.rewrite({"text": "gec: She go home."}))

        self.assertEqual(tokenizer.padding, {})

    def test_pads_the_text_up_to_its_bucket(self):
        # The cross-attention cache takes the text's length, so an unpadded
        # text would capture the decode loop afresh for every length it has.
        deployment, tokenizer, _ = _compiled_rewriter(length=40)

        asyncio.run(deployment.rewrite({"text": "gec: She go home."}))

        self.assertEqual(
            tokenizer.padding,
            {"padding": "max_length", "truncation": True, "max_length": 64},
        )

    def test_cuts_a_text_longer_than_the_last_bucket(self):
        deployment, tokenizer, _ = _compiled_rewriter(length=700)

        with self.assertLogs(SERVE, "WARNING") as logged:
            asyncio.run(deployment.rewrite({"text": "gec: She go home."}))

        self.assertEqual(tokenizer.padding["max_length"], 512)
        self.assertIn("700", logged.output[0])

    def test_caps_a_reply_at_the_cache(self):
        # A longer one would reallocate the cache and recompile decode.
        deployment, _, model = _compiled_rewriter()

        asyncio.run(deployment.rewrite({"text": "gec: She go home.", "max_new_tokens": 4096}))

        self.assertEqual(model.generate_arguments["max_new_tokens"], 512)


class TestBuckets(unittest.TestCase):
    def test_a_length_takes_the_smallest_bucket_that_holds_it(self):
        self.assertEqual(_bucket(1), 32)
        self.assertEqual(_bucket(32), 32)
        self.assertEqual(_bucket(33), 64)
        self.assertEqual(_bucket(512), 512)

    def test_a_length_past_the_last_bucket_takes_the_last(self):
        self.assertEqual(_bucket(513), 512)


def _served_by(deployment: TextRewriter, posted: dict[str, Any]) -> Any:
    """Patch the client's HTTP calls through to `deployment`, recording them in `posted`."""

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

    return mock.patch("cortexgrid_infer.protocols.rewriting.httpx.AsyncClient",
                      _FakeAsyncClientServedByTextRewriter)


class TestServedRewritingModel(unittest.IsolatedAsyncioTestCase):
    async def test_round_trips_a_rewrite_through_the_route(self):
        deployment, _, model = _deployed_rewriter()
        posted: dict[str, Any] = {}

        client = TextRewriter.client("http://h/r/Unbabel/gec-t5_small/R", "Unbabel/gec-t5_small")
        with _served_by(deployment, posted):
            rewritten = await client.rewrite("gec: She go home.", max_new_tokens=128)

        self.assertIsInstance(client, ServedRewritingModel)
        self.assertEqual(client.name, "Unbabel/gec-t5_small")
        self.assertEqual(rewritten, "She goes home.")
        self.assertEqual(posted["url"], "http://h/r/Unbabel/gec-t5_small/R/rewrite")
        self.assertEqual(model.generate_arguments["max_new_tokens"], 128)

    async def test_asks_for_512_new_tokens_unless_told_otherwise(self):
        deployment, _, model = _deployed_rewriter()
        posted: dict[str, Any] = {}

        client = TextRewriter.client("http://h/r/Unbabel/gec-t5_small/R", "Unbabel/gec-t5_small")
        with _served_by(deployment, posted):
            await client.rewrite("gec: She go home.")

        self.assertEqual(posted["json"]["max_new_tokens"], 512)
        self.assertEqual(model.generate_arguments["max_new_tokens"], 512)


if __name__ == "__main__":
    unittest.main()
