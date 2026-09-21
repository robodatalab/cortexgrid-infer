"""Tests for cortexgrid_infer.models.text2text."""

from __future__ import annotations

import torch
import unittest
from unittest import mock

from cortexgrid_infer.completion import ServedCompletingModel
from cortexgrid_infer.models.text2text import Text2Text


class TestText2TextForTheImporter(unittest.TestCase):
    def test_client_speaks_the_deployed_app(self):
        model = Text2Text.client("http://h/r/F/S/R", "Qwen/Qwen2-2.5B-Instruct")
        self.assertIsInstance(model, ServedCompletingModel)
        self.assertEqual(model.url, "http://h/r/F/S/R")
        self.assertEqual(model.name, "Qwen/Qwen2-2.5B-Instruct")

    def test_loads_every_file_the_repo_ships(self):
        # Every file a causal LM repo ships is one `from_pretrained` may read.
        self.assertIsNone(Text2Text.ignore_patterns())


class _ModelConfig:
    def __init__(self, max_position_embeddings=32768) -> None:
        if max_position_embeddings is not None:
            self.max_position_embeddings = max_position_embeddings


class _GenerationConfig:
    def __init__(self) -> None:
        self.cache_implementation = None
        self.max_cache_len = None
        self.compile_config = None


class _FakeCausalLM:
    def __init__(self, context=32768) -> None:
        self.config = _ModelConfig(context)
        self.generation_config = _GenerationConfig()
        self.compiled_with: dict | None = None
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def compile(self, **kwargs) -> None:
        self.compiled_with = kwargs


class _Device:
    def __init__(self, type_: str) -> None:
        self.type = type_


class TestText2TextCompiles(unittest.TestCase):
    SERVE = "cortexgrid_infer.models.text2text"

    def build(self, device: str = "cuda", config: dict | None = None,
              context: int = 32768) -> _FakeCausalLM:
        model = _FakeCausalLM(context)
        with mock.patch(f"{self.SERVE}.cortexgrid.load_model", return_value="/w"), \
             mock.patch(f"{self.SERVE}.cortexgrid.model_config",
                        return_value=config or {}), \
             mock.patch(f"{self.SERVE}.AutoTokenizer.from_pretrained"), \
             mock.patch(f"{self.SERVE}.AutoModelForCausalLM.from_pretrained",
                        return_value=model), \
             mock.patch(f"{self.SERVE}.detect_device", return_value=_Device(device)):
            self.deployment = Text2Text(
                "family", "suffix", "imported"
            )
        return model

    def test_pins_the_cache_to_the_budget_not_the_models_context(self):
        # Every decoded token reads the whole cache, so a 32k context costs 8x
        # the 4k default in keys and values re-read per token.
        model = self.build()

        self.assertEqual(model.generation_config.cache_implementation, "static")
        self.assertEqual(model.generation_config.max_cache_len, 4096)

    def test_takes_the_budget_from_the_model_card(self):
        model = self.build(config={"max_total_tokens": "1024"})

        self.assertEqual(model.generation_config.max_cache_len, 1024)
        self.assertEqual(self.deployment._max_total_tokens, 1024)

    def test_never_asks_for_more_than_the_model_holds(self):
        model = self.build(config={"max_total_tokens": "65536"}, context=8192)

        self.assertEqual(model.generation_config.max_cache_len, 8192)

    def test_caps_a_reply_at_half_the_budget(self):
        # Otherwise a caller asking for more output than the replica is sized
        # for would leave the prompt no room at all.
        self.build()

        self.assertEqual(self.deployment._room_for_the_reply(256), 256)
        self.assertEqual(self.deployment._room_for_the_reply(16 * 1024), 2048)

    def test_asks_for_cuda_graphs_around_decode(self):
        model = self.build()

        self.assertEqual(model.generation_config.compile_config.mode, "reduce-overhead")

    def test_leaves_the_module_for_transformers_to_compile(self):
        # Compiling it here too would compile the same forward twice.
        self.assertIsNone(self.build().compiled_with)

    def test_stays_eager_off_cuda(self):
        model = self.build(device="cpu")

        self.assertFalse(self.deployment._compiled)
        self.assertIsNone(model.generation_config.cache_implementation)
        self.assertIsNone(model.generation_config.max_cache_len)


class TestPromptTruncation(unittest.TestCase):
    """An over-long prompt would resize the KV cache and recompile the decode
    loop, which costs more than the whole generation. It is cut instead."""

    SERVE = "cortexgrid_infer.models.text2text"

    def setUp(self) -> None:
        model = _FakeCausalLM(context=8)
        with mock.patch(f"{self.SERVE}.cortexgrid.load_model", return_value="/w"), \
             mock.patch(f"{self.SERVE}.cortexgrid.model_config", return_value={}), \
             mock.patch(f"{self.SERVE}.AutoTokenizer.from_pretrained"), \
             mock.patch(f"{self.SERVE}.AutoModelForCausalLM.from_pretrained",
                        return_value=model), \
             mock.patch(f"{self.SERVE}.detect_device", return_value=_Device("cuda")):
            self.deployment = Text2Text(
                "family", "suffix", "imported"
            )

    def inputs(self, length: int) -> dict:
        row = torch.arange(length).unsqueeze(0)
        return {"input_ids": row, "attention_mask": torch.ones_like(row)}

    def test_leaves_a_prompt_within_the_limit_alone(self):
        kept = self.deployment._truncate(self.inputs(8), 8)

        self.assertEqual(kept["input_ids"].shape[-1], 8)

    def test_cuts_an_over_long_prompt_to_the_limit(self):
        with self.assertLogs(self.SERVE, "WARNING"):
            kept = self.deployment._truncate(self.inputs(20), 8)

        self.assertEqual(kept["input_ids"].shape[-1], 8)
        self.assertEqual(kept["attention_mask"].shape[-1], 8)

    def test_keeps_the_end_of_the_prompt(self):
        # A chat template puts the turn to answer last; dropping the tail would
        # cost the instruction to reply at all.
        with self.assertLogs(self.SERVE, "WARNING"):
            kept = self.deployment._truncate(self.inputs(20), 8)

        self.assertEqual(kept["input_ids"][0].tolist(), list(range(12, 20)))

    def test_says_what_it_dropped(self):
        with self.assertLogs(self.SERVE, "WARNING") as logged:
            self.deployment._truncate(self.inputs(20), 8)

        self.assertIn("20", logged.output[0])
