"""Tests for cortexgrid_infer.providers.huggingface_complete."""

from __future__ import annotations

import torch
import unittest
from unittest import mock

from cortexgrid_infer.completion import ServedCompletingModel
from cortexgrid_infer.providers.huggingface_complete import HuggingFaceCompletingImport
from cortexgrid_infer.providers.huggingface_complete_serve import (
    HuggingFaceCompletingDeployment,
)


class TestHuggingFaceCompletingImport(unittest.TestCase):
    def test_names_the_model_for_the_registry(self):
        imp = HuggingFaceCompletingImport("Qwen/Qwen2-2.5B-Instruct")
        self.assertEqual((imp.family, imp.suffix), ("Qwen2-2.5B", "Instruct"))

    def test_bundles_the_completing_serve_app(self):
        self.assertIs(
            HuggingFaceCompletingImport("Qwen/Qwen2-2.5B-Instruct").serve_app,
            HuggingFaceCompletingDeployment,
        )

    def test_client_speaks_the_deployed_app(self):
        imp = HuggingFaceCompletingImport("Qwen/Qwen2-2.5B-Instruct")
        model = imp.client("http://h/r/F/S/R")
        self.assertIsInstance(model, ServedCompletingModel)
        self.assertEqual(model.url, "http://h/r/F/S/R")
        self.assertEqual(model.name, "Qwen/Qwen2-2.5B-Instruct")

    @mock.patch("cortexgrid_infer.importing.snapshot_download")
    def test_source_downloads_the_whole_repo(self, mock_snapshot: mock.Mock):
        # A causal LM sets no ignore patterns: every file the repo ships is one
        # `from_pretrained` may read.
        with HuggingFaceCompletingImport("Qwen/Qwen2-2.5B-Instruct", "tok") as imp:
            imp.source()

        self.assertEqual(
            mock_snapshot.call_args.kwargs["repo_id"], "Qwen/Qwen2-2.5B-Instruct"
        )
        self.assertEqual(mock_snapshot.call_args.kwargs["token"], "tok")
        self.assertIsNone(mock_snapshot.call_args.kwargs["ignore_patterns"])


class _GenerationConfig:
    def __init__(self) -> None:
        self.cache_implementation = None
        self.max_cache_len = None
        self.compile_config = None


class _FakeCausalLM:
    def __init__(self) -> None:
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


class TestHuggingFaceCompletingDeploymentCompiles(unittest.TestCase):
    SERVE = "cortexgrid_infer.providers.huggingface_complete_serve"

    def build(self, device: str = "cuda") -> _FakeCausalLM:
        model = _FakeCausalLM()
        with mock.patch(f"{self.SERVE}.cortexgrid.load_model", return_value="/w"), \
             mock.patch(f"{self.SERVE}.AutoTokenizer.from_pretrained"), \
             mock.patch(f"{self.SERVE}.AutoModelForCausalLM.from_pretrained",
                        return_value=model), \
             mock.patch(f"{self.SERVE}.detect_device", return_value=_Device(device)):
            self.deployment = HuggingFaceCompletingDeployment(
                "family", "suffix", "imported"
            )
        return model

    def test_pins_the_cache_length_so_every_request_shares_one_shape(self):
        # A cache sized per request gives each prompt its own shape, and decode
        # then recompiles per request instead of reusing one capture.
        model = self.build()

        self.assertEqual(model.generation_config.cache_implementation, "static")
        self.assertEqual(model.generation_config.max_cache_len, 4096)

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


class TestMaxInputTokensSetting(unittest.TestCase):
    """How long the KV cache is pinned is the deployment's call, so it lives on
    the model card and is editable there without re-importing."""

    SERVE = "cortexgrid_infer.providers.huggingface_complete_serve"

    def build(self, config: dict | Exception) -> _FakeCausalLM:
        model = _FakeCausalLM()
        cfg = mock.Mock(side_effect=config) if isinstance(config, Exception) \
            else mock.Mock(return_value=config)
        with mock.patch(f"{self.SERVE}.cortexgrid.load_model", return_value="/w"), \
             mock.patch(f"{self.SERVE}.cortexgrid.model_config", cfg), \
             mock.patch(f"{self.SERVE}.AutoTokenizer.from_pretrained"), \
             mock.patch(f"{self.SERVE}.AutoModelForCausalLM.from_pretrained",
                        return_value=model), \
             mock.patch(f"{self.SERVE}.detect_device", return_value=_Device("cuda")):
            HuggingFaceCompletingDeployment("family", "suffix", "imported")
        return model

    def test_the_card_sets_the_prompt_limit(self):
        model = self.build({"max_input_tokens": "16384"})

        self.assertEqual(model.generation_config.max_cache_len, 16384)

    def test_falls_back_to_the_default_when_the_card_says_nothing(self):
        model = self.build({})

        self.assertEqual(model.generation_config.max_cache_len, 4096)

    def test_a_nonsense_value_costs_the_setting_not_the_replica(self):
        # Editable in a dashboard, so a bad value must not brick a deployment.
        for bad in ("lots", "0", "-1", ""):
            with self.subTest(value=bad):
                with self.assertLogs(f"{self.SERVE}", "WARNING"):
                    model = self.build({"max_input_tokens": bad})
                self.assertEqual(model.generation_config.max_cache_len, 4096)

    def test_an_unreadable_card_costs_the_setting_not_the_replica(self):
        with self.assertLogs(f"{self.SERVE}", "WARNING"):
            model = self.build(ValueError("no such model"))

        self.assertEqual(model.generation_config.max_cache_len, 4096)


class TestCompletingImportSeedsTheCard(unittest.TestCase):
    def test_says_nothing_by_default(self):
        self.assertEqual(HuggingFaceCompletingImport("org/model-x").config(), {})

    def test_carries_the_prompt_limit_to_the_card(self):
        imp = HuggingFaceCompletingImport("org/model-x", max_input_tokens=16384)

        self.assertEqual(imp.config(), {"max_input_tokens": "16384"})

    def test_rejects_a_limit_that_allows_no_prompt(self):
        with self.assertRaises(ValueError):
            HuggingFaceCompletingImport("org/model-x", max_input_tokens=0)


class TestPromptTruncation(unittest.TestCase):
    """An over-long prompt would resize the KV cache and recompile the decode
    loop, which costs more than the whole generation. It is cut instead."""

    SERVE = "cortexgrid_infer.providers.huggingface_complete_serve"

    def setUp(self) -> None:
        model = _FakeCausalLM()
        with mock.patch(f"{self.SERVE}.cortexgrid.load_model", return_value="/w"), \
             mock.patch(f"{self.SERVE}.cortexgrid.model_config",
                        return_value={"max_input_tokens": "8"}), \
             mock.patch(f"{self.SERVE}.AutoTokenizer.from_pretrained"), \
             mock.patch(f"{self.SERVE}.AutoModelForCausalLM.from_pretrained",
                        return_value=model), \
             mock.patch(f"{self.SERVE}.detect_device", return_value=_Device("cuda")):
            self.deployment = HuggingFaceCompletingDeployment(
                "family", "suffix", "imported"
            )

    def inputs(self, length: int) -> dict:
        row = torch.arange(length).unsqueeze(0)
        return {"input_ids": row, "attention_mask": torch.ones_like(row)}

    def test_leaves_a_prompt_within_the_limit_alone(self):
        kept = self.deployment._truncate(self.inputs(8))

        self.assertEqual(kept["input_ids"].shape[-1], 8)

    def test_cuts_an_over_long_prompt_to_the_limit(self):
        with self.assertLogs(self.SERVE, "WARNING"):
            kept = self.deployment._truncate(self.inputs(20))

        self.assertEqual(kept["input_ids"].shape[-1], 8)
        self.assertEqual(kept["attention_mask"].shape[-1], 8)

    def test_keeps_the_end_of_the_prompt(self):
        # A chat template puts the turn to answer last; dropping the tail would
        # cost the instruction to reply at all.
        with self.assertLogs(self.SERVE, "WARNING"):
            kept = self.deployment._truncate(self.inputs(20))

        self.assertEqual(kept["input_ids"][0].tolist(), list(range(12, 20)))

    def test_says_how_much_it_dropped_and_what_to_raise(self):
        with self.assertLogs(self.SERVE, "WARNING") as logged:
            self.deployment._truncate(self.inputs(20))

        message = logged.output[0]
        self.assertIn("20", message)
        self.assertIn("max_input_tokens", message)
