"""Tests for cortexgrid_infer.providers.huggingface_complete."""

from __future__ import annotations

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


class TestHuggingFaceCompletingDeploymentIsTunedForDecode(unittest.TestCase):
    """Decode has no stable shape to capture until the KV cache is static, and
    given one transformers compiles `generate` itself; see
    cortexgrid_infer.compiling."""

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

    def test_makes_the_kv_cache_static_so_generate_can_be_captured(self):
        model = self.build()

        self.assertTrue(self.deployment._compiled)
        self.assertEqual(model.generation_config.cache_implementation, "static")

    def test_asks_for_cuda_graphs_around_the_decode_step(self):
        model = self.build()

        self.assertEqual(model.generation_config.compile_config.mode, "reduce-overhead")

    def test_leaves_the_module_for_transformers_to_compile(self):
        # Compiling it here too would compile the same forward twice.
        model = self.build()

        self.assertIsNone(model.compiled_with)

    def test_stays_eager_off_cuda(self):
        model = self.build(device="cpu")

        self.assertFalse(self.deployment._compiled)
        self.assertIsNone(model.generation_config.cache_implementation)
        self.assertIsNone(model.generation_config.compile_config)
