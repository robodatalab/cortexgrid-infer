"""Tests for cortexgrid_infer.requirements."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from huggingface_hub.errors import NotASafetensorsRepoError

from cortexgrid_infer import requirements

_GIB = float(1 << 30)


def _sibling(name: str, size: int) -> SimpleNamespace:
    return SimpleNamespace(rfilename=name, size=size)


class TestWeightGb(unittest.TestCase):
    @mock.patch("cortexgrid_infer.requirements.get_safetensors_metadata")
    def test_converts_parameter_count_at_the_dtype_the_app_loads(
        self, mock_metadata: mock.Mock
    ):
        # The checkpoint's stored dtype doesn't matter: both serve apps load in
        # 16 bits, so a parameter costs two resident bytes whatever it was saved as.
        mock_metadata.return_value = SimpleNamespace(
            parameter_count={"BF16": 500_000_000}
        )

        self.assertAlmostEqual(
            requirements.weight_gb("Qwen/Qwen2.5-0.5B-Instruct"),
            1_000_000_000 / _GIB,
        )

    @mock.patch("cortexgrid_infer.requirements.get_safetensors_metadata")
    def test_sums_every_dtype_in_the_repo(self, mock_metadata: mock.Mock):
        mock_metadata.return_value = SimpleNamespace(
            parameter_count={"BF16": 1_000_000, "F32": 1_000_000}
        )

        self.assertAlmostEqual(requirements.weight_gb("org/mixed"), 4_000_000 / _GIB)

    @mock.patch("cortexgrid_infer.requirements.HfApi")
    @mock.patch("cortexgrid_infer.requirements.get_safetensors_metadata")
    def test_falls_back_to_file_sizes_without_a_safetensors_index(
        self, mock_metadata: mock.Mock, mock_api: mock.Mock
    ):
        # A diffusers pipeline keeps its weights in component subfolders, so the
        # safetensors helper refuses the repo outright.
        mock_metadata.side_effect = NotASafetensorsRepoError("not a safetensors repo")
        mock_api.return_value.model_info.return_value = SimpleNamespace(
            siblings=[
                _sibling("transformer/diffusion_pytorch_model.safetensors", 2 << 30),
                _sibling("vae/diffusion_pytorch_model.safetensors", 1 << 30),
                _sibling("model_index.json", 1024),
                _sibling("README.md", 4096),
            ]
        )

        self.assertAlmostEqual(requirements.weight_gb("bfl/FLUX.2-klein-base-4B"), 3.0)

    @mock.patch("cortexgrid_infer.requirements.HfApi")
    @mock.patch("cortexgrid_infer.requirements.get_safetensors_metadata")
    def test_skips_the_files_the_download_skips(
        self, mock_metadata: mock.Mock, mock_api: mock.Mock
    ):
        # A pipeline repo commonly ships a consolidated single-file checkpoint
        # next to the component weights. Counting what the download ignores
        # would size the model at several times what a replica actually holds.
        mock_metadata.side_effect = NotASafetensorsRepoError("not a safetensors repo")
        mock_api.return_value.model_info.return_value = SimpleNamespace(
            siblings=[
                _sibling("transformer/diffusion_pytorch_model.safetensors", 2 << 30),
                _sibling("flux-2-klein-base-4b.safetensors", 8 << 30),
            ]
        )

        self.assertAlmostEqual(
            requirements.weight_gb(
                "bfl/FLUX.2-klein-base-4B",
                ignore_patterns=["flux-2-klein-base-4b.safetensors"],
            ),
            2.0,
        )

    @mock.patch("cortexgrid_infer.requirements.HfApi")
    @mock.patch("cortexgrid_infer.requirements.get_safetensors_metadata")
    def test_glob_matches_a_nested_file_by_name(
        self, mock_metadata: mock.Mock, mock_api: mock.Mock
    ):
        mock_metadata.side_effect = NotASafetensorsRepoError("not a safetensors repo")
        mock_api.return_value.model_info.return_value = SimpleNamespace(
            siblings=[
                _sibling("transformer/model.safetensors", 1 << 30),
                _sibling("transformer/model.fp8.safetensors", 1 << 30),
            ]
        )

        self.assertAlmostEqual(
            requirements.weight_gb("org/repo", ignore_patterns=["*.fp8.safetensors"]),
            1.0,
        )


class TestEstimate(unittest.TestCase):
    @mock.patch("cortexgrid_infer.requirements.weight_gb")
    def test_asks_for_one_gpu_with_headroom_over_the_weights(
        self, mock_weight_gb: mock.Mock
    ):
        mock_weight_gb.return_value = 10.0

        result = requirements.estimate("org/repo")

        self.assertEqual(result.num_gpus, 1)
        self.assertGreater(result.vram_gb, 10.0)
        self.assertGreater(result.ram_gb, 10.0)

    @mock.patch("cortexgrid_infer.requirements.weight_gb")
    def test_a_tiny_model_still_asks_for_working_room(
        self, mock_weight_gb: mock.Mock
    ):
        # Activations and a KV cache do not shrink to nothing just because the
        # weights did, so the estimate never lands at zero.
        mock_weight_gb.return_value = 0.0

        result = requirements.estimate("org/tiny")

        self.assertGreater(result.vram_gb, 0.0)
        self.assertGreater(result.ram_gb, 0.0)

    @mock.patch("cortexgrid_infer.requirements.weight_gb")
    def test_scales_with_the_model(self, mock_weight_gb: mock.Mock):
        mock_weight_gb.return_value = 1.0
        small = requirements.estimate("org/small")
        mock_weight_gb.return_value = 140.0
        large = requirements.estimate("org/large")

        self.assertGreater(large.vram_gb, small.vram_gb)
        self.assertGreater(large.ram_gb, small.ram_gb)

    @mock.patch("cortexgrid_infer.requirements.weight_gb")
    def test_sizes_from_the_repo_the_caller_named(self, mock_weight_gb: mock.Mock):
        mock_weight_gb.return_value = 1.0

        requirements.estimate("org/repo", "tok", ["*.md"])

        mock_weight_gb.assert_called_once_with("org/repo", "tok", ["*.md"])
