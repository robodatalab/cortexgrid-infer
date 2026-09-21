"""Tests for cortexgrid_infer.models.base: how a serve app sizes a replica."""

from __future__ import annotations

import unittest

import cortexgrid

from cortexgrid_infer.models.base import HostedModel, LocalModel, Weights

_GIB = 1 << 30


class _Meshing(LocalModel):
    min_vram_gb = 6.0


class TestWeightGb(unittest.TestCase):
    def test_converts_parameter_count_at_the_dtype_the_app_loads(self):
        # The checkpoint's stored dtype doesn't matter: the apps load in 16
        # bits, so a parameter costs two resident bytes whatever it was saved as.
        self.assertAlmostEqual(
            LocalModel.weight_gb(Weights(params=500_000_000)), 1_000_000_000 / _GIB
        )

    def test_takes_the_file_sizes_as_they_are_without_a_parameter_count(self):
        self.assertAlmostEqual(LocalModel.weight_gb(Weights(file_bytes=3 * _GIB)), 3.0)


class TestRequirements(unittest.TestCase):
    def gb(self, size: float) -> Weights:
        return Weights(file_bytes=int(size * _GIB))

    def test_asks_for_one_gpu_with_headroom_over_the_weights(self):
        result = LocalModel.requirements(self.gb(10.0))

        self.assertEqual(result.num_gpus, 1)
        self.assertGreater(result.vram_gb, 10.0)
        self.assertGreater(result.ram_gb, 10.0)

    def test_a_tiny_model_still_asks_for_working_room(self):
        # Activations and a KV cache do not shrink to nothing just because the
        # weights did, so the estimate never lands at zero.
        result = LocalModel.requirements(self.gb(0.0))

        self.assertGreater(result.vram_gb, 0.0)
        self.assertGreater(result.ram_gb, 0.0)

    def test_scales_with_the_model(self):
        small = LocalModel.requirements(self.gb(1.0))
        large = LocalModel.requirements(self.gb(140.0))

        self.assertGreater(large.vram_gb, small.vram_gb)
        self.assertGreater(large.ram_gb, small.ram_gb)

    def test_asks_for_the_memory_the_work_takes_not_just_the_weights(self):
        needs = _Meshing.requirements(self.gb(1.0))

        self.assertEqual(needs.vram_gb, 6.0)
        self.assertEqual(needs.ram_gb, LocalModel.requirements(self.gb(1.0)).ram_gb)

    def test_the_minimum_gives_way_to_weights_that_need_more(self):
        self.assertEqual(
            _Meshing.requirements(self.gb(40.0)), LocalModel.requirements(self.gb(40.0))
        )


class TestHostedModel(unittest.TestCase):
    def test_asks_for_no_hardware(self):
        # A replica holds no weights and does no compute, so it is placeable on
        # a CPU-only node.
        self.assertEqual(HostedModel.requirements(), cortexgrid.ModelRequirements())
