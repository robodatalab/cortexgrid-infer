"""Tests for cortexgrid_infer.compiling."""

from __future__ import annotations

import unittest
from typing import Any

from cortexgrid_infer import compiling


class _Device:
    def __init__(self, type_: str) -> None:
        self.type = type_


CUDA = _Device("cuda")
MPS = _Device("mps")
CPU = _Device("cpu")


class _Module:
    """Stands in for an nn.Module: records what it was asked to compile with."""

    def __init__(self, fails: bool = False) -> None:
        self.compiled_with: dict[str, Any] | None = None
        self._fails = fails

    def compile(self, **kwargs: Any) -> None:
        if self._fails:
            raise RuntimeError("no inductor backend here")
        self.compiled_with = kwargs


class CompileModuleTest(unittest.TestCase):
    def test_compiles_the_module_in_place(self):
        module = _Module()

        self.assertTrue(compiling.compile_module(module, CUDA, compiling.Mode.GRAPHED, "denoiser"))
        self.assertEqual(module.compiled_with, {"mode": compiling.Mode.GRAPHED})

    def test_leaves_anything_but_cuda_eager(self):
        # Inductor is weakest off CUDA, and a host that outruns its accelerator
        # is a GPU problem in the first place.
        for device in (MPS, CPU, None):
            with self.subTest(device=device):
                module = _Module()
                self.assertFalse(compiling.compile_module(module, device, compiling.Mode.GRAPHED))
                self.assertIsNone(module.compiled_with)

    def test_passes_the_callers_mode_through(self):
        # compiling offers the options; the serve app picks. See GRAPHED/FUSED.
        for mode in (compiling.Mode.GRAPHED, compiling.Mode.FUSED):
            with self.subTest(mode=mode):
                module = _Module()
                compiling.compile_module(module, CUDA, mode)
                self.assertEqual(module.compiled_with, {"mode": mode})

    def test_tolerates_a_missing_module(self):
        self.assertFalse(compiling.compile_module(None, CUDA, compiling.Mode.GRAPHED))

    def test_a_backend_that_cannot_compile_still_serves(self):
        # Compilation is an optimisation; failing it must not fail a deployment.
        with self.assertLogs(compiling.log, "WARNING") as logged:
            compiled = compiling.compile_module(_Module(fails=True), CUDA, compiling.Mode.FUSED, "vae")

        self.assertFalse(compiled)
        self.assertIn("vae", logged.output[0])


class SupportedTest(unittest.TestCase):
    def test_only_on_cuda(self):
        # Inductor is weakest elsewhere, and a host that outruns its accelerator
        # is a GPU problem in the first place.
        self.assertTrue(compiling.supported(CUDA))
        for device in (MPS, CPU, None):
            with self.subTest(device=device):
                self.assertFalse(compiling.supported(device))


class _Pipeline:
    def __init__(self, **modules: Any) -> None:
        self.__dict__.update(modules)


class CompilePipelineTest(unittest.TestCase):
    def test_compiles_a_dit_denoiser_and_its_text_encoders(self):
        pipe = _Pipeline(transformer=_Module(), text_encoder=_Module(), vae=_Module())

        taken = compiling.compile_pipeline(pipe, CUDA, compiling.Mode.GRAPHED)

        self.assertEqual(taken, ["transformer", "text_encoder"])

    def test_compiles_a_unet_denoiser_just_the_same(self):
        # Nothing here knows the pipeline class, only which attributes exist, so
        # a UNet pipeline needs no special case.
        pipe = _Pipeline(unet=_Module(), text_encoder=_Module())

        taken = compiling.compile_pipeline(pipe, CUDA, compiling.Mode.GRAPHED)

        self.assertEqual(taken, ["unet", "text_encoder"])

    def test_leaves_the_vae_alone(self):
        # The smallest of the three wins and the one whose shapes move most,
        # especially with tiling on.
        vae = _Module()
        pipe = _Pipeline(transformer=_Module(), vae=vae)

        compiling.compile_pipeline(pipe, CUDA, compiling.Mode.GRAPHED)

        self.assertIsNone(vae.compiled_with)

    def test_takes_every_text_encoder_a_pipeline_has(self):
        pipe = _Pipeline(
            transformer=_Module(), text_encoder=_Module(), text_encoder_2=_Module()
        )

        taken = compiling.compile_pipeline(pipe, CUDA, compiling.Mode.GRAPHED)

        self.assertEqual(taken, ["transformer", "text_encoder", "text_encoder_2"])

    def test_leaves_anything_but_cuda_eager(self):
        pipe = _Pipeline(transformer=_Module())

        self.assertEqual(compiling.compile_pipeline(pipe, CPU, compiling.Mode.GRAPHED), [])
        self.assertIsNone(pipe.transformer.compiled_with)

    def test_one_part_failing_does_not_stop_the_others(self):
        pipe = _Pipeline(transformer=_Module(fails=True), text_encoder=_Module())

        with self.assertLogs(compiling.log, "WARNING"):
            taken = compiling.compile_pipeline(pipe, CUDA, compiling.Mode.GRAPHED)

        self.assertEqual(taken, ["text_encoder"])
