"""Tests for cortexgrid_infer.serve_apps.text2image."""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

import cortexgrid

from cortexgrid_infer.importers.huggingface import HuggingFaceImporter
from cortexgrid_infer.protocols.imaging import ServedGeneratingModel
from cortexgrid_infer.serve_apps.text2image import Text2Image


class TestText2ImageForTheImporter(unittest.TestCase):
    HF_ID = "black-forest-labs/FLUX.2-klein-base-4B"

    def test_the_model_card_serves_eager_unless_told_otherwise(self):
        self.assertEqual(Text2Image.config(), {"compile": "false"})

    def test_client_speaks_the_deployed_app(self):
        model = Text2Image.client("http://h/r/F/S/R", self.HF_ID)
        self.assertIsInstance(model, ServedGeneratingModel)
        self.assertEqual(model.url, "http://h/r/F/S/R")
        self.assertEqual(model.name, self.HF_ID)

    def test_skips_what_the_pipeline_never_loads(self):
        self.assertIn("*.md", Text2Image.ignore_patterns())

    @mock.patch.dict("os.environ", {"HF_IMAGE_SNAPSHOT_IGNORE": "single-file.safetensors"})
    def test_extra_ignores_come_from_the_environment(self):
        self.assertIn("single-file.safetensors", Text2Image.ignore_patterns())

    @mock.patch("cortexgrid_infer.importers.huggingface.snapshot_download")
    def test_an_import_skips_the_ignored_files(self, mock_snapshot: mock.Mock):
        with HuggingFaceImporter(self.HF_ID, Text2Image, "tok") as imp:
            imp.source()

        self.assertEqual(mock_snapshot.call_args.kwargs["repo_id"], self.HF_ID)
        self.assertEqual(mock_snapshot.call_args.kwargs["token"], "tok")
        self.assertIn("*.md", mock_snapshot.call_args.kwargs["ignore_patterns"])


class _FakeModule:
    def __init__(self) -> None:
        self.compiled_with: dict[str, Any] | None = None

    def compile(self, **kwargs: Any) -> None:
        self.compiled_with = kwargs


class _FakeVae(_FakeModule):
    def __init__(self) -> None:
        super().__init__()
        self.tiled = False

    def enable_tiling(self) -> None:
        self.tiled = True


class _FakePipeline:
    def __init__(self) -> None:
        self.transformer = _FakeModule()
        self.text_encoder = _FakeModule()
        self.vae = _FakeVae()
        self.device: Any = None

    def to(self, device: Any) -> "_FakePipeline":
        self.device = device
        return self

    def set_progress_bar_config(self, **_: Any) -> None:
        pass


class _Device:
    def __init__(self, type_: str) -> None:
        self.type = type_


class TestText2ImageCompiles(unittest.TestCase):
    """The deployment compiles at construction; see
    cortexgrid_infer.compiling for why the denoiser is where it pays."""

    SERVE = "cortexgrid_infer.serve_apps.text2image"

    def build(self, device: str = "cuda", compile: str = "true") -> _FakePipeline:
        pipe = _FakePipeline()
        pipeline_class = mock.Mock()
        pipeline_class.from_pretrained.return_value = pipe
        with mock.patch(f"{self.SERVE}.cortexgrid.load_model", return_value="/w"), \
             mock.patch(f"{self.SERVE}.cortexgrid.model_config",
                        return_value={"compile": compile}), \
             mock.patch(f"{self.SERVE}._pipeline_class", return_value=pipeline_class), \
             mock.patch(f"{self.SERVE}.detect_device", return_value=_Device(device)):
            self.deployment = Text2Image(cortexgrid.DeploymentKey("family", "suffix", "imported"))
        return pipe

    def test_compiles_the_denoiser_and_the_text_encoder(self):
        pipe = self.build()

        self.assertEqual(self.deployment._compiled, ["transformer", "text_encoder"])
        self.assertEqual(pipe.transformer.compiled_with, {"mode": "reduce-overhead"})

    def test_stays_eager_off_cuda(self):
        pipe = self.build(device="mps")

        self.assertEqual(self.deployment._compiled, [])
        self.assertIsNone(pipe.transformer.compiled_with)

    def test_stays_eager_unless_the_model_card_asks(self):
        pipe = self.build(compile="false")

        self.assertEqual(self.deployment._compiled, [])
        self.assertIsNone(pipe.transformer.compiled_with)

    def test_still_tiles_the_vae_and_leaves_it_uncompiled(self):
        pipe = self.build()

        self.assertTrue(pipe.vae.tiled)
        self.assertIsNone(pipe.vae.compiled_with)
