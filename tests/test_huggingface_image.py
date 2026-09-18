"""Tests for cortexgrid_infer.providers.huggingface_image."""

from __future__ import annotations

import base64
import unittest
from typing import Any
from unittest import mock

from cortexgrid_infer.providers.huggingface_image import (
    HuggingFaceImageImport,
    HuggingFaceImageModel,
)
from cortexgrid_infer.providers.huggingface_image_serve import (
    HuggingFaceImageDeployment,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeAsyncClient:
    captured: dict[str, Any] = {}

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:
        type(self).captured = {"url": url, "json": json}
        return _FakeResponse(self._payload)


def _patch_httpx_client(payload: dict[str, Any]):
    def factory(*_a: Any, **_kw: Any) -> _FakeAsyncClient:
        return _FakeAsyncClient(payload)

    return mock.patch(
        "cortexgrid_infer.providers.huggingface_image.httpx.AsyncClient", factory
    )


class TestHuggingFaceImageModelGenerate(unittest.IsolatedAsyncioTestCase):
    async def test_posts_to_generate_endpoint_and_decodes(self):
        png = b"\x89PNG\r\n\x1a\nfake-bytes"
        payload = {
            "image": base64.b64encode(png).decode("ascii"),
            "width": 1024,
            "height": 768,
            "steps": 50,
            "guidance": 4.0,
            "seed": 7,
            "duration_s": 12.3,
        }
        model = HuggingFaceImageModel(
            url="http://h:30000/r/F/S/R", model_id="bfl/FLUX.2-klein-base-4B"
        )
        with _patch_httpx_client(payload):
            result = await model.generate("a red cube", steps=50, seed=7)

        self.assertEqual(result.image, png)
        self.assertEqual((result.width, result.height), (1024, 768))
        self.assertEqual(result.params["duration_s"], 12.3)
        self.assertEqual(result.params["seed"], 7)

        captured = _FakeAsyncClient.captured
        self.assertEqual(captured["url"], "http://h:30000/r/F/S/R/generate")
        self.assertEqual(captured["json"]["prompt"], "a red cube")
        self.assertEqual(captured["json"]["steps"], 50)
        self.assertIsNone(captured["json"]["image"])

    async def test_encodes_reference_image(self):
        payload = {
            "image": base64.b64encode(b"out").decode("ascii"),
            "width": 512,
            "height": 512,
        }
        model = HuggingFaceImageModel(url="http://x/r/f/s/r", model_id="bfl/FLUX.2-klein-base-4B")
        with _patch_httpx_client(payload):
            await model.generate("stylize", image=b"reference-png")

        sent = _FakeAsyncClient.captured["json"]["image"]
        self.assertEqual(base64.b64decode(sent), b"reference-png")


class TestHuggingFaceImageImport(unittest.TestCase):
    HF_ID = "black-forest-labs/FLUX.2-klein-base-4B"

    def test_names_the_model_for_the_registry(self):
        imp = HuggingFaceImageImport(self.HF_ID)
        self.assertEqual((imp.family, imp.suffix), ("FLUX.2-klein-base", "4B"))

    def test_bundles_the_image_serve_app(self):
        self.assertIs(
            HuggingFaceImageImport(self.HF_ID).serve_app, HuggingFaceImageDeployment
        )

    def test_client_speaks_the_deployed_app(self):
        model = HuggingFaceImageImport(self.HF_ID).client("http://h/r/F/S/R")
        self.assertIsInstance(model, HuggingFaceImageModel)
        self.assertEqual(model.url, "http://h/r/F/S/R")
        self.assertEqual(model.name, self.HF_ID)

    def test_defaults_to_skipping_what_the_pipeline_never_loads(self):
        self.assertIn("*.md", HuggingFaceImageImport(self.HF_ID).ignore_patterns)

    @mock.patch.dict("os.environ", {"HF_IMAGE_SNAPSHOT_IGNORE": "single-file.safetensors"})
    def test_extra_ignores_come_from_the_environment(self):
        self.assertIn(
            "single-file.safetensors", HuggingFaceImageImport(self.HF_ID).ignore_patterns
        )

    def test_explicit_ignores_replace_the_defaults(self):
        imp = HuggingFaceImageImport(self.HF_ID, ignore_patterns=["*.onnx"])
        self.assertEqual(imp.ignore_patterns, ["*.onnx"])

    @mock.patch("cortexgrid_infer.importing.snapshot_download")
    def test_source_skips_the_ignored_files(self, mock_snapshot: mock.Mock):
        with HuggingFaceImageImport(self.HF_ID, "tok") as imp:
            imp.source()

        self.assertEqual(mock_snapshot.call_args.kwargs["repo_id"], self.HF_ID)
        self.assertEqual(mock_snapshot.call_args.kwargs["token"], "tok")
        self.assertIn("*.md", mock_snapshot.call_args.kwargs["ignore_patterns"])
