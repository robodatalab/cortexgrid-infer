"""Tests for model_gateway.providers.huggingface_image."""

from __future__ import annotations

import base64
import unittest
from typing import Any
from unittest import mock

from model_gateway.providers.huggingface_image import (
    HuggingFaceImageModel,
    deploy_huggingface_image,
)
from model_gateway.providers.huggingface_image_serve import (
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
        "model_gateway.providers.huggingface_image.httpx.AsyncClient", factory
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
            url="http://h:30000/r/F/S/R", model_id="hf-image:test"
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
        model = HuggingFaceImageModel(url="http://x/r/f/s/r", model_id="hf-image:t")
        with _patch_httpx_client(payload):
            await model.generate("stylize", image=b"reference-png")

        sent = _FakeAsyncClient.captured["json"]["image"]
        self.assertEqual(base64.b64decode(sent), b"reference-png")


class _FakeSavedModel:
    def __init__(self, family: str, suffix: str, run_name: str) -> None:
        self.family = family
        self.suffix = suffix
        self.run_name = run_name


class _FakeDeployment:
    def __init__(self, family: str, suffix: str, run_name: str, url: str) -> None:
        self.family = family
        self.suffix = suffix
        self.run_name = run_name
        self.url = url


class _FakeExperiment:
    def __init__(self, run_name: str) -> None:
        self._run_name = run_name

    def run_name(self) -> str:
        return self._run_name


class TestDeployHuggingFaceImageFlow(unittest.TestCase):
    def test_returns_none_for_non_hf_image_prefix(self):
        self.assertIsNone(deploy_huggingface_image("hf:Qwen/Qwen2-Instruct"))
        self.assertIsNone(deploy_huggingface_image("openai:gpt-4"))

    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.deploy_model")
    @mock.patch(
        "model_gateway.providers.huggingface_image.cortexflow.list_deployed_models"
    )
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.save_model")
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.list_models")
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.Experiment")
    @mock.patch("model_gateway.providers.huggingface_image.snapshot_download")
    def test_uploads_and_deploys_when_absent(
        self,
        mock_snapshot: mock.Mock,
        mock_experiment: mock.Mock,
        mock_list_models: mock.Mock,
        mock_save: mock.Mock,
        mock_list_deployed: mock.Mock,
        mock_deploy: mock.Mock,
    ):
        mock_experiment.get_instance.return_value = _FakeExperiment("run-1")
        mock_list_models.return_value = []
        mock_list_deployed.return_value = []
        mock_deploy.return_value = _FakeDeployment(
            "FLUX.2-klein-base", "4B", "run-1",
            "http://h/r/FLUX.2-klein-base/4B/run-1",
        )

        model = deploy_huggingface_image(
            "hf-image:black-forest-labs/FLUX.2-klein-base-4B"
        )

        self.assertIsNotNone(model)
        assert model is not None
        self.assertEqual(model.url, "http://h/r/FLUX.2-klein-base/4B/run-1")
        mock_snapshot.assert_called_once()
        self.assertEqual(
            mock_snapshot.call_args.kwargs["repo_id"],
            "black-forest-labs/FLUX.2-klein-base-4B",
        )
        mock_save.assert_called_once()
        self.assertEqual(mock_save.call_args.kwargs["family"], "FLUX.2-klein-base")
        self.assertEqual(mock_save.call_args.kwargs["suffix"], "4B")
        self.assertIs(mock_save.call_args.args[1], HuggingFaceImageDeployment)
        self.assertEqual(
            mock_deploy.call_args.kwargs,
            {
                "family": "FLUX.2-klein-base",
                "suffix": "4B",
                "run_name": "run-1",
                "wait": True,
                "timeout": None,
            },
        )

    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.deploy_model")
    @mock.patch(
        "model_gateway.providers.huggingface_image.cortexflow.list_deployed_models"
    )
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.save_model")
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.list_models")
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.Experiment")
    @mock.patch("model_gateway.providers.huggingface_image.snapshot_download")
    def test_reuses_existing_deployment(
        self,
        mock_snapshot: mock.Mock,
        mock_experiment: mock.Mock,
        mock_list_models: mock.Mock,
        mock_save: mock.Mock,
        mock_list_deployed: mock.Mock,
        mock_deploy: mock.Mock,
    ):
        mock_experiment.get_instance.return_value = _FakeExperiment("run-1")
        mock_list_models.return_value = [
            _FakeSavedModel("FLUX.2-klein-base", "4B", "run-1")
        ]
        mock_list_deployed.return_value = [
            _FakeDeployment("FLUX.2-klein-base", "4B", "run-1", "http://existing/url")
        ]

        model = deploy_huggingface_image(
            "hf-image:black-forest-labs/FLUX.2-klein-base-4B"
        )

        assert model is not None
        self.assertEqual(model.url, "http://existing/url")
        mock_snapshot.assert_not_called()
        mock_save.assert_not_called()
        mock_deploy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
