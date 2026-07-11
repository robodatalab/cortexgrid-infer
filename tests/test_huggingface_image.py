"""Tests for model_gateway.providers.huggingface_image."""

from __future__ import annotations

import base64
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from model_gateway.providers.huggingface_image import (
    HuggingFaceImageModel,
    _ingest_huggingface_image,
    delete_huggingface_image,
    deploy_huggingface_image,
    upload_huggingface_image,
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
    @mock.patch(
        "model_gateway.providers.huggingface_image.cortexflow.model_registry_status",
        create=True,
    )
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.Experiment")
    def test_deploys_when_registry_ready(
        self,
        mock_experiment: mock.Mock,
        mock_registry: mock.Mock,
        mock_list_deployed: mock.Mock,
        mock_deploy: mock.Mock,
    ):
        mock_experiment.get_instance.return_value = _FakeExperiment("run-1")
        mock_registry.return_value = SimpleNamespace(phase="ready")
        mock_list_deployed.return_value = []
        mock_deploy.return_value = _FakeDeployment(
            "FLUX.2-klein-base", "4B", "run-1",
            "http://h/r/FLUX.2-klein-base/4B/run-1",
        )

        model = deploy_huggingface_image(
            "hf-image:black-forest-labs/FLUX.2-klein-base-4B"
        )

        assert model is not None
        self.assertEqual(model.url, "http://h/r/FLUX.2-klein-base/4B/run-1")
        self.assertEqual(
            (model.family, model.suffix, model.run_name),
            ("FLUX.2-klein-base", "4B", "run-1"),
        )
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
    @mock.patch(
        "model_gateway.providers.huggingface_image.cortexflow.model_registry_status",
        create=True,
    )
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.Experiment")
    def test_reuses_existing_deployment(
        self,
        mock_experiment: mock.Mock,
        mock_registry: mock.Mock,
        mock_list_deployed: mock.Mock,
        mock_deploy: mock.Mock,
    ):
        mock_experiment.get_instance.return_value = _FakeExperiment("run-1")
        mock_registry.return_value = SimpleNamespace(phase="ready")
        mock_list_deployed.return_value = [
            _FakeDeployment("FLUX.2-klein-base", "4B", "run-1", "http://existing/url")
        ]

        model = deploy_huggingface_image(
            "hf-image:black-forest-labs/FLUX.2-klein-base-4B"
        )

        assert model is not None
        self.assertEqual(model.url, "http://existing/url")
        mock_deploy.assert_not_called()

    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.deploy_model")
    @mock.patch(
        "model_gateway.providers.huggingface_image.cortexflow.model_registry_status",
        create=True,
    )
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.Experiment")
    def test_raises_when_not_registry_ready(
        self,
        mock_experiment: mock.Mock,
        mock_registry: mock.Mock,
        mock_deploy: mock.Mock,
    ):
        mock_experiment.get_instance.return_value = _FakeExperiment("run-1")
        for status in (None, SimpleNamespace(phase="uploading")):
            mock_registry.return_value = status
            with self.assertRaises(RuntimeError):
                deploy_huggingface_image(
                    "hf-image:black-forest-labs/FLUX.2-klein-base-4B"
                )
        mock_deploy.assert_not_called()


class TestUploadHuggingFaceImage(unittest.TestCase):
    def test_returns_none_for_non_hf_image_prefix(self):
        self.assertIsNone(upload_huggingface_image("hf:Qwen/Qwen2-Instruct"))

    @mock.patch.dict("os.environ", {"HF_TOKEN": "tok"})
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.remote", create=True)
    @mock.patch(
        "model_gateway.providers.huggingface_image.cortexflow.model_registry_status",
        create=True,
    )
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.Experiment")
    def test_submits_remote_ingest_when_absent(
        self,
        mock_experiment: mock.Mock,
        mock_registry: mock.Mock,
        mock_remote: mock.Mock,
    ):
        mock_experiment.get_instance.return_value = _FakeExperiment("run-1")
        mock_registry.return_value = None
        mock_remote.return_value = "job-1"

        job = upload_huggingface_image(
            "hf-image:black-forest-labs/FLUX.2-klein-base-4B"
        )

        self.assertEqual(job, "job-1")
        mock_remote.assert_called_once()
        args = mock_remote.call_args.args
        self.assertIs(args[0], _ingest_huggingface_image)
        self.assertEqual(
            args[1:],
            ("black-forest-labs/FLUX.2-klein-base-4B", "FLUX.2-klein-base", "4B", "tok"),
        )
        self.assertEqual(mock_remote.call_args.kwargs, {"num_gpus": 0, "num_cpus": 2})

    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.remote", create=True)
    @mock.patch(
        "model_gateway.providers.huggingface_image.cortexflow.model_registry_status",
        create=True,
    )
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.Experiment")
    def test_skips_when_already_registered(
        self,
        mock_experiment: mock.Mock,
        mock_registry: mock.Mock,
        mock_remote: mock.Mock,
    ):
        mock_experiment.get_instance.return_value = _FakeExperiment("run-1")
        for phase in ("uploading", "ready"):
            mock_registry.return_value = SimpleNamespace(phase=phase)
            self.assertIsNone(
                upload_huggingface_image(
                    "hf-image:black-forest-labs/FLUX.2-klein-base-4B"
                )
            )
        mock_remote.assert_not_called()

    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.save_model")
    @mock.patch("model_gateway.providers.huggingface_image.snapshot_download")
    def test_ingest_downloads_and_saves(
        self, mock_snapshot: mock.Mock, mock_save: mock.Mock
    ):
        _ingest_huggingface_image(
            "black-forest-labs/FLUX.2-klein-base-4B", "FLUX.2-klein-base", "4B", "tok"
        )
        mock_snapshot.assert_called_once()
        self.assertEqual(
            mock_snapshot.call_args.kwargs["repo_id"],
            "black-forest-labs/FLUX.2-klein-base-4B",
        )
        self.assertEqual(mock_snapshot.call_args.kwargs["token"], "tok")
        self.assertIn("ignore_patterns", mock_snapshot.call_args.kwargs)
        mock_save.assert_called_once()
        self.assertIs(mock_save.call_args.args[1], HuggingFaceImageDeployment)
        self.assertEqual(
            mock_save.call_args.kwargs, {"family": "FLUX.2-klein-base", "suffix": "4B"}
        )


class TestUndeploy(unittest.TestCase):
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.undeploy_model")
    def test_undeploy_tears_down_serve_app(self, mock_undeploy: mock.Mock):
        model = HuggingFaceImageModel(
            url="http://h/r/F/S/R", model_id="hf-image:x",
            family="FLUX.2-klein-base", suffix="4B", run_name="run-1",
        )
        model.undeploy()
        mock_undeploy.assert_called_once_with("FLUX.2-klein-base", "4B", "run-1")

    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.undeploy_model")
    def test_undeploy_is_noop_without_identifiers(self, mock_undeploy: mock.Mock):
        HuggingFaceImageModel(url="u", model_id="hf-image:x").undeploy()
        mock_undeploy.assert_not_called()

    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.deploy_model")
    @mock.patch(
        "model_gateway.providers.huggingface_image.cortexflow.list_deployed_models"
    )
    @mock.patch(
        "model_gateway.providers.huggingface_image.cortexflow.model_registry_status",
        create=True,
    )
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.Experiment")
    def test_deployed_model_carries_identifiers(
        self,
        mock_experiment: mock.Mock,
        mock_registry: mock.Mock,
        mock_list_deployed: mock.Mock,
        mock_deploy: mock.Mock,
    ):
        mock_experiment.get_instance.return_value = _FakeExperiment("run-1")
        mock_registry.return_value = SimpleNamespace(phase="ready")
        mock_list_deployed.return_value = [
            _FakeDeployment("FLUX.2-klein-base", "4B", "run-1", "http://existing/url")
        ]

        model = deploy_huggingface_image(
            "hf-image:black-forest-labs/FLUX.2-klein-base-4B"
        )

        assert model is not None
        self.assertEqual(
            (model.family, model.suffix, model.run_name),
            ("FLUX.2-klein-base", "4B", "run-1"),
        )


class TestImageDeploymentStatus(unittest.TestCase):
    def test_returns_none_for_non_hf_image_prefix(self):
        from model_gateway.providers.huggingface_image import image_deployment_status
        self.assertIsNone(image_deployment_status("hf:Qwen/Qwen2-Instruct"))

    @mock.patch(
        "model_gateway.providers.huggingface_image.cortexflow.model_registry_status",
        create=True,
    )
    @mock.patch(
        "model_gateway.providers.huggingface_image.cortexflow.model_serving_status",
        create=True,
    )
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.Experiment")
    def test_reports_serving_status_once_deployed(
        self,
        mock_experiment: mock.Mock,
        mock_serving: mock.Mock,
        mock_registry: mock.Mock,
    ):
        from model_gateway.providers.huggingface_image import image_deployment_status
        mock_experiment.get_instance.return_value = _FakeExperiment("run-1")
        mock_serving.return_value = SimpleNamespace(phase="running")
        result = image_deployment_status(
            "hf-image:black-forest-labs/FLUX.2-klein-base-4B"
        )
        self.assertEqual(result.phase, "running")
        mock_serving.assert_called_once_with("FLUX.2-klein-base", "4B", "run-1")
        mock_registry.assert_not_called()

    @mock.patch(
        "model_gateway.providers.huggingface_image.cortexflow.model_registry_status",
        create=True,
    )
    @mock.patch(
        "model_gateway.providers.huggingface_image.cortexflow.model_serving_status",
        create=True,
    )
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.Experiment")
    def test_falls_back_to_registry_status_before_deploy(
        self,
        mock_experiment: mock.Mock,
        mock_serving: mock.Mock,
        mock_registry: mock.Mock,
    ):
        from model_gateway.providers.huggingface_image import image_deployment_status
        mock_experiment.get_instance.return_value = _FakeExperiment("run-1")
        mock_serving.return_value = SimpleNamespace(phase="not_deployed")
        mock_registry.return_value = SimpleNamespace(phase="uploading")
        result = image_deployment_status(
            "hf-image:black-forest-labs/FLUX.2-klein-base-4B"
        )
        self.assertEqual(result.phase, "uploading")
        mock_registry.assert_called_once_with("FLUX.2-klein-base", "4B", "run-1")

    @mock.patch(
        "model_gateway.providers.huggingface_image.cortexflow.model_registry_status",
        create=True,
    )
    @mock.patch(
        "model_gateway.providers.huggingface_image.cortexflow.model_serving_status",
        create=True,
    )
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.Experiment")
    def test_core_dispatch_routes_hf_image(
        self,
        mock_experiment: mock.Mock,
        mock_serving: mock.Mock,
        mock_registry: mock.Mock,
    ):
        import model_gateway
        mock_experiment.get_instance.return_value = _FakeExperiment("run-1")
        mock_serving.return_value = SimpleNamespace(phase="running")
        result = model_gateway.deployment_status("hf-image:org/Model-4B")
        self.assertEqual(result.phase, "running")

    def test_core_dispatch_none_for_unhandled_prefix(self):
        import model_gateway
        self.assertIsNone(model_gateway.deployment_status("openai:gpt-4"))


class TestDeleteHuggingFaceImage(unittest.TestCase):
    def test_noop_for_non_hf_image_prefix(self):
        self.assertIsNone(delete_huggingface_image("hf:Qwen/Qwen2-Instruct"))

    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.delete_model")
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.undeploy_model")
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.Experiment")
    def test_undeploys_then_deletes(
        self,
        mock_experiment: mock.Mock,
        mock_undeploy: mock.Mock,
        mock_delete: mock.Mock,
    ):
        mock_experiment.get_instance.return_value = _FakeExperiment("run-1")
        delete_huggingface_image("hf-image:black-forest-labs/FLUX.2-klein-base-4B")
        mock_undeploy.assert_called_once_with("FLUX.2-klein-base", "4B", "run-1")
        mock_delete.assert_called_once_with("FLUX.2-klein-base", "4B", "run-1")

    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.delete_model")
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.undeploy_model")
    @mock.patch("model_gateway.providers.huggingface_image.cortexflow.Experiment")
    def test_core_dispatch_routes_hf_image(
        self,
        mock_experiment: mock.Mock,
        mock_undeploy: mock.Mock,
        mock_delete: mock.Mock,
    ):
        import model_gateway
        mock_experiment.get_instance.return_value = _FakeExperiment("run-1")
        model_gateway.delete_model("hf-image:black-forest-labs/FLUX.2-klein-base-4B")
        mock_delete.assert_called_once_with("FLUX.2-klein-base", "4B", "run-1")


if __name__ == "__main__":
    unittest.main()
