"""Tests for cortexgrid_infer.providers.huggingface_image."""

from __future__ import annotations

import base64
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import cortexgrid

from cortexgrid_infer.providers.huggingface_image import (
    HuggingFaceImageModel,
    _import_huggingface_image,
    delete_huggingface_image,
    deploy_huggingface_image,
    image_deployment_status,
    upload_huggingface_image,
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


class TestDeployHuggingFaceImageFlow(unittest.TestCase):
    def test_returns_none_for_non_hf_image_prefix(self):
        self.assertIsNone(deploy_huggingface_image("hf:Qwen/Qwen2-Instruct"))
        self.assertIsNone(deploy_huggingface_image("openai:gpt-4"))

    @mock.patch("cortexgrid_infer.providers.huggingface_image.ensure_serving")
    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.model_registry_status")
    def test_serves_imported_model_within_timeout(
        self,
        mock_registry: mock.Mock,
        mock_ensure_serving: mock.Mock,
    ):
        mock_registry.return_value = SimpleNamespace(phase="ready")
        mock_ensure_serving.return_value = "http://h/r/FLUX.2-klein-base/4B/imported"

        model = deploy_huggingface_image(
            "hf-image:black-forest-labs/FLUX.2-klein-base-4B", timeout=120.0
        )

        assert model is not None
        self.assertEqual(model.url, "http://h/r/FLUX.2-klein-base/4B/imported")
        mock_registry.assert_called_once_with(
            "FLUX.2-klein-base", "4B", cortexgrid.IMPORTED
        )
        mock_ensure_serving.assert_called_once_with(
            "FLUX.2-klein-base", "4B", cortexgrid.IMPORTED, 120.0
        )

    @mock.patch("cortexgrid_infer.providers.huggingface_image.ensure_serving")
    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.model_registry_status")
    def test_raises_when_not_registry_ready(
        self,
        mock_registry: mock.Mock,
        mock_deploy: mock.Mock,
    ):
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
    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.import_model")
    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.remote")
    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.model_registry_status")
    def test_submits_remote_import_when_absent_or_failed(
        self,
        mock_registry: mock.Mock,
        mock_remote: mock.Mock,
        mock_import: mock.Mock,
    ):
        mock_remote.return_value = "job-1"
        for status in (
            None,
            SimpleNamespace(phase="upload_failed"),
            SimpleNamespace(phase="broken"),
        ):
            mock_registry.return_value = status
            mock_remote.reset_mock()

            job = upload_huggingface_image(
                "hf-image:black-forest-labs/FLUX.2-klein-base-4B"
            )

            self.assertEqual(job, "job-1")
            mock_remote.assert_called_once_with(
                _import_huggingface_image,
                "black-forest-labs/FLUX.2-klein-base-4B",
                "FLUX.2-klein-base",
                "4B",
                "tok",
                num_gpus=0,
                num_cpus=2,
            )
        mock_registry.assert_called_with("FLUX.2-klein-base", "4B", cortexgrid.IMPORTED)
        mock_import.assert_not_called()

    @mock.patch("cortexgrid_infer.utils.snapshot_download")
    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.import_model")
    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.remote")
    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.model_registry_status")
    def test_imports_in_process_when_ready(
        self,
        mock_registry: mock.Mock,
        mock_remote: mock.Mock,
        mock_import: mock.Mock,
        mock_snapshot: mock.Mock,
    ):
        # Every run calls import_model, so it gets its tag and a re-bundle of
        # changed serve code - without a cluster job or a download.
        mock_registry.return_value = SimpleNamespace(phase="ready")

        self.assertIsNone(
            upload_huggingface_image("hf-image:black-forest-labs/FLUX.2-klein-base-4B")
        )

        mock_remote.assert_not_called()
        mock_import.assert_called_once()
        self.assertIs(mock_import.call_args.args[1], HuggingFaceImageDeployment)
        self.assertEqual(
            mock_import.call_args.kwargs, {"family": "FLUX.2-klein-base", "suffix": "4B"}
        )
        mock_snapshot.assert_not_called()

    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.import_model")
    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.remote")
    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.model_registry_status")
    def test_skips_while_another_process_uploads(
        self,
        mock_registry: mock.Mock,
        mock_remote: mock.Mock,
        mock_import: mock.Mock,
    ):
        mock_registry.return_value = SimpleNamespace(phase="uploading")
        self.assertIsNone(
            upload_huggingface_image("hf-image:black-forest-labs/FLUX.2-klein-base-4B")
        )
        mock_remote.assert_not_called()
        mock_import.assert_not_called()

    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.import_model")
    @mock.patch("cortexgrid_infer.utils.snapshot_download")
    def test_import_downloads_only_when_source_is_called(
        self, mock_snapshot: mock.Mock, mock_import: mock.Mock
    ):
        _import_huggingface_image(
            "black-forest-labs/FLUX.2-klein-base-4B", "FLUX.2-klein-base", "4B", "tok"
        )

        mock_snapshot.assert_not_called()
        mock_import.assert_called_once()
        self.assertIs(mock_import.call_args.args[1], HuggingFaceImageDeployment)
        self.assertEqual(
            mock_import.call_args.kwargs, {"family": "FLUX.2-klein-base", "suffix": "4B"}
        )

        mock_import.call_args.args[0]()
        mock_snapshot.assert_called_once()
        self.assertEqual(
            mock_snapshot.call_args.kwargs["repo_id"],
            "black-forest-labs/FLUX.2-klein-base-4B",
        )
        self.assertEqual(mock_snapshot.call_args.kwargs["token"], "tok")
        self.assertIn("*.md", mock_snapshot.call_args.kwargs["ignore_patterns"])


class TestUndeploy(unittest.TestCase):
    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.undeploy_model")
    def test_undeploy_tears_down_serve_app(self, mock_undeploy: mock.Mock):
        model = HuggingFaceImageModel(
            url="http://h/r/F/S/R", model_id="hf-image:x",
            family="FLUX.2-klein-base", suffix="4B", run_name="run-1",
        )
        model.undeploy()
        mock_undeploy.assert_called_once_with("FLUX.2-klein-base", "4B", "run-1")

    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.undeploy_model")
    def test_undeploy_is_noop_without_identifiers(self, mock_undeploy: mock.Mock):
        HuggingFaceImageModel(url="u", model_id="hf-image:x").undeploy()
        mock_undeploy.assert_not_called()

    @mock.patch("cortexgrid_infer.providers.huggingface_image.ensure_serving", return_value="http://existing/url")
    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.model_registry_status")
    def test_deployed_model_carries_identifiers(
        self,
        mock_registry: mock.Mock,
        _mock_ensure_serving: mock.Mock,
    ):
        mock_registry.return_value = SimpleNamespace(phase="ready")

        model = deploy_huggingface_image(
            "hf-image:black-forest-labs/FLUX.2-klein-base-4B"
        )

        assert model is not None
        self.assertEqual(
            (model.family, model.suffix, model.run_name),
            ("FLUX.2-klein-base", "4B", cortexgrid.IMPORTED),
        )


class TestImageDeploymentStatus(unittest.TestCase):
    def test_returns_none_for_non_hf_image_prefix(self):
        self.assertIsNone(image_deployment_status("hf:Qwen/Qwen2-Instruct"))

    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.model_registry_status")
    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.model_serving_status")
    def test_reports_serving_status_once_deployed(
        self,
        mock_serving: mock.Mock,
        mock_registry: mock.Mock,
    ):
        mock_serving.return_value = SimpleNamespace(phase="running")
        result = image_deployment_status(
            "hf-image:black-forest-labs/FLUX.2-klein-base-4B"
        )
        self.assertEqual(result.phase, "running")
        mock_serving.assert_called_once_with(
            "FLUX.2-klein-base", "4B", cortexgrid.IMPORTED
        )
        mock_registry.assert_not_called()

    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.model_registry_status")
    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.model_serving_status")
    def test_falls_back_to_registry_status_before_deploy(
        self,
        mock_serving: mock.Mock,
        mock_registry: mock.Mock,
    ):
        mock_serving.return_value = SimpleNamespace(phase="not_deployed")
        mock_registry.return_value = SimpleNamespace(phase="uploading")
        result = image_deployment_status(
            "hf-image:black-forest-labs/FLUX.2-klein-base-4B"
        )
        self.assertEqual(result.phase, "uploading")
        mock_registry.assert_called_once_with(
            "FLUX.2-klein-base", "4B", cortexgrid.IMPORTED
        )

    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.model_registry_status")
    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.model_serving_status")
    def test_core_dispatch_routes_hf_image(
        self,
        mock_serving: mock.Mock,
        mock_registry: mock.Mock,
    ):
        import cortexgrid_infer
        mock_serving.return_value = SimpleNamespace(phase="running")
        result = cortexgrid_infer.deployment_status("hf-image:org/Model-4B")
        self.assertEqual(result.phase, "running")

    def test_core_dispatch_none_for_unhandled_prefix(self):
        import cortexgrid_infer
        self.assertIsNone(cortexgrid_infer.deployment_status("openai:gpt-4"))


class TestDeleteHuggingFaceImage(unittest.TestCase):
    def test_noop_for_non_hf_image_prefix(self):
        self.assertIsNone(delete_huggingface_image("hf:Qwen/Qwen2-Instruct"))

    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.delete_model")
    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.undeploy_model")
    def test_undeploys_then_deletes(
        self,
        mock_undeploy: mock.Mock,
        mock_delete: mock.Mock,
    ):
        delete_huggingface_image("hf-image:black-forest-labs/FLUX.2-klein-base-4B")
        mock_undeploy.assert_called_once_with(
            "FLUX.2-klein-base", "4B", cortexgrid.IMPORTED
        )
        mock_delete.assert_called_once_with(
            "FLUX.2-klein-base", "4B", cortexgrid.IMPORTED
        )

    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.delete_model")
    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.undeploy_model")
    def test_core_dispatch_routes_hf_image(
        self,
        mock_undeploy: mock.Mock,
        mock_delete: mock.Mock,
    ):
        import cortexgrid_infer
        cortexgrid_infer.delete_model("hf-image:black-forest-labs/FLUX.2-klein-base-4B")
        mock_delete.assert_called_once_with(
            "FLUX.2-klein-base", "4B", cortexgrid.IMPORTED
        )


if __name__ == "__main__":
    unittest.main()
