"""Tests for cortexgrid_infer.providers.huggingface_image."""

from __future__ import annotations

import base64
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from cortexgrid_infer.providers.huggingface_image import (
    HuggingFaceImageModel,
    _ingest_huggingface_image,
    delete_huggingface_image,
    deploy_huggingface_image,
    upload_huggingface_image,
)
from cortexgrid_infer.providers.huggingface_image_serve import (
    HuggingFaceImageDeployment,
)


def _fake_snapshot_download(*, local_dir: str, **_kwargs: Any) -> None:
    """Lay out what `snapshot_download(local_dir=...)` writes: the model files
    plus HuggingFace's download bookkeeping under `.cache/huggingface/`."""
    root = Path(local_dir)
    (root / "config.json").write_text("{}")
    metadata = root / ".cache" / "huggingface" / "download" / "config.json.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("etag")


def _files_under(local_dir: str) -> set[str]:
    root = Path(local_dir)
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


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


class _FakeSavedModel:
    """Stands in for a cortexgrid SavedModel record from ``list_models()``."""

    def __init__(
        self,
        family: str,
        suffix: str,
        run_name: str,
        phase: str,
        created_at: str = "2026-01-01T00:00:00+00:00",
    ) -> None:
        self.family = family
        self.suffix = suffix
        self.run_name = run_name
        self.phase = phase
        self.created_at = created_at


class TestDeployHuggingFaceImageFlow(unittest.TestCase):
    def test_returns_none_for_non_hf_image_prefix(self):
        self.assertIsNone(deploy_huggingface_image("hf:Qwen/Qwen2-Instruct"))
        self.assertIsNone(deploy_huggingface_image("openai:gpt-4"))

    @mock.patch("cortexgrid_infer.providers.huggingface_image.ensure_serving")
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_image.cortexgrid.list_models",
        create=True,
    )
    def test_serves_newest_ready_version_within_timeout(
        self,
        mock_list_models: mock.Mock,
        mock_ensure_serving: mock.Mock,
    ):
        mock_list_models.return_value = [
            _FakeSavedModel("FLUX.2-klein-base", "4B", "run-1", "ready")
        ]
        mock_ensure_serving.return_value = "http://h/r/FLUX.2-klein-base/4B/run-1"

        model = deploy_huggingface_image(
            "hf-image:black-forest-labs/FLUX.2-klein-base-4B", timeout=120.0
        )

        assert model is not None
        self.assertEqual(model.url, "http://h/r/FLUX.2-klein-base/4B/run-1")
        mock_ensure_serving.assert_called_once_with(
            "FLUX.2-klein-base", "4B", "run-1", 120.0
        )

    @mock.patch("cortexgrid_infer.providers.huggingface_image.ensure_serving")
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_image.cortexgrid.list_models",
        create=True,
    )
    def test_raises_when_not_registry_ready(
        self,
        mock_list_models: mock.Mock,
        mock_deploy: mock.Mock,
    ):
        # No versions at all, or only a still-uploading one: nothing ready to deploy.
        for versions in (
            [],
            [_FakeSavedModel("FLUX.2-klein-base", "4B", "run-1", "uploading")],
        ):
            mock_list_models.return_value = versions
            with self.assertRaises(RuntimeError):
                deploy_huggingface_image(
                    "hf-image:black-forest-labs/FLUX.2-klein-base-4B"
                )
        mock_deploy.assert_not_called()


class TestUploadHuggingFaceImage(unittest.TestCase):
    def test_returns_none_for_non_hf_image_prefix(self):
        self.assertIsNone(upload_huggingface_image("hf:Qwen/Qwen2-Instruct"))

    @mock.patch.dict("os.environ", {"HF_TOKEN": "tok"})
    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.remote", create=True)
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_image.cortexgrid.list_models",
        create=True,
    )
    def test_submits_remote_ingest_when_absent(
        self,
        mock_list_models: mock.Mock,
        mock_remote: mock.Mock,
    ):
        mock_list_models.return_value = []
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

    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.remote", create=True)
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_image.cortexgrid.list_models",
        create=True,
    )
    def test_skips_when_already_registered(
        self,
        mock_list_models: mock.Mock,
        mock_remote: mock.Mock,
    ):
        # Registered under some (possibly older) run: never re-ingest.
        for phase in ("uploading", "ready"):
            mock_list_models.return_value = [
                _FakeSavedModel("FLUX.2-klein-base", "4B", "run-1", phase)
            ]
            self.assertIsNone(
                upload_huggingface_image(
                    "hf-image:black-forest-labs/FLUX.2-klein-base-4B"
                )
            )
        mock_remote.assert_not_called()

    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.save_model")
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_image.snapshot_download",
        side_effect=_fake_snapshot_download,
    )
    def test_ingest_does_not_save_hf_download_metadata(
        self, _mock_snapshot: mock.Mock, mock_save: mock.Mock
    ):
        saved: list[set[str]] = []
        mock_save.side_effect = lambda d, *_a, **_k: saved.append(_files_under(d))

        _ingest_huggingface_image(
            "black-forest-labs/FLUX.2-klein-base-4B", "FLUX.2-klein-base", "4B", "tok"
        )

        self.assertEqual(saved, [{"config.json"}])

    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.save_model")
    @mock.patch("cortexgrid_infer.providers.huggingface_image.snapshot_download")
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
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_image.cortexgrid.list_models",
        create=True,
    )
    def test_deployed_model_carries_identifiers(
        self,
        mock_list_models: mock.Mock,
        _mock_ensure_serving: mock.Mock,
    ):
        mock_list_models.return_value = [
            _FakeSavedModel("FLUX.2-klein-base", "4B", "run-1", "ready")
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
        from cortexgrid_infer.providers.huggingface_image import image_deployment_status
        self.assertIsNone(image_deployment_status("hf:Qwen/Qwen2-Instruct"))

    @mock.patch(
        "cortexgrid_infer.providers.huggingface_image.cortexgrid.model_registry_status",
        create=True,
    )
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_image.cortexgrid.model_serving_status",
        create=True,
    )
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_image.cortexgrid.list_models",
        create=True,
    )
    def test_reports_serving_status_once_deployed(
        self,
        mock_list_models: mock.Mock,
        mock_serving: mock.Mock,
        mock_registry: mock.Mock,
    ):
        from cortexgrid_infer.providers.huggingface_image import image_deployment_status
        mock_list_models.return_value = [
            _FakeSavedModel("FLUX.2-klein-base", "4B", "run-1", "ready")
        ]
        mock_serving.return_value = SimpleNamespace(phase="running")
        result = image_deployment_status(
            "hf-image:black-forest-labs/FLUX.2-klein-base-4B"
        )
        self.assertEqual(result.phase, "running")
        mock_serving.assert_called_once_with("FLUX.2-klein-base", "4B", "run-1")
        mock_registry.assert_not_called()

    @mock.patch(
        "cortexgrid_infer.providers.huggingface_image.cortexgrid.model_registry_status",
        create=True,
    )
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_image.cortexgrid.model_serving_status",
        create=True,
    )
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_image.cortexgrid.list_models",
        create=True,
    )
    def test_falls_back_to_registry_status_before_deploy(
        self,
        mock_list_models: mock.Mock,
        mock_serving: mock.Mock,
        mock_registry: mock.Mock,
    ):
        from cortexgrid_infer.providers.huggingface_image import image_deployment_status
        # No ready version yet: report the newest (in-flight) version's registry phase.
        mock_list_models.return_value = [
            _FakeSavedModel("FLUX.2-klein-base", "4B", "run-1", "uploading")
        ]
        mock_registry.return_value = SimpleNamespace(phase="uploading")
        result = image_deployment_status(
            "hf-image:black-forest-labs/FLUX.2-klein-base-4B"
        )
        self.assertEqual(result.phase, "uploading")
        mock_serving.assert_not_called()

    @mock.patch(
        "cortexgrid_infer.providers.huggingface_image.cortexgrid.model_registry_status",
        create=True,
    )
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_image.cortexgrid.model_serving_status",
        create=True,
    )
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_image.cortexgrid.list_models",
        create=True,
    )
    def test_core_dispatch_routes_hf_image(
        self,
        mock_list_models: mock.Mock,
        mock_serving: mock.Mock,
        mock_registry: mock.Mock,
    ):
        import cortexgrid_infer
        mock_list_models.return_value = [
            _FakeSavedModel("Model", "4B", "run-1", "ready")
        ]
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
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_image.cortexgrid.list_models",
        create=True,
    )
    def test_undeploys_then_deletes(
        self,
        mock_list_models: mock.Mock,
        mock_undeploy: mock.Mock,
        mock_delete: mock.Mock,
    ):
        mock_list_models.return_value = [
            _FakeSavedModel("FLUX.2-klein-base", "4B", "run-1", "ready")
        ]
        delete_huggingface_image("hf-image:black-forest-labs/FLUX.2-klein-base-4B")
        mock_undeploy.assert_called_once_with("FLUX.2-klein-base", "4B", "run-1")
        mock_delete.assert_called_once_with("FLUX.2-klein-base", "4B", "run-1")

    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.delete_model")
    @mock.patch("cortexgrid_infer.providers.huggingface_image.cortexgrid.undeploy_model")
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_image.cortexgrid.list_models",
        create=True,
    )
    def test_core_dispatch_routes_hf_image(
        self,
        mock_list_models: mock.Mock,
        mock_undeploy: mock.Mock,
        mock_delete: mock.Mock,
    ):
        import cortexgrid_infer
        mock_list_models.return_value = [
            _FakeSavedModel("FLUX.2-klein-base", "4B", "run-1", "ready")
        ]
        cortexgrid_infer.delete_model("hf-image:black-forest-labs/FLUX.2-klein-base-4B")
        mock_delete.assert_called_once_with("FLUX.2-klein-base", "4B", "run-1")


if __name__ == "__main__":
    unittest.main()
