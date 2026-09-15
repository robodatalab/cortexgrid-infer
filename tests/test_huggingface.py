"""Tests for cortexgrid_infer.providers.huggingface_complete."""

from __future__ import annotations

import unittest
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from cortexgrid_infer.providers.huggingface_complete import (
    HuggingFaceCompletingModel,
    _ingest_huggingface,
    delete_huggingface,
    deploy_huggingface,
    upload_huggingface,
)
from cortexgrid_infer.providers.huggingface_complete_serve import (
    HuggingFaceCompletingDeployment,
)
from cortexgrid_infer.utils import parse_hf_id


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


class TestParseHFId(unittest.TestCase):
    def test_org_and_dash(self):
        self.assertEqual(
            parse_hf_id("Qwen/Qwen2-2.5B-Instruct"), ("Qwen2-2.5B", "Instruct")
        )

    def test_org_no_dash(self):
        self.assertEqual(parse_hf_id("openai/gpt2"), ("gpt2", "base"))

    def test_no_org_no_dash(self):
        self.assertEqual(parse_hf_id("gpt2"), ("gpt2", "base"))

    def test_no_org_with_dash(self):
        self.assertEqual(parse_hf_id("model-v1"), ("model", "v1"))

    def test_multiple_dashes_split_on_last(self):
        self.assertEqual(parse_hf_id("a/b-c-d-e"), ("b-c-d", "e"))


# --- HTTP streaming fakes for HuggingFaceCompletingModel.complete ---


class _FakeResponse:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    def raise_for_status(self) -> None:
        pass

    async def aiter_text(self) -> AsyncIterator[str]:
        for c in self._chunks:
            yield c


class _FakeStreamCtx:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def __aenter__(self) -> _FakeResponse:
        return _FakeResponse(self._chunks)

    async def __aexit__(self, *args: Any) -> None:
        pass


class _FakeAsyncClient:
    captured: dict[str, Any] = {}

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    def stream(self, method: str, url: str, json: dict[str, Any]) -> _FakeStreamCtx:
        type(self).captured = {"method": method, "url": url, "json": json}
        return _FakeStreamCtx(self._chunks)


def _patch_httpx_client(chunks: list[str]):
    def factory(*_a: Any, **_kw: Any) -> _FakeAsyncClient:
        return _FakeAsyncClient(chunks)

    return mock.patch("cortexgrid_infer.providers.huggingface_complete.httpx.AsyncClient", factory)


class TestHuggingFaceCompletingModelComplete(unittest.IsolatedAsyncioTestCase):
    async def test_streams_plain_text(self):
        model = HuggingFaceCompletingModel(url="http://x/r/f/s/r", model_id="hf:test")
        with _patch_httpx_client(["hello", " ", "world"]):
            out = [
                c
                async for c in model.complete(
                    messages=[{"role": "user", "content": "hi"}]
                )
            ]
        self.assertEqual("".join(c.content for c in out), "hello world")
        self.assertEqual(out[-1].finish_reason, "stop")

    async def test_posts_to_complete_endpoint_with_params(self):
        model = HuggingFaceCompletingModel(
            url="http://h:30000/r/F/S/R", model_id="hf:test"
        )
        with _patch_httpx_client(["ok"]):
            _ = [
                c
                async for c in model.complete(
                    messages=[{"role": "user", "content": "hi"}],
                    temperature=0.5,
                    max_new_tokens=42,
                )
            ]
        captured = _FakeAsyncClient.captured
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["url"], "http://h:30000/r/F/S/R/complete")
        self.assertEqual(captured["json"]["temperature"], 0.5)
        self.assertEqual(captured["json"]["max_new_tokens"], 42)
        self.assertEqual(
            captured["json"]["messages"], [{"role": "user", "content": "hi"}]
        )

    async def test_parses_tool_calls_from_stream(self):
        def add(a: int, b: int) -> int:
            return a + b

        chunks = [
            "Hello ",
            "<tool_call>",
            '{"name": "add", "arguments": {"a": 2, "b": 3}}',
            "</tool_call>",
        ]
        model = HuggingFaceCompletingModel(url="http://x/r/f/s/r", model_id="hf:test")
        with _patch_httpx_client(chunks):
            out = [
                c
                async for c in model.complete(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=[add],
                )
            ]
        text = "".join(c.content for c in out).strip()
        self.assertEqual(text, "Hello")
        tool_chunks = [c for c in out if c.has_tool_calls]
        self.assertEqual(len(tool_chunks), 1)
        tc = tool_chunks[0].tool_calls[0]
        self.assertEqual(tc.name, "add")
        self.assertEqual(tc.arguments, {"a": 2, "b": 3})
        self.assertEqual(tc(), 5)


# --- deploy_huggingface flow ---


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


class TestDeployHuggingFaceFlow(unittest.TestCase):
    def test_returns_none_for_non_hf_prefix(self):
        self.assertIsNone(deploy_huggingface("openai:gpt-4"))
        self.assertIsNone(deploy_huggingface("Qwen/Qwen2-2.5B-Instruct"))

    @mock.patch("cortexgrid_infer.providers.huggingface_complete.cortexgrid.deploy_model")
    @mock.patch("cortexgrid_infer.providers.huggingface_complete.cortexgrid.list_deployed_models")
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_complete.cortexgrid.model_registry_status",
        create=True,
    )
    @mock.patch("cortexgrid_infer.providers.huggingface_complete.cortexgrid.Experiment")
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
            "Qwen2-2.5B", "Instruct", "run-1", "http://h/r/Qwen2-2.5B/Instruct/run-1"
        )

        model = deploy_huggingface("hf:Qwen/Qwen2-2.5B-Instruct")

        assert model is not None
        self.assertEqual(model.url, "http://h/r/Qwen2-2.5B/Instruct/run-1")
        self.assertEqual(model.model_id, "hf:Qwen/Qwen2-2.5B-Instruct")
        self.assertEqual(
            (model.family, model.suffix, model.run_name),
            ("Qwen2-2.5B", "Instruct", "run-1"),
        )
        mock_deploy.assert_called_once()
        self.assertEqual(
            mock_deploy.call_args.kwargs,
            {
                "family": "Qwen2-2.5B",
                "suffix": "Instruct",
                "run_name": "run-1",
                "wait": True,
                "timeout": None,
            },
        )

    @mock.patch("cortexgrid_infer.providers.huggingface_complete.cortexgrid.deploy_model")
    @mock.patch("cortexgrid_infer.providers.huggingface_complete.cortexgrid.list_deployed_models")
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_complete.cortexgrid.model_registry_status",
        create=True,
    )
    @mock.patch("cortexgrid_infer.providers.huggingface_complete.cortexgrid.Experiment")
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
            _FakeDeployment("Qwen2-2.5B", "Instruct", "run-1", "http://existing/url")
        ]

        model = deploy_huggingface("hf:Qwen/Qwen2-2.5B-Instruct")

        assert model is not None
        self.assertEqual(model.url, "http://existing/url")
        mock_deploy.assert_not_called()

    @mock.patch("cortexgrid_infer.providers.huggingface_complete.cortexgrid.deploy_model")
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_complete.cortexgrid.model_registry_status",
        create=True,
    )
    @mock.patch("cortexgrid_infer.providers.huggingface_complete.cortexgrid.Experiment")
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
                deploy_huggingface("hf:Qwen/Qwen2-2.5B-Instruct")
        mock_deploy.assert_not_called()


class TestUploadHuggingFace(unittest.TestCase):
    def test_returns_none_for_non_hf_prefix(self):
        self.assertIsNone(upload_huggingface("openai:gpt-4"))

    @mock.patch.dict("os.environ", {"HF_TOKEN": "tok"})
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_complete.cortexgrid.remote", create=True
    )
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_complete.cortexgrid.model_registry_status",
        create=True,
    )
    @mock.patch("cortexgrid_infer.providers.huggingface_complete.cortexgrid.Experiment")
    def test_submits_remote_ingest_when_absent(
        self,
        mock_experiment: mock.Mock,
        mock_registry: mock.Mock,
        mock_remote: mock.Mock,
    ):
        mock_experiment.get_instance.return_value = _FakeExperiment("run-1")
        mock_registry.return_value = None
        mock_remote.return_value = "job-1"

        job = upload_huggingface("hf:Qwen/Qwen2-2.5B-Instruct")

        self.assertEqual(job, "job-1")
        mock_remote.assert_called_once()
        args = mock_remote.call_args.args
        self.assertIs(args[0], _ingest_huggingface)
        self.assertEqual(
            args[1:], ("Qwen/Qwen2-2.5B-Instruct", "Qwen2-2.5B", "Instruct", "tok")
        )
        self.assertEqual(
            mock_remote.call_args.kwargs, {"num_gpus": 0, "num_cpus": 2}
        )

    @mock.patch(
        "cortexgrid_infer.providers.huggingface_complete.cortexgrid.remote", create=True
    )
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_complete.cortexgrid.model_registry_status",
        create=True,
    )
    @mock.patch("cortexgrid_infer.providers.huggingface_complete.cortexgrid.Experiment")
    def test_skips_when_already_registered(
        self,
        mock_experiment: mock.Mock,
        mock_registry: mock.Mock,
        mock_remote: mock.Mock,
    ):
        mock_experiment.get_instance.return_value = _FakeExperiment("run-1")
        for phase in ("uploading", "ready"):
            mock_registry.return_value = SimpleNamespace(phase=phase)
            self.assertIsNone(upload_huggingface("hf:Qwen/Qwen2-2.5B-Instruct"))
        mock_remote.assert_not_called()

    @mock.patch("cortexgrid_infer.providers.huggingface_complete.cortexgrid.save_model")
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_complete.snapshot_download",
        side_effect=_fake_snapshot_download,
    )
    def test_ingest_does_not_save_hf_download_metadata(
        self, _mock_snapshot: mock.Mock, mock_save: mock.Mock
    ):
        saved: list[set[str]] = []
        mock_save.side_effect = lambda d, *_a, **_k: saved.append(_files_under(d))

        _ingest_huggingface("Qwen/Qwen2-2.5B-Instruct", "Qwen2-2.5B", "Instruct", "tok")

        self.assertEqual(saved, [{"config.json"}])

    @mock.patch("cortexgrid_infer.providers.huggingface_complete.cortexgrid.save_model")
    @mock.patch("cortexgrid_infer.providers.huggingface_complete.snapshot_download")
    def test_ingest_downloads_and_saves(
        self, mock_snapshot: mock.Mock, mock_save: mock.Mock
    ):
        _ingest_huggingface(
            "Qwen/Qwen2-2.5B-Instruct", "Qwen2-2.5B", "Instruct", "tok"
        )
        mock_snapshot.assert_called_once()
        self.assertEqual(
            mock_snapshot.call_args.kwargs["repo_id"], "Qwen/Qwen2-2.5B-Instruct"
        )
        self.assertEqual(mock_snapshot.call_args.kwargs["token"], "tok")
        mock_save.assert_called_once()
        self.assertIs(mock_save.call_args.args[1], HuggingFaceCompletingDeployment)
        self.assertEqual(
            mock_save.call_args.kwargs, {"family": "Qwen2-2.5B", "suffix": "Instruct"}
        )

    @mock.patch(
        "cortexgrid_infer.providers.huggingface_complete.cortexgrid.remote", create=True
    )
    @mock.patch(
        "cortexgrid_infer.providers.huggingface_complete.cortexgrid.model_registry_status",
        create=True,
    )
    @mock.patch("cortexgrid_infer.providers.huggingface_complete.cortexgrid.Experiment")
    def test_core_dispatch_routes_hf(
        self,
        mock_experiment: mock.Mock,
        mock_registry: mock.Mock,
        mock_remote: mock.Mock,
    ):
        import cortexgrid_infer

        mock_experiment.get_instance.return_value = _FakeExperiment("run-1")
        mock_registry.return_value = None
        mock_remote.return_value = "job-9"
        self.assertEqual(
            cortexgrid_infer.upload_model("hf:Qwen/Qwen2-2.5B-Instruct"), "job-9"
        )

    def test_core_dispatch_none_for_unhandled_prefix(self):
        import cortexgrid_infer

        self.assertIsNone(cortexgrid_infer.upload_model("openai:gpt-4"))


class TestDeleteHuggingFace(unittest.TestCase):
    def test_noop_for_non_hf_prefix(self):
        # No cortexgrid calls patched: a non-match must not touch the platform.
        self.assertIsNone(delete_huggingface("openai:gpt-4"))

    @mock.patch("cortexgrid_infer.providers.huggingface_complete.cortexgrid.delete_model")
    @mock.patch("cortexgrid_infer.providers.huggingface_complete.cortexgrid.undeploy_model")
    @mock.patch("cortexgrid_infer.providers.huggingface_complete.cortexgrid.Experiment")
    def test_undeploys_then_deletes(
        self,
        mock_experiment: mock.Mock,
        mock_undeploy: mock.Mock,
        mock_delete: mock.Mock,
    ):
        mock_experiment.get_instance.return_value = _FakeExperiment("run-1")
        delete_huggingface("hf:Qwen/Qwen2-2.5B-Instruct")
        mock_undeploy.assert_called_once_with("Qwen2-2.5B", "Instruct", "run-1")
        mock_delete.assert_called_once_with("Qwen2-2.5B", "Instruct", "run-1")

    @mock.patch("cortexgrid_infer.providers.huggingface_complete.cortexgrid.delete_model")
    @mock.patch("cortexgrid_infer.providers.huggingface_complete.cortexgrid.undeploy_model")
    @mock.patch("cortexgrid_infer.providers.huggingface_complete.cortexgrid.Experiment")
    def test_core_dispatch_routes_hf(
        self,
        mock_experiment: mock.Mock,
        mock_undeploy: mock.Mock,
        mock_delete: mock.Mock,
    ):
        import cortexgrid_infer

        mock_experiment.get_instance.return_value = _FakeExperiment("run-1")
        cortexgrid_infer.delete_model("hf:Qwen/Qwen2-2.5B-Instruct")
        mock_delete.assert_called_once_with("Qwen2-2.5B", "Instruct", "run-1")

    def test_core_dispatch_noop_for_unhandled_prefix(self):
        import cortexgrid_infer

        self.assertIsNone(cortexgrid_infer.delete_model("openai:gpt-4"))


if __name__ == "__main__":
    unittest.main()
