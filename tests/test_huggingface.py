"""Tests for model_gateway.providers.huggingface."""

from __future__ import annotations

import unittest
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

from model_gateway.providers.huggingface import (
    HuggingFaceModel,
    _parse_hf_id,
    deploy_huggingface,
)


class TestParseHFId(unittest.TestCase):
    def test_org_and_dash(self):
        self.assertEqual(
            _parse_hf_id("Qwen/Qwen2-2.5B-Instruct"), ("Qwen2-2.5B", "Instruct")
        )

    def test_org_no_dash(self):
        self.assertEqual(_parse_hf_id("openai/gpt2"), ("gpt2", "base"))

    def test_no_org_no_dash(self):
        self.assertEqual(_parse_hf_id("gpt2"), ("gpt2", "base"))

    def test_no_org_with_dash(self):
        self.assertEqual(_parse_hf_id("model-v1"), ("model", "v1"))

    def test_multiple_dashes_split_on_last(self):
        self.assertEqual(_parse_hf_id("a/b-c-d-e"), ("b-c-d", "e"))


# --- HTTP streaming fakes for HuggingFaceModel.complete ---


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

    def stream(
        self, method: str, url: str, json: dict[str, Any]
    ) -> _FakeStreamCtx:
        type(self).captured = {"method": method, "url": url, "json": json}
        return _FakeStreamCtx(self._chunks)


def _patch_httpx_client(chunks: list[str]):
    def factory(*_a: Any, **_kw: Any) -> _FakeAsyncClient:
        return _FakeAsyncClient(chunks)

    return mock.patch(
        "model_gateway.providers.huggingface.httpx.AsyncClient", factory
    )


class TestHuggingFaceModelComplete(unittest.IsolatedAsyncioTestCase):
    async def test_streams_plain_text(self):
        model = HuggingFaceModel(url="http://x/r/f/s/r", model_id="hf:test")
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
        model = HuggingFaceModel(url="http://h:30000/r/F/S/R", model_id="hf:test")
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
        model = HuggingFaceModel(url="http://x/r/f/s/r", model_id="hf:test")
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


class TestDeployHuggingFaceFlow(unittest.TestCase):
    def test_returns_none_for_non_hf_prefix(self):
        self.assertIsNone(deploy_huggingface("openai:gpt-4"))
        self.assertIsNone(deploy_huggingface("Qwen/Qwen2-2.5B-Instruct"))

    @mock.patch("model_gateway.providers.huggingface.cortexflow.deploy_model")
    @mock.patch("model_gateway.providers.huggingface.cortexflow.list_deployed_models")
    @mock.patch("model_gateway.providers.huggingface.cortexflow.save_model")
    @mock.patch("model_gateway.providers.huggingface.cortexflow.list_models")
    @mock.patch("model_gateway.providers.huggingface.cortexflow.Experiment")
    @mock.patch("model_gateway.providers.huggingface.snapshot_download")
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
            "Qwen2-2.5B", "Instruct", "run-1",
            "http://h/r/Qwen2-2.5B/Instruct/run-1",
        )

        model = deploy_huggingface("hf:Qwen/Qwen2-2.5B-Instruct")

        self.assertIsNotNone(model)
        assert model is not None
        self.assertEqual(model.url, "http://h/r/Qwen2-2.5B/Instruct/run-1")
        self.assertEqual(model.model_id, "hf:Qwen/Qwen2-2.5B-Instruct")

        mock_snapshot.assert_called_once()
        self.assertEqual(
            mock_snapshot.call_args.kwargs["repo_id"], "Qwen/Qwen2-2.5B-Instruct"
        )
        mock_save.assert_called_once()
        self.assertEqual(mock_save.call_args.kwargs["family"], "Qwen2-2.5B")
        self.assertEqual(mock_save.call_args.kwargs["suffix"], "Instruct")
        mock_deploy.assert_called_once()
        self.assertEqual(
            mock_deploy.call_args.kwargs,
            {"family": "Qwen2-2.5B", "suffix": "Instruct", "run_name": "run-1"},
        )

    @mock.patch("model_gateway.providers.huggingface.cortexflow.deploy_model")
    @mock.patch("model_gateway.providers.huggingface.cortexflow.list_deployed_models")
    @mock.patch("model_gateway.providers.huggingface.cortexflow.save_model")
    @mock.patch("model_gateway.providers.huggingface.cortexflow.list_models")
    @mock.patch("model_gateway.providers.huggingface.cortexflow.Experiment")
    @mock.patch("model_gateway.providers.huggingface.snapshot_download")
    def test_skips_upload_when_already_saved(
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
            _FakeSavedModel("Qwen2-2.5B", "Instruct", "run-1")
        ]
        mock_list_deployed.return_value = []
        mock_deploy.return_value = _FakeDeployment(
            "Qwen2-2.5B", "Instruct", "run-1",
            "http://h/r/Qwen2-2.5B/Instruct/run-1",
        )

        deploy_huggingface("hf:Qwen/Qwen2-2.5B-Instruct")

        mock_snapshot.assert_not_called()
        mock_save.assert_not_called()
        mock_deploy.assert_called_once()

    @mock.patch("model_gateway.providers.huggingface.cortexflow.deploy_model")
    @mock.patch("model_gateway.providers.huggingface.cortexflow.list_deployed_models")
    @mock.patch("model_gateway.providers.huggingface.cortexflow.save_model")
    @mock.patch("model_gateway.providers.huggingface.cortexflow.list_models")
    @mock.patch("model_gateway.providers.huggingface.cortexflow.Experiment")
    @mock.patch("model_gateway.providers.huggingface.snapshot_download")
    def test_skips_deploy_when_already_deployed(
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
            _FakeSavedModel("Qwen2-2.5B", "Instruct", "run-1")
        ]
        mock_list_deployed.return_value = [
            _FakeDeployment(
                "Qwen2-2.5B", "Instruct", "run-1", "http://existing/url"
            )
        ]

        model = deploy_huggingface("hf:Qwen/Qwen2-2.5B-Instruct")

        assert model is not None
        self.assertEqual(model.url, "http://existing/url")
        mock_deploy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
