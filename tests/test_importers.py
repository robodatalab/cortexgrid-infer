"""Tests for cortexgrid_infer.importers."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import cloudpickle
import cortexgrid
from huggingface_hub.errors import NotASafetensorsRepoError

from cortexgrid_infer.core import DeployedModel
from cortexgrid_infer.importers.huggingface import HuggingFaceImporter, repo_weights
from cortexgrid_infer.serve_apps.base import LocalModel, Weights

HF = "cortexgrid_infer.importers.huggingface"


class _FakeModel(DeployedModel):
    def __init__(self, url: str, name: str) -> None:
        self.url = url
        self._name = name

    @property
    def name(self) -> str:
        return self._name


class _FakeApp(LocalModel):
    """A serve app that loads everything and asks for a fixed amount of hardware."""

    @classmethod
    def requirements(cls, weights: Weights) -> cortexgrid.ModelRequirements:
        return cortexgrid.ModelRequirements(num_gpus=1, vram_gb=weights.params or 0)

    @classmethod
    def client(cls, url: str, name: str) -> DeployedModel:
        return _FakeModel(url, name)


class _PrunedApp(_FakeApp):
    """A serve app that never loads the docs."""

    @classmethod
    def ignore_patterns(cls) -> list[str]:
        return ["*.md"]


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


def _sibling(name: str, size: int) -> SimpleNamespace:
    return SimpleNamespace(rfilename=name, size=size)


class TestIdentity(unittest.TestCase):
    def test_derives_the_registry_key_from_the_repo_id(self):
        imp = HuggingFaceImporter("Qwen/Qwen2-2.5B-Instruct", _FakeApp)
        self.assertEqual((imp.family, imp.suffix), ("Qwen2-2.5B", "Instruct"))

    def test_bundles_the_serve_app_it_was_given(self):
        self.assertIs(HuggingFaceImporter("Qwen/Qwen2-2.5B-Instruct", _FakeApp).serve_app, _FakeApp)


class TestTheServeAppDecides(unittest.TestCase):
    """What depends on the model's task comes from the serve app, not the source."""

    def test_client_is_the_serve_app_s_named_for_the_repo(self):
        model = HuggingFaceImporter("Qwen/Qwen2-2.5B-Instruct", _FakeApp).client("http://h/r/F/S/R")

        self.assertIsInstance(model, _FakeModel)
        self.assertEqual((model.url, model.name), ("http://h/r/F/S/R", "Qwen/Qwen2-2.5B-Instruct"))

    @mock.patch(f"{HF}.repo_weights", return_value=Weights(params=7))
    def test_requirements_are_the_serve_app_s_for_the_repo_s_weights(self, weights: mock.Mock):
        imp = HuggingFaceImporter("Qwen/Qwen2-2.5B-Instruct", _FakeApp, "tok")

        self.assertEqual(imp.requirements(), cortexgrid.ModelRequirements(num_gpus=1, vram_gb=7))
        weights.assert_called_once_with("Qwen/Qwen2-2.5B-Instruct", "tok", None)

    def test_skips_what_the_serve_app_never_loads(self):
        self.assertEqual(HuggingFaceImporter("org/repo", _PrunedApp).ignore_patterns, ["*.md"])

    def test_explicit_ignores_replace_the_serve_app_s(self):
        imp = HuggingFaceImporter("org/repo", _PrunedApp, ignore_patterns=["*.onnx"])

        self.assertEqual(imp.ignore_patterns, ["*.onnx"])

    @mock.patch(f"{HF}.repo_weights", return_value=Weights(file_bytes=1))
    def test_sizes_only_what_it_downloads(self, weights: mock.Mock):
        HuggingFaceImporter("org/repo", _PrunedApp).requirements()

        weights.assert_called_once_with("org/repo", None, ["*.md"])

    def test_the_model_card_carries_the_serve_app_s_settings(self):
        class _ThinkingApp(_FakeApp):
            @classmethod
            def config(cls) -> dict[str, str]:
                return {"enable_thinking": "true"}

        self.assertEqual(
            HuggingFaceImporter("org/repo", _ThinkingApp).config(), {"enable_thinking": "true"}
        )
        self.assertEqual(HuggingFaceImporter("org/repo", _FakeApp).config(), {"compile": "false"})


class TestScratchDirectory(unittest.TestCase):
    def test_source_outside_a_context_is_an_error(self):
        # import_model reads the directory after source() returns, so the
        # lifetime has to be opened deliberately rather than per-download.
        with self.assertRaises(RuntimeError):
            HuggingFaceImporter("Qwen/Qwen2-2.5B-Instruct", _FakeApp).source()

    @mock.patch(f"{HF}.snapshot_download")
    def test_downloads_into_the_scratch_directory(self, mock_snapshot: mock.Mock):
        with HuggingFaceImporter("Qwen/Qwen2-2.5B-Instruct", _FakeApp, "tok") as imp:
            path = imp.source()
            self.assertTrue(os.path.isdir(path))
            self.assertEqual(mock_snapshot.call_args.kwargs["local_dir"], path)

    @mock.patch(f"{HF}.snapshot_download", side_effect=_fake_snapshot_download)
    def test_leaves_the_registry_only_the_model_files(self, mock_snapshot: mock.Mock):
        # snapshot_download writes its own etag/commit bookkeeping alongside the
        # weights; left in place the registry upload would store that too.
        with HuggingFaceImporter("Qwen/Qwen2-2.5B-Instruct", _PrunedApp, "tok") as imp:
            path = imp.source()

            self.assertEqual(_files_under(path), {"config.json"})
        mock_snapshot.assert_called_once_with(
            repo_id="Qwen/Qwen2-2.5B-Instruct",
            local_dir=path,
            ignore_patterns=["*.md"],
            token="tok",
        )

    @mock.patch(f"{HF}.snapshot_download")
    def test_a_serve_app_that_loads_everything_downloads_the_whole_repo(
        self, mock_snapshot: mock.Mock
    ):
        with HuggingFaceImporter("Qwen/Qwen2-2.5B-Instruct", _FakeApp) as imp:
            imp.source()

        self.assertIsNone(mock_snapshot.call_args.kwargs["ignore_patterns"])

    @mock.patch(f"{HF}.snapshot_download")
    def test_scratch_directory_is_removed_on_exit(self, _mock_snapshot: mock.Mock):
        with HuggingFaceImporter("Qwen/Qwen2-2.5B-Instruct", _FakeApp) as imp:
            path = imp.source()
        self.assertFalse(os.path.exists(path))

    @mock.patch(f"{HF}.snapshot_download")
    def test_source_is_an_error_again_after_exit(self, _mock_snapshot: mock.Mock):
        imp = HuggingFaceImporter("Qwen/Qwen2-2.5B-Instruct", _FakeApp)
        with imp:
            pass
        with self.assertRaises(RuntimeError):
            imp.source()


class TestTravelsToTheCluster(unittest.TestCase):
    def test_unentered_importer_survives_cloudpickle(self):
        # cortexgrid.remote cloudpickles the importer into the import job, which
        # is where the download belongs; nothing is open until it is entered.
        imp = HuggingFaceImporter("Qwen/Qwen2-2.5B-Instruct", _PrunedApp, "tok")
        restored = cloudpickle.loads(cloudpickle.dumps(imp))
        self.assertEqual(restored.model_id, imp.model_id)
        self.assertIs(restored.serve_app, _PrunedApp)
        self.assertEqual(restored.token, imp.token)
        self.assertEqual(restored.ignore_patterns, imp.ignore_patterns)
        self.assertEqual((restored.family, restored.suffix), (imp.family, imp.suffix))


class TestRepoWeights(unittest.TestCase):
    @mock.patch(f"{HF}.get_safetensors_metadata")
    def test_counts_the_parameters_in_the_safetensors_headers(self, mock_metadata: mock.Mock):
        mock_metadata.return_value = SimpleNamespace(parameter_count={"BF16": 500_000_000})

        self.assertEqual(repo_weights("Qwen/Qwen2.5-0.5B-Instruct"), Weights(params=500_000_000))

    @mock.patch(f"{HF}.get_safetensors_metadata")
    def test_sums_every_dtype_in_the_repo(self, mock_metadata: mock.Mock):
        mock_metadata.return_value = SimpleNamespace(
            parameter_count={"BF16": 1_000_000, "F32": 1_000_000}
        )

        self.assertEqual(repo_weights("org/mixed"), Weights(params=2_000_000))

    @mock.patch(f"{HF}.HfApi")
    @mock.patch(f"{HF}.get_safetensors_metadata")
    def test_falls_back_to_file_sizes_without_a_safetensors_index(
        self, mock_metadata: mock.Mock, mock_api: mock.Mock
    ):
        # A diffusers pipeline keeps its weights in component subfolders, so the
        # safetensors helper refuses the repo outright.
        mock_metadata.side_effect = NotASafetensorsRepoError("not a safetensors repo")
        mock_api.return_value.model_info.return_value = SimpleNamespace(
            siblings=[
                _sibling("transformer/diffusion_pytorch_model.safetensors", 2 << 30),
                _sibling("vae/diffusion_pytorch_model.safetensors", 1 << 30),
                _sibling("model_index.json", 1024),
                _sibling("README.md", 4096),
            ]
        )

        self.assertEqual(repo_weights("bfl/FLUX.2-klein-base-4B"), Weights(file_bytes=3 << 30))

    @mock.patch(f"{HF}.HfApi")
    @mock.patch(f"{HF}.get_safetensors_metadata")
    def test_skips_the_files_the_download_skips(
        self, mock_metadata: mock.Mock, mock_api: mock.Mock
    ):
        # A pipeline repo commonly ships a consolidated single-file checkpoint
        # next to the component weights. Counting what the download ignores
        # would size the model at several times what a replica actually holds.
        mock_metadata.side_effect = NotASafetensorsRepoError("not a safetensors repo")
        mock_api.return_value.model_info.return_value = SimpleNamespace(
            siblings=[
                _sibling("transformer/diffusion_pytorch_model.safetensors", 2 << 30),
                _sibling("flux-2-klein-base-4b.safetensors", 8 << 30),
            ]
        )

        self.assertEqual(
            repo_weights(
                "bfl/FLUX.2-klein-base-4B",
                ignore_patterns=["flux-2-klein-base-4b.safetensors"],
            ),
            Weights(file_bytes=2 << 30),
        )

    @mock.patch(f"{HF}.HfApi")
    @mock.patch(f"{HF}.get_safetensors_metadata")
    def test_glob_matches_a_nested_file_by_name(
        self, mock_metadata: mock.Mock, mock_api: mock.Mock
    ):
        mock_metadata.side_effect = NotASafetensorsRepoError("not a safetensors repo")
        mock_api.return_value.model_info.return_value = SimpleNamespace(
            siblings=[
                _sibling("transformer/model.safetensors", 1 << 30),
                _sibling("transformer/model.fp8.safetensors", 1 << 30),
            ]
        )

        self.assertEqual(
            repo_weights("org/repo", ignore_patterns=["*.fp8.safetensors"]),
            Weights(file_bytes=1 << 30),
        )
