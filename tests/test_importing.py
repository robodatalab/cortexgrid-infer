"""Tests for cortexgrid_infer.importing."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import cloudpickle

from cortexgrid_infer.core import DeployedModel
from cortexgrid_infer.importing import HuggingFaceImport, split_model_id


class _FakeModel(DeployedModel):
    def __init__(self, url: str) -> None:
        self.url = url

    @property
    def name(self) -> str:
        return self.url


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


class _FakeImport(HuggingFaceImport):
    serve_app = object

    def client(self, url: str) -> DeployedModel:
        return _FakeModel(url)


class TestSplitModelId(unittest.TestCase):
    def test_org_and_dash(self):
        self.assertEqual(
            split_model_id("Qwen/Qwen2-2.5B-Instruct"), ("Qwen2-2.5B", "Instruct")
        )

    def test_org_no_dash(self):
        self.assertEqual(split_model_id("openai/gpt2"), ("gpt2", "base"))

    def test_no_org_no_dash(self):
        self.assertEqual(split_model_id("gpt2"), ("gpt2", "base"))

    def test_no_org_with_dash(self):
        self.assertEqual(split_model_id("model-v1"), ("model", "v1"))

    def test_multiple_dashes_split_on_last(self):
        self.assertEqual(split_model_id("a/b-c-d-e"), ("b-c-d", "e"))


class TestIdentity(unittest.TestCase):
    def test_derives_the_registry_key_from_the_repo_id(self):
        imp = _FakeImport("Qwen/Qwen2-2.5B-Instruct")
        self.assertEqual((imp.family, imp.suffix), ("Qwen2-2.5B", "Instruct"))


class TestScratchDirectory(unittest.TestCase):
    def test_source_outside_a_context_is_an_error(self):
        # import_model reads the directory after source() returns, so the
        # lifetime has to be opened deliberately rather than per-download.
        with self.assertRaises(RuntimeError):
            _FakeImport("Qwen/Qwen2-2.5B-Instruct").source()

    @mock.patch("cortexgrid_infer.importing.snapshot_download")
    def test_downloads_into_the_scratch_directory(self, mock_snapshot: mock.Mock):
        with _FakeImport("Qwen/Qwen2-2.5B-Instruct", "tok") as imp:
            path = imp.source()
            self.assertTrue(os.path.isdir(path))
            self.assertEqual(mock_snapshot.call_args.kwargs["local_dir"], path)

    @mock.patch(
        "cortexgrid_infer.importing.snapshot_download",
        side_effect=_fake_snapshot_download,
    )
    def test_leaves_the_registry_only_the_model_files(self, mock_snapshot: mock.Mock):
        # snapshot_download writes its own etag/commit bookkeeping alongside the
        # weights; left in place the registry upload would store that too.
        with _FakeImport("Qwen/Qwen2-2.5B-Instruct", "tok", ["*.md"]) as imp:
            path = imp.source()

            self.assertEqual(_files_under(path), {"config.json"})
        mock_snapshot.assert_called_once_with(
            repo_id="Qwen/Qwen2-2.5B-Instruct",
            local_dir=path,
            ignore_patterns=["*.md"],
            token="tok",
        )

    @mock.patch("cortexgrid_infer.importing.snapshot_download")
    def test_scratch_directory_is_removed_on_exit(self, _mock_snapshot: mock.Mock):
        with _FakeImport("Qwen/Qwen2-2.5B-Instruct") as imp:
            path = imp.source()
        self.assertFalse(os.path.exists(path))

    @mock.patch("cortexgrid_infer.importing.snapshot_download")
    def test_source_is_an_error_again_after_exit(self, _mock_snapshot: mock.Mock):
        imp = _FakeImport("Qwen/Qwen2-2.5B-Instruct")
        with imp:
            pass
        with self.assertRaises(RuntimeError):
            imp.source()


class TestTravelsToTheCluster(unittest.TestCase):
    def test_unentered_importer_survives_cloudpickle(self):
        # cortexgrid.remote cloudpickles the importer into the import job, which
        # is where the download belongs; nothing is open until it is entered.
        imp = _FakeImport("Qwen/Qwen2-2.5B-Instruct", "tok", ["*.md"])
        restored = cloudpickle.loads(cloudpickle.dumps(imp))
        self.assertEqual(restored.hf_id, imp.hf_id)
        self.assertEqual(restored.token, imp.token)
        self.assertEqual(restored.ignore_patterns, imp.ignore_patterns)
        self.assertEqual((restored.family, restored.suffix), (imp.family, imp.suffix))


class TestRequirements(unittest.TestCase):
    @mock.patch("cortexgrid_infer.requirements.estimate")
    def test_estimates_from_the_repo_it_was_built_for(self, mock_estimate: mock.Mock):
        imp = _FakeImport("Qwen/Qwen2-2.5B-Instruct", "tok", ["*.md"])

        self.assertIs(imp.requirements(), mock_estimate.return_value)
        mock_estimate.assert_called_once_with("Qwen/Qwen2-2.5B-Instruct", "tok", ["*.md"])
