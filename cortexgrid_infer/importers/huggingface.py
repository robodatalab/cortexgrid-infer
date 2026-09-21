"""Import a model from the HuggingFace Hub.

The weights are downloaded with `snapshot_download`, and sized beforehand from
the repo's own metadata - safetensors headers where the repo has them, weight
file sizes otherwise - both read over HTTP without downloading a single weight.
"""

from __future__ import annotations

import fnmatch
import os
import shutil

from huggingface_hub import HfApi, get_safetensors_metadata, snapshot_download
from huggingface_hub.errors import NotASafetensorsRepoError

from cortexgrid_infer.importers.base import Importer
from cortexgrid_infer.models.base import LocalModel, Weights

# Files that are weights rather than configs, tokenizers or docs. Used for the
# fallback estimate, where a repo has no top-level safetensors index.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".msgpack")


def repo_weights(
    hf_id: str,
    token: str | None = None,
    ignore_patterns: list[str] | None = None,
) -> Weights:
    """What `hf_id`'s metadata says about the weights a download would fetch.

    Prefers the repo's safetensors headers, which give an exact parameter
    count. Repos those do not cover - a diffusers pipeline keeps its weights in
    component subfolders with no top-level index - fall back to summing the
    size of the weight files on the Hub.

    `ignore_patterns` must be the same globs the download uses. A pipeline repo
    commonly ships several dtype variants of the same component plus a
    consolidated single-file checkpoint; counting the files the download skips
    overestimates by multiples.
    """
    try:
        metadata = get_safetensors_metadata(hf_id, token=token)
    except NotASafetensorsRepoError:
        pass
    else:
        return Weights(params=sum(metadata.parameter_count.values()))

    info = HfApi().model_info(hf_id, files_metadata=True, token=token)
    total = 0
    for sibling in info.siblings or ():
        name = sibling.rfilename
        if not name.endswith(_WEIGHT_SUFFIXES):
            continue
        if _ignored(name, ignore_patterns):
            continue
        total += sibling.size or 0
    return Weights(file_bytes=total)


def _ignored(name: str, ignore_patterns: list[str] | None) -> bool:
    """Whether `snapshot_download` would skip this file for these globs.

    Matches the glob against the full repo-relative path and against the bare
    file name, because the patterns in use are a mix of both ("*.md" is meant to
    catch a nested README, "flux-2-klein-base-4b.safetensors" names one file)."""
    if not ignore_patterns:
        return False
    base = name.rsplit("/", 1)[-1]
    return any(
        fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(base, pattern)
        for pattern in ignore_patterns
    )


def _remove_download_metadata(local_dir: str) -> None:
    """Drop the `.cache/huggingface/` folder `snapshot_download(local_dir=...)`
    writes into `local_dir`: HuggingFace's own download bookkeeping (etags,
    commit hashes), not part of the model. Left in place, the registry upload
    would store it along with the weights."""
    shutil.rmtree(os.path.join(local_dir, ".cache", "huggingface"), ignore_errors=True)


class HuggingFaceImporter(Importer):
    """A model from the HuggingFace Hub, run by `serve_app`.

    `ignore_patterns` defaults to the serve app's own - the files it never
    loads - and replaces them when given. Either way the download and the
    hardware estimate use the same list, so the figures describe what actually
    gets staged. `token` is for gated and private repos; it travels with the
    importer to the node that downloads."""

    def __init__(
        self,
        hf_id: str,
        serve_app: type[LocalModel],
        token: str | None = None,
        ignore_patterns: list[str] | None = None,
    ) -> None:
        super().__init__(hf_id, serve_app)
        self.token = token
        self.ignore_patterns = (
            serve_app.ignore_patterns() if ignore_patterns is None else ignore_patterns
        )

    def download(self, local_dir: str) -> None:
        snapshot_download(
            repo_id=self.model_id,
            local_dir=local_dir,
            ignore_patterns=self.ignore_patterns,
            token=self.token,
        )
        _remove_download_metadata(local_dir)

    def weights(self) -> Weights:
        return repo_weights(self.model_id, self.token, self.ignore_patterns)
