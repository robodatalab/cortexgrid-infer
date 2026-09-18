"""The pieces `cortexgrid.import_model` needs but cannot work out for itself.

cortexgrid stores whatever weights it is handed under whatever key it is given,
and serves them with whatever class it was told to bundle. It deliberately knows
nothing about where a model came from or what shape it is. An importer supplies
exactly that missing knowledge for one family of models - the registry identity,
the serve app, the hardware it needs, the weights if it has any, and the client
- and drives no lifecycle of its own: importing, deploying and deleting stay the
caller's calls against the cortexgrid SDK.

    imp = HuggingFaceCompletingImport("Qwen/Qwen2.5-0.5B-Instruct")
    with imp:
        cortexgrid.import_model(
            imp.source, imp.serve_app,
            family=imp.family, suffix=imp.suffix,
            requirements=imp.requirements(),
        )

An importer is cheap to construct and carries no open resources until it is
entered, so it can be handed to `cortexgrid.remote` and rebuilt on the node that
runs the import - which is where the download belongs, so the weights travel
HuggingFace -> node -> registry and never transit the client.
"""

from __future__ import annotations

import abc
import os
import shutil
import tempfile
from types import TracebackType
from typing import ClassVar

import cortexgrid
from huggingface_hub import snapshot_download

from cortexgrid_infer import requirements
from cortexgrid_infer.core import DeployedModel


def split_model_id(model_id: str) -> tuple[str, str]:
    """Map a model id to a cortexgrid (family, suffix).

    Strips the org (anything before the first '/'), which an HF repo id has and
    a hosted model's name does not. Splits the remainder on the last '-': the
    part before becomes family, the part after becomes suffix. If there is no
    '-', suffix defaults to 'base'.
    """
    name = model_id.split("/", 1)[-1]
    if "-" not in name:
        return name, "base"
    family, _, suffix = name.rpartition("-")
    return family, suffix


def _remove_download_metadata(local_dir: str) -> None:
    """Drop the `.cache/huggingface/` folder `snapshot_download(local_dir=...)`
    writes into `local_dir`: HuggingFace's own download bookkeeping (etags,
    commit hashes), not part of the model. Left in place, the registry upload
    would store it along with the weights."""
    shutil.rmtree(os.path.join(local_dir, ".cache", "huggingface"), ignore_errors=True)


class ModelImport(abc.ABC):
    """What `cortexgrid.import_model` needs to take one model into the registry.

    Not every model has weights. One that does supplies them through
    `HuggingFaceImport.source`; one that does not - a serve app that forwards to
    a hosted API - is registered with no source at all, and carries what it needs
    to reach that API in its requirements' `params`."""

    # The class cortexgrid bundles and instantiates on the cluster. Set by each
    # concrete importer.
    serve_app: ClassVar[type]

    # The registry key this model is filed under, with cortexgrid.IMPORTED.
    family: str
    suffix: str

    @abc.abstractmethod
    def requirements(self) -> cortexgrid.ModelRequirements:
        """What one replica of this model needs to be placed and to run."""

    @abc.abstractmethod
    def client(self, url: str) -> DeployedModel:
        """Return a client for this model's serve app, deployed at `url`."""


class HuggingFaceImport(ModelImport):
    """Base for the HuggingFace importers: identity, download, requirements.

    Subclasses add the two things that differ per model family - the serve app
    cortexgrid bundles, and the client that speaks its routes."""

    def __init__(
        self,
        hf_id: str,
        token: str | None = None,
        ignore_patterns: list[str] | None = None,
    ) -> None:
        self.hf_id = hf_id
        self.token = token
        self.ignore_patterns = ignore_patterns
        self.family, self.suffix = split_model_id(hf_id)
        self._scratch: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> "HuggingFaceImport":
        """Open the scratch directory `source` downloads into.

        `import_model` reads the directory after `source` returns, so its
        lifetime has to span the whole import call rather than the download."""
        self._scratch = tempfile.TemporaryDirectory()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None

    def source(self) -> str:
        """Download the weights and return the directory holding them.

        Shaped as `import_model`'s source, which calls it only when the weights
        actually have to be uploaded - so an already-imported model downloads
        nothing and just has its serve code re-bundled if it changed."""
        if self._scratch is None:
            raise RuntimeError(
                f"{type(self).__name__}.source() needs a scratch directory; "
                "use the importer as a context manager (`with imp: ...`)"
            )
        snapshot_download(
            repo_id=self.hf_id,
            local_dir=self._scratch.name,
            ignore_patterns=self.ignore_patterns,
            token=self.token,
        )
        _remove_download_metadata(self._scratch.name)
        return self._scratch.name

    def requirements(self) -> cortexgrid.ModelRequirements:
        """Estimate what one replica needs, from repo metadata alone.

        No weights are downloaded. See `cortexgrid_infer.requirements` for what
        the figures are built from and when to override them."""
        return requirements.estimate(self.hf_id, self.token, self.ignore_patterns)
