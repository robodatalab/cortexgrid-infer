"""The base of the importers: a model whose weights come from somewhere, run on
the cluster by a `LocalModel` serve app.

An importer knows the source - how a model id names itself in the registry,
how to download the weights, and how big they are - and nothing about the
model's task. That is the serve app's, which the importer is handed, so one
importer takes any model its source holds:

    imp = HuggingFaceImporter("Qwen/Qwen2.5-0.5B-Instruct", Text2Text)
    with imp:
        cortexgrid.import_model(
            imp.source, imp.serve_app,
            family=imp.family, suffix=imp.suffix,
            requirements=imp.requirements(),
        )

An importer is cheap to construct and carries no open resources until it is
entered, so it can be handed to `cortexgrid.remote` and rebuilt on the node that
runs the import - which is where the download belongs, so the weights travel
source -> node -> registry and never transit the client.
"""

from __future__ import annotations

import abc
import tempfile
from types import TracebackType
from typing import Self

import cortexgrid

from cortexgrid_infer.core import DeployedModel
from cortexgrid_infer.models.base import LocalModel, Weights
from cortexgrid_infer.registry import ModelEntry, split_model_id


class Importer(ModelEntry):
    """Identity, download and sizing for one model from one source."""

    serve_app: type[LocalModel]

    def __init__(self, model_id: str, serve_app: type[LocalModel]) -> None:
        self.model_id = model_id
        self.serve_app = serve_app
        self.family, self.suffix = split_model_id(model_id)
        self._scratch: tempfile.TemporaryDirectory[str] | None = None

    @abc.abstractmethod
    def download(self, local_dir: str) -> None:
        """Fetch the weights the serve app loads into `local_dir`, and nothing else."""

    @abc.abstractmethod
    def weights(self) -> Weights:
        """What the source records about the weights, read without downloading them."""

    def __enter__(self) -> Self:
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
        self.download(self._scratch.name)
        return self._scratch.name

    def requirements(self) -> cortexgrid.ModelRequirements:
        """Estimate what one replica needs, from the source's metadata alone.

        No weights are downloaded. These are estimates: pass a
        `cortexgrid.ModelRequirements` of your own to `import_model` where you
        know better, or edit the figures on the model card afterwards -
        `import_model` only fills in requirements a version does not already
        have, so a hand-set value is never overwritten."""
        return self.serve_app.requirements(self.weights())

    def client(self, url: str) -> DeployedModel:
        return self.serve_app.client(url, self.model_id)
