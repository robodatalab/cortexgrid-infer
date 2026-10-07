"""The base of every image-to-mesh serve app: the `mesh` endpoint and loading the
model's files from the cortexgrid registry, around the two steps that differ per
model.

There is no one loader for image-to-mesh models the way `AutoModelForCausalLM`
or a diffusers `model_index.json` serves a whole family of repos: each model
ships its own code. So a model's serve app subclasses `Image2Mesh` and supplies
`load` and `make_mesh`; the subclass inherits the endpoint, whose contract is
`cortexgrid_infer.core.MeshingModel`'s, and the client that calls it.
cortexgrid bundles the subclass's own file, so the model's code travels with it.

    class MyMesh(Image2Mesh):
        min_vram_gb = 6.0
        def load(self, path, device): ...
        def make_mesh(self, image, **options) -> GeneratedMesh: ...
        def compile(self, device): ...  # optional; see `Image2Mesh.compile`

    imp = HuggingFaceImporter("org/my-mesh-model", MyMesh)
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import cortexgrid
from cortexgrid import serve
from PIL import Image, ImageOps
from pydantic import Base64Bytes
import torch

from cortexgrid_infer import compiling
from cortexgrid_infer.core import GeneratedMesh, MeshingModel
from cortexgrid_infer.device import detect_device
from cortexgrid_infer.serve_apps.base import COMPILE_PARAM, LocalModel


@serve.ingress
class Image2Mesh(LocalModel, MeshingModel):
    """One replica of an image-to-mesh model.

    A subclass sets `min_vram_gb` where meshing takes memory the weights do not
    show - querying the model over a whole 3D grid usually does."""

    def __init__(self, deployment: cortexgrid.DeploymentKey) -> None:
        self.device = detect_device()
        path = cortexgrid.load_model(deployment.family, deployment.suffix, deployment.run_name)
        self.load(Path(path), self.device)
        settings = cortexgrid.model_config(deployment)
        requested = settings.get(COMPILE_PARAM, "false") == "true"
        self.compiled = requested and compiling.supported(self.device)
        if self.compiled:
            self.compile(self.device)

    def load(self, path: Path, device: torch.device) -> None:
        """Build the model from its files at `path`, on `device`."""
        raise NotImplementedError

    def compile(self, device: torch.device) -> None:
        """Compile what `load` built, on `device`.

        Called after `load`, and only when the model card asks for it on a
        device worth compiling on. Does nothing unless overridden: which of a
        model's modules pay back compiling, and in which `compiling.Mode`, is
        the model's to know - a denoising loop of fixed shapes suits
        `Mode.GRAPHED`, a decoder queried over a grid whose size moves suits
        `Mode.FUSED`. `compiling.compile_module` does the rest."""

    def make_mesh(self, image: Image.Image, **options: Any) -> GeneratedMesh:
        """The mesh of the object in `image`, in the frame `GeneratedMesh`
        documents, with the options the request carried; `params` says what the
        model resolved them to."""
        raise NotImplementedError

    @serve.endpoint
    async def mesh(
        self, image: Base64Bytes, options: dict[str, Any] | None = None
    ) -> GeneratedMesh:
        picture = Image.open(io.BytesIO(image))
        upright = ImageOps.exif_transpose(picture)
        requested_options = options or {}
        given_options = {
            key: value for key, value in requested_options.items() if value is not None
        }

        t0 = time.time()
        made = self.make_mesh(upright, **given_options)
        duration = time.time() - t0

        meshed = GeneratedMesh(
            vertices=made.vertices,
            faces=made.faces,
            colours=made.colours,
            params={**made.params, "duration_s": round(duration, 2)},
        )
        return meshed
