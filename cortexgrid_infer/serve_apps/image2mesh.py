"""The base of every image-to-mesh serve app: the `POST /mesh` route and loading the
model's files from the cortexgrid registry, around the two steps that differ per
model.

There is no one loader for image-to-mesh models the way `AutoModelForCausalLM`
or a diffusers `model_index.json` serves a whole family of repos: each model
ships its own code. So a model's serve app subclasses `Image2Mesh` and supplies
`load` and `make_mesh`; the subclass inherits the route, whose protocol is
`cortexgrid_infer.protocols.meshing`'s, and the client that speaks it.
cortexgrid bundles the subclass's own file, so the model's code travels with it.

    class MyMesh(Image2Mesh):
        min_vram_gb = 6.0
        def load(self, path, device): ...
        def make_mesh(self, image, **options) -> GeneratedMesh: ...
        def compile(self, device): ...  # optional; see `Image2Mesh.compile`

    imp = HuggingFaceImporter("org/my-mesh-model", MyMesh)
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

import cortexgrid
from cortexgrid import serve
from fastapi import FastAPI
from PIL import Image, ImageOps
import torch

from cortexgrid_infer import compiling
from cortexgrid_infer.core import GeneratedMesh
from cortexgrid_infer.device import detect_device
from cortexgrid_infer.protocols import meshing
from cortexgrid_infer.protocols.meshing import ServedMeshingModel
from cortexgrid_infer.serve_apps.base import COMPILE_PARAM, LocalModel


_app = FastAPI()


@serve.ingress(_app)
class Image2Mesh(LocalModel):
    """One replica of an image-to-mesh model.

    A subclass sets `min_vram_gb` where meshing takes memory the weights do not
    show - querying the model over a whole 3D grid usually does."""

    @classmethod
    def client(cls, url: str, name: str) -> ServedMeshingModel:
        return ServedMeshingModel(url=url, model_id=name)

    def __init__(self, family: str, suffix: str, run_name: str) -> None:
        self.device = detect_device()
        self.load(Path(cortexgrid.load_model(family, suffix, run_name)), self.device)
        settings = cortexgrid.model_config(family, suffix, run_name)
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

    @_app.post("/mesh")
    async def mesh(self, body: dict[str, Any]) -> dict[str, Any]:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(body["image"]))))
        options = {key: value for key, value in body.items() if key != "image" and value is not None}

        t0 = time.time()
        made = self.make_mesh(image, **options)
        duration = time.time() - t0

        return {
            "vertices": meshing.encode(made.vertices, meshing.VERTICES),
            "faces": meshing.encode(made.faces, meshing.FACES),
            "colours": meshing.encode(made.colours, meshing.COLOURS),
            "params": made.params,
            "duration_s": round(duration, 2),
        }
