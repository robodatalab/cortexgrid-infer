"""The base of every image-to-mesh serve app: the `POST /mesh` route and loading the
model's files from the cortexgrid registry, around the two steps that differ per
model.

There is no one loader for image-to-mesh models the way `AutoModelForCausalLM`
or a diffusers `model_index.json` serves a whole family of repos: each model
ships its own code. So a model's serve app subclasses `MeshingDeployment` and
supplies `load` and `make_mesh`; the subclass inherits the route, whose protocol
is `cortexgrid_infer.meshing`'s. cortexgrid bundles the subclass's own file, so
the model's code travels with it.

    class MyMeshDeployment(MeshingDeployment):
        def load(self, path, device): ...
        def make_mesh(self, image, **options) -> GeneratedMesh: ...
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

from cortexgrid_infer import meshing
from cortexgrid_infer.core import GeneratedMesh
from cortexgrid_infer.device import detect_device


_app = FastAPI()


@serve.ingress(_app)
class MeshingDeployment:
    """One replica of an image-to-mesh model."""

    def __init__(self, family: str, suffix: str, run_name: str) -> None:
        self.device = detect_device()
        self.load(Path(cortexgrid.load_model(family, suffix, run_name)), self.device)

    def load(self, path: Path, device: torch.device) -> None:
        """Build the model from its files at `path`, on `device`."""
        raise NotImplementedError

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
