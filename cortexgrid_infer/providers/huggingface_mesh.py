"""HuggingFace image-to-mesh provider: the importer an image-to-mesh model on the
Hub is taken into the cortexgrid registry with, and the client of what it serves.

Each such model brings its own code (see `cortexgrid_infer.meshing_serve`), so
this is a base: a model's own importer subclasses it and names its serve app.

    class MyMeshImport(HuggingFaceMeshImport):
        serve_app = MyMeshDeployment   # a MeshingDeployment subclass
        vram_gb = 6.0
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import cortexgrid

from cortexgrid_infer.importing import HuggingFaceImport
from cortexgrid_infer.meshing import ServedMeshingModel


class HuggingFaceMeshImport(HuggingFaceImport):
    """What `cortexgrid.import_model` needs to take an image-to-mesh model from
    the Hub, less the serve app, which the subclass names."""

    # What one replica needs at least, whatever the weights come to: meshing
    # queries the model over a whole 3D grid, which takes memory the weights do
    # not show.
    vram_gb: ClassVar[float] = 0.0

    def requirements(self) -> cortexgrid.ModelRequirements:
        estimate = super().requirements()
        return dataclasses.replace(estimate, vram_gb=max(estimate.vram_gb, self.vram_gb))

    def client(self, url: str) -> ServedMeshingModel:
        return ServedMeshingModel(url=url, model_id=self.hf_id)
