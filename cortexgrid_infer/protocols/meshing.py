"""The wire protocol between a cortexgrid-served image-to-mesh app and its client.

An image-to-mesh serve app answers `POST /mesh`: the request carries a picture
of an object as a base64 PNG under `image`, plus whatever options the model
takes (a marching-cubes resolution, say) as further fields. The reply carries
the mesh as base64 little-endian arrays - `vertices` float32 x, y, z, `faces`
int32 indices, `colours` uint8 r, g, b per vertex - in the frame
`GeneratedMesh` documents, and `params`, what the model resolved.

Every such app speaks it through
`cortexgrid_infer.serve_apps.image2mesh.Image2Mesh`, so one client,
`ServedMeshingModel`, serves every model of the task whoever made it.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from cortexgrid_infer.core import GeneratedMesh, MeshingModel

# How each array travels: its type, and how many values make one row.
VERTICES = ("<f4", 3)
FACES = ("<i4", 3)
COLOURS = ("u1", 3)


def encode(array: np.ndarray, kind: tuple[str, int]) -> str:
    """`array` as the protocol sends it."""
    dtype, _ = kind
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def decode(data: str, kind: tuple[str, int]) -> np.ndarray:
    """An array the protocol sent, one row per vertex or triangle."""
    dtype, columns = kind
    return np.frombuffer(base64.b64decode(data), dtype=dtype).reshape(-1, columns)


@dataclass
class ServedMeshingModel(MeshingModel):
    """The client of any image-to-mesh serve app deployed at `url`."""

    url: str
    model_id: str

    @property
    def name(self) -> str:
        return self.model_id

    async def mesh(self, image: bytes, **options: Any) -> GeneratedMesh:
        body: dict[str, Any] = {"image": base64.b64encode(image).decode("ascii"), **options}
        # Meshing takes seconds; no client-side timeout.
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(f"{self.url}/mesh", json=body)
            response.raise_for_status()
            data = response.json()

        return GeneratedMesh(
            vertices=decode(data["vertices"], VERTICES),
            faces=decode(data["faces"], FACES),
            colours=decode(data["colours"], COLOURS),
            params={
                **data.get("params", {}),
                "duration_s": data.get("duration_s"),
                "model_id": self.model_id,
            },
        )
