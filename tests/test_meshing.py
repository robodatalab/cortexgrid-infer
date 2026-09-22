"""Tests for the image-to-mesh task: the /mesh protocol's client and the serve
app every model of the task subclasses."""

from __future__ import annotations

import asyncio
import base64
import io
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from cortexgrid import serve
import numpy as np
from PIL import Image

from cortexgrid_infer.core import GeneratedMesh
from cortexgrid_infer.protocols.meshing import ServedMeshingModel
from cortexgrid_infer.serve_apps.base import Weights
from cortexgrid_infer.serve_apps.image2mesh import Image2Mesh

TRIANGLE = GeneratedMesh(
    vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
    faces=np.array([[0, 1, 2]], dtype=np.int32),
    colours=np.array([[255, 0, 0]] * 3, dtype=np.uint8),
    params={"resolution": 64},
)


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeAsyncClient:
    captured: dict[str, Any] = {}

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:
        type(self).captured = {"url": url, "json": json}
        return _FakeResponse(self._payload)


class _TriangleDeployment(Image2Mesh):
    """A model that answers every picture with one triangle."""

    min_vram_gb = 6.0

    def load(self, path: Path, device: Any) -> None:
        self.loaded = (path, device)

    def make_mesh(self, image: Image.Image, **options: Any) -> GeneratedMesh:
        self.asked = (image.size, options)
        return TRIANGLE


class TestImage2Mesh(unittest.TestCase):
    SERVE = "cortexgrid_infer.serve_apps.image2mesh"

    def setUp(self):
        with mock.patch(f"{self.SERVE}.cortexgrid.load_model", return_value="/weights"), \
             mock.patch(f"{self.SERVE}.detect_device", return_value="cpu"):
            self.deployment = _TriangleDeployment("family", "suffix", "imported")

    def mesh(self, **body: Any) -> dict[str, Any]:
        image = base64.b64encode(_png(Image.new("RGBA", (40, 20)))).decode("ascii")
        return asyncio.run(self.deployment.mesh({"image": image, **body}))

    def test_loads_the_model_from_the_registry(self):
        self.assertEqual(self.deployment.loaded, (Path("/weights"), "cpu"))

    def test_hands_the_model_the_picture_and_the_options_given(self):
        self.mesh(resolution=128, unused=None)

        self.assertEqual(self.deployment.asked, ((40, 20), {"resolution": 128}))

    def test_answers_with_the_mesh_and_what_the_model_resolved(self):
        reply = self.mesh()

        vertices = np.frombuffer(base64.b64decode(reply["vertices"]), "<f4").reshape(-1, 3)
        np.testing.assert_array_equal(vertices, TRIANGLE.vertices)
        self.assertEqual(list(np.frombuffer(base64.b64decode(reply["faces"]), "<i4")), [0, 1, 2])
        self.assertEqual(reply["params"], {"resolution": 64})
        self.assertIn("duration_s", reply)

    def test_a_model_s_serve_app_is_fronted_by_the_family_s_route(self):
        self.assertIs(serve.ingress_app(_TriangleDeployment), serve.ingress_app(Image2Mesh))


class TestServedMeshingModel(unittest.IsolatedAsyncioTestCase):
    async def test_round_trips_a_mesh_through_the_protocol(self):
        with mock.patch("cortexgrid_infer.serve_apps.image2mesh.cortexgrid.load_model", return_value="/w"):
            deployment = _TriangleDeployment("family", "suffix", "imported")
        model = ServedMeshingModel(url="http://h/r/Tri/base/R", model_id="org/Tri")
        payload: dict[str, Any] = {}

        class _Served(_FakeAsyncClient):
            async def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:
                type(self).captured = {"url": url, "json": json}
                return _FakeResponse(await deployment.mesh(dict(json)))

        with mock.patch("cortexgrid_infer.protocols.meshing.httpx.AsyncClient", lambda *_a, **_kw: _Served(payload)):
            made = await model.mesh(_png(Image.new("RGBA", (8, 8))), resolution=32)

        np.testing.assert_array_equal(made.vertices, TRIANGLE.vertices)
        np.testing.assert_array_equal(made.faces, TRIANGLE.faces)
        np.testing.assert_array_equal(made.colours, TRIANGLE.colours)
        self.assertEqual(made.params["resolution"], 64)
        self.assertEqual(made.params["model_id"], "org/Tri")
        self.assertEqual(_Served.captured["url"], "http://h/r/Tri/base/R/mesh")
        self.assertEqual(deployment.asked[1], {"resolution": 32})


class TestImage2MeshForTheImporter(unittest.TestCase):
    def test_client_speaks_the_task_s_protocol(self):
        model = _TriangleDeployment.client("http://h/r/Tri/small/R", "org/Tri-small")
        self.assertIsInstance(model, ServedMeshingModel)
        self.assertEqual((model.url, model.name), ("http://h/r/Tri/small/R", "org/Tri-small"))

    def test_asks_for_the_memory_meshing_takes_not_just_the_weights(self):
        needs = _TriangleDeployment.requirements(Weights(params=1_000_000))

        self.assertEqual((needs.num_gpus, needs.vram_gb), (1, 6.0))


if __name__ == "__main__":
    unittest.main()
