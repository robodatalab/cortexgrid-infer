"""Tests for the image-to-mesh task: the /mesh protocol's client and the serve
app every model of the task subclasses."""

from __future__ import annotations

import asyncio
import io
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import cortexgrid
from cortexgrid import serve
import numpy as np
from PIL import Image
import torch

from cortexgrid_infer.core import GeneratedMesh
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


class _TriangleDeployment(Image2Mesh):
    """A model that answers every picture with one triangle."""

    min_vram_gb = 6.0

    def load(self, path: Path, device: Any) -> None:
        self.loaded = (path, device)

    def make_mesh(self, image: Image.Image, **options: Any) -> GeneratedMesh:
        self.asked = (image.size, options)
        return TRIANGLE


class _CompilingTriangleDeployment(_TriangleDeployment):
    """The triangle model, compiling what it loaded when asked to."""

    compiled_on: Any = None

    def compile(self, device: Any) -> None:
        self.compiled_on = device


class TestImage2Mesh(unittest.TestCase):
    SERVE = "cortexgrid_infer.serve_apps.image2mesh"

    def setUp(self):
        with mock.patch(f"{self.SERVE}.cortexgrid.load_model", return_value="/weights"), \
             mock.patch(f"{self.SERVE}.cortexgrid.model_config", return_value={}), \
             mock.patch(f"{self.SERVE}.detect_device", return_value="cpu"):
            self.deployment = _TriangleDeployment(cortexgrid.DeploymentKey("family", "suffix", "imported"))

    def mesh(self, **options: Any) -> GeneratedMesh:
        image = _png(Image.new("RGBA", (40, 20)))
        return asyncio.run(self.deployment.mesh(image, options))

    def test_loads_the_model_from_the_registry(self):
        self.assertEqual(self.deployment.loaded, (Path("/weights"), "cpu"))

    def test_hands_the_model_the_picture_and_the_options_given(self):
        self.mesh(resolution=128, unused=None)

        self.assertEqual(self.deployment.asked, ((40, 20), {"resolution": 128}))

    def test_answers_with_the_mesh_and_what_the_model_resolved(self):
        reply = self.mesh()

        np.testing.assert_array_equal(reply.vertices, TRIANGLE.vertices)
        np.testing.assert_array_equal(reply.faces, TRIANGLE.faces)
        np.testing.assert_array_equal(reply.colours, TRIANGLE.colours)
        self.assertEqual(reply.params, {"resolution": 64, "duration_s": mock.ANY})

    def test_a_model_s_serve_app_is_fronted_by_the_family_s_endpoint(self):
        self.assertIs(serve.ingress_app(_TriangleDeployment), serve.ingress_app(Image2Mesh))


class TestImage2MeshCompiles(unittest.TestCase):
    SERVE = "cortexgrid_infer.serve_apps.image2mesh"

    def build(self, card: dict[str, str], device: Any = torch.device("cuda"),
              app: type[Image2Mesh] = _CompilingTriangleDeployment) -> Any:
        with mock.patch(f"{self.SERVE}.cortexgrid.load_model", return_value="/weights"), \
             mock.patch(f"{self.SERVE}.cortexgrid.model_config", return_value=card), \
             mock.patch(f"{self.SERVE}.detect_device", return_value=device):
            return app(cortexgrid.DeploymentKey("family", "suffix", "imported"))

    def test_hands_the_model_its_device_to_compile_on_when_the_card_asks(self):
        deployment = self.build({"compile": "true"})

        self.assertTrue(deployment.compiled)
        self.assertEqual(deployment.compiled_on, torch.device("cuda"))

    def test_compiles_what_was_loaded(self):
        # The hook has the model to compile only once `load` has built it.
        deployment = self.build({"compile": "true"})

        self.assertEqual(deployment.loaded, (Path("/weights"), torch.device("cuda")))

    def test_stays_eager_unless_the_model_card_asks(self):
        deployment = self.build({})

        self.assertFalse(deployment.compiled)
        self.assertIsNone(deployment.compiled_on)

    def test_stays_eager_off_cuda(self):
        deployment = self.build({"compile": "true"}, device=torch.device("cpu"))

        self.assertFalse(deployment.compiled)
        self.assertIsNone(deployment.compiled_on)

    def test_a_model_that_compiles_nothing_still_serves(self):
        deployment = self.build({"compile": "true"}, app=_TriangleDeployment)

        self.assertEqual(deployment.loaded, (Path("/weights"), torch.device("cuda")))


class TestImage2MeshForTheImporter(unittest.TestCase):
    def test_the_model_card_serves_eager_unless_told_otherwise(self):
        self.assertEqual(_TriangleDeployment.config(), {"compile": "false"})

    def test_client_calls_the_family_s_endpoint(self):
        deployment = cortexgrid.Deployment(
            key=cortexgrid.DeploymentKey("Tri", "small", "R"),
            config={},
            url="http://h/r/Tri/small/R",
            phase="running",
            bundle_fingerprint="",
            replaced_bundle_fingerprint="",
            experiment_name="",
            class_import_path="tests.test_meshing:_TriangleDeployment",
        )

        model = _TriangleDeployment.client(deployment)

        self.assertIs(type(model), Image2Mesh.client)
        self.assertEqual(model.key, cortexgrid.DeploymentKey("Tri", "small", "R"))
        self.assertEqual(model.url, "http://h/r/Tri/small/R")

    def test_asks_for_the_memory_meshing_takes_not_just_the_weights(self):
        needs = _TriangleDeployment.requirements(Weights(params=1_000_000))

        self.assertEqual((needs.num_gpus, needs.vram_gb), (1, 6.0))


if __name__ == "__main__":
    unittest.main()
