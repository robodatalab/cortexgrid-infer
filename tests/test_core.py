"""Tests for public interfaces in cortexgrid_infer.core."""

from __future__ import annotations

import unittest
from collections.abc import AsyncIterator

import numpy as np
import torch
from parameterized import parameterized
from pydantic import TypeAdapter

from cortexgrid_infer.core import (
    CompletingModel,
    CompletionChunk,
    GeneratedImage,
    GeneratedMesh,
    Message,
    Tensor,
    Tool,
    ToolCall,
    complete,
)


class _StubModel(CompletingModel):
    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[CompletionChunk]:
        yield CompletionChunk(content="stub", finish_reason="stop")


class TestToolCall(unittest.TestCase):
    def test_fields(self):
        tc = ToolCall(
            id="call_x",
            name="noop",
            arguments={"k": "v"},
        )
        self.assertEqual(tc.id, "call_x")
        self.assertEqual(tc.name, "noop")
        self.assertEqual(tc.arguments, {"k": "v"})


class TestCompletionChunk(unittest.TestCase):
    def test_defaults(self):
        chunk = CompletionChunk()
        self.assertEqual(chunk.content, "")
        self.assertEqual(chunk.tool_calls, [])
        self.assertIsNone(chunk.finish_reason)
        self.assertFalse(chunk.has_text)
        self.assertFalse(chunk.has_tool_calls)

    def test_has_text(self):
        chunk = CompletionChunk(content="hello")
        self.assertTrue(chunk.has_text)

    def test_has_tool_calls(self):
        tc = ToolCall(id="1", name="f", arguments={})
        chunk = CompletionChunk(tool_calls=[tc])
        self.assertTrue(chunk.has_tool_calls)

    def test_finish_reason(self):
        chunk = CompletionChunk(finish_reason="stop")
        self.assertEqual(chunk.finish_reason, "stop")


class TestComplete(unittest.IsolatedAsyncioTestCase):
    async def test_yields_chunks(self):
        model = _StubModel()
        chunks = [c async for c in complete(model, [{"role": "user", "content": "hi"}])]
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, "stub")
        self.assertEqual(chunks[0].finish_reason, "stop")


class TestTensor(unittest.TestCase):
    @parameterized.expand([
        ("scalar", torch.tensor(2.0), {"dtype": "float32", "shape": [], "data": "AAAAQA=="}),
        ("empty", torch.zeros(0), {"dtype": "float32", "shape": [0], "data": ""}),
        (
            "vector",
            torch.tensor([1.0, 2.0]),
            {"dtype": "float32", "shape": [2], "data": "AACAPwAAAEA="},
        ),
        (
            "matrix",
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            {"dtype": "float32", "shape": [2, 2], "data": "AACAPwAAAAAAAAAAAACAPw=="},
        ),
        (
            "integers",
            torch.tensor([1, 2]),
            {"dtype": "int64", "shape": [2], "data": "AQAAAAAAAAACAAAAAAAAAA=="},
        ),
        (
            "bfloat16",
            torch.tensor([1.0, -2.0], dtype=torch.bfloat16),
            {"dtype": "bfloat16", "shape": [2], "data": "gD8AwA=="},
        ),
    ])
    def test_goes_to_json_as_its_raw_bytes_with_its_dtype_and_shape(
        self, _case: str, tensor: torch.Tensor, expected: object
    ) -> None:
        tensor_on_the_wire = TypeAdapter(Tensor)

        answered = tensor_on_the_wire.dump_python(tensor, mode="json")

        self.assertEqual(answered, expected)

    @parameterized.expand([
        ("scalar", {"dtype": "float32", "shape": [], "data": "AAAAQA=="}, torch.tensor(2.0)),
        ("empty", {"dtype": "float32", "shape": [0], "data": ""}, torch.zeros(0)),
        (
            "vector",
            {"dtype": "float32", "shape": [2], "data": "AACAPwAAAEA="},
            torch.tensor([1.0, 2.0]),
        ),
        (
            "matrix",
            {"dtype": "float32", "shape": [2, 2], "data": "AACAPwAAAAAAAAAAAACAPw=="},
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ),
        (
            "integers",
            {"dtype": "int64", "shape": [2], "data": "AQAAAAAAAAACAAAAAAAAAA=="},
            torch.tensor([1, 2]),
        ),
        (
            "bfloat16",
            {"dtype": "bfloat16", "shape": [2], "data": "gD8AwA=="},
            torch.tensor([1.0, -2.0], dtype=torch.bfloat16),
        ),
    ])
    def test_comes_from_json_as_a_tensor_of_its_dtype_and_shape(
        self, _case: str, answered: object, expected: torch.Tensor
    ) -> None:
        tensor_on_the_wire = TypeAdapter(Tensor)

        tensor = tensor_on_the_wire.validate_python(answered)

        self.assertEqual(tensor.dtype, expected.dtype)
        self.assertTrue(torch.equal(tensor, expected))


class TestCompletionChunkOnTheWire(unittest.TestCase):
    @parameterized.expand([
        ("content", CompletionChunk(content="hi"), {"content": "hi", "thinking": "", "tool_calls": [], "finish_reason": None}),
        ("thinking", CompletionChunk(thinking="hmm"), {"content": "", "thinking": "hmm", "tool_calls": [], "finish_reason": None}),
        ("finish", CompletionChunk(finish_reason="stop"), {"content": "", "thinking": "", "tool_calls": [], "finish_reason": "stop"}),
        (
            "tool_call",
            CompletionChunk(tool_calls=[ToolCall(id="call_1", name="add", arguments={"a": 2})]),
            {"content": "", "thinking": "", "tool_calls": [{"id": "call_1", "name": "add", "arguments": {"a": 2}}], "finish_reason": None},
        ),
    ])
    def test_crosses_as_its_fields(
        self, _case: str, chunk: CompletionChunk, expected: dict
    ) -> None:
        chunk_on_the_wire = TypeAdapter(CompletionChunk)

        answered = chunk_on_the_wire.dump_python(chunk, mode="json")
        chunk_back = chunk_on_the_wire.validate_python(answered)

        self.assertEqual(answered, expected)
        self.assertEqual(chunk_back, chunk)


class TestGeneratedImage(unittest.TestCase):
    @parameterized.expand([
        ("bytes", b"\x89PNG", "iVBORw=="),
        ("no_bytes", b"", ""),
    ])
    def test_crosses_with_its_picture_in_base64(
        self, _case: str, picture: bytes, expected_image: str
    ) -> None:
        image_on_the_wire = TypeAdapter(GeneratedImage)
        generated = GeneratedImage(image=picture, width=2, height=1, params={"seed": 7})

        answered = image_on_the_wire.dump_python(generated, mode="json")
        generated_back = image_on_the_wire.validate_python(answered)

        self.assertEqual(
            answered, {"image": expected_image, "width": 2, "height": 1, "params": {"seed": 7}}
        )
        self.assertEqual(generated_back, generated)


class TestGeneratedMesh(unittest.TestCase):
    def test_crosses_with_its_arrays_in_base64(self) -> None:
        mesh_on_the_wire = TypeAdapter(GeneratedMesh)
        mesh = GeneratedMesh(
            vertices=np.array([[0.0, 1.0, 2.0]], dtype=np.float32),
            faces=np.array([[0, 0, 0]], dtype=np.int32),
            colours=np.array([[255, 0, 1]], dtype=np.uint8),
            params={"resolution": 64},
        )

        answered = mesh_on_the_wire.dump_python(mesh, mode="json")
        mesh_back = mesh_on_the_wire.validate_python(answered)

        self.assertEqual(
            answered,
            {
                "vertices": "AAAAAAAAgD8AAABA",
                "faces": "AAAAAAAAAAAAAAAA",
                "colours": "/wAB",
                "params": {"resolution": 64},
            },
        )
        np.testing.assert_array_equal(mesh_back.vertices, mesh.vertices)
        np.testing.assert_array_equal(mesh_back.faces, mesh.faces)
        np.testing.assert_array_equal(mesh_back.colours, mesh.colours)
        self.assertEqual(mesh_back.vertices.dtype, np.float32)
        self.assertEqual(mesh_back.faces.dtype, np.int32)
        self.assertEqual(mesh_back.colours.dtype, np.uint8)
