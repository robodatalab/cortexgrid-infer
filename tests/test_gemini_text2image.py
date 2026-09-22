"""Tests for cortexgrid_infer.serve_apps.gemini.text2image, registered as a Hosted entry."""

from __future__ import annotations

import asyncio
import base64
import io
import unittest
from unittest import mock

import cortexgrid
from fastapi import HTTPException
from google.genai import types
from PIL import Image

from cortexgrid_infer.protocols.imaging import ServedGeneratingModel
from cortexgrid_infer.registry import Hosted
from cortexgrid_infer.serve_apps.gemini.text2image import GeminiText2Image, image_size

BASE = "cortexgrid_infer.serve_apps.gemini.base"


def _picture(format: str, size: tuple[int, int] = (8, 4)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "red").save(buffer, format=format)
    return buffer.getvalue()


def _response(*parts: types.Part, **fields) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))],
        **fields,
    )


def _image_part(data: bytes, mime_type: str = "image/png") -> types.Part:
    return types.Part(inline_data=types.Blob(data=data, mime_type=mime_type))


class TestHostedGeminiText2Image(unittest.TestCase):
    def test_names_the_model_for_the_registry(self):
        entry = Hosted("gemini-2.5-flash-image", GeminiText2Image)
        self.assertEqual((entry.family, entry.suffix), ("gemini-2.5-flash", "image"))

    def test_client_is_the_shared_generating_client(self):
        # The serve app answers the same /generate route as Text2Image, so
        # there is nothing Gemini-specific left on the client side.
        model = Hosted("gemini-2.5-flash-image", GeminiText2Image).client("http://h")
        self.assertIsInstance(model, ServedGeneratingModel)
        self.assertEqual(model.name, "gemini-2.5-flash-image")

    def test_asks_for_no_hardware(self):
        self.assertEqual(
            Hosted("gemini-2.5-flash-image", GeminiText2Image).requirements(),
            cortexgrid.ModelRequirements(),
        )


class TestImageSize(unittest.TestCase):
    def test_up_to_1k_is_left_to_the_model(self):
        self.assertIsNone(image_size(512))
        self.assertIsNone(image_size(1024))

    def test_larger_sizes_pick_the_tier_that_covers_them(self):
        self.assertEqual(image_size(1536), "2K")
        self.assertEqual(image_size(2048), "2K")
        self.assertEqual(image_size(4096), "4K")


class TestGenerate(unittest.TestCase):
    def _generate(self, response, body):
        with (
            mock.patch(f"{BASE}.cortexgrid") as mock_cortexgrid,
            mock.patch(f"{BASE}.genai") as mock_genai,
        ):
            mock_cortexgrid.model_config.return_value = {
                "model": "gemini-2.5-flash-image",
                "api_key_secret": "GEMINI_API_KEY",
            }
            call = mock.AsyncMock(return_value=response)
            mock_genai.Client.return_value.aio.models.generate_content = call
            deployment = GeminiText2Image("gemini-2.5-flash", "image", "imported")

            return asyncio.run(GeminiText2Image.generate(deployment, body)), call

    def test_returns_the_picture_as_png_with_its_size(self):
        png = _picture("PNG")

        result, _call = self._generate(_response(_image_part(png)), {"prompt": "a fox"})

        self.assertEqual(base64.b64decode(result["image"]), png)
        self.assertEqual((result["width"], result["height"]), (8, 4))

    def test_converts_other_formats_to_png(self):
        result, _call = self._generate(
            _response(_image_part(_picture("JPEG"), "image/jpeg")), {"prompt": "a fox"}
        )

        image = Image.open(io.BytesIO(base64.b64decode(result["image"])))
        self.assertEqual(image.format, "PNG")

    def test_reports_diffusion_settings_as_unused(self):
        result, _call = self._generate(
            _response(_image_part(_picture("PNG"))),
            {"prompt": "a fox", "steps": 30, "guidance": 3.5, "seed": 7},
        )

        self.assertIsNone(result["steps"])
        self.assertIsNone(result["guidance"])
        self.assertEqual(result["seed"], 7)

    def test_asks_for_a_square_image_of_the_requested_size(self):
        _result, call = self._generate(
            _response(_image_part(_picture("PNG"))),
            {"prompt": "a fox", "size": 2048, "seed": 7},
        )

        self.assertEqual(
            call.call_args.kwargs,
            {
                "model": "gemini-2.5-flash-image",
                "contents": "a fox",
                "config": {
                    "response_modalities": ["IMAGE"],
                    "image_config": {"aspect_ratio": "1:1", "image_size": "2K"},
                    "seed": 7,
                },
            },
        )
        types.GenerateContentConfig.model_validate(call.call_args.kwargs["config"])

    def test_the_default_size_sends_no_tier(self):
        _result, call = self._generate(
            _response(_image_part(_picture("PNG"))), {"prompt": "a fox", "size": None}
        )

        self.assertEqual(
            call.call_args.kwargs["config"]["image_config"], {"aspect_ratio": "1:1"}
        )

    def test_refuses_a_reference_image(self):
        with self.assertRaises(HTTPException) as caught:
            self._generate(_response(), {"prompt": "a fox", "image": "aGk="})

        self.assertEqual(caught.exception.status_code, 400)

    def test_says_what_the_model_said_instead_of_a_picture(self):
        # A refused prompt comes back as text, not as an error.
        with self.assertRaises(HTTPException) as caught:
            self._generate(
                _response(types.Part(text="I can't draw that.")), {"prompt": "a fox"}
            )

        self.assertEqual(caught.exception.status_code, 502)
        self.assertIn("I can't draw that.", caught.exception.detail)

    def test_says_when_the_prompt_was_blocked(self):
        response = types.GenerateContentResponse(
            candidates=[],
            prompt_feedback=types.GenerateContentResponsePromptFeedback(
                block_reason=types.BlockedReason.SAFETY
            ),
        )

        with self.assertRaises(HTTPException) as caught:
            self._generate(response, {"prompt": "a fox"})

        self.assertIn("blocked", caught.exception.detail)
