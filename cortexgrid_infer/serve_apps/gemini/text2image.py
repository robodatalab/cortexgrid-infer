"""The text-to-image serve app that forwards to the Gemini API.

It answers the same `POST /generate` route as `Text2Image` (see
`cortexgrid_infer.protocols.imaging`), so the same client talks to it. It takes a prompt
only. Of the sampling options, `size` and `seed` have a Gemini counterpart;
`steps` and `guidance` are diffusion settings, and are reported back as unused.
"""

from __future__ import annotations

import base64
import io
import time
from typing import Any

from cortexgrid import serve
from fastapi import FastAPI, HTTPException
from google.genai import types
from PIL import Image

from cortexgrid_infer.protocols.imaging import ServedGeneratingModel
from cortexgrid_infer.serve_apps.gemini.base import GeminiModel

_app = FastAPI()


def image_size(size: int) -> str | None:
    """The Gemini size tier that covers `size` pixels a side.

    None up to 1K, the model's default, so that it is left unsent: not every
    Gemini image model takes a size tier."""
    if size <= 1024:
        return None
    if size <= 2048:
        return "2K"
    return "4K"


def to_png(data: bytes) -> tuple[bytes, int, int]:
    """`data`, a picture in whatever format Gemini returned it, as PNG bytes
    along with its width and height."""
    image = Image.open(io.BytesIO(data))
    if image.format == "PNG":
        return data, image.width, image.height
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), image.width, image.height


def no_image_reason(response: types.GenerateContentResponse) -> str:
    """Why `response` carries no picture: what the model said instead, or why
    it stopped."""
    feedback = response.prompt_feedback
    if feedback and feedback.block_reason:
        return f"the prompt was blocked ({feedback.block_reason})"
    text = " ".join(part.text for part in response.parts or [] if part.text)
    if text:
        return text
    candidate = response.candidates[0] if response.candidates else None
    return f"finish reason {candidate.finish_reason if candidate else None}"


@serve.ingress(_app)
class GeminiText2Image(GeminiModel):
    @classmethod
    def client(cls, url: str, name: str) -> ServedGeneratingModel:
        return ServedGeneratingModel(url=url, model_id=name)

    @_app.post("/generate")
    async def generate(self, body: dict[str, Any]) -> dict[str, Any]:
        if body.get("image"):
            raise HTTPException(
                status_code=400,
                detail=f"{type(self).__name__} takes a prompt only, not an image",
            )
        size = int(body.get("size") or 1024)
        seed = body.get("seed")

        # Square, like `Text2Image` without a reference picture.
        image_config: dict[str, Any] = {"aspect_ratio": "1:1"}
        tier = image_size(size)
        if tier:
            image_config["image_size"] = tier
        config: dict[str, Any] = {
            "response_modalities": ["IMAGE"],
            "image_config": image_config,
        }
        if seed is not None:
            config["seed"] = int(seed)

        t0 = time.time()
        response = await self._client.aio.models.generate_content(
            model=self._model, contents=body["prompt"], config=config
        )
        duration = time.time() - t0

        data = next(
            (part.inline_data.data for part in response.parts or [] if part.inline_data),
            None,
        )
        if data is None:
            raise HTTPException(
                status_code=502,
                detail=f"{self._model} returned no image: {no_image_reason(response)}",
            )

        png, width, height = to_png(data)
        return {
            "image": base64.b64encode(png).decode("ascii"),
            "width": width,
            "height": height,
            "steps": None,
            "guidance": None,
            "seed": seed,
            "duration_s": round(duration, 2),
        }
