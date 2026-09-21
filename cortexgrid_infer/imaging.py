"""The wire protocol between a cortexgrid-served text-to-image app and its client.

A text-to-image serve app answers `POST /generate`: the request carries the
`prompt`, optionally a reference picture as a base64 PNG under `image`, and the
sampling options (`steps`, `guidance`, `size`, `seed`) - any left out, or sent
as null, take the app's defaults. The reply carries the picture as a base64 PNG
under `image`, its `width` and `height`, the options the app resolved, and
`duration_s`.

`cortexgrid_infer.models.text2image.Text2Image` speaks it for any diffusers
pipeline, so one client, `ServedGeneratingModel`, serves every model of the
task whoever made it.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import httpx

from cortexgrid_infer.core import GeneratedImage, GeneratingModel


@dataclass
class ServedGeneratingModel(GeneratingModel):
    """The client of any text-to-image serve app deployed at `url`."""

    url: str
    model_id: str

    @property
    def name(self) -> str:
        return self.model_id

    async def generate(
        self,
        prompt: str,
        *,
        image: bytes | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        size: int = 1024,
        seed: int | None = None,
        **kwargs: Any,
    ) -> GeneratedImage:
        body: dict[str, Any] = {
            "prompt": prompt,
            "image": base64.b64encode(image).decode("ascii") if image else None,
            "steps": steps,
            "guidance": guidance,
            "size": size,
            "seed": seed,
            **kwargs,
        }
        # Diffusion is slow (tens of seconds); no client-side timeout.
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(f"{self.url}/generate", json=body)
            response.raise_for_status()
            data = response.json()

        return GeneratedImage(
            image=base64.b64decode(data["image"]),
            width=data["width"],
            height=data["height"],
            params={
                "steps": data.get("steps", steps),
                "guidance": data.get("guidance", guidance),
                "seed": data.get("seed", seed),
                "duration_s": data.get("duration_s"),
                "model_id": self.model_id,
            },
        )
