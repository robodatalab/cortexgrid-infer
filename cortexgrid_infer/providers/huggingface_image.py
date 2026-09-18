"""HuggingFace image provider: deploys a diffusers pipeline via cortexgrid and
clients it over HTTP.

The image analogue of :mod:`cortexgrid_infer.providers.huggingface_complete`:
the importer cortexgrid serves a diffusers pipeline from, and the client that
speaks the deployed app's routes.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import os
from typing import Any

import httpx

from cortexgrid_infer.core import GeneratedImage, GeneratingModel
from cortexgrid_infer.importing import HuggingFaceImport
from cortexgrid_infer.providers.huggingface_image_serve import HuggingFaceImageDeployment


# `from_pretrained` reads the component subfolders + configs; it never touches the
# example images, docs, or a repo's consolidated single-file checkpoint. Skipping
# those keeps the snapshot (and the S3 upload that follows) to just the weights the
# pipeline loads — e.g. FLUX.2 ships a ~7.75 GB single-file checkpoint on top of the
# ~16 GB of component weights. Extend per-model via HF_IMAGE_SNAPSHOT_IGNORE
# (comma-separated globs), e.g. "flux-2-klein-base-4b.safetensors".
_DEFAULT_SNAPSHOT_IGNORE = [
    "*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.webp", "*.md", ".gitattributes",
]


def _snapshot_ignore_patterns() -> list[str]:
    extra = os.environ.get("HF_IMAGE_SNAPSHOT_IGNORE", "")
    return _DEFAULT_SNAPSHOT_IGNORE + [p.strip() for p in extra.split(",") if p.strip()]


@dataclass
class HuggingFaceImageModel(GeneratingModel):
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


class HuggingFaceImageImport(HuggingFaceImport):
    """What `cortexgrid.import_model` needs to take a diffusers pipeline.

    Defaults `ignore_patterns` to the globs the pipeline never loads, which both
    keeps the snapshot to the weights `from_pretrained` reads and keeps the
    hardware estimate off the variants that are skipped."""

    serve_app = HuggingFaceImageDeployment

    def __init__(
        self,
        hf_id: str,
        token: str | None = None,
        ignore_patterns: list[str] | None = None,
    ) -> None:
        super().__init__(
            hf_id,
            token,
            _snapshot_ignore_patterns() if ignore_patterns is None else ignore_patterns,
        )

    def client(self, url: str) -> HuggingFaceImageModel:
        return HuggingFaceImageModel(url=url, model_id=self.hf_id)
