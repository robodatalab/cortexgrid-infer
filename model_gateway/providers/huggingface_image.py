"""HuggingFace image provider: deploys a diffusers pipeline via cortexflow and
clients it over HTTP.

The image analogue of :mod:`model_gateway.providers.huggingface`. Model ids use
the ``hf-image:`` prefix (e.g. ``hf-image:black-forest-labs/FLUX.2-klein-base-4B``)
so they don't collide with the ``hf:`` completion provider.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import tempfile
from typing import Any

import cortexflow
import httpx
from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError

from model_gateway.core import GeneratedImage, GeneratingModel, register_provider
from model_gateway.providers.huggingface_complete import _parse_hf_id
from model_gateway.providers.huggingface_image_serve import HuggingFaceImageDeployment


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


def deploy_huggingface_image(model_id: str) -> HuggingFaceImageModel | None:
    if not model_id.startswith("hf-image:"):
        return None
    hf_id = model_id[len("hf-image:") :]
    family, suffix = _parse_hf_id(hf_id)
    run_name = cortexflow.Experiment.get_instance().run_name()

    already_saved = any(
        m.family == family and m.suffix == suffix and m.run_name == run_name
        for m in cortexflow.list_models()
    )
    if not already_saved:
        try:
            with tempfile.TemporaryDirectory() as d:
                snapshot_download(repo_id=hf_id, local_dir=d)
                cortexflow.save_model(
                    d, HuggingFaceImageDeployment, family=family, suffix=suffix
                )
        except (RepositoryNotFoundError, OSError):
            return None

    existing = next(
        (
            d
            for d in cortexflow.list_deployed_models()
            if d.family == family and d.suffix == suffix and d.run_name == run_name
        ),
        None,
    )
    if existing is not None:
        url = existing.url
    else:
        deployment = cortexflow.deploy_model(
            family=family, suffix=suffix, run_name=run_name, wait=True, timeout=None
        )
        url = deployment.url

    return HuggingFaceImageModel(url=url, model_id=model_id)


register_provider("hf-image:", deploy_huggingface_image)
