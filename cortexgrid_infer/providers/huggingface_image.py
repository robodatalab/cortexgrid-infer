"""HuggingFace image provider: deploys a diffusers pipeline via cortexgrid and
clients it over HTTP.

The image analogue of :mod:`cortexgrid_infer.providers.huggingface`. Model ids use
the ``hf-image:`` prefix (e.g. ``hf-image:black-forest-labs/FLUX.2-klein-base-4B``)
so they don't collide with the ``hf:`` completion provider.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import partial
import os
import tempfile
from typing import Any

import cortexgrid
import httpx

from cortexgrid_infer.core import (
    GeneratedImage,
    GeneratingModel,
    register_deleter,
    register_provider,
    register_status_provider,
    register_uploader,
)
from cortexgrid_infer.providers.huggingface_image_serve import HuggingFaceImageDeployment
from cortexgrid_infer.providers.serving import ensure_serving
from cortexgrid_infer.utils import download_hf_snapshot, parse_hf_id


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
    # cortexgrid deployment identity, carried so the owner can tear it down
    # without re-deriving it from the active experiment at shutdown.
    family: str = ""
    suffix: str = ""
    run_name: str = ""

    @property
    def name(self) -> str:
        return self.model_id

    def undeploy(self) -> None:
        """Tear down the Ray Serve app backing this model (frees its GPU).

        The weights + bundle stay in the cortexgrid registry, so a later
        deploy re-schedules the app without re-uploading."""
        if self.family and self.suffix and self.run_name:
            cortexgrid.undeploy_model(self.family, self.suffix, self.run_name)

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


def _import_huggingface_image(
    hf_id: str, family: str, suffix: str, token: str | None
) -> None:
    """Import the HF pipeline weights into the cortexgrid registry, or reuse the
    imported copy.

    ``import_model`` downloads only when the weights still have to be uploaded. On
    a model that is already ``ready`` it just re-bundles the serve code if it
    changed, and either way tags the ambient Experiment's run with the model."""
    with tempfile.TemporaryDirectory() as d:
        cortexgrid.import_model(
            partial(
                download_hf_snapshot,
                hf_id,
                d,
                token,
                ignore_patterns=_snapshot_ignore_patterns(),
            ),
            HuggingFaceImageDeployment,
            family=family,
            suffix=suffix,
        )


def upload_huggingface_image(model_id: str) -> str | None:
    """Import *model_id*'s weights into the registry. Call it on every run.

    When the model is already ``ready``, imports it in-process, which uploads no
    weights: it re-bundles changed serve code before the next deploy and tags this
    run with the model, then returns None. When the model is absent or its upload
    failed, submits the import as a ``cortexgrid.remote`` job, so the weights travel
    HuggingFace -> cluster node -> registry and never transit the client - which
    matters most here, where image pipelines run to tens of GB - and returns the job
    id. Returns None while another process is uploading it. Poll progress via
    ``deployment_status(model_id)``; deploy once `ready`."""
    if not model_id.startswith("hf-image:"):
        return None
    hf_id = model_id[len("hf-image:") :]
    family, suffix = parse_hf_id(hf_id)
    token = os.environ.get("HF_TOKEN")

    status = cortexgrid.model_registry_status(family, suffix, cortexgrid.IMPORTED)
    if status is not None and status.phase == "uploading":
        return None
    if status is not None and status.phase == "ready":
        _import_huggingface_image(hf_id, family, suffix, token)
        return None

    return cortexgrid.remote(
        _import_huggingface_image,
        hf_id,
        family,
        suffix,
        token,
        num_gpus=0,
        num_cpus=2,
    )


def deploy_huggingface_image(
    model_id: str, timeout: float | None = None
) -> HuggingFaceImageModel | None:
    if not model_id.startswith("hf-image:"):
        return None
    hf_id = model_id[len("hf-image:") :]
    family, suffix = parse_hf_id(hf_id)
    run_name = cortexgrid.IMPORTED

    status = cortexgrid.model_registry_status(family, suffix, run_name)
    if status is None or status.phase != "ready":
        phase = None if status is None else status.phase
        raise RuntimeError(
            f"Model '{model_id}' is not registry-ready (phase={phase}); call "
            f"upload_model('{model_id}') and wait for phase 'ready' before deploying."
        )

    return HuggingFaceImageModel(
        url=ensure_serving(family, suffix, run_name, timeout),
        model_id=model_id,
        family=family,
        suffix=suffix,
        run_name=run_name,
    )


def image_deployment_status(model_id: str) -> Any:
    """Live phase of the deployment for *model_id*, delegated to cortexgrid.

    Read-only - safe to poll from a status endpoint while a deploy is in flight.
    Resolves the same (family, suffix, IMPORTED) identity `deploy` uses. Reports
    the serving lifecycle (`cortexgrid.model_serving_status`) once a Serve app
    exists; before that - while the weights are still uploading to the registry -
    it falls back to the registry lifecycle (`cortexgrid.model_registry_status`),
    so a poll stays meaningful during weight staging / scheduling too."""
    if not model_id.startswith("hf-image:"):
        return None
    family, suffix = parse_hf_id(model_id[len("hf-image:") :])
    serving = cortexgrid.model_serving_status(family, suffix, cortexgrid.IMPORTED)
    if serving.phase != "not_deployed":
        return serving
    return cortexgrid.model_registry_status(family, suffix, cortexgrid.IMPORTED)


def delete_huggingface_image(model_id: str) -> None:
    """Undeploy (if running) and delete this model's weights + serve bundle.

    The imported model is shared by every run, so this removes it for all of them.
    Idempotent: safe whether or not the model is deployed or registered, so it is
    the inverse of `upload_huggingface_image` + `deploy_huggingface_image`."""
    if not model_id.startswith("hf-image:"):
        return
    family, suffix = parse_hf_id(model_id[len("hf-image:") :])
    cortexgrid.undeploy_model(family, suffix, cortexgrid.IMPORTED)
    cortexgrid.delete_model(family, suffix, cortexgrid.IMPORTED)


register_provider("hf-image:", deploy_huggingface_image)
register_status_provider("hf-image:", image_deployment_status)
register_uploader("hf-image:", upload_huggingface_image)
register_deleter("hf-image:", delete_huggingface_image)
