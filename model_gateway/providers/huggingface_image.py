"""HuggingFace image provider: deploys a diffusers pipeline via cortexgrid and
clients it over HTTP.

The image analogue of :mod:`model_gateway.providers.huggingface`. Model ids use
the ``hf-image:`` prefix (e.g. ``hf-image:black-forest-labs/FLUX.2-klein-base-4B``)
so they don't collide with the ``hf:`` completion provider.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import os
import tempfile
from typing import Any

import cortexgrid
import httpx
from huggingface_hub import snapshot_download

from model_gateway.core import (
    GeneratedImage,
    GeneratingModel,
    register_deleter,
    register_provider,
    register_status_provider,
    register_uploader,
)
from model_gateway.providers.huggingface_complete import _parse_hf_id
from model_gateway.providers.huggingface_image_serve import HuggingFaceImageDeployment


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


def _ingest_huggingface_image(
    hf_id: str, family: str, suffix: str, token: str | None
) -> None:
    """Download the HF pipeline weights and register them in the cortexgrid registry.

    Submitted to the cluster via ``cortexgrid.remote`` (see `upload_huggingface_image`),
    so the weights travel HuggingFace -> cluster node -> registry and never transit
    the client - which matters most here, where image pipelines run to tens of GB.
    ``save_model`` scopes the version to the ambient Experiment, which the remote job
    inherits from the submitter, so it lands under the same run the deploy later reads."""
    with tempfile.TemporaryDirectory() as d:
        snapshot_download(
            repo_id=hf_id,
            local_dir=d,
            ignore_patterns=_snapshot_ignore_patterns(),
            token=token,
        )
        cortexgrid.save_model(
            d, HuggingFaceImageDeployment, family=family, suffix=suffix
        )


def _family_versions(family: str, suffix: str) -> list[Any]:
    """Every registry version of this model, newest first, across all runs.

    cortexgrid scopes a version to the run that uploaded it, but a HuggingFace
    base model is immutable, so any run's copy is interchangeable. Resolving by
    family/suffix (not by the active experiment run) is what lets the backend
    start a fresh run every boot without re-ingesting multi-GB weights each time."""
    versions = [
        m
        for m in cortexgrid.list_models()
        if m.family == family and m.suffix == suffix
    ]
    return sorted(versions, key=lambda m: m.created_at, reverse=True)


def _ready_run_name(family: str, suffix: str) -> str | None:
    """run_name of the newest ``ready`` version of this model, or None if none is."""
    return next(
        (m.run_name for m in _family_versions(family, suffix) if m.phase == "ready"),
        None,
    )


def upload_huggingface_image(model_id: str) -> str | None:
    """Start ingesting *model_id*'s weights into the registry, on the cluster.

    Returns the id of the background ``cortexgrid.remote`` job, or None if the
    model is already registered under *any* run (phase `uploading`/`ready`) so
    there is nothing to submit. Poll progress via ``deployment_status(model_id)``;
    deploy once `ready`."""
    if not model_id.startswith("hf-image:"):
        return None
    hf_id = model_id[len("hf-image:") :]
    family, suffix = _parse_hf_id(hf_id)

    if any(m.phase in ("uploading", "ready") for m in _family_versions(family, suffix)):
        return None

    # First upload: the ingest job registers the version under the active run.
    return cortexgrid.remote(
        _ingest_huggingface_image,
        hf_id,
        family,
        suffix,
        os.environ.get("HF_TOKEN"),
        num_gpus=0,
        num_cpus=2,
    )


def deploy_huggingface_image(model_id: str) -> HuggingFaceImageModel | None:
    if not model_id.startswith("hf-image:"):
        return None
    hf_id = model_id[len("hf-image:") :]
    family, suffix = _parse_hf_id(hf_id)
    run_name = _ready_run_name(family, suffix)
    if run_name is None:
        raise RuntimeError(
            f"Model '{model_id}' has no registry-ready version; call "
            f"upload_model('{model_id}') and wait for phase 'ready' before deploying."
        )

    existing = next(
        (
            d
            for d in cortexgrid.list_deployed_models()
            if d.family == family and d.suffix == suffix and d.run_name == run_name
        ),
        None,
    )
    if existing is not None:
        url = existing.url
    else:
        deployment = cortexgrid.deploy_model(
            family=family, suffix=suffix, run_name=run_name, wait=True, timeout=None
        )
        url = deployment.url

    return HuggingFaceImageModel(
        url=url,
        model_id=model_id,
        family=family,
        suffix=suffix,
        run_name=run_name,
    )


def image_deployment_status(model_id: str) -> Any:
    """Live phase of the deployment for *model_id*, delegated to cortexgrid.

    Read-only - safe to poll from a status endpoint while a deploy is in flight.
    Resolves the same (family, suffix, run_name) identity `deploy` uses. Reports
    the serving lifecycle (`cortexgrid.model_serving_status`) once a Serve app
    exists; before that - while the weights are still uploading to the registry -
    it falls back to the registry lifecycle (`cortexgrid.model_registry_status`),
    so a poll stays meaningful during weight staging / scheduling too."""
    if not model_id.startswith("hf-image:"):
        return None
    family, suffix = _parse_hf_id(model_id[len("hf-image:") :])
    run_name = _ready_run_name(family, suffix)
    if run_name is not None:
        serving = cortexgrid.model_serving_status(family, suffix, run_name)
        if serving.phase != "not_deployed":
            return serving
        return cortexgrid.model_registry_status(family, suffix, run_name)
    # Nothing ready yet: report the newest in-flight/failed version, if any.
    versions = _family_versions(family, suffix)
    return versions[0] if versions else None


def delete_huggingface_image(model_id: str) -> None:
    """Undeploy (if running) and delete this model's weights + serve bundle.

    Removes every registered version of the model across runs (deploy/upload
    resolve by family/suffix, not by the active run, so a single run_name no
    longer identifies the weights). Idempotent: safe whether or not the model is
    deployed or registered, so it is the inverse of `upload_huggingface_image` +
    `deploy_huggingface_image`."""
    if not model_id.startswith("hf-image:"):
        return
    family, suffix = _parse_hf_id(model_id[len("hf-image:") :])
    for run_name in {m.run_name for m in _family_versions(family, suffix)}:
        cortexgrid.undeploy_model(family, suffix, run_name)
        cortexgrid.delete_model(family, suffix, run_name)


register_provider("hf-image:", deploy_huggingface_image)
register_status_provider("hf-image:", image_deployment_status)
register_uploader("hf-image:", upload_huggingface_image)
register_deleter("hf-image:", delete_huggingface_image)
