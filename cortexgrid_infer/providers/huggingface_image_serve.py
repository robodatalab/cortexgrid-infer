"""Ray Serve deployment that loads a HuggingFace diffusion pipeline from the
cortexgrid registry and exposes a single POST /generate endpoint.

The pipeline class is not hardcoded: it is read from the model's
``model_index.json`` (``_class_name``) and resolved against ``diffusers``, so
one deployment serves any diffusers text-to-image pipeline (FLUX.2, SD, ...).
Pipelines that also accept a reference ``image`` (e.g. FLUX.2) get img2img for
free — the reference is forwarded when the request carries one.
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Any


import cortexgrid
from cortexgrid import serve
import diffusers
from fastapi import FastAPI
from PIL import Image, ImageOps
import torch

from cortexgrid_infer.device import detect_device


_app = FastAPI()

# FLUX.2 base (non-distilled) defaults from the model card; applied when the
# caller omits them. Reasonable for other diffusers pipelines too.
_DEFAULT_STEPS = 50
_DEFAULT_GUIDANCE = 4.0


def _round_to_multiple(x: int, base: int = 16) -> int:
    return max(base, (x // base) * base)


def _decode_image(image_b64: str, max_side: int) -> Any:
    raw = base64.b64decode(image_b64)
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        w, h = int(w * scale), int(h * scale)
    return img.resize(
        (_round_to_multiple(w), _round_to_multiple(h)), Image.Resampling.LANCZOS
    )


def _pipeline_class(model_dir: Path) -> Any:
    """Resolve the concrete diffusers pipeline class named in model_index.json."""
    class_name = json.loads((model_dir / "model_index.json").read_text())["_class_name"]
    return getattr(diffusers, class_name)


@serve.ingress(_app)
class HuggingFaceImageDeployment:
    num_gpus = 1
    num_replicas = 1

    def __init__(self, family: str, suffix: str, run_name: str) -> None:
        path = cortexgrid.load_model(family, suffix, run_name)
        self._device = detect_device()
        pipe = _pipeline_class(path).from_pretrained(path, torch_dtype=torch.bfloat16)
        pipe = pipe.to(self._device)
        pipe.set_progress_bar_config(disable=True)
        vae = getattr(pipe, "vae", None)
        if vae is not None and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        self._pipe = pipe

    @_app.post("/generate")
    async def generate(self, body: dict[str, Any]) -> dict[str, Any]:
        prompt = body["prompt"]
        size = int(body.get("size") or 1024)
        steps = int(body.get("steps") or _DEFAULT_STEPS)
        guidance = float(
            body["guidance"] if body.get("guidance") is not None else _DEFAULT_GUIDANCE
        )
        seed = body.get("seed")

        ref = _decode_image(body["image"], size) if body.get("image") else None
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))

        # Output follows the reference's (fitted) aspect ratio; text2img is a
        # square of the requested size.
        if ref is not None:
            height, width = ref.height, ref.width
        else:
            height = width = _round_to_multiple(size)

        call: dict[str, Any] = dict(
            prompt=prompt,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
        )
        if ref is not None:
            call["image"] = ref

        t0 = time.time()
        result = self._pipe(**call)
        duration = time.time() - t0

        image = result.images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return {
            "image": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "width": image.width,
            "height": image.height,
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
            "duration_s": round(duration, 2),
        }
