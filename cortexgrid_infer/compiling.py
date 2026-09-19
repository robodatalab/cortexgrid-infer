"""`torch.compile` for the serve apps that run a model in-process."""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

log = logging.getLogger(__name__)

# Inductor is weakest off CUDA, and a host outrunning its accelerator is a GPU problem.
_ACCELERATOR = "cuda"

# Parts of a diffusers pipeline worth compiling. Not the VAE: smallest win, shakiest shapes.
_DENOISER_ATTRS = ("transformer", "unet")
_TEXT_ENCODER_ATTRS = ("text_encoder", "text_encoder_2", "text_encoder_3")


class Mode(StrEnum):
    """How far `torch.compile` should go. A serve app picks by the shape of its work."""

    GRAPHED = "reduce-overhead"
    """Fuse, then capture a CUDA graph. For a loop of identically shaped forwards;
    captures per shape, so it wants a settled set of them."""

    FUSED = "default"
    """Fuse only. For work whose shapes move under it, leaving dynamo free to
    compile one dynamic kernel set for all of them."""


def supported(device: Any) -> bool:
    """Whether compiling is worth arming on this device."""
    return getattr(device, "type", None) == _ACCELERATOR


def compile_module(module: Any, device: Any, mode: Mode, label: str = "model") -> bool:
    """Arm compilation of one module's forward, in place. Returns whether it was armed."""
    if module is None or not supported(device):
        return False
    try:
        _fall_back_to_eager_on_failure()
        module.compile(mode=mode)
    except Exception as exc:  # noqa: BLE001 - an optimisation, never fatal
        log.warning("torch.compile(%s, mode=%s) failed: %s", label, mode, exc)
        return False
    log.info("compiling %s with torch.compile(mode=%s)", label, mode)
    return True


def compile_pipeline(pipe: Any, device: Any, mode: Mode) -> list[str]:
    """Arm compilation of a diffusers pipeline's denoiser and text encoders."""
    armed = []
    for attr in (*_DENOISER_ATTRS, *_TEXT_ENCODER_ATTRS):
        module = getattr(pipe, attr, None)
        if module is not None and compile_module(module, device, mode, label=attr):
            armed.append(attr)
    return armed


def _fall_back_to_eager_on_failure() -> None:
    """Make dynamo warn rather than raise out of a request. Process-wide."""
    try:
        import torch._dynamo.config as dynamo_config

        dynamo_config.suppress_errors = True
    except Exception:  # noqa: BLE001 - best effort on a private flag
        pass
