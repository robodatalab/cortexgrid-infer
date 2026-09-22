"""What the rest of the library needs to know about a serve app, carried on the
serve app itself.

A serve app is named for the task it runs - `Text2Text`, `Text2Image`,
`Image2Mesh` - not for where its model came from, and runs any model of that
task whatever the source. Two kinds:

- `LocalModel` runs weights an importer staged in the registry, so it tells the
  importer which files it never loads and how much hardware a replica needs for
  weights of a given size.
- `HostedModel` forwards to a model hosted elsewhere, so there is nothing to
  import; it tells `registry.Hosted` what config it reads instead.

Both name the client that speaks their routes. cortexgrid's `serve.ingress`
leaves the class exactly as written, so all of this rides along as class
attributes and classmethods, with no separate description of the model to keep
in step with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import cortexgrid

from cortexgrid_infer.core import DeployedModel

_GIB = float(1 << 30)

ENABLE_THINKING_PARAM = "enable_thinking"

# Whether a replica runs its model through `torch.compile` (see
# `cortexgrid_infer.compiling`). Off unless the deployment's config says
# "true": compiled serving pays a warm-up per input shape, which is worth it
# only where traffic repeats those shapes.
COMPILE_PARAM = "compile"

# Weights are only part of what a replica holds. The multiplier covers CUDA
# context, activations and fragmentation; the floor covers the KV cache of a
# long context, which does not scale with model size the way activations do.
_VRAM_HEADROOM = 1.25
_VRAM_FLOOR_GB = 2.0

# `from_pretrained` materialises the state dict in host memory before `.to()`
# moves it to the device, so host RAM has to hold the weights transiently too.
_HOST_HEADROOM = 1.1
_HOST_FLOOR_GB = 2.0


@dataclass(frozen=True)
class Weights:
    """What an importer can tell about a model's weights without downloading them.

    `params` is the exact parameter count, where the source records one (a
    safetensors header does); `file_bytes` the size of the weight files the
    download fetches, for a source that records nothing better."""

    params: int | None = None
    file_bytes: int = 0


class LocalModel:
    """Base of the serve apps that run a model's weights on the cluster."""

    # Bytes one parameter takes once loaded. Every app here loads in a 16-bit
    # dtype, whatever the checkpoint was stored as.
    bytes_per_param: ClassVar[int] = 2

    # What one replica needs at least, whatever the weights come to - for a
    # model whose work takes memory its weights do not show.
    min_vram_gb: ClassVar[float] = 0.0

    @classmethod
    def ignore_patterns(cls) -> list[str] | None:
        """Globs of the files in a model's repo this app never loads.

        The importer skips them in both the download and the size estimate, so
        the figures describe what actually gets staged. None skips nothing."""
        return None

    @classmethod
    def weight_gb(cls, weights: Weights) -> float:
        """Resident size in GiB of `weights` once this app has loaded them."""
        if weights.params is not None:
            return weights.params * cls.bytes_per_param / _GIB
        return weights.file_bytes / _GIB

    @classmethod
    def requirements(cls, weights: Weights) -> cortexgrid.ModelRequirements:
        """What one replica needs to hold `weights` and run them.

        Always exactly one GPU: every app here moves the whole model onto a
        single device, so a replica is never spread across cards. A model too
        large for any one card in the cluster therefore needs an app that
        shards it, not a larger `num_gpus` - the requirement would simply never
        be placed."""
        size = cls.weight_gb(weights)
        return cortexgrid.ModelRequirements(
            num_gpus=1,
            ram_gb=round(size * _HOST_HEADROOM + _HOST_FLOOR_GB, 1),
            vram_gb=max(round(size * _VRAM_HEADROOM + _VRAM_FLOOR_GB, 1), cls.min_vram_gb),
        )

    @classmethod
    def config(cls) -> dict[str, str]:
        return {COMPILE_PARAM: "false"}

    @classmethod
    def client(cls, url: str, name: str) -> DeployedModel:
        """A client for this app deployed at `url`, serving the model `name`."""
        raise NotImplementedError


class HostedModel:
    """Base of the serve apps that forward to a model hosted elsewhere."""

    @classmethod
    def config(cls, model_id: str, **settings: str) -> dict[str, str]:
        """The registry entry's config for the hosted model `model_id`: what the
        app reads with `cortexgrid.model_config` to reach it."""
        raise NotImplementedError

    @classmethod
    def requirements(cls) -> cortexgrid.ModelRequirements:
        """None: a replica holds no weights and does no compute of its own, so
        it is placed on any node, CPU-only included."""
        return cortexgrid.ModelRequirements()

    @classmethod
    def client(cls, url: str, name: str) -> DeployedModel:
        """A client for this app deployed at `url`, serving the model `name`."""
        raise NotImplementedError
