"""Estimate the hardware a HuggingFace model needs, from repo metadata alone.

`cortexgrid.ModelRequirements` decides which node Ray will place a replica on,
and nothing in the registry can derive it: cortexgrid stores whatever weights it
is handed, and cortexgrid-infer imports whatever repo id it is handed. So the
figures have to come from somewhere, and the cheapest honest source is the
repo's own metadata - safetensors headers and file sizes, both readable over
HTTP without downloading a single weight.

These are estimates. Pass a `cortexgrid.ModelRequirements` of your own to
`import_model` where you know better, or edit the figures on the model card in
the dashboard afterwards - `import_model` only fills in requirements a version
does not already have, so a hand-set value is never overwritten.
"""

from __future__ import annotations

import fnmatch

import cortexgrid
from huggingface_hub import HfApi, get_safetensors_metadata
from huggingface_hub.errors import NotASafetensorsRepoError

_GIB = float(1 << 30)

# Both serve apps load in a 16-bit dtype (float16 for causal LMs, bfloat16 for
# diffusers pipelines), so parameter count converts to resident bytes at this
# rate whatever the checkpoint was stored as.
_BYTES_PER_PARAM = 2

# Weights are only part of what a replica holds. The multiplier covers CUDA
# context, activations and fragmentation; the floor covers the KV cache of a
# long context, which does not scale with model size the way activations do.
_VRAM_HEADROOM = 1.25
_VRAM_FLOOR_GB = 2.0

# `from_pretrained` materialises the state dict in host memory before `.to()`
# moves it to the device, so host RAM has to hold the weights transiently too.
_HOST_HEADROOM = 1.1
_HOST_FLOOR_GB = 2.0

# Files that are weights rather than configs, tokenizers or docs. Used for the
# fallback estimate, where a repo has no top-level safetensors index.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".msgpack")


def weight_gb(
    hf_id: str,
    token: str | None = None,
    ignore_patterns: list[str] | None = None,
) -> float:
    """Resident size in GiB of the weights `hf_id` would load, without downloading.

    Prefers the repo's safetensors headers, which give an exact parameter count
    and so an exact size at the dtype the serve app loads in. Repos those do not
    cover - a diffusers pipeline keeps its weights in component subfolders with
    no top-level index - fall back to summing the size of the weight files on
    the Hub.

    `ignore_patterns` must be the same globs the download uses. A pipeline repo
    commonly ships several dtype variants of the same component plus a
    consolidated single-file checkpoint; counting the files the download skips
    overestimates by multiples.
    """
    try:
        metadata = get_safetensors_metadata(hf_id, token=token)
    except NotASafetensorsRepoError:
        pass
    else:
        params = sum(metadata.parameter_count.values())
        return params * _BYTES_PER_PARAM / _GIB

    info = HfApi().model_info(hf_id, files_metadata=True, token=token)
    total = 0
    for sibling in info.siblings or ():
        name = sibling.rfilename
        if not name.endswith(_WEIGHT_SUFFIXES):
            continue
        if _ignored(name, ignore_patterns):
            continue
        total += sibling.size or 0
    return total / _GIB


def _ignored(name: str, ignore_patterns: list[str] | None) -> bool:
    """Whether `snapshot_download` would skip this file for these globs.

    Matches the glob against the full repo-relative path and against the bare
    file name, because the patterns in use are a mix of both ("*.md" is meant to
    catch a nested README, "flux-2-klein-base-4b.safetensors" names one file)."""
    if not ignore_patterns:
        return False
    base = name.rsplit("/", 1)[-1]
    return any(
        fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(base, pattern)
        for pattern in ignore_patterns
    )


def estimate(
    hf_id: str,
    token: str | None = None,
    ignore_patterns: list[str] | None = None,
) -> cortexgrid.ModelRequirements:
    """Estimate what one replica of `hf_id` needs, from repo metadata alone.

    Always asks for exactly one GPU: both serve apps move the whole model onto a
    single device, so a replica is never spread across cards. A model too large
    for any one card in the cluster therefore needs a serve app that shards it,
    not a larger `num_gpus` here - the requirement would simply never be placed.
    """
    weights = weight_gb(hf_id, token, ignore_patterns)
    return cortexgrid.ModelRequirements(
        num_gpus=1,
        ram_gb=round(weights * _HOST_HEADROOM + _HOST_FLOOR_GB, 1),
        vram_gb=round(weights * _VRAM_HEADROOM + _VRAM_FLOOR_GB, 1),
    )
