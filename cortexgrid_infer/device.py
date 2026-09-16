"""Device detection for PyTorch workloads."""

from __future__ import annotations

import torch


def detect_device() -> torch.device:
    """Return the best available accelerator (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
