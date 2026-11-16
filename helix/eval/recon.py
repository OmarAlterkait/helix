"""The autocast context every evaluation forward runs under (helix.eval.events, helix.eval.denoise).

The masked-reconstruction metrics that lived here (grid-free var_expl / charge closure, the
pooled loss terms, the seeded evaluation mask) served the in-training evaluator retired on
2026-11-09 and had no caller left; they were removed with it.
"""

from __future__ import annotations

import contextlib

import torch


def autocast(device, dtype=torch.bfloat16, enabled=True):
    """bf16 autocast on CUDA, nothing elsewhere."""
    return torch.autocast(device_type=device.type, dtype=dtype) if enabled and device.type == "cuda" \
        else contextlib.nullcontext()
