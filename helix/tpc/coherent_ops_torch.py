"""Torch backend for TPC coherent-noise removal (group/wire ops).

Completes the backend matrix: ``coherent_gate_ops`` and ``core.wavelet_ops``
already had all three, but ``coherent_ops`` had only numpy and jax — so
``pipeline(removal='multipass')``, the legacy R1 image-space path, was
unreachable on torch, which is now the DEFAULT backend.

Two places here would silently disagree with numpy if written naively:

* **median tie-break.** ``np.median`` averages the two middle elements for an
  even count; ``torch.median`` returns the LOWER one. That difference already
  cost this codebase once, in band thresholding, where it changed which
  coefficients survived. Both medians here go through the shared
  ``helix.core.backend.torch_q50``, which is the numpy-equivalent.

* **dilation window.** ``scipy.ndimage.maximum_filter1d(size=k)`` centres an
  even-k window asymmetrically — ``k//2`` left, ``k-1-k//2`` right — while
  ``max_pool1d`` only pads symmetrically. The padding is applied explicitly.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from helix.core.backend import torch_q50


def group_median(image, group_size: int):
    """Per-group, per-tick median. Matches ``np.median`` on even group sizes."""
    nw, nt = image.shape
    n_full = nw // group_size
    rem = nw % group_size
    out = []
    if n_full > 0:
        full = image[:n_full * group_size].reshape(n_full, group_size, nt)
        out.append(torch_q50(full, dim=1))
    if rem > 0:
        out.append(torch_q50(image[n_full * group_size:], dim=0).unsqueeze(0))
    return torch.cat(out, dim=0) if len(out) > 1 else out[0]


def broadcast_groups(group_arr, n_wires: int, group_size: int):
    idx = torch.arange(n_wires, device=group_arr.device) // group_size
    return group_arr[idx]


def signal_mask(residual, sigma, nsigma: float):
    return residual.abs() > nsigma * sigma[:, None]


def temporal_dilate(mask, ticks: int):
    """Max-filter along time. Window placement follows scipy's, which is
    asymmetric for even ``ticks`` — hence the explicit pad."""
    if ticks <= 1:
        return mask
    left, right = ticks // 2, ticks - 1 - ticks // 2
    x = mask.to(torch.float32).unsqueeze(0)             # (1, nw, nt)
    x = F.pad(x, (left, right), value=0.0)
    return F.max_pool1d(x, kernel_size=ticks, stride=1).squeeze(0) > 0.5


def masked_group_mean(image, mask, group_size: int):
    """Mean of unflagged wires per (group, tick) -> (estimate, n_unflagged)."""
    nw, nt = image.shape
    n_full = nw // group_size
    rem = nw % group_size
    est, nuf = [], []
    if n_full > 0:
        fi = image[:n_full * group_size].reshape(n_full, group_size, nt)
        unflag = (~mask[:n_full * group_size].reshape(n_full, group_size, nt)).to(fi.dtype)
        n = unflag.sum(dim=1)
        est.append((fi * unflag).sum(dim=1) / n.clamp(min=1.0))
        nuf.append(n)
    if rem > 0:
        li = image[n_full * group_size:]
        unflag = (~mask[n_full * group_size:]).to(li.dtype)
        n = unflag.sum(dim=0)
        est.append(((li * unflag).sum(dim=0) / n.clamp(min=1.0)).unsqueeze(0))
        nuf.append(n.unsqueeze(0))
    cat = lambda xs: torch.cat(xs, dim=0) if len(xs) > 1 else xs[0]
    return cat(est).to(torch.float32), cat(nuf).to(torch.float32)


def mad_sigma_per_wire(residual):
    """Per-wire MAD noise sigma -> (n_wires,) float32."""
    return (torch_q50(residual.abs(), dim=1) / 0.6745).to(torch.float32)


def xblock_kernel(estimate, beta: float):
    """3-tap spatial high-pass across groups: (-beta, 1, -beta)."""
    out = estimate.clone()
    out[0] -= beta * estimate[0]
    out[1:] -= beta * estimate[:-1]
    out[-1] -= beta * estimate[-1]
    out[:-1] -= beta * estimate[1:]
    return out
