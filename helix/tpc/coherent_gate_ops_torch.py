"""Torch backend for coefficient-space smart gating (the qualified R2).

Backend module of the ``helix.tpc.coherent_gate_ops`` family — dispatched via
``backend.ops()``; call it through :func:`helix.tpc.coherent_gate.coherent_gate`.

A line-for-line port of :mod:`helix.tpc.coherent_gate_ops_numpy` — same
mechanism, same qualified defaults (kgate=3.0, ksig=3.0, npass=2, group_size=64,
A-parity ``sigc``), same fail-open on non-finite bands. See that module's
docstring for the algorithm and ``research/r2_qualification/REPORT.md`` for the
evidence behind the defaults.

Two things this port must get exactly right, because both silently change gate
decisions rather than failing:

1. **quantile(0.5), not median.** ``torch.median`` returns the LOWER of the two
   middle values; ``np.quantile(..., 0.5)`` AVERAGES them. They differ by ~1e-4
   relative on even-length inputs, which flips a handful of gate decisions per
   event. ``_q50`` below reproduces the averaging convention. (This is the exact
   distinction ``sigc_mode='quantile'`` names, and why the legacy ``'median'``
   mode is kept selectable but is not the default.)
2. **Signal exclusion.** The ``~smf`` term in the common-mode estimate excludes
   wires already flagged as signal. Dropping it is nearly invisible on aggregate
   metrics but degrades the 2-pass result it exists to produce.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import torch

from helix.core.backend import torch_q50 as _q50

_EPS = 1e-6


def _mad(x: torch.Tensor, dim: int) -> torch.Tensor:
    """MAD-derived sigma = quantile(|x|, 0.5) / 0.6745 (A-parity)."""
    return torch.clamp_min(_q50(x.abs(), dim) / 0.6745, _EPS)


def _block_common_mode(b: torch.Tensor, ksig: float, sigmask: torch.Tensor,
                       group_size: int) -> torch.Tensor:
    """Per-(block, position) robust common mode M, shape (n_blocks, Lb)."""
    W, Lb = b.shape
    ngf = W // group_size
    Ms = []
    if ngf > 0:
        bf = b[:ngf * group_size].reshape(ngf, group_size, Lb)
        smf = sigmask[:ngf * group_size].reshape(ngf, group_size, Lb)
        med = _q50(bf, dim=1)                                   # (ngf, Lb)
        resid = bf - med[:, None]
        sg = _mad(resid.reshape(ngf, -1), dim=1)                # (ngf,)
        uf = (resid.abs() <= ksig * sg[:, None, None]) & (~smf)
        nuf = uf.sum(1)
        mean = (bf * uf).sum(1) / torch.clamp_min(nuf, 1)
        Ms.append(torch.where(nuf > 0, mean, med))
    rem = W - ngf * group_size
    if rem > 0:
        blk = b[ngf * group_size:]
        smr = sigmask[ngf * group_size:]
        med = _q50(blk, dim=0)
        resid = blk - med
        sg = _mad(resid.reshape(-1), dim=0)
        uf = (resid.abs() <= ksig * sg) & (~smr)
        nuf = uf.sum(0)
        mean = (blk * uf).sum(0) / torch.clamp_min(nuf, 1)
        Ms.append(torch.where(nuf > 0, mean, med)[None, :])
    return torch.cat(Ms, dim=0)                                  # (n_blocks, Lb)


def _detect_signal(cleaned_band: torch.Tensor, ksig: float, group_size: int) -> torch.Tensor:
    """Boolean (W, Lb) mask of coeffs that stick out (> ksig·MAD) within their block."""
    W, Lb = cleaned_band.shape
    ngf = W // group_size
    sm = torch.zeros_like(cleaned_band, dtype=torch.bool)
    cf = cleaned_band.abs()
    if ngf > 0:
        cb = cf[:ngf * group_size].reshape(ngf, group_size, Lb)
        csg = _mad(cb.reshape(ngf, -1), dim=1)                   # (ngf,)
        sm[:ngf * group_size] = (cb > ksig * csg[:, None, None]).reshape(ngf * group_size, Lb)
    rem = W - ngf * group_size
    if rem > 0:
        cr = cf[ngf * group_size:]
        csg = _mad(cr.reshape(-1), dim=0)
        sm[ngf * group_size:] = cr > ksig * csg
    return sm


def _sigc(M: torch.Tensor, mode: str) -> torch.Tensor:
    """Coherent scale — q50 for BOTH modes, exactly as the numpy backend does.

    ``'median'`` is an alias, not a second algorithm: numpy implements it with
    ``np.median``, which is bit-identical to ``np.quantile(..., 0.5)``. The
    distinction its docstring draws is against the OLD torch.median (lower of two
    middles), which no backend reproduces any more — jax rejects the mode
    outright, and nothing in production sets it (``sigc_mode`` is not even a
    DetectorConfig field). Implementing it here with ``torch.median`` would make
    the same mode name mean different arithmetic on different backends, in a
    branch nothing exercises — a silent divergence with no upside.
    """
    return torch.clamp_min(_q50(M.abs().reshape(-1), dim=0) / 0.6745, _EPS)


def gate_band(b: torch.Tensor, *, group_size: int, kgate, ksig: float,
              npass: int, sigc_mode: str = "quantile") -> torch.Tensor:
    """Coherent-gate one band ``(W, Lb)`` → cleaned band."""
    W = b.shape[0]
    kg = list(kgate) if isinstance(kgate, (list, tuple, np.ndarray)) else [kgate] * npass
    if len(kg) > npass:
        # Silently dropping trailing entries ships a DIFFERENT corpus than the
        # provenance records: pipeline.py stamps removal_json with the kgate the
        # caller passed, so --kgate 2.5,3.5 at npass=1 produced a k=2.5 corpus
        # labelled [2.5, 3.5]. A per-pass sequence longer than the pass count is
        # a config error, not something to truncate.
        raise ValueError(
            f"kgate has {len(kg)} per-pass entries but npass={npass}; "
            f"the extra entries would be silently ignored and the recorded "
            f"provenance would not describe the coefficients produced")
    if len(kg) < npass:
        kg = kg + [kg[-1]] * (npass - len(kg))
    n_blocks = (W + group_size - 1) // group_size
    idx = torch.clamp(torch.arange(W, device=b.device) // group_size, max=n_blocks - 1)
    sm = torch.zeros_like(b, dtype=torch.bool)
    cleaned = b
    for p in range(npass):
        M = _block_common_mode(b, ksig, sm, group_size)
        sc = _sigc(M, sigc_mode)
        Mc = torch.where(M.abs() < kg[p] * sc, M, torch.zeros((), dtype=M.dtype, device=M.device))
        cleaned = b - Mc[idx]
        if p + 1 < npass:
            sm = _detect_signal(cleaned, ksig, group_size)
    return cleaned


def gate_bands(
    bands: Sequence[torch.Tensor],
    *,
    group_size: int = 64,
    kgate=3.0,
    ksig: float = 3.0,
    npass: int = 2,
    gate_approx: bool = True,
    sigc_mode: str = "quantile",
) -> list[torch.Tensor]:
    """Coherent-remove a plane's DWT bands ``[cA, cD_L, …, cD_1]`` (list in, list out).

    ``gate_approx=False`` leaves the approx band untouched. ``kgate`` may be a
    scalar or a per-pass sequence. A non-finite band fails OPEN — returned
    unchanged, whole-band, matching the numpy backend exactly — so a bad event
    never propagates corrupted coefficients.
    """
    out = []
    for i, b in enumerate(bands):
        if not isinstance(b, torch.Tensor):
            b = torch.as_tensor(np.asarray(b, np.float32))
        if b.dtype not in (torch.float32, torch.float64):
            b = b.float()
        if not bool(torch.isfinite(b).all()):
            warnings.warn(f"coherent_gate: non-finite band {i}; failing open (no removal)")
            out.append(b)
            continue
        if i == 0 and not gate_approx:
            out.append(b)
            continue
        out.append(gate_band(b, group_size=group_size, kgate=kgate, ksig=ksig,
                             npass=npass, sigc_mode=sigc_mode))
    return out
