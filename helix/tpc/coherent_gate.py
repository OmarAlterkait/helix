"""Coefficient-space coherent-noise gate on contiguous wire blocks (the qualified R2): the TPC
chain's ``block_gate`` (:class:`helix.tpc.chain.TPCChain`, simulation). Torch, any device.

Given the per-band DWT coefficients of a plane, it removes the common-mode coherent noise
that is rank-1 within `group_size`-wire blocks, *without* an image round-trip (the
DWT is linear, so the cleaned image's bands == band − gated block common-mode).

Mechanism, per band, per `npass`:
  1. robust per-(block, coeff-position) common mode M — a ksig-masked mean over the
     wires of each block, excluding wires flagged as signal (so signal doesn't bias
     the estimate);
  2. coherent scale `sigc` = MAD of M over blocks;
  3. gate: keep `|M| < kgate·sigc` (that is the coherent part → subtract), drop the
     rest (large ⇒ real signal → protect);
  4. subtract the gated common-mode from the band.
Between passes, signal is (re)detected on the cleaned band and excluded from the
next common-mode estimate — a purer estimate that removes more noise at no signal
cost (the qualified 2-pass result).

Qualified default: kgate=3.0, ksig=3.0, npass=2, group_size=64, A-parity `sigc`
(`quantile(0.5)`). See research/r2_qualification/REPORT.md. Ported from
`measure_coeffs.smart_gate_bands` (1-pass) + `r2_qualification/grid.py` (2-pass).

A corpus is architecture-specific: float32 reductions depend on summation order, the
threshold comparisons on those floats are discrete, and one block decision is broadcast to
all its wires, so one GPU generation builds a whole corpus.

Two things must be exactly right, because both silently change gate decisions rather than
failing:

1. **np.median, not torch.median.** ``torch.median`` returns the LOWER of the two
   middle values; :func:`helix.core.robust.q50` AVERAGES them, as ``np.quantile(..., 0.5)``
   does. They differ by ~1e-4 relative on even-length inputs, which flips a handful of
   gate decisions per event.
2. **Signal exclusion.** The ``~smf`` term in the common-mode estimate excludes
   wires already flagged as signal. Dropping it is nearly invisible on aggregate
   metrics but degrades the 2-pass result it exists to produce.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import torch

from helix.core.robust import mad, q50 as _q50

_EPS = 1e-6


def _block_common_mode(b: torch.Tensor, ksig: float, sigmask: torch.Tensor,
                       group_size: int, return_occ: bool = False):
    """Per-(block, position) robust common mode M, shape (n_blocks, Lb).

    ``return_occ`` also returns ``1 - nuf/block_size``, the fraction of the
    block's wires the k-sigma test rejected — already computed as the mean's
    denominator. Uses each block's REAL width so the partial trailing block is
    judged on the same scale.
    """
    W, Lb = b.shape
    ngf = W // group_size
    Ms, Os = [], []
    if ngf > 0:
        bf = b[:ngf * group_size].reshape(ngf, group_size, Lb)
        smf = sigmask[:ngf * group_size].reshape(ngf, group_size, Lb)
        med = _q50(bf, dim=1)                                   # (ngf, Lb)
        resid = bf - med[:, None]
        sg = mad(resid.reshape(ngf, -1), 1, _EPS)                # (ngf,)
        uf = (resid.abs() <= ksig * sg[:, None, None]) & (~smf)
        nuf = uf.sum(1)
        mean = (bf * uf).sum(1) / torch.clamp_min(nuf, 1)
        Ms.append(torch.where(nuf > 0, mean, med))
        if return_occ:
            Os.append(1.0 - nuf.to(b.dtype) / float(group_size))
    rem = W - ngf * group_size
    if rem > 0:
        blk = b[ngf * group_size:]
        smr = sigmask[ngf * group_size:]
        med = _q50(blk, dim=0)
        resid = blk - med
        sg = mad(resid.reshape(-1), 0, _EPS)
        uf = (resid.abs() <= ksig * sg) & (~smr)
        nuf = uf.sum(0)
        mean = (blk * uf).sum(0) / torch.clamp_min(nuf, 1)
        Ms.append(torch.where(nuf > 0, mean, med)[None, :])
        if return_occ:
            Os.append((1.0 - nuf.to(b.dtype) / float(rem))[None, :])
    M = torch.cat(Ms, dim=0)                                     # (n_blocks, Lb)
    return (M, torch.cat(Os, dim=0)) if return_occ else M


def _detect_signal(cleaned_band: torch.Tensor, ksig: float, group_size: int) -> torch.Tensor:
    """Boolean (W, Lb) mask of coeffs that stick out (> ksig·MAD) within their block."""
    W, Lb = cleaned_band.shape
    ngf = W // group_size
    sm = torch.zeros_like(cleaned_band, dtype=torch.bool)
    cf = cleaned_band.abs()
    if ngf > 0:
        cb = cf[:ngf * group_size].reshape(ngf, group_size, Lb)
        csg = mad(cb.reshape(ngf, -1), 1, _EPS)                   # (ngf,)
        sm[:ngf * group_size] = (cb > ksig * csg[:, None, None]).reshape(ngf * group_size, Lb)
    rem = W - ngf * group_size
    if rem > 0:
        cr = cf[ngf * group_size:]
        csg = mad(cr.reshape(-1), 0, _EPS)
        sm[ngf * group_size:] = cr > ksig * csg
    return sm


def _sigc(M: torch.Tensor, mode: str) -> torch.Tensor:
    """Coherent scale: the MAD of M over the band (``sigc_mode`` 'median' and 'quantile' are the
    same np.median arithmetic; it is not a chain config key)."""
    return mad(M.reshape(-1), 0, _EPS)


def gate_band(b: torch.Tensor, *, group_size: int, kgate, ksig: float,
              npass: int, tau=None, sigc_mode: str = "quantile") -> torch.Tensor:
    """Coherent-gate one band ``(W, Lb)`` → cleaned band.

    ``tau`` adds the occupancy condition — refuse only when |M| is large AND more
    than ``tau`` of the block's wires were flagged. ``None`` reproduces the legacy
    magnitude-only rule bit-for-bit.

    The rule the magnitude test alone gets wrong: coherent noise is identical on
    every wire of a block, so it CANNOT produce an outlier. An empty mask is
    therefore positive evidence that nothing but coherent noise is present and M
    is the plain mean of all wires — refusing it because |M| landed in the
    coherent's own Gaussian tail leaves the whole component behind, one block
    wide, which is the leftover-coherent "blip" population.

    Occupancy MUST come from pass 1. At a cell pass 1 refused, the cleaned band
    still holds the full coherent, so ``_detect_signal`` flags every wire and
    pass-2 occupancy is 1.0 by construction — the statistic destroys itself.
    """
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
    occ1 = None
    for p in range(npass):
        if p == 0:
            M, occ1 = _block_common_mode(b, ksig, sm, group_size, return_occ=True)
        else:
            M = _block_common_mode(b, ksig, sm, group_size)
        sc = _sigc(M, sigc_mode)
        refuse = M.abs() >= kg[p] * sc
        if tau is not None:
            refuse = refuse & (occ1 > tau)
        Mc = torch.where(refuse, torch.zeros((), dtype=M.dtype, device=M.device), M)
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
    tau=None,
    sigc_mode: str = "quantile",
) -> list[torch.Tensor]:
    """Coherent-remove a plane's DWT bands ``[cA, cD_L, …, cD_1]`` (list in, list out).

    ``gate_approx=False`` leaves the approx band untouched. ``kgate`` may be a
    scalar or a per-pass sequence. A non-finite band fails OPEN — returned
    unchanged, whole-band — so a bad event never propagates corrupted coefficients.
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
                             npass=npass, tau=tau, sigc_mode=sigc_mode))
    return out
