"""NumPy backend for coefficient-space smart gating (the qualified R2).

Backend module of the ``helix.tpc.coherent_gate_ops`` family — dispatched via
``backend.ops()``; call it through :func:`helix.tpc.coherent_gate.coherent_gate`.

One self-contained, parameterized function: `coherent_gate(bands, …)`. Given the
per-band DWT coefficients of a plane, it removes the common-mode coherent noise
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
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

_EPS = 1e-6


def _mad(x: np.ndarray, axis=None) -> np.ndarray:
    """MAD-derived sigma = quantile(|x|, 0.5) / 0.6745 (A-parity)."""
    return np.maximum(np.quantile(np.abs(x), 0.5, axis=axis) / 0.6745, _EPS)


def _block_common_mode(b: np.ndarray, ksig: float, sigmask: np.ndarray,
                       group_size: int) -> np.ndarray:
    """Per-(block, position) robust common mode M, shape (n_blocks, Lb).

    Blocks are contiguous runs of `group_size` wires; a trailing partial block is
    handled separately. Within a block, M is the mean over wires whose residual
    from the block median is within `ksig` MADs AND not flagged as signal; blocks
    with no such wires fall back to the median.
    """
    W, Lb = b.shape
    ngf = W // group_size
    Ms = []
    if ngf > 0:
        bf = b[:ngf * group_size].reshape(ngf, group_size, Lb)
        smf = sigmask[:ngf * group_size].reshape(ngf, group_size, Lb)
        med = np.quantile(bf, 0.5, axis=1)                      # (ngf, Lb)
        resid = bf - med[:, None]
        sg = _mad(resid.reshape(ngf, -1), axis=1)               # (ngf,)
        uf = (np.abs(resid) <= ksig * sg[:, None, None]) & (~smf)
        nuf = uf.sum(1)
        mean = (bf * uf).sum(1) / np.maximum(nuf, 1)
        Ms.append(np.where(nuf > 0, mean, med))
    rem = W - ngf * group_size
    if rem > 0:
        blk = b[ngf * group_size:]
        smr = sigmask[ngf * group_size:]
        med = np.quantile(blk, 0.5, axis=0)
        resid = blk - med
        sg = _mad(resid.reshape(-1))
        uf = (np.abs(resid) <= ksig * sg) & (~smr)
        nuf = uf.sum(0)
        mean = (blk * uf).sum(0) / np.maximum(nuf, 1)
        Ms.append(np.where(nuf > 0, mean, med)[None, :])
    return np.concatenate(Ms, axis=0)                            # (n_blocks, Lb)


def _detect_signal(cleaned_band: np.ndarray, ksig: float, group_size: int) -> np.ndarray:
    """Boolean (W, Lb) mask of coeffs that stick out (> ksig·MAD) within their block."""
    W, Lb = cleaned_band.shape
    ngf = W // group_size
    sm = np.zeros_like(cleaned_band, dtype=bool)
    cf = np.abs(cleaned_band)
    if ngf > 0:
        cb = cf[:ngf * group_size].reshape(ngf, group_size, Lb)
        csg = _mad(cb.reshape(ngf, -1), axis=1)                  # (ngf,)
        sm[:ngf * group_size] = (cb > ksig * csg[:, None, None]).reshape(ngf * group_size, Lb)
    rem = W - ngf * group_size
    if rem > 0:
        cr = cf[ngf * group_size:]
        csg = _mad(cr.reshape(-1))
        sm[ngf * group_size:] = cr > ksig * csg
    return sm


def _sigc(M: np.ndarray, mode: str) -> float:
    # NB: 'median' uses np.median (average of the two middles) — it does NOT reproduce
    # the old torch.median (lower of the two middles); the two differ by the even-n
    # tie-break (~1e-4 rel), which flips a handful of gate decisions on some events.
    # The shipped default is 'quantile' (A-parity, quantile(0.5)) — the deliberate
    # canonical, not the reference's accidental torch.median. See COEFF_CORPUS_DESIGN.
    if mode == "median":
        return max(float(np.median(np.abs(M))) / 0.6745, _EPS)
    return max(float(np.quantile(np.abs(M), 0.5)) / 0.6745, _EPS)   # A-parity (default)


def gate_band(b: np.ndarray, *, group_size: int, kgate, ksig: float,
              npass: int, sigc_mode: str = "quantile") -> np.ndarray:
    """Coherent-gate one band ``(W, Lb)`` → cleaned band. See module docstring."""
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
    idx = np.minimum(np.arange(W) // group_size, n_blocks - 1)
    sm = np.zeros_like(b, dtype=bool)
    cleaned = b
    for p in range(npass):
        M = _block_common_mode(b, ksig, sm, group_size)
        sc = _sigc(M, sigc_mode)
        Mc = np.where(np.abs(M) < kg[p] * sc, M, 0.0)
        cleaned = b - Mc[idx]
        if p + 1 < npass:
            sm = _detect_signal(cleaned, ksig, group_size)
    return cleaned.astype(b.dtype, copy=False)


def gate_bands(
    bands: Sequence[np.ndarray],
    *,
    group_size: int = 64,
    kgate=3.0,
    ksig: float = 3.0,
    npass: int = 2,
    gate_approx: bool = True,
    sigc_mode: str = "quantile",
) -> list[np.ndarray]:
    """Coherent-remove a plane's DWT bands ``[cA, cD_L, …, cD_1]`` (list in, list out).

    `gate_approx=False` leaves the approx band (index 0) untouched. `kgate` may be a
    scalar or a per-pass sequence.

    NaN/inf in a band FAILS OPEN — the band is returned unchanged. Note what that
    means downstream: the band is then un-gated (coherent noise still in it), and
    thresholding it yields a NaN sigma, so `|c| >= NaN` is all-False and the band
    is silently ZEROED. `audit_shard` checks for non-finite VALUES and a zeroed
    band has none, so such an event ships looking plausible. No NaN has ever been
    observed here (source shards and the 1000-event pilot are both clean, and the
    frozen research gate carries no such guard), so this is defensive only — but
    if one ever appears, raising would be strictly more informative than the
    quiet zero this produces.
    """
    out = []
    for i, b in enumerate(bands):
        b = np.asarray(b, dtype=np.float32)
        if not np.isfinite(b).all():
            warnings.warn(f"coherent_gate: non-finite band {i}; failing open (no removal)")
            out.append(b)
            continue
        if i == 0 and not gate_approx:
            out.append(b)
            continue
        out.append(gate_band(b, group_size=group_size, kgate=kgate, ksig=ksig,
                             npass=npass, sigc_mode=sigc_mode))
    return out
