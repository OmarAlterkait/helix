"""JAX/GPU backend for coefficient-space smart gating (the qualified R2).

Backend module of the ``helix.tpc.coherent_gate_ops`` family — dispatched via
``backend.ops()``; call it through :func:`helix.tpc.coherent_gate.coherent_gate`.

Same algorithm as the numpy backend, expressed as jitted whole-band ops. The
partial trailing wire block is handled by NaN-padding the wire axis to a multiple
of ``group_size`` and using ``nanquantile`` (the idiom already used by
``coherent_ops_jax.group_median``), so no dynamic shapes enter the jit.

**Numerics:** ``jnp.quantile`` matches ``np.quantile`` semantics exactly (both
average the two middle values), so this backend reproduces the numpy A-parity
``sigc`` bit-for-bit — measured max abs diff 1e-13…1e-16 with median relative
diff 0.0 on a real 1969x4336 plane. That is *not* true of ``torch.median``,
which takes the lower middle; JAX is therefore the faithful GPU backend here.

Measured ~42x over numpy for the 2-pass gate on an RTX 2080 Ti.
"""
from __future__ import annotations

import functools

import numpy as np
import jax
import jax.numpy as jnp

_EPS = 1e-6


def _pad_wires(b, gs):
    """NaN-pad the wire axis up to a multiple of ``gs``; return (padded, n_real)."""
    W = b.shape[0]
    rem = (-W) % gs
    if rem:
        b = jnp.concatenate([b, jnp.full((rem, b.shape[1]), jnp.nan, b.dtype)], axis=0)
    return b, W


def _gate_band_core(b, gs, kgate, ksig, npass, tau):
    """Gate one band ``(W, Lb)`` -> cleaned band ``(W, Lb)``.

    ``kgate`` is a per-pass vector (length ``npass``) so the pass loop unrolls
    without retracing on scalar values. ``tau`` < 0 disables the occupancy
    condition (the legacy magnitude-only rule); it is a float rather than None so
    it does not become a retracing static argument.
    """
    bp, W = _pad_wires(b, gs)
    nb, Lb = bp.shape[0] // gs, bp.shape[1]
    blk = bp.reshape(nb, gs, Lb)
    valid = ~jnp.isnan(blk)
    blk0 = jnp.nan_to_num(blk)
    sm = jnp.zeros_like(blk, dtype=bool)          # signal mask (per pass)
    cleaned = bp
    nvalid = valid.sum(axis=1)                    # real wires per block (NaN pad)
    occ1 = None

    for p in range(npass):
        med = jnp.nanquantile(blk, 0.5, axis=1)                       # (nb, Lb)
        resid = blk - med[:, None, :]
        sg = jnp.maximum(
            jnp.nanquantile(jnp.abs(resid).reshape(nb, -1), 0.5, axis=1) / 0.6745, _EPS)
        uf = (jnp.abs(resid) <= ksig * sg[:, None, None]) & (~sm) & valid
        nuf = uf.sum(axis=1)
        mean = jnp.where(uf, blk0, 0.0).sum(axis=1) / jnp.maximum(nuf, 1)
        M = jnp.where(nuf > 0, mean, med)                             # (nb, Lb)
        if p == 0:                       # occupancy from PASS 1 only; pass 2 is
            occ1 = 1.0 - nuf / jnp.maximum(nvalid, 1)   # 1.0 by construction
        sigc = jnp.maximum(jnp.quantile(jnp.abs(M), 0.5) / 0.6745, _EPS)
        refuse = jnp.abs(M) >= kgate[p] * sigc
        refuse = jnp.where(tau < 0, refuse, refuse & (occ1 > tau))
        Mc = jnp.where(refuse, 0.0, M)
        cleaned = bp - jnp.repeat(Mc, gs, axis=0)
        if p + 1 < npass:                                             # re-detect signal
            cb = cleaned.reshape(nb, gs, Lb)
            csg = jnp.maximum(
                jnp.nanquantile(jnp.abs(cb).reshape(nb, -1), 0.5, axis=1) / 0.6745, _EPS)
            sm = jnp.abs(cb) > ksig * csg[:, None, None]
    # fail open on non-finite input, decided ON DEVICE: a host-side
    # `bool(isfinite(...))` per band cost 30 device stalls per event.
    return jnp.where(jnp.isfinite(b).all(), cleaned[:W], b)


@functools.partial(jax.jit, static_argnames=("gs", "npass", "lens", "skip_first"))
def _gate_all(flat, gs, kgate, ksig, npass, lens, skip_first, tau):
    """Gate EVERY band of a plane in ONE compiled kernel.

    ``flat`` is the band-concatenated plane ``(W, sum(lens))``. The band loop runs
    inside the trace, so it unrolls into a single kernel instead of dispatching
    one op sequence per band — a Python loop outside jit costs 5 dispatches per
    plane (30 per event) and serialises them.
    """
    out, off = [], 0
    for i, L in enumerate(lens):
        seg = flat[:, off:off + L]
        out.append(seg if (i == 0 and skip_first)
                   else _gate_band_core(seg, gs, kgate, ksig, npass, tau))
        off += L
    return jnp.concatenate(out, axis=1)


def gate_bands(bands, *, group_size=64, kgate=3.0, ksig=3.0, npass=2,
               gate_approx=True, tau=None, sigc_mode="quantile"):
    """Coherent-gate a plane's DWT bands ``[cA, cD_L, …, cD_1]`` (list in, list out).

    ``sigc_mode`` is accepted for API parity with the numpy backend; JAX only
    implements the canonical ``'quantile'`` (A-parity) convention — ``'median'``
    exists solely as the numpy back-compat path for the old torch reference.
    """
    if sigc_mode != "quantile":
        raise ValueError(
            "the jax backend implements only sigc_mode='quantile' (A-parity); "
            "use the numpy backend for the legacy 'median' convention")
    kg = list(kgate) if isinstance(kgate, (list, tuple)) else [float(kgate)] * npass
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
    kvec = jnp.asarray(kg[:npass], dtype=jnp.float32)

    from helix.core.wavelet import FlatBands
    if isinstance(bands, FlatBands):             # already flat: zero copies
        lens, flat = bands.lens, bands.flat
    else:
        lens = tuple(int(np.asarray(b).shape[-1]) for b in bands)
        flat = jnp.concatenate([jnp.asarray(b, jnp.float32) for b in bands], axis=1)
    # tau<0 is the in-trace sentinel for 'no occupancy condition'; passing
    # None would make it a static arg and retrace per value.
    gated = _gate_all(flat, int(group_size), kvec, float(ksig), int(npass),
                      lens, not gate_approx,
                      float(-1.0 if tau is None else tau))
    return FlatBands(gated, lens)
