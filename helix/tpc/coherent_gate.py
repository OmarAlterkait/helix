"""Coherent-noise removal by coefficient-space smart gating (the qualified R2).

One self-contained, parameterized removal — dispatched to the active backend
(``coherent_gate_ops_{numpy,jax}``). Given the per-band DWT coefficients of a
plane it removes the common-mode coherent noise that is rank-1 within
``group_size``-wire blocks, *without* an image round-trip (the DWT is linear, so
the cleaned image's bands == band − gated block common-mode).

Mechanism, per band, per pass:
  1. robust per-(block, coeff-position) common mode M — a ksig-masked mean over
     the wires of each block, excluding wires flagged as signal so signal does
     not bias the estimate;
  2. coherent scale ``sigc`` = MAD of M over blocks;
  3. gate: keep ``|M| < kgate·sigc`` (that is the coherent part → subtract), drop
     the rest (large ⇒ real signal → protect);
  4. subtract the gated common mode from the band.
Between passes, signal is re-detected on the cleaned band and excluded from the
next estimate — a purer estimate that removes more noise at no signal cost (the
qualified 2-pass result).

Qualified default: kgate=3.0, ksig=3.0, npass=2, group_size=64, A-parity ``sigc``
(``quantile(0.5)``). See ``research/r2_qualification/REPORT.md``.

Backends: ``numpy`` (reference, also the legacy ``sigc_mode='median'`` path),
``jax`` (GPU, ~42x, bit-identical to numpy — ``jnp.quantile`` matches
``np.quantile``) and ``torch``.

The torch port has to work around one thing: ``torch.median`` returns the LOWER
of the two middle values while ``np.quantile(..., 0.5)`` AVERAGES them, so a
naive port cannot reproduce A-parity ``sigc`` — they differ by ~1e-4 relative on
even-length inputs, which flips a handful of gate decisions per event.
``coherent_gate_ops_torch._q50`` reproduces the averaging convention instead.
(An earlier version of this docstring concluded from that discrepancy that a
torch backend was impossible and said none existed; the backend was written
anyway, and the workaround is what makes it agree.)
"""
from __future__ import annotations

from helix.core import backend

_OPS = "helix.tpc.coherent_gate_ops"


def coherent_gate(bands, *, group_size=64, kgate=3.0, ksig=3.0, npass=2,
                  gate_approx=True, sigc_mode="quantile"):
    """Coherent-gate a plane's DWT bands ``[cA, cD_L, …, cD_1]`` (list in, list out).

    ``gate_approx=False`` leaves the approximation band (index 0) untouched.
    ``kgate`` may be a scalar or a per-pass sequence. Non-finite input fails open
    (the band is returned unchanged) so a bad event never propagates corrupted
    coefficients. Dispatches to the active backend.
    """
    return backend.ops(_OPS).gate_bands(
        bands, group_size=group_size, kgate=kgate, ksig=ksig, npass=npass,
        gate_approx=gate_approx, sigc_mode=sigc_mode)


# The single-band helper stays importable from the numpy backend for tests that
# exercise the reference implementation directly.
def gate_band(b, **kw):
    """Gate a single band with the NUMPY reference implementation."""
    from helix.tpc import coherent_gate_ops_numpy as _np_ops
    return _np_ops.gate_band(b, **kw)
