"""Hysteresis (two-threshold) selection, shared by the TPC and optical chains.

HELIX's universal threshold keeps an element when it alone is above
kappa sigma sqrt(2 ln N): the largest of N noise values. Hysteresis (Canny) keeps, in
addition, every element above a low ``floor`` that is CONNECTED to such a seed through
elements that also pass the floor. A faint element is kept where it continues a
structure that is certainly there (a track's faint end, a pulse's tail) and dropped
where noise crosses the floor in isolation.

:func:`grow` (seed, candidate) -> kept, connectivity ``conn``:
  * ``"8"``: 2-D, 8-connected (TPC: wire x coefficient within a band);
  * ``"1d"``: along axis 1 only (optical: samples of one waveform).
:func:`window_any`: any(m) in a window along an axis (dilation of a mask).

Backends: numpy (reference) and torch (``helix.core.hysteresis_ops_{numpy,torch}``),
chosen by the input's type or the active backend.
"""
from __future__ import annotations

from helix.core import backend

_OPS = "helix.core.hysteresis_ops"


def _ops_for(x):
    if backend.kind_of(x) == "torch" or backend.get_backend() == "torch":
        return backend.import_backend_module(_OPS, "torch")
    return backend.import_backend_module(_OPS, "numpy")


def grow(seed, cand, conn="8"):
    """Elements of ``seed | cand`` connected (``conn``) to a ``seed`` element through
    ``cand`` elements. Bool, 2-D (rows, columns)."""
    if conn not in ("8", "1d"):
        raise ValueError(f"conn must be '8' or '1d', got {conn!r}")
    return _ops_for(seed).grow(seed, cand, conn)


def grow_many(seeds, cands, conn="8"):
    """:func:`grow` of each (seed, cand) pair."""
    return _ops_for(seeds[0]).grow_many(seeds, cands, conn)


def window_any(m, left, right, dim=1):
    """out[i] = any(m[i-left : i+right+1]) along ``dim`` (bool, 2-D)."""
    return _ops_for(m).window_any(m, left, right, dim)
