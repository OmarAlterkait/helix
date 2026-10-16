"""NumPy backend of hysteresis selection (the reference).

Backend module of the ``helix.core.hysteresis_ops`` family; call it through
:mod:`helix.core.hysteresis`, whose docstring describes the method.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label

_STRUCT = {"8": np.ones((3, 3), bool), "1d": np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], bool)}


def grow(seed, cand, conn="8"):
    seed = np.asarray(seed, bool)
    cand = np.asarray(cand, bool) | seed
    lab, n = label(cand, structure=_STRUCT[conn])
    good = np.zeros(n + 1, bool)
    good[lab[seed]] = True
    good[0] = False
    return good[lab]


def grow_many(seeds, cands, conn="8"):
    return [grow(s_, c_, conn) for s_, c_ in zip(seeds, cands)]


def window_any(m, left, right, dim=1):
    """Prefix sums: exact for any (left, right)."""
    m = np.asarray(m, bool)
    if dim == 0:
        return window_any(m.T, left, right, 1).T
    T = m.shape[1]
    c = np.zeros((m.shape[0], T + 1), np.int64)
    np.cumsum(m, axis=1, out=c[:, 1:])
    i = np.arange(T)
    return (c[:, np.minimum(i + right + 1, T)] - c[:, np.maximum(i - left, 0)]) > 0
