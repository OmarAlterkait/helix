"""NumPy backend of the live-row threshold (the reference).

Backend module of the ``helix.tpc.threshold_rows_ops`` family; call it through
:mod:`helix.tpc.threshold_rows`, whose docstring describes the rules.
"""
from __future__ import annotations

import numpy as np

from helix.core.hysteresis_ops_numpy import grow


def _mad(x, axis=None):
    return np.median(np.abs(x), axis=axis) / 0.6745


def band_sigma_rows(c, valid, sigma_scope="plane", groups=None):
    """Per-row sigma, shape (rows, 1)."""
    c = np.asarray(c)
    valid = np.asarray(valid, bool)
    if sigma_scope == "plane":
        return np.full((len(c), 1), _mad(c[valid]) if valid.any() else 0.0, np.float32)
    if sigma_scope == "channel":
        return _mad(c, axis=1)[:, None].astype(np.float32)
    if sigma_scope == "group":
        out = np.zeros((len(c), 1), np.float32)
        g = np.asarray(groups)
        for lab in np.unique(g[valid]):
            m = (g == lab) & valid
            out[g == lab] = _mad(c[m])
        return out
    raise ValueError(sigma_scope)


def threshold_bands_rows(coeffs, kappa=1.0, valid=None, sigma_scope="plane", groups=None,
                         threshold_approx=True, floor=2.0):
    coeffs = [np.asarray(c, np.float32) for c in coeffs]
    n = coeffs[0].shape[0]
    valid = np.ones(n, bool) if valid is None else np.asarray(valid, bool)
    out, kept, total, sig = [], 0, 0, []
    for i, c in enumerate(coeffs):
        total += c.size
        if i == 0 and not threshold_approx:
            o = np.where(valid[:, None], c, 0.0).astype(np.float32)
            out.append(o); kept += int(np.count_nonzero(o)); sig.append(np.nan)
            continue
        s = band_sigma_rows(c, valid, sigma_scope, groups)
        # a zero sigma marks a dead channel (channel scope); a plane / group with sigma 0 is
        # noise-free (simulation): its nonzero coefficients are all above it
        live = ((s > 0) | (sigma_scope != "channel")) & valid[:, None]
        a = np.abs(c)
        live = live & (a > 0)
        keep = (a >= kappa * s * np.sqrt(2.0 * np.log(max(c.shape[-1], 2)))) & live
        if floor is not None:
            keep = grow(keep, (a >= floor * s) & live, "8")
        out.append(np.where(keep, c, 0.0).astype(np.float32))
        kept += int(keep.sum())
        sig.append(float(np.median(s[valid])) if valid.any() else 0.0)
    return out, kept, total, np.asarray(sig, np.float32)


def threshold_planes_rows(planes):
    """One :func:`threshold_bands_rows` per (bands, kwargs)."""
    return [threshold_bands_rows(b, **kw) for b, kw in planes]
