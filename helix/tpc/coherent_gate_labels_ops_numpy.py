"""NumPy backend of the labeled coherent gate (the reference).

Backend module of the ``helix.tpc.coherent_gate_labels_ops`` family; call it through
:mod:`helix.tpc.coherent_gate_labels`, whose docstring describes the algorithm.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from helix.tpc.coherent_gate_ops_numpy import gate_band as _gate_band_contiguous
from helix.tpc.coherent_gate_labels import ChannelGroups

_EPS = 1e-6


def _masked_median(x, v, axis):
    """np.quantile(x[v], 0.5) along ``axis`` for each slice (NaN where no valid entry)."""
    xs = np.sort(np.where(v, x, np.inf), axis=axis)
    n = v.sum(axis=axis, keepdims=True)
    lo = np.maximum((n - 1) // 2, 0)
    hi = np.maximum(n // 2, 0)
    a = np.take_along_axis(xs, lo, axis=axis)
    b = np.take_along_axis(xs, hi, axis=axis)
    med = 0.5 * (a + b)
    return np.squeeze(np.where(n > 0, med, np.nan), axis=axis)


def _masked_mad(x, v):
    """Per group (axis 0): MAD sigma of all valid entries of x (G, g_max, L)."""
    G = x.shape[0]
    return np.maximum(_masked_median(np.abs(x).reshape(G, -1), v.reshape(G, -1), axis=1) / 0.6745, _EPS)


def _block_common_mode(X, V, n, ksig, sm, return_occ=False):
    med = _masked_median(X, V, axis=1)                                   # (G, L)
    med = np.nan_to_num(med)
    resid = X - med[:, None, :]
    sg = _masked_mad(resid, np.broadcast_to(V, X.shape))                 # (G,)
    uf = (np.abs(resid) <= ksig * sg[:, None, None]) & ~sm & V
    nuf = uf.sum(1)
    mean = (X * uf).sum(1) / np.maximum(nuf, 1)
    M = np.where(nuf > 0, mean, med)
    if return_occ:
        return M, 1.0 - nuf / n[:, None]
    return M


def _detect_signal(cleaned, V, ksig):
    csg = _masked_mad(np.abs(cleaned), np.broadcast_to(V, cleaned.shape))
    return (np.abs(cleaned) > ksig * csg[:, None, None]) & V


def _sigc(M, mode, scope="global"):
    """Coherent scale for the magnitude test: one for the band ("global", HELIX) or one
    per group ("group": each group's |M| median over positions, shape (G, 1))."""
    if scope == "group":
        return np.maximum(np.quantile(np.abs(M), 0.5, axis=1) / 0.6745, _EPS)[:, None]
    if mode == "median":
        return max(float(np.median(np.abs(M))) / 0.6745, _EPS)
    return max(float(np.quantile(np.abs(M), 0.5)) / 0.6745, _EPS)


def gate_band_labeled(b, groups: ChannelGroups, *, kgate=3.0, ksig=3.0, npass=2, tau=None,
                      sigc_mode="quantile", sigc_scope="global", return_parts=False):
    """Gate one band ``(W, L)`` with channel groups. See the module docstring.

    ``sigc_scope``: the coherent scale the magnitude test (refuse |M| >= kgate sigc)
    compares with. "global" (HELIX): one for the band. "group": one per group, so each
    group's common mode is judged against its own coherent level. Use "group" when the
    groups' coherent amplitudes differ (SBND: boards differ up to ~4x). With a global
    scale, noisy groups exceed kgate sigc often and only the occupancy rule (tau)
    subtracts them - and tau also subtracts signals that are uniform across a group
    (e.g. tracks at constant drift time on a single-plane board). "group" with
    tau=None avoids both."""
    b = np.asarray(b, dtype=np.float32)
    out = b.copy()
    if groups.n_groups == 0:
        return (out, None) if return_parts else out
    X = groups.gather(b)
    if groups.full and sigc_scope == "global" and not return_parts:
        flat = X.reshape(groups.n_groups * groups.g_max, -1)
        cl = _gate_band_contiguous(flat, group_size=groups.g_max, kgate=kgate, ksig=ksig, npass=npass,
                                   tau=tau, sigc_mode=sigc_mode)
        out[groups.rows] = cl
        return out
    kg = list(kgate) if isinstance(kgate, (list, tuple, np.ndarray)) else [kgate] * npass
    if len(kg) > npass:
        raise ValueError(f"kgate has {len(kg)} per-pass entries but npass={npass}")
    kg = kg + [kg[-1]] * (npass - len(kg))
    V = groups.valid[:, :, None]
    n = groups.size.astype(np.float64)
    X = np.where(V, X, 0.0).astype(np.float32)
    sm = np.zeros(X.shape, bool)
    cleaned = X
    occ1 = None
    for p in range(npass):
        if p == 0:
            M, occ1 = _block_common_mode(X, V, n, ksig, sm, return_occ=True)
        else:
            M = _block_common_mode(X, V, n, ksig, sm)
        sc = _sigc(M, sigc_mode, sigc_scope)
        refuse = np.abs(M) >= kg[p] * sc
        if tau is not None:
            refuse = refuse & (occ1 > tau)
        Mc = np.where(refuse, 0.0, M)
        cleaned = X - Mc[:, None, :]
        if p + 1 < npass:
            sm = _detect_signal(cleaned, V, ksig)
    out[groups.index[groups.valid]] = cleaned[groups.valid]
    out = out.astype(np.float32, copy=False)
    if return_parts:
        return out, dict(M=M, refuse=refuse, sigc=sc, X=X)
    return out


def coherent_gate_labeled(bands: Sequence[np.ndarray], labels_or_groups, *, kgate=3.0, ksig=3.0, npass=2,
                          gate_approx=True, tau=None, sigc_mode="quantile", sigc_scope="global") -> list[np.ndarray]:
    """Coherent-gate a plane's (or stacked planes') DWT bands ``[cA, cD_L, ..., cD_1]``
    with channel groups. ``labels_or_groups``: per-row integer labels (-1 = excluded)
    or a prebuilt :class:`ChannelGroups` (reuse it across events: it depends only on
    the channel map). Same parameters and fail-open behaviour as
    :func:`helix.tpc.coherent_gate_ops_numpy.gate_bands`."""
    groups = labels_or_groups if isinstance(labels_or_groups, ChannelGroups) else ChannelGroups(labels_or_groups)
    out = []
    for i, b in enumerate(bands):
        b = np.asarray(b, dtype=np.float32)
        if not np.isfinite(b).all():
            warnings.warn(f"coherent_gate_labeled: non-finite band {i}; failing open (no removal)")
            out.append(b)
            continue
        if i == 0 and not gate_approx:
            out.append(b)
            continue
        out.append(gate_band_labeled(b, groups, kgate=kgate, ksig=ksig, npass=npass, tau=tau,
                                     sigc_mode=sigc_mode, sigc_scope=sigc_scope))
    return out
