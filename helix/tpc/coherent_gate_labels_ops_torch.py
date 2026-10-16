"""Torch backend of the labeled coherent gate.

Backend module of the ``helix.tpc.coherent_gate_labels_ops`` family; call it through
:mod:`helix.tpc.coherent_gate_labels`. A port of the numpy reference
(:mod:`helix.tpc.coherent_gate_labels_ops_numpy`): same gather / masked kernel /
scatter, same conventions. As for the contiguous gate, float32 reduction order
differs, so the two agree to ~1e-4 ADC, not bit-exactly.

Medians are ``np.quantile(., 0.5)`` of the VALID entries: a sort with invalid entries
set to +inf and the average of the ``(n-1)//2`` and ``n//2`` order statistics.
Never ``torch.median`` / ``torch.nanmedian`` (lower middle; see
:func:`helix.core.backend.torch_q50`).
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from helix.core.backend import torch_q50 as _q50
from helix.tpc.coherent_gate_labels import ChannelGroups

_EPS = 1e-6


def _as_tensor(b):
    if not isinstance(b, torch.Tensor):
        b = torch.as_tensor(np.asarray(b, np.float32))
    if b.dtype not in (torch.float32, torch.float64):
        b = b.float()
    return b


def _masked_median(x, v, dim):
    """np.quantile(x[v], 0.5) along ``dim`` per slice (NaN where no valid entry)."""
    inf = torch.full((), float("inf"), dtype=x.dtype, device=x.device)
    xs, _ = torch.sort(torch.where(v, x, inf), dim=dim)
    n = v.sum(dim=dim, keepdim=True)
    lo = torch.clamp_min((n - 1) // 2, 0)
    hi = torch.clamp_min(n // 2, 0)
    a = torch.gather(xs, dim, lo)
    b = torch.gather(xs, dim, hi)
    med = 0.5 * (a + b)
    return torch.where(n > 0, med, torch.full_like(med, float("nan"))).squeeze(dim)


def _masked_mad(x, v):
    """Per group (dim 0): MAD sigma of all valid entries of x (G, g_max, L)."""
    G = x.shape[0]
    return torch.clamp_min(_masked_median(x.abs().reshape(G, -1), v.reshape(G, -1), dim=1) / 0.6745, _EPS)


def _block_common_mode(X, V, n, ksig, sm, return_occ=False):
    med = torch.nan_to_num(_masked_median(X, V.expand_as(X), dim=1))       # (G, L)
    resid = X - med[:, None, :]
    sg = _masked_mad(resid, V.expand_as(X))                                 # (G,)
    uf = (resid.abs() <= ksig * sg[:, None, None]) & ~sm & V
    nuf = uf.sum(1)
    mean = (X * uf).sum(1) / torch.clamp_min(nuf, 1)
    M = torch.where(nuf > 0, mean, med)
    if return_occ:
        return M, 1.0 - nuf.to(X.dtype) / n[:, None]
    return M


def _detect_signal(cleaned, V, ksig):
    csg = _masked_mad(cleaned.abs(), V.expand_as(cleaned))
    return (cleaned.abs() > ksig * csg[:, None, None]) & V


def _sigc(M, scope="global"):
    """Coherent scale: one for the band ("global") or per group ("group", shape (G, 1)).
    The legacy 'median' mode is an alias of quantile here, as in the contiguous torch gate."""
    if scope == "group":
        return torch.clamp_min(_q50(M.abs(), dim=1) / 0.6745, _EPS)[:, None]
    return torch.clamp_min(_q50(M.abs().reshape(-1), dim=0) / 0.6745, _EPS)


def gate_band_labeled(b, groups: ChannelGroups, *, kgate=3.0, ksig=3.0, npass=2, tau=None,
                      sigc_mode="quantile", sigc_scope="global", return_parts=False):
    """Gate one band ``(W, L)``; see :func:`helix.tpc.coherent_gate_labels.gate_band_labeled`."""
    b = _as_tensor(b)
    out = b.clone()
    if groups.n_groups == 0:
        return (out, None) if return_parts else out
    T = groups.torch(b.device)
    X = b[T["index"]]                                                        # (G, g_max, L)
    if groups.full and sigc_scope == "global" and not return_parts:
        from helix.tpc.coherent_gate_ops_torch import gate_band as _gate_band_contiguous
        flat = X.reshape(groups.n_groups * groups.g_max, -1)
        out[T["rows"]] = _gate_band_contiguous(flat, group_size=groups.g_max, kgate=kgate, ksig=ksig, npass=npass,
                                               tau=tau, sigc_mode=sigc_mode)
        return out
    kg = list(kgate) if isinstance(kgate, (list, tuple, np.ndarray)) else [kgate] * npass
    if len(kg) > npass:
        raise ValueError(f"kgate has {len(kg)} per-pass entries but npass={npass}")
    kg = kg + [kg[-1]] * (npass - len(kg))
    V = T["valid"][:, :, None]
    n = T["size"].to(b.dtype)
    X = torch.where(V, X, torch.zeros((), dtype=b.dtype, device=b.device))
    sm = torch.zeros(X.shape, dtype=torch.bool, device=b.device)
    cleaned, occ1 = X, None
    for p in range(npass):
        if p == 0:
            M, occ1 = _block_common_mode(X, V, n, ksig, sm, return_occ=True)
        else:
            M = _block_common_mode(X, V, n, ksig, sm)
        sc = _sigc(M, sigc_scope)
        refuse = M.abs() >= kg[p] * sc
        if tau is not None:
            refuse = refuse & (occ1 > tau)
        cleaned = X - torch.where(refuse, torch.zeros((), dtype=M.dtype, device=M.device), M)[:, None, :]
        if p + 1 < npass:
            sm = _detect_signal(cleaned, V, ksig)
    out[T["dest"]] = cleaned.reshape(-1, cleaned.shape[-1])[T["flat_valid"]]
    if return_parts:
        return out, dict(M=M, refuse=refuse, sigc=sc, X=X)
    return out


def coherent_gate_labeled(bands: Sequence, groups: ChannelGroups, *, kgate=3.0, ksig=3.0, npass=2,
                          gate_approx=True, tau=None, sigc_mode="quantile", sigc_scope="global") -> list:
    """See :func:`helix.tpc.coherent_gate_labels.coherent_gate_labeled`."""
    bands = [_as_tensor(b) for b in bands]
    finite = torch.stack([torch.isfinite(b).all() for b in bands]).tolist()        # one host sync
    out = []
    for i, b in enumerate(bands):
        if not finite[i]:
            warnings.warn(f"coherent_gate_labeled: non-finite band {i}; failing open (no removal)")
            out.append(b)
            continue
        if i == 0 and not gate_approx:
            out.append(b)
            continue
        out.append(gate_band_labeled(b, groups, kgate=kgate, ksig=ksig, npass=npass, tau=tau,
                                     sigc_mode=sigc_mode, sigc_scope=sigc_scope))
    return out
