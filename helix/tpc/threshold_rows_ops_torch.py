"""Torch backend of the live-row threshold.

Backend module of the ``helix.tpc.threshold_rows_ops`` family; call it through
:mod:`helix.tpc.threshold_rows`. A port of :mod:`helix.tpc.threshold_rows_ops_numpy`;
medians via :func:`helix.core.backend.torch_q50` (np.median convention), the group
scope via the :class:`helix.tpc.coherent_gate_labels.ChannelGroups` table and a
masked median instead of a loop over labels.
"""
from __future__ import annotations

import numpy as np
import torch

from helix.core.backend import torch_q50 as _q50
from helix.core.hysteresis_ops_torch import grow, grow_many


def band_sigma_rows(c, valid, sigma_scope="plane", groups=None, any_valid=None, vidx=None):
    """``vidx``: the valid rows' indices (``valid.nonzero()``), reused across bands so the
    boolean selection does not sync the host once per band."""
    if sigma_scope == "plane":
        if vidx is None:
            vidx = valid.nonzero()[:, 0]
        if any_valid is None:
            any_valid = vidx.numel() > 0
        s = _q50(c.index_select(0, vidx).abs().reshape(-1), dim=0) / 0.6745 if any_valid else c.new_zeros(())
        return s.reshape(1, 1).expand(len(c), 1)
    if sigma_scope == "channel":
        return (_q50(c.abs(), dim=1) / 0.6745)[:, None]
    if sigma_scope == "group":
        from helix.tpc.coherent_gate_labels import ChannelGroups
        from helix.tpc.coherent_gate_labels_ops_torch import _masked_median
        g = groups if isinstance(groups, ChannelGroups) else ChannelGroups(
            np.where(valid.cpu().numpy(), np.asarray(groups), -1))
        T = g.torch(c.device)
        X = c[T["index"]].abs().reshape(g.n_groups, -1)                     # (G, g_max * L)
        V = (T["valid"][:, :, None] & valid[T["index"]][:, :, None]).expand(-1, -1, c.shape[1]).reshape(g.n_groups, -1)
        sg = torch.nan_to_num(_masked_median(X, V, dim=1) / 0.6745)
        out = torch.zeros((len(c), 1), dtype=c.dtype, device=c.device)
        out[T["dest"], 0] = sg[:, None].expand(-1, g.g_max).reshape(-1)[T["flat_valid"]]
        return out
    raise ValueError(sigma_scope)


def _prepare(coeffs, kappa=1.0, valid=None, sigma_scope="plane", groups=None,
             threshold_approx=True, floor=2.0):
    """Sigma, seeds and candidates of one plane; hysteresis bands are left for the grow."""
    coeffs = [c if isinstance(c, torch.Tensor) else torch.as_tensor(np.asarray(c, np.float32)) for c in coeffs]
    dev = coeffs[0].device
    n = coeffs[0].shape[0]
    if valid is None:
        valid = torch.ones(n, dtype=torch.bool, device=dev)
    elif not isinstance(valid, torch.Tensor):
        valid = torch.as_tensor(np.asarray(valid, bool), device=dev)
    zero = torch.zeros((), dtype=coeffs[0].dtype, device=dev)
    vidx = valid.nonzero()[:, 0]                                         # one host sync
    any_valid = vidx.numel() > 0
    st = dict(coeffs=coeffs, zero=zero, dev=dev, out=[], kept=[], total=0, sig=[], grow={})
    for i, c in enumerate(coeffs):
        st["total"] += c.numel()
        if i == 0 and not threshold_approx:
            o = torch.where(valid[:, None], c, zero)
            st["out"].append(o); st["kept"].append(torch.count_nonzero(o))
            st["sig"].append(torch.full((), float("nan"), device=dev))
            continue
        s = band_sigma_rows(c, valid, sigma_scope, groups, any_valid=any_valid, vidx=vidx)
        # a zero sigma marks a dead channel (channel scope); a plane / group with sigma 0 is
        # noise-free (simulation): its nonzero coefficients are all above it
        live = ((s > 0) | (sigma_scope != "channel")) & valid[:, None]
        a = c.abs()
        live = live & (a > 0)
        keep = (a >= kappa * s * float(np.sqrt(2.0 * np.log(max(c.shape[-1], 2))))) & live
        st["sig"].append(_q50(s.index_select(0, vidx).reshape(-1), dim=0) if any_valid else zero)
        if floor is not None:
            st["grow"][i] = (keep, (a >= floor * s) & live)
            st["out"].append(None)
            continue
        st["out"].append(torch.where(keep, c, zero))
        st["kept"].append(keep.sum())
    return st


def _finish(st, grown):
    for i, keep in grown.items():
        st["out"][i] = torch.where(keep, st["coeffs"][i], st["zero"])
        st["kept"].append(keep.sum())
    dev = st["dev"]
    both = torch.cat([torch.stack([torch.as_tensor(k, device=dev) for k in st["kept"]]).sum().reshape(1).to(torch.float64),
                      torch.stack([x.reshape(()).to(torch.float64) for x in st["sig"]])]).cpu().numpy()      # one host sync
    return st["out"], int(both[0]), st["total"], both[1:].astype(np.float32)


def threshold_bands_rows(coeffs, **kw):
    st = _prepare(coeffs, **kw)
    idx = list(st["grow"])
    grown = grow_many([st["grow"][i][0] for i in idx], [st["grow"][i][1] for i in idx], "8") if idx else []
    return _finish(st, dict(zip(idx, grown)))


def threshold_planes_rows(planes):
    """``planes``: list of (bands, kwargs) -> one ``threshold_bands_rows`` result per plane.
    Hysteresis bands of the same index are grown in ONE call over all planes, stacked by
    rows with an all-False row between planes (nothing connects across): an SBND event's
    grow goes from 30 calls to 5 (0.18 -> 0.11 s on a 2080 Ti, identical result)."""
    sts = [_prepare(b, **kw) for b, kw in planes]
    grown = [dict() for _ in sts]
    for i in sorted({i for st in sts for i in st["grow"]}):
        who = [p for p, st in enumerate(sts) if i in st["grow"]]
        S, C, cuts = [], [], [0]
        for p in who:
            s_, c_ = sts[p]["grow"][i]
            gap = torch.zeros((1, s_.shape[1]), dtype=torch.bool, device=s_.device)
            S += [s_, gap]; C += [c_, gap]; cuts.append(cuts[-1] + s_.shape[0] + 1)
        k = grow(torch.cat(S[:-1]), torch.cat(C[:-1]), "8")
        for p, a, b in zip(who, cuts[:-1], cuts[1:]):
            grown[p][i] = k[a:b - 1]
    return [_finish(st, g) for st, g in zip(sts, grown)]
