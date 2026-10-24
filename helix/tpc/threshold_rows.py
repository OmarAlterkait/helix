"""Hysteresis threshold with the noise sigma estimated on LIVE rows.

HELIX's universal threshold (hard, kappa * sigma_band * sqrt(2 ln N_band), per-band MAD
sigma, approximation thresholded) computes each band's MAD over ALL rows. Dead or missing channels are zero rows, so
they bias sigma low; when many are missing (SBND run 18260: 1024 of 1983 U channels of a
TPC absent in one event) sigma collapses and every coefficient of the live channels is
kept (3.3 M instead of ~35 k). :func:`threshold_bands_rows`:
  * estimates sigma on ``valid`` rows only and zeroes the others;
  * optionally per group (``sigma_scope="group"``, e.g. one sigma per front-end board)
    or per row (``"channel"``: each channel's own MAD in that band);
  * hysteresis (:mod:`helix.core.hysteresis`): the universal threshold
    kappa sigma sqrt(2 ln N_band) gives the seeds; every coefficient with |c| >= ``floor``
    sigma that is 8-connected (row x coefficient, within the band) to a seed through such
    coefficients is kept too. SBND (data, SBND MC and HELIX sim; SBND-data-helix study):
    floor 2 keeps no more noise than plain universal thresholding one step of kappa lower,
    with ~25% fewer FM tokens, and lowers the 50%-efficiency energy of isolated deposits by
    11-20%. ``floor=None``: the seeds alone (HELIX's universal threshold; with all rows
    valid and ``sigma_scope="plane"`` HELIX's universal threshold).

Torch on any device. Medians are :func:`helix.core.robust.q50` (np.median convention); the
group scope uses the :class:`helix.tpc.coherent_gate.ChannelGroups` table and a masked
median instead of a loop over labels.
"""
from __future__ import annotations

import numpy as np
import torch

from helix.core.hysteresis import grow, grow_many
from helix.core.robust import mad, masked_mad, q50 as _q50


def band_sigma_rows(c, valid, sigma_scope="plane", groups=None, any_valid=None, vidx=None):
    """Per-row sigma, shape (rows, 1). ``vidx``: the valid rows' indices (``valid.nonzero()``),
    reused across bands so the boolean selection does not sync the host once per band."""
    if sigma_scope == "plane":
        if vidx is None:
            vidx = valid.nonzero()[:, 0]
        if any_valid is None:
            any_valid = vidx.numel() > 0
        s = mad(c.index_select(0, vidx).reshape(-1), dim=0) if any_valid else c.new_zeros(())
        return s.reshape(1, 1).expand(len(c), 1)
    if sigma_scope == "channel":
        return mad(c, dim=1)[:, None]
    if sigma_scope == "group":
        from helix.tpc.coherent_gate import ChannelGroups
        g = groups if isinstance(groups, ChannelGroups) else ChannelGroups(
            np.where(valid.cpu().numpy(), np.asarray(groups), -1))
        T = g.torch(c.device)
        X = c[T["index"]].reshape(g.n_groups, -1)                           # (G, g_max * L)
        V = (T["valid"][:, :, None] & valid[T["index"]][:, :, None]).expand(-1, -1, c.shape[1]).reshape(g.n_groups, -1)
        sg = torch.nan_to_num(masked_mad(X, V, 1))
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


def threshold_bands_rows(coeffs, kappa=1.0, valid=None, sigma_scope="plane", groups=None,
                         threshold_approx=True, floor=2.0):
    """Hysteresis hard threshold per band with live-row sigma (``floor=None``: seeds only).

    Returns (bands, n_kept, n_total, sigma_per_band); sigma_per_band is the median of the
    per-row sigmas over live rows."""
    st = _prepare(coeffs, kappa=kappa, valid=valid, sigma_scope=sigma_scope, groups=groups,
                  threshold_approx=threshold_approx, floor=floor)
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
