"""Coherent gate with arbitrary channel groups (e.g. front-end boards).

The R2 smart gate (:mod:`helix.tpc.coherent_gate`) estimates the common mode of
contiguous ``group_size``-wire blocks. On a real detector the coherent noise follows
the readout electronics: on SBND one FEMB reads 128 channels, which on the top boards
are 32 U + 32 V + 64 Y wires, so a board's group spans three planes and is not a
contiguous run of any plane's wires. This module runs the same gate on groups
given by integer labels (one per row; -1 = row excluded, e.g. a dead channel), so a
group may take rows from several planes (stack the planes' bands; see
:func:`coherent_gate_planes`).

Algorithm (the labels are fixed by the channel map, so all bookkeeping is done once):
  * :class:`ChannelGroups` sorts the rows by label once and builds a padded
    (n_groups, g_max) row-index table (-1 = empty slot) and its validity mask.
  * per band: one gather ``b[index] -> (n_groups, g_max, L)``, the gate, one scatter.
  * if every group is complete (all the same size, no empty slot) the gathered
    tensor is exactly what the contiguous-block kernel sees, so it runs that kernel
    unchanged (``gate_band`` with ``group_size = g_max``): identical numbers, cost =
    the contiguous gate + one gather/scatter.
  * otherwise a masked twin of the kernel: medians / MADs over each group's valid
    entries by one sort per axis with empty slots set to +inf and the two middle
    order statistics picked per group (:func:`helix.core.robust.masked_median`), means
    and occupancies over valid rows only. All shapes are static.
Excluded rows (label -1) are returned unchanged. Torch; the bands' device.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import torch

from helix.core.robust import mad, masked_mad, masked_median
from helix.tpc.coherent_gate import gate_band as _gate_band_contiguous

_EPS = 1e-6


class ChannelGroups:
    """Row bookkeeping for integer group labels (-1 = excluded), built once.

    Plain numpy; :meth:`torch` returns the same tables as tensors on a device (cached),
    so a table built from the channel map is reused across events and backends."""

    def __init__(self, labels):
        labels = np.asarray(labels).astype(np.int64)
        self.n_rows = len(labels)
        valid_rows = np.where(labels >= 0)[0]
        order = valid_rows[np.argsort(labels[valid_rows], kind="stable")]
        self.labels_unique, counts = np.unique(labels[order], return_counts=True)
        self.n_groups = len(counts)
        self.size = counts
        self.g_max = int(counts.max()) if len(counts) else 0
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        slot = np.arange(len(order)) - np.repeat(starts, counts)
        gid = np.repeat(np.arange(self.n_groups), counts)
        self.index = np.full((self.n_groups, self.g_max), -1, np.int64)
        self.index[gid, slot] = order
        self.valid = self.index >= 0
        self.full = bool(self.valid.all())
        self.rows = order                                    # valid rows in group-major order
        self._dev = {}

    def torch(self, device):
        """dict(index, valid, size, rows, flat_index, flat_rows) as tensors on ``device``."""
        import torch
        key = str(device)
        if key not in self._dev:
            t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
            self._dev[key] = dict(index=t(np.where(self.valid, self.index, 0)), valid=t(self.valid),
                                  size=t(self.size), rows=t(self.rows),
                                  flat_valid=t(np.flatnonzero(self.valid)), dest=t(self.index[self.valid]))
        return self._dev[key]


def _groups(labels_or_groups):
    return labels_or_groups if isinstance(labels_or_groups, ChannelGroups) else ChannelGroups(labels_or_groups)


def _as_tensor(b):
    if not isinstance(b, torch.Tensor):
        b = torch.as_tensor(np.asarray(b, np.float32))
    if b.dtype not in (torch.float32, torch.float64):
        b = b.float()
    return b


def _masked_mad(x, v):
    """Per group (dim 0): MAD sigma of all valid entries of x (G, g_max, L)."""
    G = x.shape[0]
    return masked_mad(x.reshape(G, -1), v.reshape(G, -1), 1, _EPS)


def _block_common_mode(X, V, n, ksig, sm, return_occ=False):
    med = torch.nan_to_num(masked_median(X, V.expand_as(X), dim=1))       # (G, L)
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
    """Coherent scale: one for the band ("global") or per group ("group", shape (G, 1))."""
    if scope == "group":
        return mad(M, 1, _EPS)[:, None]
    return mad(M.reshape(-1), 0, _EPS)


def gate_band_labeled(b, groups, *, kgate=3.0, ksig=3.0, npass=2, tau=None,
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
    groups = _groups(groups)
    b = _as_tensor(b)
    out = b.clone()
    if groups.n_groups == 0:
        return (out, None) if return_parts else out
    T = groups.torch(b.device)
    X = b[T["index"]]                                                        # (G, g_max, L)
    if groups.full and sigc_scope == "global" and not return_parts:
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


def coherent_gate_labeled(bands: Sequence, labels_or_groups, *, kgate=3.0, ksig=3.0, npass=2,
                          gate_approx=True, tau=None, sigc_mode="quantile", sigc_scope="global") -> list:
    """Coherent-gate a plane's (or stacked planes') DWT bands ``[cA, cD_L, ..., cD_1]``
    with channel groups. ``labels_or_groups``: per-row integer labels (-1 = excluded)
    or a prebuilt :class:`ChannelGroups` (reuse it across events: it depends only on
    the channel map). Same parameters and fail-open behaviour as
    :func:`helix.tpc.coherent_gate.gate_bands`."""
    groups = _groups(labels_or_groups)
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


def coherent_gate_planes(bands_by_plane: Sequence[Sequence], labels_by_plane, **kw):
    """Cross-plane gate: groups (e.g. front-end boards) may take rows from several
    planes. ``bands_by_plane``: one band list per plane (same number of bands and
    band lengths, i.e. the same n_ticks and wavelet). ``labels_by_plane``: one label
    array per plane, labels shared across planes (or ``groups=`` a prebuilt
    :class:`ChannelGroups` over the stacked rows). Returns one band list per plane."""
    n_rows = [len(bl[0]) for bl in bands_by_plane]
    cuts = np.concatenate([[0], np.cumsum(n_rows)])
    stacked = [torch.cat([_as_tensor(bl[i]) for bl in bands_by_plane]) for i in range(len(bands_by_plane[0]))]
    groups = kw.pop("groups", None)
    if groups is None:
        groups = np.concatenate([np.asarray(l) for l in labels_by_plane])
    g = coherent_gate_labeled(stacked, groups, **kw)
    return [[gi[cuts[p]:cuts[p + 1]] for gi in g] for p in range(len(bands_by_plane))]
