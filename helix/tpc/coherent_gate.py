"""The coherent-noise gate: the common mode of each electronics group, removed in coefficient space.

Coherent noise follows the readout electronics: every channel of a group (an SBND front-end board,
128 channels over three planes; a block of 64 contiguous wires in the simulation's noise model) carries
the same waveform. The groups come from the readout (:meth:`helix.tpc.readout.ReadoutMap.labels`: one
integer per row, -1 = row excluded, e.g. a dead channel), so a group may take rows from several planes
(stack the planes' bands: :func:`coherent_gate_planes`). The DWT is linear, so the cleaned image's
bands are the bands minus the gated common mode: no image round trip.

Per band, per pass:
  1. the common mode M per (group, coefficient position): a ksig-clipped mean over the group's rows,
     excluding rows flagged as signal (so signal does not bias it);
  2. the coherent scale sigc: the MAD of M over the band (``sigc_scope="global"``) or per group
     ("group": when groups' coherent levels differ, SBND boards up to ~4x);
  3. the gate: subtract M where |M| < kgate sigc (coherent), keep it where larger (signal) - and with
     ``tau``, subtract a large M too unless more than ``tau`` of the group's rows were flagged in pass 1.
     Coherent noise is identical on every row of a group, so it cannot produce an outlier: an empty
     mask is evidence that only coherent noise is there, and refusing M because it landed in the
     coherent's own Gaussian tail leaves the whole component behind, one group wide. Occupancy comes
     from pass 1: where pass 1 refused, the band still holds the coherent noise and pass 2 would flag
     every row;
  4. between passes, signal is re-detected on the cleaned band (> ksig MAD within the group) and
     excluded from the next estimate.

Two things change gate decisions silently rather than failing: medians are np.median's (the AVERAGE
of the two middles, :func:`helix.core.robust.masked_median`; torch.median takes the lower), and the
signal exclusion in step 1. Float32 reductions depend on summation order, the comparisons on them
are discrete and a group's decision covers all its rows, so one GPU generation builds a corpus.

Bookkeeping, done once per channel map (:class:`ChannelGroups`): rows sorted by label into a padded
(n_groups, g_max) index table and its validity mask; per band one gather to (n_groups, g_max, L),
medians / MADs over each group's valid entries (one sort, empty slots +inf), one scatter. Excluded rows
are returned unchanged. Torch, on the bands' device.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import torch

from helix.core.robust import mad, masked_mad, masked_median

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
                      sigc_scope="global", return_parts=False):
    """Gate one band ``(W, L)`` with channel groups. See the module docstring.

    ``tau``: refuse (keep as signal) only when |M| is large AND more than ``tau`` of the
    group's rows were flagged in pass 1; None: the magnitude test alone.

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
                          gate_approx=True, tau=None, sigc_scope="global") -> list:
    """Coherent-gate a plane's (or stacked planes') DWT bands ``[cA, cD_L, ..., cD_1]``
    with channel groups. ``labels_or_groups``: per-row integer labels (-1 = excluded)
    or a prebuilt :class:`ChannelGroups` (reuse it across events: it depends only on
    the channel map). ``gate_approx=False`` leaves the approximation band untouched; a
    non-finite band fails OPEN (returned unchanged), so a bad event never propagates
    corrupted coefficients."""
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
                                     sigc_scope=sigc_scope))
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
