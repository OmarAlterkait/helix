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
    order statistics picked per group (exactly ``np.quantile(., 0.5)`` of the valid
    values), means and occupancies over valid rows only. All shapes are static.
Excluded rows (label -1) are returned unchanged.

Backends: ``numpy`` (reference) and ``torch``
(``helix.tpc.coherent_gate_labels_ops_{numpy,torch}``). A call runs on torch when its
bands are torch tensors or the active backend is torch, else on numpy; torch keeps
the bands' device.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from helix.core import backend

_OPS = "helix.tpc.coherent_gate_labels_ops"


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

    def gather(self, b):
        return b[np.where(self.valid, self.index, 0)]        # (G, g_max, L)

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


def _ops_for(x):
    if backend.kind_of(x) == "torch" or backend.get_backend() == "torch":
        return backend.import_backend_module(_OPS, "torch")
    if backend.kind_of(x) == "jax":
        raise NotImplementedError("coherent_gate_labels: no jax backend; use numpy or torch")
    return backend.import_backend_module(_OPS, "numpy")


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
    return _ops_for(b).gate_band_labeled(b, _groups(groups), kgate=kgate, ksig=ksig, npass=npass, tau=tau,
                                         sigc_mode=sigc_mode, sigc_scope=sigc_scope, return_parts=return_parts)


def coherent_gate_labeled(bands: Sequence, labels_or_groups, *, kgate=3.0, ksig=3.0, npass=2,
                          gate_approx=True, tau=None, sigc_mode="quantile", sigc_scope="global") -> list:
    """Coherent-gate a plane's (or stacked planes') DWT bands ``[cA, cD_L, ..., cD_1]``
    with channel groups. ``labels_or_groups``: per-row integer labels (-1 = excluded)
    or a prebuilt :class:`ChannelGroups` (reuse it across events: it depends only on
    the channel map). Same parameters and fail-open behaviour as
    :func:`helix.tpc.coherent_gate.coherent_gate`."""
    return _ops_for(bands[0]).coherent_gate_labeled(
        bands, _groups(labels_or_groups), kgate=kgate, ksig=ksig, npass=npass, gate_approx=gate_approx, tau=tau,
        sigc_mode=sigc_mode, sigc_scope=sigc_scope)


def coherent_gate_planes(bands_by_plane: Sequence[Sequence], labels_by_plane, **kw):
    """Cross-plane gate: groups (e.g. front-end boards) may take rows from several
    planes. ``bands_by_plane``: one band list per plane (same number of bands and
    band lengths, i.e. the same n_ticks and wavelet). ``labels_by_plane``: one label
    array per plane, labels shared across planes (or ``groups=`` a prebuilt
    :class:`ChannelGroups` over the stacked rows). Returns one band list per plane."""
    n_rows = [len(bl[0]) for bl in bands_by_plane]
    cuts = np.concatenate([[0], np.cumsum(n_rows)])
    if backend.kind_of(bands_by_plane[0][0]) == "torch":
        import torch
        cat = lambda xs: torch.cat(list(xs))  # noqa: E731
    else:
        cat = lambda xs: np.concatenate([np.asarray(x) for x in xs])  # noqa: E731
    stacked = [cat(bl[i] for bl in bands_by_plane) for i in range(len(bands_by_plane[0]))]
    groups = kw.pop("groups", None)
    if groups is None:
        groups = np.concatenate([np.asarray(l) for l in labels_by_plane])
    g = coherent_gate_labeled(stacked, groups, **kw)
    return [[gi[cuts[p]:cuts[p + 1]] for gi in g] for p in range(len(bands_by_plane))]
