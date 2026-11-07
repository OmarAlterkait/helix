"""Readout description of a wire detector: channels -> (tpc, plane, wire) and
electronics groups.

A detector ships one ``helix/tpc/data/<name>_readout.npz`` (SBND: exported from the
channel map and geometry by the SBND-data-helix study, archive/scripts/export_helix_readout.py), or is
described by its wire layout (simulation: ``{"kind": "layout", "num_wires": [[U, V, Y] per volume],
"group_size": 64}`` - channels volume-major then plane then wire, every wire instrumented, coherent
groups of ``group_size`` contiguous wires, the simulation's noise model):
per offline channel ``tpc, plane, wire, instrumented`` and one integer array per
electronics grouping (SBND: ``femb``, ``asic``) and a JSON ``meta``. (The SBND file also holds
``templates``, pulse shapes of the retired candidate-region filter; nothing reads them.)

The PLANE GEOMETRY registry (:func:`load_plane_registry`) is the detector description the noise
model needs: keyed by canonical plane gid (matching the per-point ``plane_gid`` the sensor
reader surfaces), per-plane ``n_wires``/``n_ticks``/``pedestal``/``wire_lengths`` (meters) --
what densify / noise / digitize consume::

    geom = load_plane_registry("cubic_wireplane_geometry.json")   # ships in data/

It DUPLICATES ``pimm_data.geometry``: a byte-identical loader and a byte-identical bundled
``cubic_wireplane_geometry.json``, pinned by ``tests/test_boundary.py``. The
duplication is deliberate and is the same arrangement as the shard codec: pimm-data must run
with no helix installed (it serves every detector), and helix must build a corpus with no
pimm-data installed, so the shared concern is mirrored and pinned by a test instead of imported.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parent / "data"


class ReadoutMap:
    def __init__(self, path=None, arrays=None):
        d = np.load(path, allow_pickle=False) if arrays is None else arrays
        self.path = str(path) if arrays is None else "layout"
        self.meta = json.loads(str(d["meta"]))
        self.tpc = d["tpc"].astype(np.int64)
        self.plane = d["plane"].astype(np.int64)
        self.wire = d["wire"].astype(np.int64)
        self.instrumented = d["instrumented"].astype(bool)
        self.groups = {k: d[k].astype(np.int64) for k in self.meta.get("group_kinds", [])}
        self.planes = list(self.meta["planes"])
        self.n_tpc = int(self.tpc.max()) + 1

    def rows(self, tpc, instrumented_only=True):
        """Channel indices per plane of one TPC, in wire order."""
        out = []
        for p in range(len(self.planes)):
            sel = np.where((self.tpc == tpc) & (self.plane == p))[0]
            sel = sel[np.argsort(self.wire[sel], kind="stable")]
            if instrumented_only:
                sel = sel[self.instrumented[sel]]
            out.append(sel)
        return out

    def labels(self, kind, rows, valid=None):
        """Group label per row (-1 = excluded: not instrumented or not ``valid``)."""
        lab = np.where(self.instrumented[rows], self.groups[kind][rows], -1)
        if valid is not None:
            lab = np.where(np.asarray(valid, bool), lab, -1)
        return lab


def layout_arrays(num_wires, planes=("U", "V", "Y"), group_size=64):
    """The arrays of a :class:`ReadoutMap` for a wire layout (see the module docstring)."""
    nw = np.asarray(num_wires, np.int64)
    tpc, plane, wire, grp = [], [], [], []
    g0 = 0
    for v in range(nw.shape[0]):
        for p in range(nw.shape[1]):
            n = int(nw[v, p])
            tpc.append(np.full(n, v)); plane.append(np.full(n, p)); wire.append(np.arange(n))
            grp.append(g0 + np.arange(n) // group_size); g0 += -(-n // group_size)
    meta = dict(planes=list(planes), group_kinds=["block"], group_size=int(group_size),
                num_wires=nw.tolist())
    n = int(nw.sum())
    return dict(meta=json.dumps(meta), tpc=np.concatenate(tpc), plane=np.concatenate(plane), wire=np.concatenate(wire),
                instrumented=np.ones(n, bool), block=np.concatenate(grp))


@lru_cache(maxsize=None)
def _load(key) -> ReadoutMap:
    spec = json.loads(key)
    if isinstance(spec, dict):
        if spec.get("kind") != "layout":
            raise ValueError(f"unknown readout {spec!r}")
        return ReadoutMap(arrays=layout_arrays(spec["num_wires"], tuple(spec.get("planes", ("U", "V", "Y"))),
                                               int(spec.get("group_size", 64))))
    p = Path(spec)
    if not p.suffix:
        p = _DATA / f"{spec}_readout.npz"
    return ReadoutMap(p)


def load_readout(spec="sbnd") -> ReadoutMap:
    """A packaged readout name ("sbnd"), an .npz path, or a layout dict (simulation)."""
    return _load(json.dumps(spec, sort_keys=True))


def canonical_plane_gid(tpc, plane, n_planes=3):
    """The corpus plane id: tpc * n_planes + plane index."""
    return int(tpc) * n_planes + int(plane)


# ---- plane geometry --------------------------------------------------------------------------

_PLANE_INDEX = {'U': 0, 'V': 1, 'Y': 2}
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def _resolve(path):
    for cand in (path, os.path.join(_DATA_DIR, path),
                 os.path.join(_DATA_DIR, str(path) + '.json')):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        f"plane-geometry file {path!r} not found (looked in cwd and {_DATA_DIR})")


def _wire_lengths(spec, n_wires):
    """Constant planes store a scalar (collection); varying ones a per-wire array."""
    if isinstance(spec, (int, float)):
        return np.full(int(n_wires), float(spec), dtype=np.float32)
    arr = np.asarray(spec, dtype=np.float32)
    if arr.shape != (int(n_wires),):
        raise ValueError(f"wire_lengths length {arr.shape} != n_wires {n_wires}")
    return arr


def _label_gid(label):
    """``'volume_{v}_{U|V|Y}'`` -> ``v*3 + {U:0,V:1,Y:2}`` (matches pimm-data)."""
    vol, plane = label.rsplit('_', 1)
    return canonical_plane_gid(int(vol.split('_')[1]), _PLANE_INDEX[plane])


def load_plane_registry(path):
    """JSON -> ``{plane gid: {label, n_wires, n_ticks, pedestal, wire_lengths}}``."""
    with open(_resolve(path)) as f:
        d = json.load(f)
    nts = int(d['num_time_steps'])
    reg = {}
    for label, e in d['planes'].items():
        nw = int(e['n_wires'])
        reg[_label_gid(label)] = {
            'label': label, 'n_wires': nw, 'n_ticks': nts,
            'pedestal': int(e['pedestal']),
            'wire_lengths': _wire_lengths(e['wire_lengths_m'], nw),
        }
    return reg
