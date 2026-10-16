"""Readout description of a wire detector: channels -> (tpc, plane, wire) and
electronics groups, plus the pulse templates of the candidate-region filter.

A detector ships one ``helix/tpc/data/<name>_readout.npz`` (SBND: exported from the
channel map and geometry by the SBND-data-helix study, scripts/export_helix_readout.py):
per offline channel ``tpc, plane, wire, instrumented`` and one integer array per
electronics grouping (SBND: ``femb``, ``asic``), plus ``templates`` (planes x samples,
peak at ``meta.template_pre``) and a JSON ``meta``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parent / "data"


class ReadoutMap:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.path = str(path)
        self.meta = json.loads(str(d["meta"]))
        self.tpc = d["tpc"].astype(np.int64)
        self.plane = d["plane"].astype(np.int64)
        self.wire = d["wire"].astype(np.int64)
        self.instrumented = d["instrumented"].astype(bool)
        self.groups = {k: d[k].astype(np.int64) for k in self.meta.get("group_kinds", [])}
        self.templates = d["templates"].astype(np.float32)
        self.template_pre = int(self.meta["template_pre"])
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


@lru_cache(maxsize=None)
def load_readout(name_or_path="sbnd") -> ReadoutMap:
    p = Path(name_or_path)
    if not p.suffix:
        p = _DATA / f"{name_or_path}_readout.npz"
    return ReadoutMap(p)
