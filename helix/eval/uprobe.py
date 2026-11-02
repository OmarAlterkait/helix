"""The ``uprobe`` metric: the along-wire coordinate ``u`` from frozen encoder features.

``u`` is the degree of freedom a single wire plane cannot determine -- a wire fixes only the
pitch coordinate -- so recovering it from one plane's tokens needs information the encoder
combined across planes. That is what makes it the probe target.

Truth, from the cells record: per fine cell, its dominant hits group's 3D centroid (``dom_y``,
``dom_z``) and that group's share of the cell's charge (``dom_frac``). ``u`` is the centroid
projected on the plane's along-wire axis (:data:`ALONGWIRE`), in metres.

Rows are PATCHES -- distinct tuples of covering tokens (one per band, :func:`helix.data.cells.
cell_inputs`). Cells sharing a tuple share every feature the probe can see, so they are one row
and scoring them separately would weight the metric by cell count. A row's label is the
charge-weighted mean ``u`` over its DOMINANT cells (``dom_frac >= 0.5``); a patch with none is
dropped, and so are cells no token covers.

Three arms, every one with the geometry columns (plane one-hot, mean wire, mean tick, presence
bits) appended: ``geo`` (those alone -- the floor; it includes drift time, so any ``u`` leaking
through timing is already in it), ``trained`` and ``random`` (the same architecture randomly
initialised). Scored with ``fisher_r``: Pearson r WITHIN each (event, plane), Fisher-z averaged,
so the coarse between-event and between-plane structure any position-only baseline reproduces is
removed (97.4% of ``u``'s variance is between groups). ``d_over_geo`` = trained - geo.
"""

from __future__ import annotations

import numpy as np

from helix.data.cells import FT, FW, cell_inputs, unkey
from helix.eval import windows as W
from helix.eval.probe import fisher_r, fit_probe, fit_rows, gather, split_events

#: Along-wire unit vector (y, z) per plane gid, SBND: U +120 deg, V -120 deg, Y 180 deg, in each
#: of the two volumes. The geometry registry carries no wire orientation; this was fitted
#: (``wire ~ a*y + b*z`` per event over ~600k deposits, three shards) and is exactly the
#: +-60 / 0 degree wire orientations to four decimals, residual 0.30 wires on every plane.
ALONGWIRE = np.array([[-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 2], [-1.0, 0.0]] * 2)
DOM_THRESHOLD = 0.5
MIN_ROWS = 100                    # per (event, plane) group, for a correlation to count
ARMS = ("geo", "trained", "random")


def u_of(gid, y, z):
    """``u`` (m) of points (y, z) (mm) on plane ``gid``."""
    aw = ALONGWIRE[np.asarray(gid, np.int64)]
    return ((np.asarray(y, np.float64) * aw[:, 0] + np.asarray(z, np.float64) * aw[:, 1]) / 1000.0)


def patch_rows(cells, cell_rows, n_planes, wire_scale=2000.0, tick_scale=4321.0):
    """Cells -> patch rows. ``cell_rows`` (n, n_bands): each cell's covering token per band.

    Returns ``tok`` (the rows' token tuples), ``y`` (u, m), ``geo`` and ``plane``; rows whose
    tuple is all -1 (no covering token) or that have no dominant cell are dropped.
    """
    g, fw, ft = unkey(cells["key"])
    q = np.asarray(cells["q"], np.float64)
    dom = np.asarray(cells["dom_frac"], np.float64) >= DOM_THRESHOLD
    u = u_of(g, cells["dom_y"], cells["dom_z"])
    use = (q > 0) & (cell_rows >= 0).any(1) & np.isfinite(u)
    cell_rows, g, fw, ft, q, dom, u = (x[use] for x in (cell_rows, g, fw, ft, q, dom, u))
    uniq, grp = np.unique(cell_rows, axis=0, return_inverse=True)
    grp = grp.ravel()
    ng = len(uniq)
    wq = q * dom
    wsum = np.bincount(grp, weights=wq, minlength=ng)
    keep = np.bincount(grp, weights=dom.astype(float), minlength=ng) > 0
    y = np.bincount(grp, weights=wq * u, minlength=ng) / np.maximum(wsum, 1e-12)
    n = np.bincount(grp, minlength=ng)
    wmean = np.bincount(grp, weights=fw * FW + FW / 2, minlength=ng) / np.maximum(n, 1)
    tmean = np.bincount(grp, weights=ft * FT + FT / 2, minlength=ng) / np.maximum(n, 1)
    plane = np.zeros(ng, np.int64)
    plane[grp] = g                                     # a token tuple names its plane
    geo = np.concatenate([np.eye(n_planes, dtype=np.float32)[plane],
                          (wmean / wire_scale)[:, None], (tmean / tick_scale)[:, None],
                          (uniq >= 0).astype(np.float32)], 1).astype(np.float32)
    return dict(tok=uniq[keep], y=y[keep].astype(np.float32), geo=geo[keep], plane=plane[keep])


class UProbe:
    name = "uprobe"

    def __init__(self, run, n_events, *, seed=0, n_boot=200):
        self.models = {"trained": run.model, "random": run.random_init(seed)}
        self.cfg, self.seed, self.n_boot = run.tokenizer, seed, n_boot
        self.n_planes = int(run.config["model"]["n_plane"])
        self.n_train = split_events(n_events)
        self.fit = dict(y=[], geo=[], grp=[], e=[], trained=[], random=[])   # the training events' fit rows
        self.probes, self.n_fit, self.n_patch = None, None, 0
        self.p = {a: [] for a in ARMS}                                        # the scored events'
        self.y, self.grp, self.e = [], [], []

    def add(self, ev):
        idx, _ = cell_inputs(ev.cells["key"], ev.cell_key, ev.band_lengths, self.cfg)
        pr = patch_rows(ev.cells, idx, self.n_planes)
        n = len(pr["y"])
        self.n_patch += n
        grp = ev.index * 16 + pr["plane"]
        if ev.index < self.n_train:
            if self.probes is not None:
                raise ValueError(f"uprobe: training event {ev.index} after a scored one")
            if not n:
                return
            keep = fit_rows(n, self.n_train, np.random.default_rng([self.seed, ev.index, 2]))
            f = self.fit
            f["y"].append(pr["y"][keep]); f["geo"].append(pr["geo"][keep]); f["grp"].append(grp[keep])
            f["e"].append(np.full(len(keep), ev.index))
            for a in ("trained", "random"):
                f[a].append(gather(ev.features(a, self.models[a]), pr["tok"][keep]))
            return
        if self.probes is None:
            self._fit()
        if not n:
            return
        for a in ARMS:
            X = None if a == "geo" else gather(ev.features(a, self.models[a]), pr["tok"])
            self.p[a].append(self.probes[a].predict(X, pr["geo"]))
        self.y.append(pr["y"]); self.grp.append(grp); self.e.append(np.full(n, ev.index))

    def _fit(self):
        import torch
        f, self.fit = self.fit, None
        y, geo, grp, e = (np.concatenate(f[k]) for k in ("y", "geo", "grp", "e"))
        if not len(y):
            raise ValueError("uprobe: no patch with a dominant cell in any training event")
        self.probes, self.n_fit = {}, len(y)
        for a in ARMS:
            X = None if a == "geo" else torch.cat(f.pop(a))
            self.probes[a] = fit_probe(X, y, e, extra=geo, group=grp, seed=self.seed)
            del X

    def result(self):
        if not self.y:
            raise ValueError("uprobe: no patch with a dominant cell in any scored event")
        y, grp, e = (np.concatenate(x) for x in (self.y, self.grp, self.e))
        rows_of = {v: np.nonzero(e == v)[0] for v in np.unique(e)}
        out = dict(n_train_events=self.n_train, n_fit_rows=self.n_fit, n_patch=self.n_patch)
        for a in ARMS:
            p = np.concatenate(self.p[a])

            def stat(evs, p=p):
                idx = np.concatenate([rows_of[v] for v in evs])
                return {"fisher_r": fisher_r(y[idx], p[idx], grp[idx], MIN_ROWS)}

            out[a] = dict(W.bootstrap(stat, np.unique(e), self.n_boot, self.seed),
                          val_fisher_r=round(self.probes[a].val_score, 4))
        out["d_over_geo"] = round(out["trained"]["fisher_r"][0] - out["geo"]["fisher_r"][0], 4)
        return out
