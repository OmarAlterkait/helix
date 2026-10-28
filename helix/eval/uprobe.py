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
from helix.eval.probe import fisher_r, fit_probe, gather, split_events

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
        self.rows = dict(y=[], geo=[], plane=[], e=[], trained=[], random=[])

    def add(self, ev):
        idx, _ = cell_inputs(ev.cells["key"], ev.cell_key, ev.band_lengths, self.cfg)
        pr = patch_rows(ev.cells, idx, self.n_planes)
        if not len(pr["y"]):
            return
        for k in ("y", "geo", "plane"):
            self.rows[k].append(pr[k])
        self.rows["e"].append(np.full(len(pr["y"]), ev.index))
        for a in ("trained", "random"):
            self.rows[a].append(gather(ev.features(a, self.models[a]), pr["tok"]))

    def result(self):
        import torch
        if not self.rows["y"]:
            raise ValueError("uprobe: no patch with a dominant cell in any event")
        y, geo, plane, e = (np.concatenate(self.rows[k]) for k in ("y", "geo", "plane", "e"))
        grp = e * 16 + plane
        tr = e < self.n_train
        te_e, te_y, te_g = e[~tr], y[~tr], grp[~tr]
        rows_of = {v: np.nonzero(te_e == v)[0] for v in np.unique(te_e)}
        out = dict(n_train_events=self.n_train, n_patch=int(len(y)))
        for a in ARMS:
            X = None if a == "geo" else torch.cat(self.rows[a])
            probe = fit_probe(None if X is None else X[torch.from_numpy(tr)], y[tr], e[tr], extra=geo[tr],
                              group=grp[tr], seed=self.seed)
            p = probe.predict(None if X is None else X[torch.from_numpy(~tr)], geo[~tr])

            def stat(evs, p=p):
                idx = np.concatenate([rows_of[v] for v in evs])
                return {"fisher_r": fisher_r(te_y[idx], p[idx], te_g[idx], MIN_ROWS)}

            out[a] = dict(W.bootstrap(stat, np.unique(te_e), self.n_boot, self.seed),
                          val_fisher_r=round(probe.val_score, 4))
        out["d_over_geo"] = round(out["trained"]["fisher_r"][0] - out["geo"]["fisher_r"][0], 4)
        return out
