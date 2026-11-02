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
initialised). Scored with ``fisher_r``: Pearson r WITHIN each (event, plane) of at least
:data:`helix.eval.windows.MIN_ROWS` patches, Fisher-z averaged, so the coarse between-event and
between-plane structure any position-only baseline reproduces is removed (97.4% of ``u``'s
variance is between groups). ``d_over_geo`` = trained - geo, bootstrapped paired.
"""

from __future__ import annotations

import numpy as np

from helix.data.cells import FT, FW, cell_inputs, unkey
from helix.eval.probe import fit_probe, fit_rows, gather
from helix.eval.windows import bootstrap, fisher_z, group_event, plane_group

#: Along-wire unit vector (y, z) per plane gid, SBND: U +120 deg, V -120 deg, Y 180 deg, in each
#: of the two volumes. The geometry registry carries no wire orientation; this was fitted
#: (``wire ~ a*y + b*z`` per event over ~600k deposits, three shards) and is exactly the
#: +-60 / 0 degree wire orientations to four decimals, residual 0.30 wires on every plane.
ALONGWIRE = np.array([[-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 2], [-1.0, 0.0]] * 2)
DOM_THRESHOLD = 0.5
WIRE_SCALE, TICK_SCALE = 2000.0, 4321.0          # the geometry columns' units
ARMS = ("geo", "trained", "random")


def u_of(gid, y, z):
    """``u`` (m) of points (y, z) (mm) on plane ``gid``."""
    aw = ALONGWIRE[np.asarray(gid, np.int64)]
    return ((np.asarray(y, np.float64) * aw[:, 0] + np.asarray(z, np.float64) * aw[:, 1]) / 1000.0)


def patch_rows(cells, cell_rows, n_planes):
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
                          (wmean / WIRE_SCALE)[:, None], (tmean / TICK_SCALE)[:, None],
                          (uniq >= 0).astype(np.float32)], 1).astype(np.float32)
    return dict(tok=uniq[keep], y=y[keep].astype(np.float32), geo=geo[keep], plane=plane[keep])


class UProbe:
    name = "uprobe"

    def __init__(self, ctx):
        self.ctx = ctx
        self.n_planes = int(ctx.run.config["model"]["n_plane"])
        self.fit = dict(y=[], geo=[], grp=[], e=[], trained=[], random=[])   # the training events' fit rows
        self.probes, self.n_fit, self.n_patch = None, None, 0
        self.p = {a: [] for a in ARMS}                                        # the scored events'
        self.y, self.grp = [], []

    def add(self, ev):
        import torch
        c = self.ctx
        idx, _ = cell_inputs(ev.cells["key"], ev.cell_key, ev.band_lengths, c.run.tokenizer)
        pr = patch_rows(ev.cells, idx, self.n_planes)
        n = len(pr["y"])
        self.n_patch += n
        scored = ev.index >= c.n_train
        if scored and self.probes is None:
            self._fit()
        if not n:
            return
        grp = plane_group(ev.index, pr["plane"])
        if not scored:
            keep = fit_rows(n, c.n_train, np.random.default_rng([c.seed, ev.index, 2]))
            f = self.fit
            f["y"].append(pr["y"][keep]); f["geo"].append(pr["geo"][keep]); f["grp"].append(grp[keep])
            f["e"].append(np.full(len(keep), ev.index))
            for a in ("trained", "random"):
                f[a].append(gather(ev.features(a, c.models[a]), pr["tok"][keep]).cpu())
            return
        for a in ARMS:
            X = torch.zeros((n, 0)) if a == "geo" else gather(ev.features(a, c.models[a]), pr["tok"])
            self.p[a].append(self.probes[a].predict(X, pr["geo"]))
        self.y.append(pr["y"]); self.grp.append(grp)

    def _fit(self):
        import torch
        f, self.fit = self.fit, None
        if not f["y"]:
            raise ValueError("uprobe: no patch with a dominant cell in any training event")
        y, geo, grp, e = (np.concatenate(f[k]) for k in ("y", "geo", "grp", "e"))
        self.probes, self.n_fit = {}, len(y)
        for a in ARMS:
            X = torch.zeros((len(y), 0)) if a == "geo" else torch.cat(f.pop(a))
            self.probes[a] = fit_probe(X, geo, y, e, group=grp, seed=self.ctx.seed)

    def result(self):
        c = self.ctx
        if not self.y:
            raise ValueError("uprobe: no patch with a dominant cell in any scored event")
        y, grp = np.concatenate(self.y), np.concatenate(self.grp)
        evs = np.unique(group_event(grp))
        zs = {}
        for a in ARMS:                    # each (event, plane)'s z once; a resample weighs its events
            z = fisher_z(np.concatenate(self.p[a]), y, grp)
            zs[a] = (np.array(list(z.values()), np.float64),
                     np.searchsorted(evs, group_event(np.array(list(z), np.int64))))

        def stat(m):
            r = {}
            for a, (zv, ze) in zs.items():
                w = m[ze]
                r[a] = float(np.tanh((w * zv).sum() / w.sum())) if w.sum() else float("nan")
            r["d_over_geo"] = r["trained"] - r["geo"]
            return r

        b = bootstrap(stat, len(evs), c.n_boot, c.seed)
        out = dict(n_train_events=c.n_train, n_fit_rows=self.n_fit, n_patch=self.n_patch)
        for a in ARMS:
            out[a] = dict(fisher_r=b[a], val_fisher_r=round(self.probes[a].val_score, 4))
        out["d_over_geo"] = b["d_over_geo"]
        return out
