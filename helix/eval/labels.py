"""The ``labels`` metric: which particle made a cell's charge, from frozen encoder features.

Per signal cell of the floor's map rows (truth charge > 0, covered by a token), the class of the
track carrying most of its charge (:func:`helix.data.cells.cell_class`), in pimm-data's PILArNet
vocabulary: ``motif`` (shower, track, michel, delta, le) and ``pid`` (photon, electron, muon,
pion, proton, le). Unlike the charge, the class is not in a cell's own amplitude: it is a
property of the particle, so reading it needs context -- the shape and neighbourhood of the
deposit, which is what a representation is for.

A classifier probe (:func:`helix.eval.probe.fit_probe` with ``n_classes``: class-balanced
cross-entropy) per task, inputs as the floor's (covering-token features + presence bits and
offsets). Two arms: ``trained`` and ``geo`` (the presence bits and offsets alone -- what the
token layout gives away). Scored with the balanced accuracy (``macro_recall``, the mean
per-class recall: the classes run from 53% of cells (le) to 0.4% (michel)), with a [16, 84]%
interval over an event bootstrap, and each class's recall; ``d_over_geo`` paired.
"""

from __future__ import annotations

import numpy as np

from helix.data.cells import MOTIF_NAMES, PID_NAMES, cell_class, cell_inputs
from helix.eval.probe import fit_probe, fit_rows, gather
from helix.eval.windows import bootstrap

ARMS = ("geo", "trained")
TASKS = {"motif": ("trk_motif", MOTIF_NAMES), "pid": ("trk_pid", PID_NAMES)}


def signal_rows(ev, tr, cfg):
    """The event's labelled rows: covered signal map rows -> ``(idx, aux, {task: class})``."""
    keys = np.asarray(ev.cells["key"], np.int64)
    sig = tr["mrows"][tr["mq"] > 0]
    pos = np.searchsorted(keys, sig)                     # every signal map row is a cell
    idx, aux = cell_inputs(sig, ev.cell_key, ev.band_lengths, cfg)
    cls = {t: cell_class(ev.cells, col)[pos] for t, (col, _) in TASKS.items()}
    ok = (idx >= 0).any(1) & np.logical_or.reduce([c >= 0 for c in cls.values()])
    return idx[ok], aux[ok], {t: c[ok] for t, c in cls.items()}


class Labels:
    name = "labels"

    def __init__(self, ctx):
        self.ctx = ctx
        self.fit = dict(X=[], A=[], e=[], **{t: [] for t in TASKS})
        self.probes, self.n_fit = None, None
        self.p = {(t, a): [] for t in TASKS for a in ARMS}
        self.y, self.e = {t: [] for t in TASKS}, []

    def add(self, ev):
        import torch
        c = self.ctx
        scored = ev.index >= c.n_train
        if scored and self.probes is None:
            self._fit()
        idx, aux, cls = signal_rows(ev, ev.truth(c.seed, windows=scored), c.run.tokenizer)
        if not len(idx):
            return
        if not scored:
            keep = fit_rows(len(idx), c.n_train, np.random.default_rng([c.seed, ev.index, 3]))
            f = self.fit
            f["X"].append(gather(ev.features(c.model), idx[keep]).cpu()); f["A"].append(aux[keep])
            f["e"].append(np.full(len(keep), ev.index))
            for t in TASKS:
                f[t].append(cls[t][keep])
            return
        X = {"geo": torch.zeros((len(idx), 0)), "trained": gather(ev.features(c.model), idx)}
        for t in TASKS:
            for a in ARMS:
                self.p[(t, a)].append(self.probes[(t, a)].predict(X[a], aux))
            self.y[t].append(cls[t])
        self.e.append(np.full(len(idx), ev.index))

    def _fit(self):
        import torch
        f, self.fit = self.fit, None
        if not f["e"]:
            raise ValueError("labels: no labelled signal cell in any training event")
        X, A, e = torch.cat(f.pop("X")), np.concatenate(f["A"]), np.concatenate(f["e"])
        self.probes, self.n_fit = {}, len(e)
        for t, (_, names) in TASKS.items():
            y = np.concatenate(f[t])
            ok = y >= 0
            okt = torch.from_numpy(ok)
            for a in ARMS:
                Xa = torch.zeros((int(ok.sum()), 0)) if a == "geo" else X[okt]
                self.probes[(t, a)] = fit_probe(Xa, A[ok], y[ok], e[ok], n_classes=len(names), seed=self.ctx.seed)

    def result(self):
        c = self.ctx
        if not self.e:
            raise ValueError("labels: no labelled signal cell in any scored event")
        e = np.concatenate(self.e)
        evs, we = np.unique(e, return_inverse=True)
        out = dict(n_train_events=c.n_train, n_fit_rows=self.n_fit)
        for t, (_, names) in TASKS.items():
            y = np.concatenate(self.y[t])
            ok = y >= 0
            hit = {a: np.concatenate(self.p[(t, a)])[ok] == y[ok] for a in ARMS}
            yk, wk = y[ok], we[ok]
            K = len(names)

            def stat(m, hit=hit, yk=yk, wk=wk, K=K):
                w = m[wk]
                r = {}
                for a in ARMS:
                    n_c = np.bincount(yk, w, K)
                    h_c = np.bincount(yk, w * hit[a], K)
                    r[a] = float(np.mean(h_c[n_c > 0] / n_c[n_c > 0]))
                r["d_over_geo"] = r["trained"] - r["geo"]
                return r

            b = bootstrap(stat, len(evs), c.n_boot, c.seed)
            res = dict(prior={nm: round(float((yk == i).mean()), 4) for i, nm in enumerate(names)})
            for a in ARMS:
                res[a] = dict(macro_recall=b[a], val_macro_recall=round(self.probes[(t, a)].val_score, 4),
                              recall={nm: round(float(hit[a][yk == i].mean()), 4)
                                      for i, nm in enumerate(names) if (yk == i).any()})
            res["d_over_geo"] = b["d_over_geo"]
            out[t] = res
        return out
