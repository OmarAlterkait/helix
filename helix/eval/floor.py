"""The ``floor`` metric: what a frozen encoder knows about charge on the fixed fine grid.

Every model is scored on the same 2-wire x 16-tick cells whatever its patch size, through FROZEN
per-token encoder features: each cell's input is, per band, the feature of the token that covers
it (zeros and a presence bit if none) plus the cell's continuous offset inside that token
(:func:`helix.data.cells.cell_inputs`). A bigger patch puts several cells under one feature; the
offset is what lets a probe ask where inside it.

A probe (:func:`helix.eval.probe.fit_probe`) predicts ``log1p(q / Q0)`` per cell, fitted on the
first events of the split and scored on the rest, with the map rows and windows of
:func:`helix.eval.windows.event_truth` from the cells truth. Two arms, which define the metric:
the trained encoder and the same architecture randomly initialised (the floor a trained model
must beat to have learned anything). Every number of :func:`helix.eval.windows.scalars` comes
with a [16, 84]% interval from an event bootstrap.

Also per arm, ``rankme``: the effective rank ``exp(H(s / sum s))`` of the centred token features
(``s`` their singular values, via the d x d Gram matrix) -- label-free, a necessary condition
for a useful representation (collapse), read alongside the probe.

Memory: map-row features are held on the host (float16), ~10-40 GB at a 400-event split.
"""

from __future__ import annotations

import numpy as np

from helix.data.cells import FT, FW, Q0, cell_inputs, unkey
from helix.eval import windows as W
from helix.eval.probe import fit_probe, gather, split_events

ARMS = ("trained", "random")


def score_maps(p, y, e, pl, wscore, wq, meta, cfg, n_boot=200, seed=0):
    """The floor numbers of one per-cell prediction, each ``[point, lo, hi]``.

    ``p``/``y``: prediction and truth (``log1p(q/Q0)``) on map rows of events ``e``, planes
    ``pl``; ``wscore``/``wq``: score and truth charge on the window cells of ``meta`` (each with
    its event as ``ev``), in window order.
    """
    n_cell = (W.WW // FW) * (W.WT // FT)                     # every window is the same shape
    starts = np.arange(len(meta)) * n_cell
    ends = starts + n_cell
    toff = {i: cfg.toff[i] for i in range(3)}
    Wn = W.window_stats(wscore, wq, meta, starts, ends, cfg.pw, cfg.pt, cfg.delta[0], toff)
    q = np.expm1(y) * Q0
    q30 = np.quantile(q[q > 0], 0.3) if (q > 0).any() else 0.0
    kz = W.fisher_z(p, y, e * 10 + pl)
    rows_of = {v: np.nonzero(e == v)[0] for v in np.unique(e)}
    wins_of = {v: [w for w in Wn if w["ev"] == v] for v in np.unique(e)}
    z_of = {v: [z for k, z in kz.items() if k // 10 == v] for v in np.unique(e)}

    def stat(evs):
        idx = np.concatenate([rows_of[v] for v in evs])
        return W.scalars(p[idx], q[idx], [z for v in evs for z in z_of[v]], [w for v in evs for w in wins_of[v]], q30)

    return W.bootstrap(stat, np.unique(e), n_boot, seed)


class Floor:
    name = "floor"

    def __init__(self, run, n_events, *, seed=0, n_boot=200):
        self.models = {"trained": run.model, "random": run.random_init(seed)}
        self.cfg, self.seed, self.n_boot = run.tokenizer, seed, n_boot
        self.n_train = split_events(n_events)
        self.rows = {a: dict(X=[], A=[], y=[], e=[], pl=[]) for a in ARMS}
        self.win = {a: dict(X=[], A=[], ok=[]) for a in ARMS}
        self.wq, self.meta = [], []
        self.gram = {a: None for a in ARMS}

    def add(self, ev):
        tr = W.event_truth(ev.cells, ev.coeff_cells, np.random.default_rng([self.seed, ev.index]))
        y = np.log1p(tr["mq"] / Q0).astype(np.float32)
        idx, aux = cell_inputs(tr["mrows"], ev.cell_key, ev.band_lengths, self.cfg)
        ok = (idx >= 0).any(1)                    # a cell no token covers is predicted 0, never fitted
        test = ev.index >= self.n_train
        if test and len(tr["wkey"]):
            widx, waux = cell_inputs(tr["wkey"], ev.cell_key, ev.band_lengths, self.cfg)
        for a in ARMS:
            F = ev.features(a, self.models[a])
            self._rank(a, F)
            r = self.rows[a]
            r["X"].append(gather(F, idx[ok])); r["A"].append(aux[ok]); r["y"].append(y[ok])
            r["e"].append(np.full(ok.sum(), ev.index)); r["pl"].append((unkey(tr["mrows"][ok])[0] % 3).astype(np.int8))
            if test and len(tr["wkey"]):
                self.win[a]["X"].append(gather(F, widx)); self.win[a]["A"].append(waux)
                self.win[a]["ok"].append((widx >= 0).any(1))
        if test and len(tr["wkey"]):
            self.wq.append(tr["wq"])
            self.meta += [dict(m, ev=ev.index) for m in tr["meta"]]

    def _rank(self, a, F):
        F = F.double()
        g = (F.T @ F, F.sum(0), F.shape[0])
        self.gram[a] = g if self.gram[a] is None else tuple(x + y for x, y in zip(self.gram[a], g))

    def result(self):
        import torch
        out = dict(n_train_events=self.n_train, n_windows=len(self.meta))
        for a in ARMS:
            r, w = self.rows[a], self.win[a]
            cat = lambda xs: torch.cat(xs) if isinstance(xs[0], torch.Tensor) else np.concatenate(xs)
            X, A, y, e, pl = (cat(r[k]) for k in ("X", "A", "y", "e", "pl"))
            tr = e < self.n_train
            trt = torch.from_numpy(tr)
            probe = fit_probe(X[trt], y[tr], e[tr], extra=A[tr], seed=self.seed)
            p = probe.predict(X[~trt], A[~tr])
            if self.meta:
                qh = np.expm1(np.maximum(probe.predict(cat(w["X"]), cat(w["A"])), 0)) * Q0
                # an uncovered cell carries no model information: predict nothing there (the
                # probe's output on an all-zero input is an extrapolated constant, which would
                # set every window's maximum)
                qh[~cat(w["ok"])] = 0.0
                wq = np.concatenate(self.wq)
            else:
                qh = wq = np.zeros(0)
            res = score_maps(p, y[~tr], e[~tr], pl[~tr].astype(np.int64), qh, wq, self.meta, self.cfg,
                             self.n_boot, self.seed)
            res["val_mse"] = -probe.val_score
            res["rankme"] = rankme(*self.gram[a])
            out[a] = res
        return out


def rankme(gram, total, n):
    """``exp(-sum p log p)``, ``p = s / sum s``, ``s`` the singular values of the centred feature
    matrix, from its Gram matrix ``X^T X``, column sum and row count."""
    import torch
    C = gram - torch.outer(total, total) / max(n, 1)
    ev = torch.linalg.eigvalsh(C).clamp(min=0)
    s = ev.sqrt()
    p = s / (s.sum() + 1e-12)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))
