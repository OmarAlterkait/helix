"""The ``floor`` metric: what a frozen encoder knows about charge on the fixed fine grid.

Every model is scored on the same 2-wire x 16-tick cells whatever its patch size, through FROZEN
per-token encoder features: each cell's input is, per band, the feature of the token that covers
it (zeros and a presence bit if none) plus the cell's continuous offset inside that token
(:func:`helix.data.cells.cell_inputs`). A bigger patch puts several cells under one feature; the
offset is what lets a probe ask where inside it. A cell no token covers carries no model
information: it is never fitted on and is scored as predicted 0 (the probe's output on an
all-zero input is an extrapolated constant, which would set every window's maximum).

A probe (:func:`helix.eval.probe.fit_probe`) predicts ``log1p(q / Q0)`` per cell, fitted on the
first events of the split and scored on the rest, on the map rows and windows of
:func:`helix.eval.windows.event_truth`. Two arms, which define the metric: the trained encoder
and the same architecture randomly initialised (the floor a trained model must beat to have
learned anything). Every number of :func:`helix.eval.windows.score_cells` comes with a [16, 84]%
interval from an event bootstrap.

Also per arm, ``rankme``: the effective rank ``exp(H(s / sum s))`` of the centred token features
(``s`` their singular values, via the d x d Gram matrix) -- label-free, a necessary condition
for a useful representation (collapse), read alongside the probe.

Memory: only the fit's rows hold features (float16, at most :data:`helix.eval.probe.FIT_ROWS`
per arm, ~6 GB each); a scored event keeps its predictions alone.
"""

from __future__ import annotations

import numpy as np

from helix.data.cells import Q0, cell_inputs, unkey
from helix.eval.probe import fit_probe, fit_rows, gather
from helix.eval.windows import score_cells

ARMS = ("trained", "random")


class Scored:
    """The scored events' map rows and windows, and named per-cell predictions on them, scored by
    :func:`helix.eval.windows.score_cells`: the floor's probe arms and the denoiser alike."""

    def __init__(self):
        self.y, self.e, self.g, self.wq, self.meta = [], [], [], [], []
        self.p, self.ws = {}, {}

    def add(self, ev, tr, preds):
        """``tr``: the event's truth; ``preds``: ``{name: (log1p(q/Q0) on its map rows, window
        score on its window cells)}``."""
        self.y.append(np.log1p(tr["mq"] / Q0).astype(np.float32))
        self.e.append(np.full(len(tr["mrows"]), ev.index))
        self.g.append(unkey(tr["mrows"])[0])
        self.wq.append(tr["wq"])
        self.meta += [dict(m, ev=ev.index) for m in tr["meta"]]
        for k, (p, ws) in preds.items():
            self.p.setdefault(k, []).append(p)
            self.ws.setdefault(k, []).append(ws)

    def result(self, n_boot, seed):
        if not self.y:
            raise ValueError("no scored event was added")
        y, e, g, wq = (np.concatenate(x) for x in (self.y, self.e, self.g, self.wq))
        return {k: score_cells(np.concatenate(self.p[k]), y, e, g, np.concatenate(self.ws[k]), wq, self.meta,
                               n_boot, seed) for k in self.p}


class Floor:
    name = "floor"

    def __init__(self, ctx):
        self.ctx = ctx
        self.fit = {a: dict(X=[], A=[], y=[], e=[]) for a in ARMS}   # the training events' fit rows
        self.probes, self.n_fit = None, None
        self.scored = Scored()
        self.gram = dict.fromkeys(ARMS)

    def add(self, ev):
        c = self.ctx
        scored = ev.index >= c.n_train
        tr = ev.truth(c.seed, windows=scored)
        idx, aux = cell_inputs(tr["mrows"], ev.cell_key, ev.band_lengths, c.run.tokenizer)
        ok = (idx >= 0).any(1)
        if not scored:
            keep = np.nonzero(ok)[0][fit_rows(int(ok.sum()), c.n_train, np.random.default_rng([c.seed, ev.index, 1]))]
            y = np.log1p(tr["mq"][keep] / Q0).astype(np.float32)
            for a in ARMS:
                F = ev.features(a, c.models[a])
                self._rank(a, F)
                f = self.fit[a]
                f["X"].append(gather(F, idx[keep]).cpu()); f["A"].append(aux[keep]); f["y"].append(y)
                f["e"].append(np.full(len(keep), ev.index))
            return
        if self.probes is None:
            self._fit()
        widx, waux = cell_inputs(tr["wkey"], ev.cell_key, ev.band_lengths, c.run.tokenizer)
        wok = (widx >= 0).any(1)
        preds = {}
        for a in ARMS:
            F = ev.features(a, c.models[a])
            self._rank(a, F)
            wp = self._predict(a, F, widx, waux, wok)
            preds[a] = (self._predict(a, F, idx, aux, ok), np.expm1(np.maximum(wp, 0)) * Q0)
        self.scored.add(ev, tr, preds)

    def _predict(self, a, F, idx, aux, ok):
        """The probe's ``log1p(q/Q0)`` per cell, 0 where no token covers it."""
        p = np.zeros(len(idx), np.float32)
        if ok.any():
            p[ok] = self.probes[a].predict(gather(F, idx[ok]), aux[ok])
        return p

    def _fit(self):
        import torch
        fits, self.fit, self.probes = self.fit, None, {}
        for a in ARMS:
            f = fits.pop(a)
            y = np.concatenate(f["y"])
            self.probes[a] = fit_probe(torch.cat(f.pop("X")), np.concatenate(f["A"]), y, np.concatenate(f["e"]),
                                       seed=self.ctx.seed)
            self.n_fit = len(y)

    def _rank(self, a, F):
        F = F.double()
        g = (F.T @ F, F.sum(0), F.shape[0])
        self.gram[a] = g if self.gram[a] is None else tuple(x + y for x, y in zip(self.gram[a], g))

    def result(self):
        c = self.ctx
        res = self.scored.result(c.n_boot, c.seed)
        out = dict(n_train_events=c.n_train, n_fit_rows=self.n_fit, n_windows=len(self.scored.meta))
        for a in ARMS:
            out[a] = dict(res[a], val_mse=-self.probes[a].val_score, rankme=rankme(*self.gram[a]))
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
