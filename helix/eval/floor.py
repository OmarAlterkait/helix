"""The ``floor`` metric: how faint a deposit frozen encoder features let a probe find.

Every model is scored on the same 2-wire x 16-tick cells whatever its patch size, through FROZEN
per-token encoder features: each cell's input is, per band, the feature of the token that covers
it (zeros and a presence bit if none) plus the cell's continuous offset inside that token
(:func:`helix.data.cells.cell_inputs`). A bigger patch puts several cells under one feature; the
offset is what lets a probe ask where inside it. A cell no token covers carries no model
information: it is never fitted on and is scored as predicted 0 (the probe's output on an
all-zero input is an extrapolated constant, which would set every window's maximum).

A probe (:func:`helix.eval.probe.fit_probe`) predicts ``log1p(q / Q0)`` per cell, fitted on the
map rows of the first events of the split; a window's score is its largest predicted charge,
scored on the windows of the rest (:func:`helix.eval.windows.score_windows`: efficiency at a 1%
false-positive rate per view, by energy). Two arms:

  trained    the probe on the trained encoder's features
  amplitude  no fit: a cell's largest |kept coefficient| / sigma
             (:func:`helix.data.cells.coeff_amplitude`) -- what the compressor itself sees. The
             reference a representation must beat. (The randomly initialised encoder under the
             same probe scored below it at every energy up to 1 MeV, so it is no floor.)

Memory: only the fit's rows hold features (float16, at most :data:`helix.eval.probe.FIT_ROWS`,
~6 GB); a scored event keeps its window scores alone.
"""

from __future__ import annotations

import numpy as np

from helix.data.cells import Q0, cell_inputs
from helix.eval.probe import fit_probe, fit_rows, gather
from helix.eval.windows import score_windows

ARMS = ("trained", "amplitude")


def amplitude(ev, keys):
    """The amplitude baseline's score of fine cells ``keys``: 0 where no kept coefficient."""
    uk, amp = ev.coeff_amp
    if not len(uk):
        return np.zeros(len(keys))
    p = np.clip(np.searchsorted(uk, keys), 0, len(uk) - 1)
    return np.where(uk[p] == keys, amp[p], 0.0)


class Scored:
    """The scored events' windows and named per-cell scores on them, scored by
    :func:`helix.eval.windows.score_windows`: the floor's arms and the denoiser alike."""

    def __init__(self):
        self.meta, self.s = [], {}

    def add(self, ev, tr, scores):
        """``tr``: the event's truth; ``scores``: ``{name: score per window cell}``."""
        self.meta += [dict(m, ev=ev.index) for m in tr["meta"]]
        for k, s in scores.items():
            self.s.setdefault(k, []).append(np.asarray(s, np.float32))

    def result(self, n_boot, seed):
        if not self.meta:
            raise ValueError("no scored window")
        return {k: score_windows(np.concatenate(v), self.meta, n_boot, seed) for k, v in self.s.items()}


class Floor:
    name = "floor"

    def __init__(self, ctx):
        self.ctx = ctx
        self.fit = dict(X=[], A=[], y=[], e=[])         # the training events' fit rows
        self.probe, self.n_fit = None, None
        self.scored = Scored()

    def add(self, ev):
        c = self.ctx
        scored = ev.index >= c.n_train
        tr = ev.truth(c.seed, windows=scored)
        if not scored:
            idx, aux = cell_inputs(tr["mrows"], ev.cell_key, ev.band_lengths, c.run.tokenizer)
            ok = np.nonzero((idx >= 0).any(1))[0]
            keep = ok[fit_rows(len(ok), c.n_train, np.random.default_rng([c.seed, ev.index, 1]))]
            f = self.fit
            f["X"].append(gather(ev.features(c.model), idx[keep]).cpu()); f["A"].append(aux[keep])
            f["y"].append(np.log1p(tr["mq"][keep] / Q0).astype(np.float32)); f["e"].append(np.full(len(keep), ev.index))
            return
        if self.probe is None:
            self._fit()
        widx, waux = cell_inputs(tr["wkey"], ev.cell_key, ev.band_lengths, c.run.tokenizer)
        ok = (widx >= 0).any(1)
        p = np.zeros(len(widx), np.float32)
        if ok.any():
            p[ok] = self.probe.predict(gather(ev.features(c.model), widx[ok]), waux[ok])
        self.scored.add(ev, tr, dict(trained=np.expm1(np.maximum(p, 0)) * Q0, amplitude=amplitude(ev, tr["wkey"])))

    def _fit(self):
        import torch
        f, self.fit = self.fit, None
        if not f["y"]:
            raise ValueError("floor: no covered map row in any training event")
        y = np.concatenate(f["y"])
        self.probe = fit_probe(torch.cat(f.pop("X")), np.concatenate(f["A"]), y, np.concatenate(f["e"]),
                               seed=self.ctx.seed)
        self.n_fit = len(y)

    def result(self):
        c = self.ctx
        out = dict(n_train_events=c.n_train, n_fit_rows=self.n_fit, val_mse=-self.probe.val_score,
                   **self.scored.result(c.n_boot, c.seed))
        out["d_over_amplitude"] = {k: round(out["trained"][k][0] - out["amplitude"][k][0], 4)
                                   for k in out["trained"] if k.startswith("floor_eff")}
        return out
