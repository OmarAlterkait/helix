"""The evaluation of a fine-tuned denoiser: detection on the windows, and charge on every charged
cell.

The primary evaluation of a pretrained encoder is a fixed-budget FINE-TUNE (``configs/pimm/
ft_eval.py``: the denoiser trained on 1,024 labelled events), scored here; the same protocol
with ``model.freeze_encoder=True`` is the frozen diagnostic. One scoring path for both.

The metric needs only a predictor, ``predict(event, keys) -> (y, logit)``: ``y`` the predicted
``log1p(q / Q0)`` and ``logit`` the presence log-odds per fine-cell key; a cell no token covers
is ``y = 0`` and ``logit =`` :data:`UNCOVERED_LOGIT`. :func:`load_denoiser` makes one from a
denoiser run (``configs/pimm/denoise.py``).

Detection (:mod:`helix.eval.events` windows), two scores on identical windows:

  presence   the window's largest presence LOG-ODDS. Not the probability: a confident head
             saturates P to exactly 1.0 in float32, the 99th-percentile threshold becomes 1.0
             and no window can exceed it (measured on the frozen presence probes, 58dc4f8);
             log-odds orders the cells the same way without saturating
  amplitude  the largest kept coefficient's |value| / sigma (:func:`helix.eval.events.amplitude`):
             fit-free, what the corpus gate alone gives

Detection alone does not say the charge is right, so ``cells`` scores it on EVERY charged cell of
the scored events (q > ``presence_q``, the training's presence label) -- a cell no token covers
is predicted 0, a miss, and counted (``frac_uncovered``) -- pooled (``all``) and per view
(:func:`charge_accuracy`):

  resolution    per truth-charge bin (:data:`TRUTH_BINS`): median and hw68 = (p84 - p16) / 2 of
                ``log(pred_q / true_q)``, a prediction below :data:`Q_FLOOR` counted as it
                (``frac_zero``: how many were; ``frac_uncovered``: how many no token covers)
  linearity     per PREDICTED-charge bin (:data:`PRED_BINS`): median true / pred
  closure       per event, sum pred / sum true over its charged cells of the view; the median
                over events. On charged cells only: charge predicted on empty cells (haze) is the
                near-window detection's to price, not this

and the presence probability's calibration (:func:`binary_calibration`: ECE over 15 equal-width
bins, Brier, log-loss) against ``q > presence_q`` on the COVERED cells of the charged set and of
the event's windows (iso, far and near: a charge-free population with cells near activity). An
uncovered cell is the tokenizer's, not the head's, so it is left out of the head's calibration.

The truth: cells-3 ``q`` is the hits file's electrons, cut per sample by the simulation (25 e- on
d1m); the output records the cut (``truth`` in :func:`helix.eval.evaluate`'s JSON). One noise
realisation per event, the corpus's own, as on real data: nothing is redrawn.
"""

from __future__ import annotations

import numpy as np

from helix.data.cells import Q0, unkey
from helix.model.cells_join import cell_inputs, cell_neighbors
from helix.eval.events import VIEWS, Scored, amplitude, cell_charge

#: Truth-charge bins of the resolution (e-); the last is open.
TRUTH_BINS = (1e3, 4e3, 16e3, 64e3, 256e3, np.inf)
#: Predicted-charge bins of the linearity (e-); the first edge is 0.05 Q0.
PRED_BINS = (102.0, 500.0, 1e3, 2e3, 4e3, 8e3, 16e3, np.inf)
#: The presence label's charge (e-): ``q > PRESENCE_Q``, the denoiser's training definition.
PRESENCE_Q = 25.0
#: A prediction below this (e-) counts as it in ``log(pred / true)``: a predicted 0 is a finite miss.
Q_FLOOR = 1.0
ECE_BINS = 15
#: The presence log-odds of a cell no token covers: below any head's, and finite, so quantiles hold.
UNCOVERED_LOGIT = -1e6


def _bin_names(edges):
    f = lambda x: f"{x / 1e3:g}k" if x >= 1e3 else f"{x:g}"          # noqa: E731
    return tuple(f"{f(a)}-{f(b)}" if np.isfinite(b) else f"{f(a)}+" for a, b in zip(edges[:-1], edges[1:]))


TRUTH_NAMES, PRED_NAMES = _bin_names(TRUTH_BINS), _bin_names(PRED_BINS)


def sigmoid(x):
    """The probability of log-odds ``x``, float64, without overflow."""
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, np.float64), -700, 700)))


def binary_calibration(p, y, n_bins=ECE_BINS, eps=1e-7):
    """Of probabilities ``p`` against 0/1 outcomes ``y``: ``ece`` (sum over ``n_bins`` equal-width
    bins of the bin's share times |mean p - mean y|), ``brier`` (mean (p - y)^2), ``log_loss``
    (p clipped to [eps, 1 - eps]), ``n`` and ``frac_true``."""
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    n = len(p)
    if not n:
        return dict(n=0, frac_true=float("nan"), ece=float("nan"), brier=float("nan"), log_loss=float("nan"))
    b = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    gap = np.abs(np.bincount(b, p - y, n_bins)).sum() / n
    pc = np.clip(p, eps, 1 - eps)
    return dict(n=n, frac_true=round(float(y.mean()), 6), ece=round(float(gap), 6),
                brier=round(float(np.mean((p - y) ** 2)), 6),
                log_loss=round(float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))), 6))


def _summary(x):
    if not len(x):
        return dict(n=0, median=float("nan"), hw68=float("nan"))
    p16, p50, p84 = np.percentile(x, [16, 50, 84])
    return dict(n=int(len(x)), median=round(float(p50), 6), hw68=round(float(p84 - p16) / 2, 6))


def _mean(x):
    return round(float(np.mean(x)), 6) if len(x) else float("nan")


def charge_accuracy(pred_q, true_q, view, event, covered=None, q_floor=Q_FLOOR):
    """``{"all"|"U"|"V"|"Y": {resolution, linearity, closure, frac_uncovered}}`` of predicted
    charges ``pred_q`` against truth ``true_q`` > 0 (e-), one row per cell (see the module
    docstring); ``view`` 0/1/2, ``event`` and ``covered`` (a token covers the cell; default all)
    per row. An uncovered row is scored as its prediction, which the caller sets to 0."""
    pred_q, true_q = np.asarray(pred_q, np.float64), np.asarray(true_q, np.float64)
    view, event = np.asarray(view, np.int64), np.asarray(event, np.int64)
    unc = np.zeros(len(view), bool) if covered is None else ~np.asarray(covered, bool)
    lr = np.log(np.maximum(pred_q, q_floor) / true_q)
    tb = np.searchsorted(TRUTH_BINS, true_q, side="right") - 1          # -1: below the first edge
    pb = np.searchsorted(PRED_BINS, pred_q, side="right") - 1
    out = {}
    for name, sel in [("all", np.ones(len(view), bool))] + [(VIEWS[v], view == v) for v in range(3)]:
        res = {}
        for i, nm in enumerate(TRUTH_NAMES):
            s = sel & (tb == i)
            res[nm] = dict(_summary(lr[s]), frac_zero=_mean(pred_q[s] < q_floor), frac_uncovered=_mean(unc[s]))
        lin = {}
        for i, nm in enumerate(PRED_NAMES):
            s = sel & (pb == i)
            r = _summary(true_q[s] / pred_q[s]) if s.any() else _summary([])
            lin[nm] = dict(n=r["n"], median_true_over_pred=r["median"])
        evs, inv = np.unique(event[sel], return_inverse=True)
        st = np.bincount(inv.ravel(), true_q[sel], len(evs))
        sp = np.bincount(inv.ravel(), pred_q[sel], len(evs))
        ratio = sp[st > 0] / st[st > 0]
        clo = dict(n_events=int(len(ratio)),
                   median=round(float(np.median(ratio)), 6) if len(ratio) else float("nan"),
                   pooled=round(float(sp.sum() / st.sum()), 6) if st.sum() > 0 else float("nan"))
        out[name] = dict(resolution=res, linearity=lin, closure=clo, frac_uncovered=_mean(unc[sel]))
    return out


def load_denoiser(path, device=None, *, weights="last", fm=None):
    """``(predict, tokenizer)`` from a denoiser run (``configs/pimm/denoise.py``): its run
    directory -- ``weights`` ``"last"`` (the end of the fixed budget, default) or ``"best"`` (the
    lowest validation loss) -- or one checkpoint in it (:func:`helix.model.checkpoint.load_run`;
    ``fm`` relocates its FM run). ``predict.provenance`` is that run's, ``predict.run`` the run
    itself (its config names the corpus), ``predict.dist`` its charge head."""
    import torch
    from helix.eval.recon import autocast
    from helix.model.checkpoint import load_run

    run = load_run(path, weights, device=device, fm=fm)
    model, cfg = run.model, run.tokenizer
    if not hasattr(model, "head"):
        raise ValueError(f"{path}: not a denoiser run (model type {run.config['model'].get('type')!r})")
    dev = next(model.parameters()).device
    t = lambda a: torch.as_tensor(a).to(dev)                            # noqa: E731

    def predict(ev, keys):
        idx, aux = cell_inputs(keys, ev.cell_key, ev.band_lengths, cfg)
        ok = (idx >= 0).any(1)
        y = np.zeros(len(keys), np.float32)
        logit = np.full(len(keys), UNCOVERED_LOGIT, np.float32)
        if ok.any():
            nbr = cell_neighbors(keys[ok], ev.cell_key, ev.band_lengths, cfg)
            with torch.no_grad(), autocast(dev):
                yy, lg = model.predict(ev.batch, t(idx[ok]), t(aux[ok]), t(nbr))
            y[ok], logit[ok] = yy.float().cpu().numpy(), lg.float().cpu().numpy()
        return y, logit

    predict.provenance = run.provenance          # which checkpoint (``weights``) it read, for the JSON
    predict.run = run                            # its config names the corpus and the events
    predict.dist = getattr(model.head, "dist", "point")
    return predict, cfg


class Denoise:
    """The denoiser's metric (see the module docstring): detection on the windows (``presence``,
    ``amplitude``), and ``cells`` -- charge resolution, linearity, closure and the presence
    calibration."""
    name = "denoise"
    #: Charged cells per predictor call, at most (each call encodes the event once).
    CHUNK = 200_000

    def __init__(self, ctx):
        self.ctx = ctx
        self.scored = Scored()
        self.presence_q = float(getattr(ctx, "presence_q", PRESENCE_Q))
        self.rows = dict(pred=[], true=[], view=[], ev=[], cov=[])         # every charged cell
        self.cal = []                                                      # [(p, outcome, view)]

    def add(self, ev):
        c = self.ctx
        cfg = c.run.tokenizer
        covered = lambda k: (cell_inputs(k, ev.cell_key, ev.band_lengths, cfg)[0] >= 0).any(1)   # noqa: E731
        tr = ev.truth(c.seed)
        y, lg = c.denoiser(ev, tr["wkey"])
        self.scored.add(ev, tr, {"presence": lg, "amplitude": amplitude(ev, tr["wkey"])})

        keys, q = np.asarray(ev.cells["key"], np.int64), np.asarray(ev.cells["q"], np.float64)
        ck, cq = keys[q > self.presence_q], q[q > self.presence_q]   # charged as the training's label
        cov = covered(ck)
        cy, cl = np.zeros(len(ck), np.float32), np.zeros(len(ck), np.float32)
        for s in range(0, len(ck), self.CHUNK):
            cy[s:s + self.CHUNK], cl[s:s + self.CHUNK] = c.denoiser(ev, ck[s:s + self.CHUNK])
        cy[~cov] = 0.0                                               # uncovered: predicted 0, a miss
        view = unkey(ck)[0] % 3
        r = self.rows
        r["pred"].append(np.expm1(np.maximum(cy, 0)) * Q0); r["true"].append(cq)
        r["view"].append(view); r["ev"].append(np.full(len(ck), ev.index)); r["cov"].append(cov)

        # calibration: the covered charged cells, and the windows' covered cells not among them
        wk, first = np.unique(tr["wkey"], return_index=True)
        wcov = covered(wk) & ~np.isin(wk, ck)
        p = sigmoid(np.concatenate([cl[cov], lg[first[wcov]]]))
        o = np.concatenate([cq[cov], cell_charge(ev.cells, wk[wcov])]) > self.presence_q
        v = np.concatenate([view[cov], unkey(wk[wcov])[0] % 3])
        self.cal.append((p.astype(np.float32), o, v))

    def result(self):
        c = self.ctx
        out = self.scored.result(c.n_boot, c.seed)
        out["_windows"] = self.scored.windows()          # popped by the CLI into <out>.windows.npz
        r = {k: np.concatenate(v) if v else np.zeros(0) for k, v in self.rows.items()}
        p, o, v = (np.concatenate([x[i] for x in self.cal]) if self.cal else np.zeros(0) for i in range(3))
        cal = dict(all=binary_calibration(p, o), **{VIEWS[j]: binary_calibration(p[v == j], o[v == j]) for j in range(3)})
        out["cells"] = dict(n_cells=int(len(r["true"])), frac_uncovered=_mean(~r["cov"].astype(bool)),
                            presence_q=self.presence_q, q_floor=Q_FLOOR, calibration=dict(presence=cal),
                            by_view=charge_accuracy(r["pred"], r["true"], r["view"], r["ev"], r["cov"]))
        return out
