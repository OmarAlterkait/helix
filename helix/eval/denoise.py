"""The ``denoise`` metric: a denoiser's per-cell charge, scored on the floor's windows and on
every charged cell.

The same scored events, windows and numbers (:class:`helix.eval.probe.Scored`) as ``floor`` --
only the per-cell prediction comes from a trained denoiser instead of a probe on frozen
features, so the numbers are directly comparable with the floor's ``trained`` and ``amplitude``
arms, near-activity background included. Cells no token covers are predicted 0.

The metric needs only a predictor, ``predict(event, keys) -> (y, presence)``: ``y`` the
predicted ``log1p(q / Q0)`` and ``presence`` the probability of charge per fine-cell key, both 0
where no token covers the cell. :func:`load_denoiser` makes one from a denoiser run
(``configs/pimm/denoise.py``).

Detection, per score: ``charge`` (window score = predicted charge, ``expm1(max(y, 0)) * Q0``) and
``presence`` (window score = the probability).

Detection alone does not say the charge is right, so ``cells`` scores it, on every truth-charged
cell a token covers in the scored events (:func:`charge_accuracy`), pooled (``all``) and per view:

  resolution    per truth-charge bin (:data:`TRUTH_BINS`): median and hw68 = (p84 - p16) / 2 of
                ``log(pred_q / true_q)``, a prediction below :data:`Q_FLOOR` counted as it
                (``frac_zero``: how many were)
  linearity     per PREDICTED-charge bin (:data:`PRED_BINS`): median true / pred
  closure       per event, sum pred / sum true over its charged covered cells of the view; the
                median over events. On charged cells only: charge predicted on empty cells
                (haze) is the near-window detection's to price, not this

and the presence probability's calibration (:func:`binary_calibration`: ECE over 15 equal-width
bins, Brier, log-loss) against ``q > presence_q`` -- the training's presence label, 25 e- by
default -- on the covered cells of the charged set and of the event's windows (iso, far and near:
a charge-free population with cells near activity, which every covered cell of the event is not:
enumerating those is millions of cells an event). :meth:`Denoise.calibrations` names the
(probability, outcome) pairs scored, so a head with its own distribution adds its own.

A distributional charge head (``dist`` gauss or cat, :mod:`helix.model.denoise`) also has its
per-cell uncertainty scored, on the same charged covered cells (:func:`dist_calibration`, as
``cells.dist``), per truth bin and view, in y = ``log1p(q / Q0)``:

  gauss   z = (y_true - mu) / sigma: ``cov68`` = P(|z| < 1) (0.683 if calibrated), ``cov95`` =
          P(|z| < 2) (0.954)
  cat     the PIT, the predictive CDF given charge at y_true: ``cov68`` = P(0.16 < PIT < 0.84)
          (0.68), ``cov95`` = P(0.025 < PIT < 0.975) (0.95)

with the sharpness beside them: ``width_median`` (median sigma, or the predictive std) against
``err_hw68``, the hw68 of the point estimate's error -- equal if the width matches the spread.
The predictor supplies it through an optional ``predict.calib(event, keys, y_true) -> (y,
presence, stat, width)`` (``predict.dist`` names the head); ``predict(event, keys) -> (y,
presence)`` stays the contract every predictor meets.

The truth: cells-3 ``q`` is the hits file's electrons, cut per sample by the simulation (25 e- on
d1m); the output records the cut (``truth`` in :func:`helix.eval.evaluate`'s JSON). One noise
realisation per event, the corpus's own, as on real data: nothing is redrawn.
"""

from __future__ import annotations

import numpy as np

from helix.data.cells import Q0, cell_inputs, cell_neighbors, unkey
from helix.eval.events import VIEWS
from helix.eval.probe import Scored, window_flags

#: Truth-charge bins of the resolution (e-); the last is open.
TRUTH_BINS = (1e3, 4e3, 16e3, 64e3, 256e3, np.inf)
#: Predicted-charge bins of the linearity (e-); the first edge is 0.05 Q0.
PRED_BINS = (102.0, 500.0, 1e3, 2e3, 4e3, 8e3, 16e3, np.inf)
#: The presence label's charge (e-): ``q > PRESENCE_Q``, the denoiser's training definition.
PRESENCE_Q = 25.0
#: A prediction below this (e-) counts as it in ``log(pred / true)``: a predicted 0 is a finite miss.
Q_FLOOR = 1.0
ECE_BINS = 15


def _bin_names(edges):
    f = lambda x: f"{x / 1e3:g}k" if x >= 1e3 else f"{x:g}"          # noqa: E731
    return tuple(f"{f(a)}-{f(b)}" if np.isfinite(b) else f"{f(a)}+" for a, b in zip(edges[:-1], edges[1:]))


TRUTH_NAMES, PRED_NAMES = _bin_names(TRUTH_BINS), _bin_names(PRED_BINS)


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


def charge_accuracy(pred_q, true_q, view, event, q_floor=Q_FLOOR):
    """``{"all"|"U"|"V"|"Y": {resolution, linearity, closure}}`` of predicted charges ``pred_q``
    against truth ``true_q`` > 0 (e-), one row per cell (see the module docstring); ``view``
    0/1/2 and ``event`` per row."""
    pred_q, true_q = np.asarray(pred_q, np.float64), np.asarray(true_q, np.float64)
    view, event = np.asarray(view, np.int64), np.asarray(event, np.int64)
    lr = np.log(np.maximum(pred_q, q_floor) / true_q)
    tb = np.searchsorted(TRUTH_BINS, true_q, side="right") - 1          # -1: below the first edge
    pb = np.searchsorted(PRED_BINS, pred_q, side="right") - 1
    out = {}
    for name, sel in [("all", np.ones(len(view), bool))] + [(VIEWS[v], view == v) for v in range(3)]:
        res = {}
        for i, nm in enumerate(TRUTH_NAMES):
            s = sel & (tb == i)
            zero = round(float((pred_q[s] < q_floor).mean()), 6) if s.any() else float("nan")
            res[nm] = dict(_summary(lr[s]), frac_zero=zero)
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
        out[name] = dict(resolution=res, linearity=lin, closure=clo)
    return out


#: Per distributional head: how its calibration statistic falls inside the 68% / 95% intervals.
_COVERS = {"gauss": (lambda s: np.abs(s) < 1, lambda s: np.abs(s) < 2),
           "cat": (lambda s: (s > 0.16) & (s < 0.84), lambda s: (s > 0.025) & (s < 0.975))}


def dist_calibration(kind, stat, width, err, true_q, view):
    """``{"kind", "all"|"U"|"V"|"Y": {truth bin: {n, cov68, cov95, width_median, err_hw68}}}`` of a
    distributional head (module docstring): ``stat`` per cell is z (``kind="gauss"``) or the PIT
    (``"cat"``), ``width`` its sigma or predictive std, ``err`` the point estimate's error in y;
    ``true_q`` (e-) bins the cells by :data:`TRUTH_BINS`, ``view`` 0/1/2."""
    if kind not in _COVERS:
        raise ValueError(f"no distribution calibration for kind={kind!r}; one of {sorted(_COVERS)}")
    stat, width, err = (np.asarray(a, np.float64) for a in (stat, width, err))
    true_q, view = np.asarray(true_q, np.float64), np.asarray(view, np.int64)
    c68, c95 = _COVERS[kind]
    tb = np.searchsorted(TRUTH_BINS, true_q, side="right") - 1
    out = dict(kind=kind)
    for name, sel in [("all", np.ones(len(view), bool))] + [(VIEWS[v], view == v) for v in range(3)]:
        res = {}
        for i, nm in enumerate(TRUTH_NAMES):
            s = sel & (tb == i)
            if not s.any():
                res[nm] = dict(n=0, cov68=float("nan"), cov95=float("nan"), width_median=float("nan"),
                               err_hw68=float("nan"))
                continue
            res[nm] = dict(n=int(s.sum()), cov68=round(float(c68(stat[s]).mean()), 6),
                           cov95=round(float(c95(stat[s]).mean()), 6),
                           width_median=round(float(np.median(width[s])), 6), err_hw68=_summary(err[s])["hw68"])
        out[name] = res
    return out


def load_denoiser(path, device=None):
    """``(predict, tokenizer)`` from a denoiser run (``configs/pimm/denoise.py``): its run
    directory -- the checkpoint of the lowest validation loss -- or one checkpoint in it
    (:func:`helix.model.checkpoint.load_run`). ``predict.provenance`` is that run's,
    ``predict.dist`` its charge head; a distributional head also gives ``predict.calib``."""
    import torch
    from helix.eval.recon import autocast
    from helix.model.checkpoint import load_run
    from helix.model.denoise import calibration_stats

    run = load_run(path, "best", device=device)
    model, cfg = run.model, run.tokenizer
    if not hasattr(model, "head"):
        raise ValueError(f"{path}: not a denoiser run (model type {run.config['model'].get('type')!r})")
    dev = next(model.parameters()).device
    dist = getattr(model.head, "dist", "point")
    t = lambda a: torch.as_tensor(a).to(dev)                            # noqa: E731

    def run_cells(ev, keys, y_true=None):
        idx, aux = cell_inputs(keys, ev.cell_key, ev.band_lengths, cfg)
        ok = (idx >= 0).any(1)
        out = [np.zeros(len(keys), np.float32) for _ in range(2 if y_true is None else 4)]
        if ok.any():
            nbr = cell_neighbors(keys[ok], ev.cell_key, ev.band_lengths, cfg)
            with torch.no_grad(), autocast(dev):
                o = model.outputs(ev.batch, t(idx[ok]), t(aux[ok]), t(nbr))
                got = [o[0].float(), torch.sigmoid(o[1].float())]
                if y_true is not None:
                    got += list(calibration_stats(model.head, o, t(np.asarray(y_true, np.float32)[ok])))
            for a, g in zip(out, got):
                a[ok] = g.float().cpu().numpy()
        return tuple(out)

    def predict(ev, keys):
        return run_cells(ev, keys)

    predict.provenance = run.provenance          # which checkpoint (``weights``) it read, for the JSON
    predict.dist = dist
    if dist != "point":
        predict.calib = run_cells                # (ev, keys, y_true) -> (y, presence, stat, width)
    return predict, cfg


class Denoise:
    """The ``denoise`` metric (see the module docstring): detection on the floor's windows
    (``charge``, ``presence``), and ``cells`` -- charge resolution, linearity, closure, the
    presence calibration and, for a distributional head, its distribution's calibration."""
    name = "denoise"
    #: Charged cells per predictor call, at most (each call encodes the event once).
    CHUNK = 200_000

    def __init__(self, ctx):
        self.ctx = ctx
        self.scored = Scored()
        self.presence_q = float(getattr(ctx, "presence_q", PRESENCE_Q))
        self.rows = dict(pred=[], true=[], view=[], ev=[])                 # charged covered cells
        self.cal = {}                                                      # name -> [(p, outcome, view)]
        self.calib = getattr(ctx.denoiser, "calib", None)                  # a distributional head's
        self.dist = dict(stat=[], width=[], err=[])                        # per charged covered cell

    def calibrations(self, y, presence, q):
        """``{name: (probability, outcome)}`` scored by :func:`binary_calibration` on the covered
        cells: the presence head against ``q > presence_q``. A head with its own distribution
        (P(q > t) for any t) adds its pairs here."""
        return {"presence": (presence, q > self.presence_q)}

    def add(self, ev):
        c = self.ctx
        if ev.index < c.n_train:
            return
        tr = window_flags(ev, ev.truth(c.seed, windows=True), c.run.tokenizer)   # kept / tok / shtok
        y, pr = c.denoiser(ev, tr["wkey"])
        self.scored.add(ev, tr, {"charge": np.expm1(np.maximum(y, 0)) * Q0, "presence": pr})

        cfg = c.run.tokenizer
        keys, q = np.asarray(ev.cells["key"], np.int64), np.asarray(ev.cells["q"], np.float64)
        ck, cq = keys[q > self.presence_q], q[q > self.presence_q]   # charged as the training's label
        cov = (cell_inputs(ck, ev.cell_key, ev.band_lengths, cfg)[0] >= 0).any(1)
        ck, cq = ck[cov], cq[cov]
        cy, cp = np.zeros(len(ck), np.float32), np.zeros(len(ck), np.float32)
        if self.calib is None:
            for s in range(0, len(ck), self.CHUNK):
                cy[s:s + self.CHUNK], cp[s:s + self.CHUNK] = c.denoiser(ev, ck[s:s + self.CHUNK])
        else:                                       # the same call, with the distribution's statistic
            yt = np.log1p(cq / Q0).astype(np.float32)
            st, wd = np.zeros(len(ck), np.float32), np.zeros(len(ck), np.float32)
            for s in range(0, len(ck), self.CHUNK):
                sl = slice(s, s + self.CHUNK)
                cy[sl], cp[sl], st[sl], wd[sl] = self.calib(ev, ck[sl], yt[sl])
            self.dist["stat"].append(st); self.dist["width"].append(wd); self.dist["err"].append(cy - yt)
        view = unkey(ck)[0] % 3
        r = self.rows
        r["pred"].append(np.expm1(np.maximum(cy, 0)) * Q0); r["true"].append(cq)
        r["view"].append(view); r["ev"].append(np.full(len(ck), ev.index))

        # calibration: the charged covered cells, and the windows' covered cells not among them
        wk, first = np.unique(tr["wkey"], return_index=True)
        wcov = (cell_inputs(wk, ev.cell_key, ev.band_lengths, cfg)[0] >= 0).any(1) & ~np.isin(wk, ck)
        pos = np.clip(np.searchsorted(keys, wk), 0, max(len(keys) - 1, 0))
        wq = np.where(keys[pos] == wk, q[pos], 0.0) if len(keys) else np.zeros(len(wk))
        sel = first[wcov]
        allq = np.concatenate([cq, wq[wcov]])
        ally, allp = np.concatenate([cy, y[sel]]), np.concatenate([cp, pr[sel]])
        allv = np.concatenate([view, unkey(wk[wcov])[0] % 3])
        for name, (p, o) in self.calibrations(ally, allp, allq).items():
            self.cal.setdefault(name, []).append((np.asarray(p, np.float32), np.asarray(o, bool), allv))

    def result(self):
        c = self.ctx
        out = dict(n_train_events=c.n_train, **self.scored.result(c.n_boot, c.seed))
        out["_windows"] = self.scored.windows()          # popped by the CLI into <out>.windows.npz
        r = {k: np.concatenate(v) if v else np.zeros(0) for k, v in self.rows.items()}
        cal = {}
        for name, parts in self.cal.items():
            p, o, v = (np.concatenate([x[i] for x in parts]) for i in range(3))
            cal[name] = dict(all=binary_calibration(p, o),
                             **{VIEWS[j]: binary_calibration(p[v == j], o[v == j]) for j in range(3)})
        out["cells"] = dict(n_cells=int(len(r["true"])), presence_q=self.presence_q, q_floor=Q_FLOOR,
                            calibration=cal, by_view=charge_accuracy(r["pred"], r["true"], r["view"], r["ev"]))
        if self.calib is not None:
            d = {k: np.concatenate(v) if v else np.zeros(0) for k, v in self.dist.items()}
            out["cells"]["dist"] = dist_calibration(c.denoiser.dist, d["stat"], d["width"], d["err"],
                                                    r["true"], r["view"])
        return out
