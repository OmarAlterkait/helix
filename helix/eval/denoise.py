"""The evaluation of a fine-tuned denoiser: detection (on the windows and per cell), charge, and
the presence probability's calibration.

The primary evaluation of a pretrained encoder is a fixed-budget FINE-TUNE (``configs/pimm/
ft_eval.py``: the denoiser trained on 1,024 labelled events), scored here; the same protocol
with ``model.freeze_encoder=True`` is the frozen diagnostic. One scoring path for both.

The metric needs only a predictor, ``predict(event, keys) -> (y, logit[, q_mean[, y_sd]])`` per
fine-cell key: ``y`` the point estimate of ``log1p(q / Q0)`` given charge, ``logit`` the presence
log-odds, ``q_mean`` the MEAN charge (e-) given charge (default ``Q0 expm1(y)``) and ``y_sd`` the
predictive width of y given charge (NaN for a point head). A cell no token covers is ``y = 0``,
``logit =`` :data:`UNCOVERED_LOGIT`. :func:`load_denoiser` makes one from a denoiser run
(``configs/pimm/denoise.py``); ``predict.conditional`` False marks a head whose charge is
regressed on every cell rather than given charge (``cond_charge=False``).

Definitions. ``q`` is a cell's truth charge (e-); a cell is CHARGED when ``q > presence_q`` (the
training's presence label, 25 e-; the simulation's own per-sample cut on d1m); ``p =
sigmoid(logit)``. Per cell, two charge estimates and one comparison:

  call        ``q_call = Q0 expm1(y)`` where ``p > CALL_P`` (0.5, the presence decision), else 0:
              what a reconstruction that keeps the cells it believes charged reports
  expectation ``q_exp = p q_mean`` (``q_mean`` alone for an unconditional head): the cell's
              expected charge, whose sum over cells is the predicted total. ``q_mean`` comes
              from the head's distribution where it has one (cat: the mean of ``Q0 expm1(y)``
              over its bins, y uniform within a bin; gauss: ``Q0 (exp(mu + s^2 / 2) - 1)``),
              not ``Q0 expm1(E[y])``, which is low by Jensen on a wide faint cell
  log ratio   ``log(max(pred, F) / max(q, F))``, ``F = presence_q``: a charge at or below the
              label's own zero counts as it, so a miss and a false call are finite

The NATURAL COVERED CELLS are every fine cell inside a token's footprint (:func:`covered_cells`);
6.5% of them are charged on d1m. They are represented by every charged covered cell (weight 1)
and a uniform sample of ``empty_frac`` of the others (weight ``1 / empty_frac``, drawn with
``[seed, event, 1]``). No window selection enters a per-cell number.

Detection
  windows     (:func:`helix.eval.events.window_stat`) two scores on identical windows:
              ``presence`` -- the window's largest presence LOG-ODDS (not the probability: a
              confident head saturates P to 1.0 in float32, and no window exceeds a threshold of
              1.0) -- and ``amplitude``, the largest kept coefficient's |value| / sigma (fit-free,
              what the corpus gate alone gives). ``headline``: the LAUC of both matched pairs at
              0.1-0.2 MeV, view-mean, and the old 1%-FPR efficiency beside it
  per cell    ``cells.detection``: the average precision of the presence log-odds on the
              natural covered cells (:func:`average_precision`), positives the charged cells,
              against all the others (``ap``) and against each distance stratum of them
              (:data:`STRATA`: the Chebyshev distance in cells to the nearest charged cell of the
              plane, ``adjacent`` 1, ``near`` 2..8 -- the near windows' box -- ``far`` beyond),
              each with its ``prevalence`` (the AP of an uninformative score); pooled over views
              and per view

Charge (``cells.charge``, pooled over views and per view)
  pred_bins   THE charge accuracy: in bins of the CALL (:data:`PRED_BINS`), over the natural
              covered cells called charged (charged or not, weighted): ``bias`` = the median and
              ``resolution`` = (p84 - p16) / 2 of the log ratio; ``frac_empty`` the bin's weight
              share of cells that are not charged (false calls, which the log ratio prices at F)
              and ``mean_true_over_mean_pred`` = sum q / sum q_mean over the bin
  truth_bins  a DIAGNOSTIC of the charge head alone, on every charged cell, covered or not (the
              selection is the truth's, so it is not an accuracy): per truth bin
              (:data:`TRUTH_BINS`, from the label's 25 e-) ``bias`` / ``resolution`` of
              ``log(max(Q0 expm1(y), F) / q)`` -- the charge given presence, whatever ``p``, 0 if
              uncovered -- with ``frac_detected`` (``p > CALL_P``), ``frac_uncovered`` (no token),
              ``mean_pred_over_true`` (sum q_mean / sum q) and the COVERAGE CEILING
              ``charge_frac_no_kept``: the share of the bin's charge in cells holding no kept
              coefficient
  closure     sum of the expected charge over ALL cells / the events' whole truth charge (every
              truth cell, charged or not): ``pooled`` (sums over events, with its interval),
              split into ``on_charged`` and ``on_empty`` (the haze), ``median`` over events, and
              ``point_pooled``, the same with ``q_mean`` replaced by ``Q0 expm1(y)``

Calibration (``cells.calibration``) of ``p`` against "charged" on the natural covered cells,
importance-weighted: ``raw`` (:func:`binary_calibration`: ECE over 15 equal-width bins, Brier,
log-loss), and one global log-odds ``offset`` fitted by weighted log-loss on the events at even
positions, stored, with ``eval_half`` the events at odd positions scored raw and corrected. A
denoiser trains on rows ~35% charged against 6.5% here, so its raw probabilities are
prior-shifted by construction; the offset says how much of the miscalibration that shift is.
The per-cell detection and calibration read the log-odds quantised to :data:`LOGIT_STEP` (per-event
histograms, so a bootstrap resample is a weighted sum).

Intervals: central ``level`` (95%) over an event bootstrap that resamples the same events for the
windows, the per-cell detection and the closure; the charge tables are point values.

``dump_cells``: a directory each event's per-cell outputs are written to (:meth:`Denoise.dump`).

The truth: cells-3 ``q`` is the hits file's electrons, cut per sample by the simulation (25 e- on
d1m); the output records the cut (``truth`` in :func:`helix.eval.evaluate`'s JSON). One noise
realisation per event, the corpus's own, as on real data: nothing is redrawn.
"""

from __future__ import annotations

import os

import numpy as np

from helix.data.cells import Q0, Grid, fkey, unkey
from helix.model.cells_join import cell_inputs, cell_neighbors, covering_tokens
from helix.eval.events import HEADLINE, LEVEL, PAD, SECONDARY, VIEWS, Scored, amplitude, bootstrap, cell_charge

#: Truth-charge bins of the diagnostic (e-), from the presence label; the last is open.
TRUTH_BINS = (25.0, 1e3, 4e3, 16e3, 64e3, 256e3, np.inf)
#: Bins of the CALLED charge (e-) of the charge accuracy, from the presence label; the last is open.
PRED_BINS = (25.0, 100.0, 500.0, 1e3, 2e3, 4e3, 16e3, 64e3, np.inf)
#: The presence label's charge (e-): ``q > PRESENCE_Q``, the denoiser's training definition.
PRESENCE_Q = 25.0
#: The presence decision: a cell is called charged when ``p > CALL_P``.
CALL_P = 0.5
#: The share of the uncharged covered cells sampled per event (each weighted by its inverse).
EMPTY_FRAC = 0.05
#: The near stratum's reach (cells, Chebyshev): the near windows' +-16 wires / +-128 ticks.
#: The distance strata of the uncharged covered cells: the adjacent cell, then within the near
#: windows' box (:data:`helix.eval.events.PAD`, wires x ticks, so it is the same physical box at
#: every truth grid), then beyond.
STRATA = ("adjacent", "near", "far")
#: The log-odds quantisation of the per-cell detection and calibration, and its clipped range.
LOGIT_STEP, LOGIT_MAX = 0.025, 40.0
LOGIT_EDGES = np.arange(-LOGIT_MAX, LOGIT_MAX + LOGIT_STEP / 2, LOGIT_STEP)
LOGIT_CENTERS = np.concatenate([[-LOGIT_MAX], (LOGIT_EDGES[1:] + LOGIT_EDGES[:-1]) / 2, [LOGIT_MAX]])
ECE_BINS = 15
#: The presence log-odds of a cell no token covers: below any head's, and finite, so quantiles hold.
UNCOVERED_LOGIT = -1e6
#: Bits of a dumped cell's ``flags``.
F_CHARGED, F_COVERED, F_KEPT, F_WINDOW, F_SAMPLED = 1, 2, 4, 8, 16
_F_CALLED = 32                       # in the charged-cell table only


def _bin_names(edges):
    f = lambda x: f"{x / 1e3:g}k" if x >= 1e3 else f"{x:g}"          # noqa: E731
    return tuple(f"{f(a)}-{f(b)}" if np.isfinite(b) else f"{f(a)}+" for a, b in zip(edges[:-1], edges[1:]))


TRUTH_NAMES, PRED_NAMES = _bin_names(TRUTH_BINS), _bin_names(PRED_BINS)


def sigmoid(x):
    """The probability of log-odds ``x``, float64, without overflow."""
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, np.float64), -700, 700)))


def binary_calibration(p, y, w=None, n_bins=ECE_BINS, eps=1e-7):
    """Of probabilities ``p`` against 0/1 outcomes ``y``, rows weighted by ``w`` (default 1):
    ``ece`` (sum over ``n_bins`` equal-width bins of the bin's weight share times |mean p -
    mean y|), ``brier`` (mean (p - y)^2), ``log_loss`` (p clipped to [eps, 1 - eps]), ``n`` (the
    total weight), ``frac_true`` and ``mean_p``."""
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    w = np.ones(len(p)) if w is None else np.asarray(w, np.float64)
    W = w.sum()
    if not W > 0:
        return dict(n=0, frac_true=float("nan"), mean_p=float("nan"), ece=float("nan"), brier=float("nan"),
                    log_loss=float("nan"))
    b = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    gap = np.abs(np.bincount(b, w * (p - y), n_bins)).sum() / W
    pc = np.clip(p, eps, 1 - eps)
    r = lambda x: round(float(x), 6)                                     # noqa: E731
    return dict(n=r(W), frac_true=r((w * y).sum() / W), mean_p=r((w * p).sum() / W), ece=r(gap),
                brier=r((w * (p - y) ** 2).sum() / W),
                log_loss=r(-(w * (y * np.log(pc) + (1 - y) * np.log(1 - pc))).sum() / W))


def hist_calibration(hpos, hneg, offset=0.0):
    """:func:`binary_calibration` of log-odds histograms (weight per :data:`LOGIT_CENTERS` bin of
    the charged and of the uncharged cells), each bin at its centre plus ``offset``."""
    p = sigmoid(LOGIT_CENTERS + offset)
    return binary_calibration(np.concatenate([p, p]), np.r_[np.ones(len(p)), np.zeros(len(p))],
                              np.concatenate([hpos, hneg]))


def fit_logit_offset(hpos, hneg, iters=100):
    """The single log-odds offset ``b`` minimising the weighted log-loss of ``sigmoid(logit +
    b)`` on log-odds histograms (Newton on the intercept, steps capped at 5: the loss is convex
    in ``b``)."""
    b, W = 0.0, hpos + hneg
    for _ in range(iters):
        p = sigmoid(LOGIT_CENTERS + b)
        g, h = (W * p - hpos).sum(), (W * p * (1 - p)).sum()
        if not h > 0:
            break
        step = float(np.clip(g / h, -5, 5))
        b -= step
        if abs(step) < 1e-10:
            break
    return b


def average_precision(hpos, hneg):
    """The average precision of a score from its histograms over ascending score bins (``hpos``
    the positives' weight per bin, ``hneg`` the negatives'): the sum over thresholds, from the
    top, of the recall gained times the precision there (sklearn's step definition); a bin's
    cells are tied, one threshold. NaN without positives."""
    tp, fp = np.cumsum(hpos[::-1]), np.cumsum(hneg[::-1])
    P = tp[-1] if len(tp) else 0.0
    if not P > 0:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    return float((hpos[::-1] * prec).sum() / P)


def covered_cells(ev, cfg):
    """The natural covered cells of an event, sorted: every fine cell of a token's footprint in
    its band -- the token's wires, and the ticks its taus round from, within one cell -- for which
    :func:`helix.model.cells_join.covering_tokens` (``cell_inputs``' own lookup) finds a token.
    A cell beyond every footprint (before or past the readout, which ``tau_of_tick``'s clipping
    would attach to an edge token) is not a natural cell."""
    from helix.model.tokenize import tick_of_tau, unpack_cell_key
    g, b, wb, tb = unpack_cell_key(np.asarray(ev.cell_key, np.int64))
    if not len(g):
        return np.zeros(0, np.int64)
    FW, FT = ev.grid.fw, ev.grid.ft                   # the truth's own grid
    nwc = -(-cfg.pw // FW)
    cand = []
    for band in np.unique(b).tolist():
        s = b == band
        gg = g[s]
        lo = tick_of_tau(tb[s] * cfg.pt - 0.5, gg, np.full(len(gg), band), cfg)
        nt = int(np.ceil(cfg.pt * (1 << int(cfg.lev[band])) / FT)) + 3      # the span, +-1 cell
        ft0 = np.floor(lo / FT).astype(np.int64) - 1
        fw0 = wb[s] * cfg.pw // FW
        sh = (len(gg), nwc, nt)
        cand.append(fkey(np.broadcast_to(gg[:, None, None], sh),
                         np.broadcast_to(fw0[:, None, None] + np.arange(nwc)[None, :, None], sh),
                         np.broadcast_to(ft0[:, None, None] + np.arange(nt)[None, None, :], sh)).ravel())
    cand = np.unique(np.concatenate(cand))
    keep = np.zeros(len(cand), bool)
    for s in range(0, len(cand), 500_000):
        keep[s:s + 500_000] = (covering_tokens(cand[s:s + 500_000], ev.cell_key, ev.band_lengths, cfg, ev.grid) >= 0).any(1)
    return cand[keep]


def activity_class(keys, pos_keys, grid, near=None):
    """Per cell of ``keys`` (on ``grid``): 0 if it is one of ``pos_keys`` (the charged cells),
    else its distance stratum to the nearest of them on its plane -- 1 ``adjacent`` (within one
    cell), 2 ``near`` (within ``near`` = (wires, ticks), default :data:`~helix.eval.events.PAD`),
    3 ``far`` (beyond, or none on the plane)."""
    grid = Grid(*grid)
    rw, rt = grid.cells(*(PAD if near is None else near))
    r = max(rw, rt)
    keys, pos_keys = np.asarray(keys, np.int64), np.asarray(pos_keys, np.int64)
    out = np.full(len(keys), 3, np.int8)
    if not len(keys) or not len(pos_keys):
        return out
    g, w, t = unkey(keys)
    pg, pw, pt = unkey(pos_keys)
    for plane in np.unique(g).tolist():
        m, pm = g == plane, pg == plane
        if not pm.any():
            continue
        w0 = int(min(w[m].min(), pw[pm].min())) - r - 1
        t0 = int(min(t[m].min(), pt[pm].min())) - r - 1
        shape = (int(max(w[m].max(), pw[pm].max())) - w0 + r + 2, int(max(t[m].max(), pt[pm].max())) - t0 + r + 2)
        occ = np.zeros(shape, np.int32)
        occ[pw[pm] - w0, pt[pm] - t0] = 1
        S = np.pad(occ.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        i, j = w[m] - w0, t[m] - t0

        def box(a, b):
            return S[i + a + 1, j + b + 1] - S[i - a, j + b + 1] - S[i + a + 1, j - b] + S[i - a, j - b]
        out[m] = np.where(occ[i, j] > 0, 0, np.where(box(1, 1) > 0, 1, np.where(box(rw, rt) > 0, 2, 3)))
    return out


def wsummary(x, w=None):
    """``n`` (rows), ``n_w`` (their weight), ``bias`` (the weighted median) and ``resolution``
    ((p84 - p16) / 2) of ``x``; a weighted percentile is the smallest value whose cumulative
    weight share reaches it."""
    x = np.asarray(x, np.float64)
    if not len(x):
        return dict(n=0, n_w=0.0, bias=float("nan"), resolution=float("nan"))
    w = np.ones(len(x)) if w is None else np.asarray(w, np.float64)
    o = np.argsort(x, kind="stable")
    xs, cw = x[o], np.cumsum(w[o])
    p16, p50, p84 = xs[np.minimum(np.searchsorted(cw, np.array([0.16, 0.5, 0.84]) * cw[-1]), len(xs) - 1)]
    return dict(n=int(len(x)), n_w=round(float(cw[-1]), 3), bias=round(float(p50), 6),
                resolution=round(float(p84 - p16) / 2, 6))


def _ratio(a, b):
    return round(float(a / b), 6) if b > 0 else float("nan")


def charge_tables(chg, emp, w_emp, floor):
    """``{"all"|"U"|"V"|"Y": {pred_bins, truth_bins, n_charged, frac_uncovered,
    charge_frac_no_kept}}`` (module docstring); a table is a list of rows in bin order, each
    naming its ``bin`` and edges ``lo`` / ``hi`` (e-). ``chg``: every charged cell -- ``q``, ``qp``
    (``Q0 expm1(y)``, 0 if uncovered), ``qm`` (q_mean, 0 if uncovered), ``flags``
    (:data:`F_COVERED`, :data:`F_KEPT`, and ``_F_CALLED``), ``view``; ``emp``: the sampled
    uncharged covered cells that are CALLED -- ``q``, ``qp``, ``qm``, ``view`` -- each of weight
    ``w_emp``; ``floor``: F of the log ratio."""
    q, qp, qm = (np.asarray(chg[k], np.float64) for k in ("q", "qp", "qm"))
    fl, v = np.asarray(chg["flags"]), np.asarray(chg["view"])
    cov, kept, called = (fl & F_COVERED) > 0, (fl & F_KEPT) > 0, (fl & _F_CALLED) > 0
    eq, eqp, eqm = (np.asarray(emp[k], np.float64) for k in ("q", "qp", "qm"))
    lr_t = np.log(np.maximum(qp, floor) / np.maximum(q, floor))
    tb = np.searchsorted(TRUTH_BINS, q, side="right") - 1
    # the calls: charged cells called (weight 1) and sampled uncharged cells called (weight w_emp)
    cq, cqp, cqm = (np.concatenate([a[called], b]) for a, b in ((q, eq), (qp, eqp), (qm, eqm)))
    cv = np.concatenate([v[called], np.asarray(emp["view"])])
    cw = np.concatenate([np.ones(int(called.sum())), np.full(len(eq), float(w_emp))])
    lr_p = np.log(np.maximum(cqp, floor) / np.maximum(cq, floor))
    pb = np.searchsorted(PRED_BINS, cqp, side="right") - 1
    out = {}
    for name, view in [("all", None)] + [(VIEWS[j], j) for j in range(3)]:
        tsel = np.ones(len(q), bool) if view is None else v == view
        psel = np.ones(len(cq), bool) if view is None else cv == view
        truth = []
        for i, nm in enumerate(TRUTH_NAMES):
            s = tsel & (tb == i)
            truth.append(dict(bin=nm, lo=TRUTH_BINS[i], hi=TRUTH_BINS[i + 1], **wsummary(lr_t[s]), frac_detected=_ratio(called[s].sum(), s.sum()),
                             frac_uncovered=_ratio((~cov[s]).sum(), s.sum()),
                             mean_pred_over_true=_ratio(qm[s].sum(), q[s].sum()),
                             charge_frac_no_kept=_ratio(q[s & ~kept].sum(), q[s].sum())))
        pred = []
        for i, nm in enumerate(PRED_NAMES):
            s = psel & (pb == i)
            pred.append(dict(bin=nm, lo=PRED_BINS[i], hi=PRED_BINS[i + 1], **wsummary(lr_p[s], cw[s]),
                             frac_empty=_ratio((cw[s] * (cq[s] <= floor)).sum(), cw[s].sum()),
                             mean_true_over_mean_pred=_ratio((cw[s] * cq[s]).sum(), (cw[s] * cqm[s]).sum())))
        out[name] = dict(pred_bins=pred, truth_bins=truth, n_charged=int(tsel.sum()),
                         frac_uncovered=_ratio((~cov[tsel]).sum(), tsel.sum()),
                         charge_frac_no_kept=_ratio(q[tsel & ~kept].sum(), q[tsel].sum()))
    return out


def head_charge(head, out):
    """``(q_mean, y_sd)`` per cell from a :class:`helix.model.denoise.CellDecoder`'s output, as
    float64 tensors: the mean charge (e-) given charge from the head's distribution -- cat: the
    mean of ``Q0 expm1(y)`` over its bins 1..K, y uniform within a bin; gauss: ``Q0 (exp(mu +
    s^2 / 2) - 1)``; point: ``Q0 expm1(y)``, having none -- and the predictive std of y given
    charge (NaN for point)."""
    import torch
    from helix.model.denoise import LOG_SIGMA_RANGE
    dist = getattr(head, "dist", "point")
    y = out[0].double()
    if dist == "cat":
        e = head.edges.double()
        a, b = e[:-1], e[1:]
        bin_mean = Q0 * ((torch.exp(b) - torch.exp(a)) / (b - a) - 1.0)          # (K,)
        pc = out[2].double()[:, 1:].softmax(-1)
        c = head.centers.double()
        m = (pc * c).sum(-1, keepdim=True)
        return (pc * bin_mean).sum(-1), (pc * (c - m) ** 2).sum(-1).sqrt()
    if dist == "gauss":
        s = torch.exp(out[2].double().clamp(*LOG_SIGMA_RANGE))
        return Q0 * torch.expm1(y + s * s / 2), s
    return Q0 * torch.expm1(y.clamp(min=0)), torch.full_like(y, float("nan"))


def load_denoiser(path, device=None, *, weights="last", fm=None):
    """``(predict, tokenizer)`` from a denoiser run (``configs/pimm/denoise.py``): its run
    directory -- ``weights`` ``"last"`` (the end of the fixed budget, default) or ``"best"`` (the
    lowest validation loss) -- or one checkpoint in it (:func:`helix.model.checkpoint.load_run`;
    ``fm`` relocates its FM run). ``predict(ev, keys) -> (y, logit, q_mean, y_sd)`` (module
    docstring); the event is encoded once (:meth:`helix.eval.events.Event.features`, cached for
    this encoder) however many calls read it. ``predict.provenance`` is that run's,
    ``predict.run`` the run itself (its config names the corpus), ``predict.dist`` its charge
    head, ``predict.conditional`` whether its charge is given presence."""
    import torch
    from helix.eval.recon import autocast
    from helix.model.checkpoint import load_run

    run = load_run(path, weights, device=device, fm=fm)
    model, cfg = run.model, run.tokenizer
    if not hasattr(model, "head"):
        raise ValueError(f"{path}: not a denoiser run (model type {run.config['model'].get('type')!r})")
    dev = next(model.parameters()).device
    t = lambda a: torch.as_tensor(a).to(dev)                            # noqa: E731

    def predict(ev, keys, grid=None):
        """``grid``: the grid ``keys`` are on, default the event truth's."""
        keys = np.asarray(keys, np.int64)
        grid = ev.grid if grid is None else Grid(*grid)
        idx, aux = cell_inputs(keys, ev.cell_key, ev.band_lengths, cfg, grid)
        ok = (idx >= 0).any(1)
        y = np.zeros(len(keys), np.float32)
        logit = np.full(len(keys), UNCOVERED_LOGIT, np.float32)
        qm, sd = np.zeros(len(keys)), np.full(len(keys), np.nan)
        if ok.any():
            nbr = cell_neighbors(keys[ok], ev.cell_key, ev.band_lengths, cfg, grid)
            feats = ev.features(model.fm)                            # what DenoiseModel.outputs encodes
            with torch.no_grad(), autocast(dev):
                out = model.head(feats, t(idx[ok]), t(aux[ok]), t(nbr), ev.batch)
            y[ok], logit[ok] = out[0].float().cpu().numpy(), out[1].float().cpu().numpy()
            m, s = head_charge(model.head, out)
            qm[ok], sd[ok] = m.cpu().numpy(), s.cpu().numpy()
        return y, logit, qm, sd

    predict.provenance = run.provenance          # which checkpoint (``weights``) it read, for the JSON
    predict.run = run                            # its config names the corpus and the events
    predict.dist = getattr(model.head, "dist", "point")
    predict.conditional = bool(getattr(model, "cond_charge", True))
    return predict, cfg


def sub_cells(keys, coarse, fine):
    """``(n, m)``: the keys, on the ``fine`` grid, of the ``m`` sub-cells of each cell of ``keys``
    (on the ``coarse`` grid, each side a multiple of ``fine``'s), wire-major."""
    coarse, fine = Grid(*coarse), Grid(*fine)
    if coarse.fw % fine.fw or coarse.ft % fine.ft:
        raise ValueError(f"a {coarse} cell is not a whole number of {fine} cells")
    nw, nt = coarse.fw // fine.fw, coarse.ft // fine.ft
    g, w, t = unkey(np.asarray(keys, np.int64))
    i, j = np.meshgrid(np.arange(nw), np.arange(nt), indexing="ij")
    return fkey(g[:, None], w[:, None] * nw + i.ravel()[None], t[:, None] * nt + j.ravel()[None])


def _log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def coarsen(predict, fine, chunk=200_000):
    """A predictor on a truth grid coarser than ``fine``, the grid ``predict`` was trained on (its
    run's truth): each cell's prediction is aggregated from its sub-cells' (:func:`sub_cells`),
    treating them as independent given the event --

      presence  noisy-OR, ``P = 1 - prod(1 - p_i)``, returned as its log-odds (exact at any p_i)
      q_mean    the mean of the summed charge given the cell is charged, ``sum p_i q_mean_i / P``
      y         the same with each sub-cell's point charge ``Q0 expm1(y_i)``, as ``log1p(. / Q0)``
      y_sd      NaN (the sum has no width here)

    so the cell's expected charge ``P q_mean`` is its sub-cells' summed expected charge, and the
    closure is the fine predictor's. A sub-cell no token covers is ``p_i = 0``; a cell none of
    whose sub-cells is covered is uncovered (``y = 0``, :data:`UNCOVERED_LOGIT`). On ``fine``
    itself it is ``predict``. Conditional heads only: an unconditional head's charge is not given
    presence, and these sums would not hold. ``chunk``: sub-cells per ``predict`` call, at most."""
    fine = Grid(*fine)
    if not getattr(predict, "conditional", True):
        raise ValueError("coarsen: the head's charge is not given presence (cond_charge=False), so its "
                         "sub-cells do not aggregate as a hurdle")

    def coarse(ev, keys, grid=None):
        grid = ev.grid if grid is None else Grid(*grid)
        keys = np.asarray(keys, np.int64)
        if grid == fine:
            return predict(ev, keys, grid=fine)
        sk = sub_cells(keys, grid, fine)
        n, m = sk.shape
        flat = sk.ravel()
        parts = [predict(ev, flat[s:s + chunk], grid=fine) for s in range(0, len(flat), chunk)]
        y, lg, qm = (np.concatenate([np.asarray(p[i], np.float64) for p in parts]).reshape(n, m) if parts
                     else np.zeros((n, m)) for i in range(3))
        cov = lg > UNCOVERED_LOGIT
        anyc = cov.any(1)
        lp = np.where(cov, _log_sigmoid(lg), -np.inf)               # log p_i
        s = np.where(cov, _log_sigmoid(-lg), 0.0).sum(1)            # log(1 - P) = sum log(1 - p_i)
        with np.errstate(divide="ignore", invalid="ignore"):
            # log P: log(-expm1(s)); where 1 - P rounds to 1, log(sum p_i), its limit
            logp = np.where(s < 0, np.log(-np.expm1(np.minimum(s, -1e-300))), np.logaddexp.reduce(lp, 1))
            w = np.where(cov, np.exp(lp - logp[:, None]), 0.0)     # p_i / P
        w = np.where(anyc[:, None], w, 0.0)
        qp = Q0 * np.expm1(np.maximum(y, 0.0))
        out_y = np.where(anyc, np.log1p((w * qp).sum(1) / Q0), 0.0).astype(np.float32)
        out_l = np.where(anyc, logp - s, UNCOVERED_LOGIT).astype(np.float32)
        out_q = np.where(anyc, (w * qm).sum(1), 0.0)
        return out_y, out_l, out_q, np.full(n, np.nan)

    for k in ("provenance", "run", "dist", "conditional"):
        if hasattr(predict, k):
            setattr(coarse, k, getattr(predict, k))
    coarse.grid = fine
    return coarse


class Denoise:
    """The denoiser's metric (module docstring): detection on the windows (``presence``,
    ``amplitude``, ``headline``) and ``cells`` -- per-cell detection, charge, closure and the
    presence calibration on the natural covered cells."""
    name = "denoise"
    #: Cells per predictor call, at most.
    CHUNK = 200_000

    def __init__(self, ctx):
        self.ctx = ctx
        self.scored = Scored()
        self.presence_q = float(getattr(ctx, "presence_q", PRESENCE_Q))
        self.empty_frac = float(getattr(ctx, "empty_frac", EMPTY_FRAC))
        if not 0 < self.empty_frac <= 1:
            raise ValueError(f"empty_frac={self.empty_frac}: a share of the uncharged cells, in (0, 1]")
        self.dump_dir = getattr(ctx, "dump_cells", None)
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)
        self.hist = []         # per event: (3 views, 1 + len(STRATA), bins) weight per log-odds bin
        self.sums = []         # per event: (3 views, 5) truth; q_exp on charged, on the rest; point same
        self.chg = dict(q=[], qp=[], qm=[], flags=[], view=[])
        self.emp = dict(q=[], qp=[], qm=[], view=[])

    def _predict(self, ev, keys):
        """The predictor on ``keys`` in chunks -> ``(y, logit, q_mean, y_sd)`` (the module
        docstring's defaults for a predictor returning fewer)."""
        keys = np.asarray(keys, np.int64)
        n = len(keys)
        y, lg = np.zeros(n, np.float32), np.full(n, UNCOVERED_LOGIT, np.float32)
        qm, sd = np.zeros(n), np.full(n, np.nan)
        for s in range(0, n, self.CHUNK):
            r = self.ctx.denoiser(ev, keys[s:s + self.CHUNK])
            sl = slice(s, s + self.CHUNK)
            y[sl], lg[sl] = r[0], r[1]
            qm[sl] = r[2] if len(r) > 2 else Q0 * np.expm1(np.maximum(np.asarray(r[0], np.float64), 0))
            if len(r) > 3:
                sd[sl] = r[3]
        return y, lg, qm, sd

    def add(self, ev):
        c = self.ctx
        cfg, pq, f = c.run.tokenizer, self.presence_q, self.empty_frac
        cond = bool(getattr(c.denoiser, "conditional", True))
        tr = ev.truth(c.seed)
        win = self._predict(ev, tr["wkey"])
        self.scored.add(ev, tr, {"presence": win[1], "amplitude": amplitude(ev, tr["wkey"])})

        K = covered_cells(ev, cfg)
        keys, q = np.asarray(ev.cells["key"], np.int64), np.asarray(ev.cells["q"], np.float64)
        pos = q > pq
        ck, cq = keys[pos], q[pos]
        cov, kept = np.isin(ck, K), np.isin(ck, ev.coeff_cells)
        y, lg = np.zeros(len(ck), np.float32), np.full(len(ck), UNCOVERED_LOGIT, np.float32)
        qm, sd = np.zeros(len(ck)), np.full(len(ck), np.nan)
        if cov.any():
            y[cov], lg[cov], qm[cov], sd[cov] = self._predict(ev, ck[cov])
        p = np.where(cov, sigmoid(lg), 0.0)
        qp = np.where(cov, Q0 * np.expm1(np.maximum(y.astype(np.float64), 0)), 0.0)
        qm = np.where(cov, qm, 0.0)
        called = cov & (p > CALL_P)
        cview = (unkey(ck)[0] % 3).astype(np.int64)

        # a uniform sample of the covered cells without charge
        ek = K[~np.isin(K, ck)]
        sk = ek[np.random.default_rng([c.seed, ev.index, 1]).random(len(ek)) < f]
        sy, sl, sqm, ssd = self._predict(ev, sk)
        sq, sp = cell_charge(ev.cells, sk), sigmoid(sl)
        scls = activity_class(sk, ck, ev.grid)
        sview = (unkey(sk)[0] % 3).astype(np.int64)
        sqp = Q0 * np.expm1(np.maximum(sy.astype(np.float64), 0))

        nb, ns = len(LOGIT_CENTERS), 1 + len(STRATA)
        H = np.bincount((cview[cov] * ns) * nb + np.searchsorted(LOGIT_EDGES, lg[cov], side="right"),
                        minlength=3 * ns * nb).astype(np.float64)
        H += np.bincount((sview * ns + scls) * nb + np.searchsorted(LOGIT_EDGES, sl, side="right"),
                         minlength=3 * ns * nb) / f
        self.hist.append(H.reshape(3, ns, nb).astype(np.float32))

        ex_c, ex_s = (p * qm, sp * sqm) if cond else (qm, sqm)
        pt_c, pt_s = (p * qp, sp * sqp) if cond else (qp, sqp)
        bc = lambda v, x: np.bincount(v, x, 3)                                    # noqa: E731
        self.sums.append(np.stack([bc(unkey(keys)[0] % 3, q), bc(cview, ex_c), bc(sview, ex_s) / f,
                                   bc(cview, pt_c), bc(sview, pt_s) / f], 1))

        flags = (cov * F_COVERED | kept * F_KEPT | called * _F_CALLED).astype(np.uint8)
        for k, x in (("q", cq), ("qp", qp), ("qm", qm)):
            self.chg[k].append(x.astype(np.float32))
        self.chg["flags"].append(flags)
        self.chg["view"].append(cview.astype(np.uint8))
        sc = sp > CALL_P
        for k, x in (("q", sq), ("qp", sqp), ("qm", sqm)):
            self.emp[k].append(x[sc].astype(np.float32))
        self.emp["view"].append(sview[sc].astype(np.uint8))

        if self.dump_dir:
            self.dump(ev, tr, K, (ck, cq, y, lg, qm, sd, cov), (sk, sq, sy, sl, sqm, ssd, scls), win)

    def dump(self, ev, tr, K, charged, sampled, win):
        """Write ``<dump_cells>/ev<index>.npz``: the event's charged cells, its window cells and
        its sample of uncharged covered cells, one row per cell -- ``key``, ``q`` (truth e-),
        ``y``, ``logit``, ``q_mean``, ``y_sd`` (the predictor's), ``w`` (the natural covered
        cells' weight: 1 for a charged covered cell, ``1 / empty_frac`` for a sampled one, 0 for a
        row outside that population -- an uncovered charged cell, a window cell neither charged
        nor sampled), ``flags`` (:data:`F_CHARGED` | :data:`F_COVERED` | :data:`F_KEPT` |
        :data:`F_WINDOW` | :data:`F_SAMPLED`) and ``dclass`` (:func:`activity_class`) -- and the
        event's ``index``, ``n_covered``, ``q_total`` (all its truth charge, per view),
        ``presence_q`` and ``empty_frac``."""
        ck, cq, y, lg, qm, sd, cov = charged
        sk, sq, sy, sl, sqm, ssd, scls = sampled
        wy, wl, wqm, wsd = win
        wk, first = np.unique(tr["wkey"], return_index=True)
        rest = ~np.isin(wk, ck) & ~np.isin(wk, sk)
        rk, ri = wk[rest], first[rest]
        key = np.concatenate([ck, sk, rk])
        fl = np.zeros(len(key), np.uint8)
        fl[:len(ck)] |= F_CHARGED
        fl[np.isin(key, K)] |= F_COVERED
        fl[np.isin(key, np.asarray(ev.coeff_cells, np.int64))] |= F_KEPT
        fl[np.isin(key, wk)] |= F_WINDOW
        fl[len(ck):len(ck) + len(sk)] |= F_SAMPLED
        w = np.concatenate([cov.astype(np.float32), np.full(len(sk), 1.0 / self.empty_frac, np.float32),
                            np.zeros(len(rk), np.float32)])
        cat = lambda a, b, c: np.concatenate([a, b, c[ri]]).astype(np.float32)   # noqa: E731
        allk = np.asarray(ev.cells["key"], np.int64)
        np.savez_compressed(os.path.join(self.dump_dir, f"ev{ev.index:06d}.npz"),
                            key=key, q=np.concatenate([cq, sq, cell_charge(ev.cells, rk)]).astype(np.float32),
                            y=cat(y, sy, wy), logit=cat(lg, sl, wl), q_mean=cat(qm, sqm, wqm), y_sd=cat(sd, ssd, wsd),
                            w=w, flags=fl,
                            dclass=np.concatenate([np.zeros(len(ck), np.int8), scls, activity_class(rk, ck, ev.grid)]),
                            index=ev.index, n_covered=len(K), presence_q=self.presence_q,
                            empty_frac=self.empty_frac,
                            q_total=np.bincount(unkey(allk)[0] % 3, np.asarray(ev.cells["q"], np.float64), 3))

    def result(self):
        c = self.ctx
        level = float(getattr(c, "level", LEVEL))
        out = self.scored.result(c.n_boot, c.seed, level=level)
        out["headline"] = {k: {h: out[k][h] for h in HEADLINE + SECONDARY} for k in ("presence", "amplitude")}
        out["_windows"] = self.scored.windows()          # popped by the CLI into <out>.windows.npz
        H = np.stack(self.hist).astype(np.float64)       # (events, 3, 1 + strata, bins)
        S = np.stack(self.sums)                          # (events, 3, 5)
        groups = [("all", None)] + [(VIEWS[v], v) for v in range(3)]
        sfx = lambda name: "" if name == "all" else f"_{name}"                   # noqa: E731
        pick = lambda a, v: a.sum(0) if v is None else a[v]                       # noqa: E731

        def stat(m):
            m = np.asarray(m, np.float64)
            h, s = np.tensordot(m, H, 1), np.tensordot(m, S, 1)
            r = {}
            for name, v in groups:
                hv, sv, t = pick(h, v), pick(s, v), sfx(name)
                r[f"ap{t}"] = average_precision(hv[0], hv[1:].sum(0))
                for j, st in enumerate(STRATA):
                    r[f"ap_{st}{t}"] = average_precision(hv[0], hv[1 + j])
                r[f"ece{t}"] = hist_calibration(hv[0], hv[1:].sum(0))["ece"]
                r[f"closure{t}"] = _ratio(sv[1] + sv[2], sv[0])
                r[f"closure_on_charged{t}"] = _ratio(sv[1], sv[0])
                r[f"closure_on_empty{t}"] = _ratio(sv[2], sv[0])
                r[f"closure_point{t}"] = _ratio(sv[3] + sv[4], sv[0])
            return r

        b = bootstrap(stat, len(H), c.n_boot, c.seed, level=level)
        tot = H.sum(0)
        det = {k: v for k, v in b.items() if k.startswith("ap")}
        for name, v in groups:
            hv, t = pick(tot, v), sfx(name)
            P = hv[0].sum()
            det[f"prevalence{t}"] = _ratio(P, P + hv[1:].sum())
            for j, st in enumerate(STRATA):
                det[f"prevalence_{st}{t}"] = _ratio(P, P + hv[1 + j].sum())

        def cal(hh, off=0.0):
            return {name: hist_calibration(pick(hh, v)[0], pick(hh, v)[1:].sum(0), off) for name, v in groups}
        even, odd = H[0::2].sum(0), H[1::2].sum(0)
        off = fit_logit_offset(even.sum(0)[0], even.sum(0)[1:].sum(0))
        calib = dict(population="natural covered cells, importance-weighted", raw=cal(tot),
                     ece={name: b[f"ece{sfx(name)}"] for name, _ in groups},
                     offset=round(off, 6), offset_fit="events at even positions",
                     eval_half=dict(events="odd positions", raw=cal(odd), corrected=cal(odd, off)))

        chg = {k: np.concatenate(v) for k, v in self.chg.items()}
        emp = {k: np.concatenate(v) for k, v in self.emp.items()}
        charge = charge_tables(chg, emp, 1.0 / self.empty_frac, self.presence_q)
        for name, v in groups:
            e, t = (S.sum(1) if v is None else S[:, v]), sfx(name)
            ok = e[:, 0] > 0
            ratio = (e[ok, 1] + e[ok, 2]) / e[ok, 0]
            charge[name]["closure"] = dict(pooled=b[f"closure{t}"], on_charged=b[f"closure_on_charged{t}"],
                                           on_empty=b[f"closure_on_empty{t}"], point_pooled=b[f"closure_point{t}"],
                                           median=round(float(np.median(ratio)), 6) if ok.any() else float("nan"),
                                           n_events=int(ok.sum()))
        out["cells"] = dict(n_events=len(H), n_charged=int(len(chg["q"])), presence_q=self.presence_q,
                            q_floor=self.presence_q, call_p=CALL_P, empty_frac=self.empty_frac,
                            conditional=bool(getattr(c.denoiser, "conditional", True)), level=level,
                            frac_uncovered=charge["all"]["frac_uncovered"],
                            detection=det, calibration=calib, charge=charge)
        return out


#: A bright parent of the sub-cell score: its truth charge (e-) at least this (the 2026-11 share
#: review's selection, /pscratch/sd/o/oalter/helix_work/decide/denoise/share_probe.py).
BRIGHT_Q = 5000.0


class SubCell:
    """How a fine-grid denoiser shares a coarse cell's charge over its sub-cells, against spreading
    a coarse-grid denoiser's charge evenly over them.

    Events carry the FINE truth (``ev.grid``, the ``fine`` predictor's grid); each fine cell's
    parent is its cell on ``coarse`` (the coarse predictor's grid), with the summed truth charge
    (:func:`helix.data.cells.aggregate`'s rule). Scored: every OCCUPIED sub-cell (q >
    ``presence_q``) of every BRIGHT parent (q >= ``bright_q``) that the coarse predictor covers.
    Per sub-cell, with ``pred`` floored at 1 e- as in that review::

        err = |log(max(pred, 1) / q)|
        fine         pred = Q0 expm1(y), the fine predictor's charge given presence
        even         pred = Q0 expm1(y_parent) / m, the coarse predictor's spread over its m sub-cells
        even_truth   pred = q_parent / m: the even spread of the TRUE parent charge (its floor)

    ``result()``: per view and pooled, ``n`` and each arm's median ``err`` ([point, lo, hi] over an
    event bootstrap), and ``even_over_fine`` = median even / median fine. Also the fine
    predictor's ``frac_called`` (p > :data:`CALL_P`) on the scored sub-cells."""
    ARMS = ("fine", "even", "even_truth")

    def __init__(self, fine, coarse, coarse_grid, *, bright_q=BRIGHT_Q, presence_q=PRESENCE_Q, n_boot=200,
                 seed=0, level=LEVEL):
        self.fine, self.coarse, self.coarse_grid = fine, coarse, Grid(*coarse_grid)
        self.bright_q, self.presence_q = float(bright_q), float(presence_q)
        self.n_boot, self.seed, self.level = int(n_boot), int(seed), float(level)
        self.rows = dict(ev=[], view=[], called=[], **{a: [] for a in self.ARMS})

    def add(self, ev):
        G, fg = self.coarse_grid, ev.grid
        m = sub_cells(np.zeros(1, np.int64), G, fg).shape[1]
        keys, q = np.asarray(ev.cells["key"], np.int64), np.asarray(ev.cells["q"], np.float64)
        g, w, t = unkey(keys)
        par = fkey(g, w // (G.fw // fg.fw), t // (G.ft // fg.ft))
        up, inv = np.unique(par, return_inverse=True)
        inv = inv.ravel()
        qpar = np.bincount(inv, q, len(up))
        bright = np.nonzero(qpar >= self.bright_q)[0]
        yc, lc = self.coarse(ev, up[bright], grid=G)[:2]
        ok = np.zeros(len(up), bool)
        ok[bright[np.asarray(lc) > UNCOVERED_LOGIT]] = True
        even = np.zeros(len(up))
        even[bright] = Q0 * np.expm1(np.maximum(np.asarray(yc, np.float64), 0)) / m
        occ = (q > self.presence_q) & ok[inv]
        if not occ.any():
            return
        yf, lf = self.fine(ev, keys[occ], grid=fg)[:2]
        qo, po = q[occ], inv[occ]
        err = lambda pred: np.abs(np.log(np.maximum(pred, 1.0) / qo)).astype(np.float32)   # noqa: E731
        r = self.rows
        r["fine"].append(err(Q0 * np.expm1(np.maximum(np.asarray(yf, np.float64), 0))))
        r["even"].append(err(even[po]))
        r["even_truth"].append(err(qpar[po] / m))
        r["called"].append(sigmoid(np.asarray(lf, np.float64)) > CALL_P)
        r["view"].append((g[occ] % 3).astype(np.uint8))
        r["ev"].append(np.full(int(occ.sum()), ev.index, np.int64))

    def result(self):
        r = {k: np.concatenate(v) if v else np.zeros(0) for k, v in self.rows.items()}
        evs, ev = np.unique(r["ev"].astype(np.int64), return_inverse=True)
        out = dict(bright_q=self.bright_q, presence_q=self.presence_q, coarse_grid=str(self.coarse_grid),
                   n_events=int(len(evs)), level=self.level)
        if not len(evs):
            return out
        for name, v in [("all", None)] + [(VIEWS[j], j) for j in range(3)]:
            sel = np.ones(len(ev), bool) if v is None else r["view"] == v
            srt = {}
            for a in self.ARMS:                                     # each arm's errors sorted, with their events
                o = np.argsort(r[a][sel], kind="stable")
                srt[a] = (r[a][sel][o], ev[sel][o])

            def stat(mult, srt=srt):
                res = {}
                for a, (x, e) in srt.items():                       # the event-weighted median
                    cw = np.cumsum(mult[e])
                    res[a] = float(x[min(np.searchsorted(cw, 0.5 * cw[-1]), len(x) - 1)]) \
                        if len(x) and cw[-1] > 0 else float("nan")
                res["even_over_fine"] = res["even"] / res["fine"] if res["fine"] > 0 else float("nan")
                return res
            b = bootstrap(stat, len(evs), self.n_boot, self.seed, level=self.level)
            out[name] = dict(n=int(sel.sum()), frac_called=_ratio(r["called"][sel].sum(), sel.sum()),
                             median_abs_log_err={a: b[a] for a in self.ARMS}, even_over_fine=b["even_over_fine"])
        return out
