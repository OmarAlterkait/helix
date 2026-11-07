"""The probe: an MLP fitted in memory on frozen features. ONE implementation, used by ``floor``
(charge per fine cell), ``uprobe`` (the along-wire coordinate per patch) and ``labels`` (a
particle class per signal cell).

Nonlinear on purpose: a linear (ridge) readout of the same features measured 0.29 vs 0.35
floor efficiency below 0.1 MeV and within-group R^2 0.30 vs 0.53 along the wire (0.01 vs 0.48
on the Y view). Its known weakness is the converse: thousands of columns that carry nothing
drown a weak signal (Gaussian-noise features scored below the geometry columns alone), so a
probe arm is compared with a baseline that is not itself a probe fit.

Protocol, the same for every arm of every metric:

* rows are split by EVENT, never by row (rows of one event are not independent): the first
  :func:`split_events` events are fitted on, the rest scored. They come in that order, so a
  metric fits once the training events are in and scores each later event as it arrives,
  holding only its predictions;
* the fit sees at most :data:`FIT_ROWS` rows, an equal seeded sample of each training event
  (:func:`fit_rows`), so its memory and time do not grow with the split; every scored row is
  scored;
* inside the training events the last ~10% are held out for early stopping, scored by the
  metric the caller reports -- validation MSE; with ``group`` the pooled within-group R^2
  (:func:`helix.eval.events.group_r2`; selection by MSE halts a head that has only learned
  which (event, plane) a row belongs to when, as for the along-wire coordinate, almost all the
  target's variance is between groups); with ``n_classes`` the balanced accuracy (mean
  per-class recall), the loss then being cross-entropy weighted by inverse class frequency,
  so the argmax is the balanced decision and a rare class (a Michel electron: 0.4% of signal
  cells) is not traded away for the common ones;
* every input column -- features and ``extra`` (presence bits, offsets, geometry) alike -- is
  STANDARDISED from the training rows: one massive-activation channel (measured: 4836 against a
  next-largest 462) otherwise sets the input scale and the score tracks it, not the
  representation (two seeds of one recipe: 0.82 and 0.68 on raw inputs); and ``extra`` left in
  its own units next to thousands of unit-variance features is ignored (the along-wire probe's
  random arm scored 0.005 with the geometry columns that alone score 0.121). A column constant
  over the training rows is centred, not scaled, and no column is scaled by more than 1e3;
* the budget is in EPOCHS (at least :data:`MIN_STEPS` steps), so a larger split does not train
  the head less; seeded;
* every fitted arm is fitted ``Context.probe_seeds`` times on the same rows, seeds ``seed + k``.
  The numbers with their bootstrap interval are seed ``seed``'s (the single fit of before);
  beside them ``seed_mean`` / ``seed_half_range``, (max - min) / 2 over the fits -- the probe's
  own variance, which an event bootstrap of one fitted probe leaves out;
* beside the trained arm, the ``raw`` arm: the same probe on the covering tokens' RAW tokenizer
  inputs (:meth:`helix.eval.events.Event.raw_inputs`: per band the token's ``inp`` and ``occ``,
  2 x n_slot) instead of their encoder features -- what the input already carries, so
  ``d_over_raw`` (trained - raw, bootstrapped paired) is what pretraining adds;
* the fit takes at most ``FIT_ROWS // n_train`` rows per event but every scored row is scored,
  so a dense event weighs more in a pooled number; each pooled number has an ``_ev`` twin in
  which every scored event weighs the same.

The three probe metrics follow it, each class documenting its own: :class:`Floor` (``floor``),
:class:`UProbe` (``uprobe``) and :class:`Labels` (``labels``).
"""

from __future__ import annotations

import numpy as np

from helix.data.cells import FT, FW, MOTIF_NAMES, PID_NAMES, Q0, cell_class, cell_inputs, unkey
from helix.eval.events import (VIEWS, bootstrap, group_event, group_r2, plane_group, score_windows, seed_spread,
                               window_stat)

#: Fraction of the evaluated events the probes are fitted on; the rest are scored.
TRAIN_FRAC = 2 / 3
#: Rows one probe is fitted on, at most. A row holds n_bands x d float16 features (6 KB at
#: 4 x 768), so ~6 GB per probe; an event has ~6k floor and ~9k uprobe rows (SBND campaign), so
#: the budget binds from ~110-170 training events on.
FIT_ROWS = 1_000_000
EPOCHS, MIN_STEPS, BATCH, LR = 6, 300, 4096, 1e-3
CHUNK = 16384                       # rows per inference / statistics chunk


def split_events(n_events):
    """How many of ``n_events`` (in evaluation order) are probe-training events."""
    if n_events < 3:
        raise ValueError(f"{n_events} events: a probe needs at least 3 (fit, early stop, score)")
    return min(n_events - 1, max(2, int(round(TRAIN_FRAC * n_events))))


def fit_rows(n, n_train, rng):
    """The rows (sorted indices into ``n``) one of ``n_train`` training events gives the fit: all
    of them, or a uniform sample of ``FIT_ROWS // n_train``."""
    k = max(1, FIT_ROWS // n_train)
    return np.arange(n) if n <= k else np.sort(rng.choice(n, k, replace=False))


def gather(feats, idx):
    """Per row, the features of its covering token in each band, concatenated -> (n, nb * d)
    float16 on the features' device; zeros where a band has no token (``idx == -1``; the caller
    carries the absence as presence bits)."""
    import torch
    idx = torch.as_tensor(np.asarray(idx, np.int64), device=feats.device)
    X = feats.half()[idx.clamp(min=0)] * (idx >= 0)[..., None]
    return X.reshape(idx.shape[0], idx.shape[1] * feats.shape[1])


class Probe:
    """A fitted head: ``predict(X, extra) -> np.ndarray`` (a classifier's: the class). ``X`` (n, d)
    features (zero columns for a geometry-only arm) and ``extra`` (n, k): torch on any device, or
    numpy."""

    def __init__(self, net, mu, sd, device, n_classes=None):
        self.net, self.mu, self.sd, self.device, self.n_classes = net, mu, sd, device, n_classes
        self.val_score = None

    def inputs(self, X, extra):
        import torch
        A = extra if isinstance(extra, torch.Tensor) else torch.from_numpy(np.asarray(extra, np.float32))
        return (torch.cat([X.to(self.device).float(), A.to(self.device).float()], 1) - self.mu) / self.sd

    def predict(self, X, extra):
        import torch
        with torch.no_grad():
            out = [self.net(self.inputs(X[s:s + CHUNK], extra[s:s + CHUNK])).float()
                   for s in range(0, len(extra), CHUNK)]
            out = [o.argmax(1).cpu() if self.n_classes else o.squeeze(-1).cpu() for o in out]
        return torch.cat(out).numpy() if out else np.zeros(0, np.int64 if self.n_classes else np.float32)


def fit_probe(X, extra, y, event, *, group=None, n_classes=None, seed=0, device=None):
    """Fit the probe head on training rows.

    ``X``: (n, d) features (torch, CPU; d = 0 for geometry alone); ``extra``: (n, k) float32
    appended to them; ``y``: (n,) target -- a class index with ``n_classes``; ``event``: (n,)
    event index; ``group``: (n,) int -- select on the within-group R^2 rather than validation
    MSE. A classifier's ``predict`` returns the class.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    event = np.asarray(event)
    y = np.asarray(y, np.int64 if n_classes else np.float32)
    n = len(y)
    A = torch.as_tensor(np.asarray(extra, np.float32))
    s1 = torch.zeros(X.shape[1] + A.shape[1], dtype=torch.float64, device=dev)
    s2 = torch.zeros_like(s1)
    for s in range(0, n, CHUNK):                             # chunked: no float copy of the matrix
        c = torch.cat([X[s:s + CHUNK].to(dev).double(), A[s:s + CHUNK].to(dev).double()], 1)
        s1 += c.sum(0)
        s2 += (c * c).sum(0)
    mu = (s1 / n).float()
    sd = (s2 / n - (s1 / n) ** 2).clamp(min=0).sqrt().float()
    sd = torch.where(sd > 1e-6, sd.clamp(min=1e-3), torch.ones_like(sd))
    net = nn.Sequential(nn.Linear(X.shape[1] + extra.shape[1], 1024), nn.GELU(), nn.Linear(1024, 512),
                        nn.GELU(), nn.Linear(512, n_classes or 1)).to(dev)
    probe = Probe(net, mu, sd, dev, n_classes)

    evs = np.unique(event)
    if len(evs) < 2:
        raise ValueError("a probe fit needs rows from at least 2 events (one held out to stop on)")
    va = np.isin(event, evs[-max(1, int(round(0.1 * len(evs)))):])
    tri, vai = np.nonzero(~va)[0], np.nonzero(va)[0]
    Xg, Ag = X.to(dev), A.to(dev)
    yg = torch.from_numpy(y).to(dev)
    if n_classes:
        cnt = np.bincount(y[tri], minlength=n_classes).astype(np.float64)
        wcls = torch.as_tensor(np.where(cnt > 0, cnt.sum() / np.maximum(cnt, 1) / n_classes, 0.0),
                               dtype=torch.float32, device=dev)
        loss_fn = lambda out, t: F.cross_entropy(out, t, weight=wcls)              # noqa: E731
    else:
        loss_fn = lambda out, t: F.mse_loss(out.squeeze(-1), t)                    # noqa: E731

    def score():
        p = probe.predict(Xg[vai], Ag[vai])
        if n_classes:
            yv = y[vai]
            rec = [(p[yv == c] == c).mean() for c in range(n_classes) if (yv == c).any()]
            return float(np.mean(rec)) if rec else float("nan")
        if group is None:
            return -float(np.mean((p - y[vai]) ** 2))
        _, sse, sst = group_r2(p, y[vai], np.asarray(group)[vai])
        return float(1 - sse.sum() / sst.sum()) if sst.sum() > 0 else float("nan")

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    steps = max(MIN_STEPS, EPOCHS * len(tri) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps)
    every = max(1, steps // 6)
    tri_d = torch.from_numpy(tri).to(dev)
    best, best_sd = -np.inf, None
    for s in range(steps):
        b = tri_d[torch.randint(len(tri_d), (BATCH,), device=dev)]
        loss = loss_fn(net(probe.inputs(Xg[b], Ag[b])), yg[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (s + 1) % every == 0 or s + 1 == steps:
            v = score()
            if v > best:
                best, best_sd = v, {k: t.clone() for k, t in net.state_dict().items()}
    if best_sd is None:
        raise ValueError("probe: no validation score at any check (a validation set with no "
                         "variance or no class, or a NaN loss)")
    net.load_state_dict(best_sd)
    probe.val_score = float(best)
    return probe


def fit_seeds(ctx, X, extra, y, event, **kw):
    """The probe fitted ``ctx.probe_seeds`` times on the same rows, seeds ``ctx.seed + k``; the
    first is the fit every number with an interval comes from."""
    return [fit_probe(X, extra, y, event, seed=ctx.seed + k, **kw) for k in range(max(1, int(ctx.probe_seeds)))]


def arm_features(ev, model, arm):
    """Per token, what arm ``arm`` reads: the encoder's features or the raw tokenizer inputs."""
    return ev.raw_inputs() if arm == "raw" else ev.features(model)


def seed_name(arm, k):
    """The key of seed ``k``'s predictions of ``arm``: the arm itself for the first seed."""
    return arm if k == 0 else f"{arm}.s{k}"


def _diff(a, b):
    return {k: a[k] - b[k] for k in a}


# ---- floor -----------------------------------------------------------------------------------

FLOOR_ARMS = ("trained", "raw", "amplitude")
FLOOR_FITTED = ("trained", "raw")


def amplitude(ev, keys):
    """The amplitude baseline's score of fine cells ``keys``: 0 where no kept coefficient."""
    uk, amp = ev.coeff_amp
    if not len(uk):
        return np.zeros(len(keys))
    p = np.clip(np.searchsorted(uk, keys), 0, len(uk) - 1)
    return np.where(uk[p] == keys, amp[p], 0.0)


class Scored:
    """The scored events' windows and named per-cell scores on them, scored by
    :func:`helix.eval.events.score_windows`: the floor's arms and the denoiser alike."""

    def __init__(self):
        self.meta, self.s = [], {}

    def add(self, ev, tr, scores):
        """``tr``: the event's truth; ``scores``: ``{name: score per window cell}``."""
        self.meta += [dict(m, ev=ev.index) for m in tr["meta"]]
        for k, s in scores.items():
            self.s.setdefault(k, []).append(np.asarray(s, np.float32))

    def stat(self, name):
        """:func:`helix.eval.events.window_stat` of score ``name``: ``(stat, n_events, n_windows)``."""
        if not self.meta:
            raise ValueError("no scored window")
        return window_stat(np.concatenate(self.s[name]), self.meta)

    def point(self, name):
        """Score ``name``'s numbers with every event once (no interval)."""
        stat, n, _ = self.stat(name)
        return stat(np.ones(n, np.int64))

    def paired(self, a, b, n_boot, seed):
        """``a - b`` of every number, each ``[point, lo, hi]`` over the SAME resamples."""
        sa, n, _ = self.stat(a)
        sb = self.stat(b)[0]
        return bootstrap(lambda m: _diff(sa(m), sb(m)), n, n_boot, seed)

    def result(self, n_boot, seed, names=None):
        if not self.meta:
            raise ValueError("no scored window")
        return {k: score_windows(np.concatenate(self.s[k]), self.meta, n_boot, seed) for k in (names or self.s)}


class Floor:
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
    scored on the windows of the rest (:func:`helix.eval.events.score_windows`: efficiency at a 1%
    false-positive rate per view on the far background, the same on the near background, the far
    threshold's rate on the near windows and the AUC, by energy, pooled and per view). Three arms:

      trained    the probe on the trained encoder's features
      raw        the same probe on the covering tokens' raw tokenizer inputs (``inp``, ``occ``)
      amplitude  no fit: a cell's largest |kept coefficient| / sigma
                 (:func:`helix.data.cells.coeff_amplitude`) -- what the compressor itself sees. The
                 reference a representation must beat. (The randomly initialised encoder under the
                 same probe scored below it at every energy up to 1 MeV, so it is no floor.)

    ``d_over_amplitude`` (point differences) and ``d_over_raw`` (bootstrapped paired) are trained
    minus each reference. The ``_ev`` efficiency weighs every scored event's iso windows equally.

    Memory: only the fit's rows hold features (float16, at most :data:`helix.eval.probe.FIT_ROWS`,
    ~6 GB, and ~2 GB of raw inputs); a scored event keeps its window scores alone.
    """
    name = "floor"

    def __init__(self, ctx):
        self.ctx = ctx
        self.fit = dict(X={a: [] for a in FLOOR_FITTED}, A=[], y=[], e=[])     # the training events' fit rows
        self.probes, self.n_fit = None, None
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
            for a in FLOOR_FITTED:
                f["X"][a].append(gather(arm_features(ev, c.model, a), idx[keep]).cpu())
            f["A"].append(aux[keep])
            f["y"].append(np.log1p(tr["mq"][keep] / Q0).astype(np.float32)); f["e"].append(np.full(len(keep), ev.index))
            return
        if self.probes is None:
            self._fit()
        widx, waux = cell_inputs(tr["wkey"], ev.cell_key, ev.band_lengths, c.run.tokenizer)
        ok = (widx >= 0).any(1)
        scores = dict(amplitude=amplitude(ev, tr["wkey"]))
        for a in FLOOR_FITTED:
            X = gather(arm_features(ev, c.model, a), widx[ok]) if ok.any() else None
            for k, probe in enumerate(self.probes[a]):
                p = np.zeros(len(widx), np.float32)
                if ok.any():
                    p[ok] = probe.predict(X, waux[ok])
                scores[seed_name(a, k)] = np.expm1(np.maximum(p, 0)) * Q0
        self.scored.add(ev, tr, scores)

    def _fit(self):
        import torch
        f, self.fit = self.fit, None
        if not f["y"]:
            raise ValueError("floor: no covered map row in any training event")
        y, A, e = np.concatenate(f["y"]), np.concatenate(f["A"]), np.concatenate(f["e"])
        self.probes = {}
        for a in FLOOR_FITTED:
            X = torch.cat(f["X"].pop(a))
            self.probes[a] = fit_seeds(self.ctx, X, A, y, e)
            del X
        self.n_fit = len(y)

    def result(self):
        c = self.ctx
        out = dict(n_train_events=c.n_train, n_fit_rows=self.n_fit, probe_seeds=len(self.probes["trained"]),
                   val_mse=-self.probes["trained"][0].val_score,
                   **self.scored.result(c.n_boot, c.seed, names=FLOOR_ARMS))
        out["d_over_amplitude"] = {k: round(out["trained"][k][0] - out["amplitude"][k][0], 4)
                                   for k in out["trained"] if k.startswith("floor_eff")}
        out["d_over_raw"] = self.scored.paired("trained", "raw", c.n_boot, c.seed)
        pts = {a: [self.scored.point(seed_name(a, k)) for k in range(len(self.probes[a]))] for a in FLOOR_FITTED}
        for a in FLOOR_FITTED:
            out[a].update(val_mse_seeds=[round(-p.val_score, 6) for p in self.probes[a]], **seed_spread(pts[a]))
        out["d_over_raw_seeds"] = seed_spread([_diff(t, r) for t, r in zip(pts["trained"], pts["raw"])])
        return out


# ---- uprobe ----------------------------------------------------------------------------------

#: Along-wire unit vector (y, z) per plane gid, SBND: U +120 deg, V -120 deg, Y 180 deg, in each
#: of the two volumes. The geometry registry carries no wire orientation; this was fitted
#: (``wire ~ a*y + b*z`` per event over ~600k deposits, three shards) and is exactly the
#: +-60 / 0 degree wire orientations to four decimals, residual 0.30 wires on every plane.
ALONGWIRE = np.array([[-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 2], [-1.0, 0.0]] * 2)
DOM_THRESHOLD = 0.5
WIRE_SCALE, TICK_SCALE = 2000.0, 4321.0          # the geometry columns' units
UPROBE_ARMS = ("geo", "trained", "raw")


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

    Three arms, all with the geometry columns (plane one-hot, mean wire, mean tick, presence bits)
    appended: ``geo`` (those alone -- the reference; it includes drift time, so any ``u`` leaking
    through timing is already in it), ``trained`` and ``raw`` (the covering tokens' raw tokenizer
    inputs). Scored with ``r2``: the pooled R^2 WITHIN each (event, plane)
    (:func:`helix.eval.events.group_r2`), so the coarse between-event and between-plane structure
    any position-only baseline reproduces is removed (97.4% of ``u``'s variance is between groups)
    while a prediction of the wrong scale is still charged (a correlation is not: geometry alone
    has within-group r 0.15 but R^2 0.03); also per view. ``r2_ev`` weighs every (event, plane)
    group equally (each group's sums divided by its row count) where ``r2`` pools rows.
    ``d_over_geo`` = trained - geo and ``d_over_raw`` = trained - raw, bootstrapped paired.
    """
    name = "uprobe"

    def __init__(self, ctx):
        self.ctx = ctx
        self.n_planes = int(ctx.run.config["model"]["n_plane"])
        self.fit = dict(y=[], geo=[], grp=[], e=[], trained=[], raw=[])   # the training events' fit rows
        self.probes, self.n_fit, self.n_patch = None, None, 0
        self.p = {}                                                       # the scored events', per seed name
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
            for a in ("trained", "raw"):
                f[a].append(gather(arm_features(ev, c.model, a), pr["tok"][keep]).cpu())
            return
        for a in UPROBE_ARMS:
            X = torch.zeros((n, 0)) if a == "geo" else gather(arm_features(ev, c.model, a), pr["tok"])
            for k, probe in enumerate(self.probes[a]):
                self.p.setdefault(seed_name(a, k), []).append(probe.predict(X, pr["geo"]))
        self.y.append(pr["y"]); self.grp.append(grp)

    def _fit(self):
        import torch
        f, self.fit = self.fit, None
        if not f["y"]:
            raise ValueError("uprobe: no patch with a dominant cell in any training event")
        y, geo, grp, e = (np.concatenate(f[k]) for k in ("y", "geo", "grp", "e"))
        self.probes, self.n_fit = {}, len(y)
        for a in UPROBE_ARMS:
            X = torch.zeros((len(y), 0)) if a == "geo" else torch.cat(f.pop(a))
            self.probes[a] = fit_seeds(self.ctx, X, geo, y, e, group=grp)
            del X

    def result(self):
        c = self.ctx
        if not self.y:
            raise ValueError("uprobe: no patch with a dominant cell in any scored event")
        y, grp = np.concatenate(self.y), np.concatenate(self.grp)
        evs = np.unique(group_event(grp))
        _, n_g = np.unique(grp, return_counts=True)            # group_r2's keys, in its order
        terms = {}                               # each (event, plane)'s sums once; a resample weighs its events
        for name, p in self.p.items():
            keys, sse, sst = group_r2(np.concatenate(p), y, grp)
            terms[name] = (sse, sst, np.searchsorted(evs, group_event(keys)), (keys % 16) % 3)

        def r2(w, sse, sst, s=slice(None)):
            d = (w * sst)[s].sum()
            return float(1 - (w * sse)[s].sum() / d) if d else float("nan")

        def stat_for(k):
            def stat(m):
                r = {}
                for a in UPROBE_ARMS:
                    sse, sst, ge, gv = terms[seed_name(a, k)]
                    w, w_ev = m[ge], m[ge] / n_g
                    r[a], r[f"{a}_ev"] = r2(w, sse, sst), r2(w_ev, sse, sst)
                    for v in range(3):
                        r[f"{a}_{VIEWS[v]}"] = r2(w, sse, sst, gv == v)
                        r[f"{a}_ev_{VIEWS[v]}"] = r2(w_ev, sse, sst, gv == v)
                for ref in ("geo", "raw"):
                    r[f"d_over_{ref}"] = r["trained"] - r[ref]
                    r[f"d_over_{ref}_ev"] = r["trained_ev"] - r[f"{ref}_ev"]
                return r
            return stat

        b = bootstrap(stat_for(0), len(evs), c.n_boot, c.seed)
        n_seeds = len(self.probes["trained"])
        spread = seed_spread([stat_for(k)(np.ones(len(evs), np.int64)) for k in range(n_seeds)])
        out = dict(n_train_events=c.n_train, n_fit_rows=self.n_fit, n_patch=self.n_patch, probe_seeds=n_seeds)

        def rename(a, key):                       # the stat's key of arm a -> the reported name
            return "r2" + key[len(a):]
        for a in UPROBE_ARMS:
            ks = [k for k in b if k == a or k.startswith(a + "_")]
            out[a] = dict({rename(a, k): b[k] for k in ks}, val_r2=round(self.probes[a][0].val_score, 4),
                          val_r2_seeds=[round(p.val_score, 4) for p in self.probes[a]],
                          **{s: {rename(a, k): spread[s][k] for k in ks} for s in spread})
        for k in b:
            if k.startswith("d_over_"):
                out[k] = b[k]
        out.update({s: {k: v for k, v in spread[s].items() if k.startswith("d_over_")} for s in spread})
        return out


# ---- labels ----------------------------------------------------------------------------------

LABELS_ARMS = ("geo", "trained", "raw")
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
    """The ``labels`` metric: which particle made a cell's charge, from frozen encoder features.

    Per signal cell of the floor's map rows (truth charge > 0, covered by a token), the class of the
    track carrying most of its charge (:func:`helix.data.cells.cell_class`), in pimm-data's PILArNet
    vocabulary: ``motif`` (shower, track, michel, delta, le) and ``pid`` (photon, electron, muon,
    pion, proton, le). Unlike the charge, the class is not in a cell's own amplitude: it is a
    property of the particle, so reading it needs context -- the shape and neighbourhood of the
    deposit, which is what a representation is for.

    A classifier probe (:func:`helix.eval.probe.fit_probe` with ``n_classes``: class-balanced
    cross-entropy) per task, inputs as the floor's (covering-token features + presence bits and
    offsets). Three arms: ``trained``, ``geo`` (the presence bits and offsets alone -- what the
    token layout gives away; it carries no charge at all) and ``raw`` (the covering tokens' raw
    tokenizer inputs -- what the input carries). Scored with the balanced accuracy
    (``macro_recall``, the mean per-class recall: the classes run from 53% of cells (le) to 0.4%
    (michel)), with a [16, 84]% interval over an event bootstrap, and each class's recall;
    ``macro_recall_ev`` the same with every scored event's rows weighing equally in each class's
    recall; ``d_over_geo`` and ``d_over_raw`` paired.
    """
    name = "labels"

    def __init__(self, ctx):
        self.ctx = ctx
        self.fit = dict(X={a: [] for a in ("trained", "raw")}, A=[], e=[], **{t: [] for t in TASKS})
        self.probes, self.n_fit = None, None
        self.p = {}                                          # (task, seed name) -> the scored events' classes
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
            for a in f["X"]:
                f["X"][a].append(gather(arm_features(ev, c.model, a), idx[keep]).cpu())
            f["A"].append(aux[keep])
            f["e"].append(np.full(len(keep), ev.index))
            for t in TASKS:
                f[t].append(cls[t][keep])
            return
        X = {a: torch.zeros((len(idx), 0)) if a == "geo" else gather(arm_features(ev, c.model, a), idx)
             for a in LABELS_ARMS}
        for t in TASKS:
            for a in LABELS_ARMS:
                for k, probe in enumerate(self.probes[(t, a)]):
                    self.p.setdefault((t, seed_name(a, k)), []).append(probe.predict(X[a], aux))
            self.y[t].append(cls[t])
        self.e.append(np.full(len(idx), ev.index))

    def _fit(self):
        import torch
        f, self.fit = self.fit, None
        if not f["e"]:
            raise ValueError("labels: no labelled signal cell in any training event")
        A, e = np.concatenate(f["A"]), np.concatenate(f["e"])
        self.probes, self.n_fit = {}, len(e)
        for a in LABELS_ARMS:
            X = None if a == "geo" else torch.cat(f["X"].pop(a))
            for t, (_, names) in TASKS.items():
                y = np.concatenate(f[t])
                ok = y >= 0
                Xa = torch.zeros((int(ok.sum()), 0)) if X is None else X[torch.from_numpy(ok)]
                self.probes[(t, a)] = fit_seeds(self.ctx, Xa, A[ok], y[ok], e[ok], n_classes=len(names))
            del X

    def result(self):
        c = self.ctx
        if not self.e:
            raise ValueError("labels: no labelled signal cell in any scored event")
        e = np.concatenate(self.e)
        evs, we = np.unique(e, return_inverse=True)
        n_seeds = len(self.probes[(next(iter(TASKS)), "trained")])
        out = dict(n_train_events=c.n_train, n_fit_rows=self.n_fit, probe_seeds=n_seeds)
        for t, (_, names) in TASKS.items():
            y = np.concatenate(self.y[t])
            ok = y >= 0
            hit = {nm: np.concatenate(p)[ok] == y[ok] for (tt, nm), p in self.p.items() if tt == t}
            yk, wk = y[ok], we[ok]
            inv_n = 1.0 / np.bincount(wk, minlength=len(evs))[wk]         # every event's rows weigh 1 in all
            K = len(names)

            def stat_for(k, hit=hit, yk=yk, wk=wk, inv_n=inv_n, K=K):
                def recall(w, h):
                    n_c, h_c = np.bincount(yk, w, K), np.bincount(yk, w * h, K)
                    return float(np.mean(h_c[n_c > 0] / n_c[n_c > 0]))

                def stat(m):
                    w = m[wk]
                    r = {}
                    for a in LABELS_ARMS:
                        h = hit[seed_name(a, k)]
                        r[a], r[f"{a}_ev"] = recall(w, h), recall(w * inv_n, h)
                    for ref in ("geo", "raw"):
                        r[f"d_over_{ref}"] = r["trained"] - r[ref]
                        r[f"d_over_{ref}_ev"] = r["trained_ev"] - r[f"{ref}_ev"]
                    return r
                return stat

            b = bootstrap(stat_for(0), len(evs), c.n_boot, c.seed)
            spread = seed_spread([stat_for(k)(np.ones(len(evs), np.int64)) for k in range(n_seeds)])
            res = dict(prior={nm: round(float((yk == i).mean()), 4) for i, nm in enumerate(names)})
            for a in LABELS_ARMS:
                pr = self.probes[(t, a)]
                res[a] = dict(macro_recall=b[a], macro_recall_ev=b[f"{a}_ev"],
                              val_macro_recall=round(pr[0].val_score, 4),
                              val_macro_recall_seeds=[round(p.val_score, 4) for p in pr],
                              recall={nm: round(float(hit[a][yk == i].mean()), 4)
                                      for i, nm in enumerate(names) if (yk == i).any()},
                              **{s: {"macro_recall": spread[s][a], "macro_recall_ev": spread[s][f"{a}_ev"]}
                                 for s in spread})
            for k in b:
                if k.startswith("d_over_"):
                    res[k] = b[k]
            res.update({s: {k: v for k, v in spread[s].items() if k.startswith("d_over_")} for s in spread})
            out[t] = res
        return out
