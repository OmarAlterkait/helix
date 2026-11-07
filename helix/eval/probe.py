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
  the head less; seeded.

The three probe metrics follow it, each class documenting its own: :class:`Floor` (``floor``),
:class:`UProbe` (``uprobe``) and :class:`Labels` (``labels``).
"""

from __future__ import annotations

import numpy as np

from helix.data.cells import FT, FW, MOTIF_NAMES, PID_NAMES, Q0, cell_class, cell_inputs, unkey
from helix.eval.events import VIEWS, bootstrap, group_event, group_r2, plane_group, score_windows

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


# ---- floor -----------------------------------------------------------------------------------

FLOOR_ARMS = ("trained", "amplitude")


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

    def result(self, n_boot, seed):
        if not self.meta:
            raise ValueError("no scored window")
        return {k: score_windows(np.concatenate(v), self.meta, n_boot, seed) for k, v in self.s.items()}


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
    false-positive rate per view, by energy). Two arms:

      trained    the probe on the trained encoder's features
      amplitude  no fit: a cell's largest |kept coefficient| / sigma
                 (:func:`helix.data.cells.coeff_amplitude`) -- what the compressor itself sees. The
                 reference a representation must beat. (The randomly initialised encoder under the
                 same probe scored below it at every energy up to 1 MeV, so it is no floor.)

    Memory: only the fit's rows hold features (float16, at most :data:`helix.eval.probe.FIT_ROWS`,
    ~6 GB); a scored event keeps its window scores alone.
    """
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


# ---- uprobe ----------------------------------------------------------------------------------

#: Along-wire unit vector (y, z) per plane gid, SBND: U +120 deg, V -120 deg, Y 180 deg, in each
#: of the two volumes. The geometry registry carries no wire orientation; this was fitted
#: (``wire ~ a*y + b*z`` per event over ~600k deposits, three shards) and is exactly the
#: +-60 / 0 degree wire orientations to four decimals, residual 0.30 wires on every plane.
ALONGWIRE = np.array([[-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 2], [-1.0, 0.0]] * 2)
DOM_THRESHOLD = 0.5
WIRE_SCALE, TICK_SCALE = 2000.0, 4321.0          # the geometry columns' units
UPROBE_ARMS = ("geo", "trained")


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

    Two arms, both with the geometry columns (plane one-hot, mean wire, mean tick, presence bits)
    appended: ``geo`` (those alone -- the reference; it includes drift time, so any ``u`` leaking
    through timing is already in it) and ``trained``. Scored with ``r2``: the pooled R^2 WITHIN
    each (event, plane) (:func:`helix.eval.events.group_r2`), so the coarse between-event and
    between-plane structure any position-only baseline reproduces is removed (97.4% of ``u``'s
    variance is between groups) while a prediction of the wrong scale is still charged (a
    correlation is not: geometry alone has within-group r 0.15 but R^2 0.03); also per view.
    ``d_over_geo`` = trained - geo, bootstrapped paired.
    """
    name = "uprobe"

    def __init__(self, ctx):
        self.ctx = ctx
        self.n_planes = int(ctx.run.config["model"]["n_plane"])
        self.fit = dict(y=[], geo=[], grp=[], e=[], trained=[])    # the training events' fit rows
        self.probes, self.n_fit, self.n_patch = None, None, 0
        self.p = {a: [] for a in UPROBE_ARMS}                                        # the scored events'
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
            f["trained"].append(gather(ev.features(c.model), pr["tok"][keep]).cpu())
            return
        for a in UPROBE_ARMS:
            X = torch.zeros((n, 0)) if a == "geo" else gather(ev.features(c.model), pr["tok"])
            self.p[a].append(self.probes[a].predict(X, pr["geo"]))
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
            self.probes[a] = fit_probe(X, geo, y, e, group=grp, seed=self.ctx.seed)

    def result(self):
        c = self.ctx
        if not self.y:
            raise ValueError("uprobe: no patch with a dominant cell in any scored event")
        y, grp = np.concatenate(self.y), np.concatenate(self.grp)
        evs = np.unique(group_event(grp))
        terms = {}
        for a in UPROBE_ARMS:                    # each (event, plane)'s sums once; a resample weighs its events
            keys, sse, sst = group_r2(np.concatenate(self.p[a]), y, grp)
            terms[a] = (sse, sst, np.searchsorted(evs, group_event(keys)), (keys % 16) % 3)

        def stat(m):
            r = {}
            for a, (sse, sst, ge, gv) in terms.items():
                w = m[ge]
                r[a] = float(1 - (w * sse).sum() / (w * sst).sum())
                for v in range(3):
                    s = gv == v
                    r[f"{a}_{VIEWS[v]}"] = float(1 - (w * sse)[s].sum() / (w * sst)[s].sum()) if (w * sst)[s].sum() else float("nan")
            r["d_over_geo"] = r["trained"] - r["geo"]
            return r

        b = bootstrap(stat, len(evs), c.n_boot, c.seed)
        out = dict(n_train_events=c.n_train, n_fit_rows=self.n_fit, n_patch=self.n_patch)
        for a in UPROBE_ARMS:
            out[a] = dict(r2=b[a], **{f"r2_{v}": b[f"{a}_{v}"] for v in VIEWS},
                          val_r2=round(self.probes[a].val_score, 4))
        out["d_over_geo"] = b["d_over_geo"]
        return out


# ---- labels ----------------------------------------------------------------------------------

LABELS_ARMS = ("geo", "trained")
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
    offsets). Two arms: ``trained`` and ``geo`` (the presence bits and offsets alone -- what the
    token layout gives away). Scored with the balanced accuracy (``macro_recall``, the mean
    per-class recall: the classes run from 53% of cells (le) to 0.4% (michel)), with a [16, 84]%
    interval over an event bootstrap, and each class's recall; ``d_over_geo`` paired.
    """
    name = "labels"

    def __init__(self, ctx):
        self.ctx = ctx
        self.fit = dict(X=[], A=[], e=[], **{t: [] for t in TASKS})
        self.probes, self.n_fit = None, None
        self.p = {(t, a): [] for t in TASKS for a in LABELS_ARMS}
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
            for a in LABELS_ARMS:
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
            for a in LABELS_ARMS:
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
            hit = {a: np.concatenate(self.p[(t, a)])[ok] == y[ok] for a in LABELS_ARMS}
            yk, wk = y[ok], we[ok]
            K = len(names)

            def stat(m, hit=hit, yk=yk, wk=wk, K=K):
                w = m[wk]
                r = {}
                for a in LABELS_ARMS:
                    n_c = np.bincount(yk, w, K)
                    h_c = np.bincount(yk, w * hit[a], K)
                    r[a] = float(np.mean(h_c[n_c > 0] / n_c[n_c > 0]))
                r["d_over_geo"] = r["trained"] - r["geo"]
                return r

            b = bootstrap(stat, len(evs), c.n_boot, c.seed)
            res = dict(prior={nm: round(float((yk == i).mean()), 4) for i, nm in enumerate(names)})
            for a in LABELS_ARMS:
                res[a] = dict(macro_recall=b[a], val_macro_recall=round(self.probes[(t, a)].val_score, 4),
                              recall={nm: round(float(hit[a][yk == i].mean()), 4)
                                      for i, nm in enumerate(names) if (yk == i).any()})
            res["d_over_geo"] = b["d_over_geo"]
            out[t] = res
        return out
