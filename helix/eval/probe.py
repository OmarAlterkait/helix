"""The along-wire probe (``uprobe``): an MLP fitted in memory on FROZEN encoder features.

The one frozen-feature key kept (docs/DECISIONS.md, 2026-11-09): it measures 3D information --
the coordinate a single wire plane cannot fix -- not detection, and it is the only frozen key
that has ordered pretraining arms. It stays beside the fine-tune (:mod:`helix.eval.denoise`)
until that key is shown to rank arms the same way.

Nonlinear on purpose: a linear (ridge) readout of the same features measured within-group R^2
0.30 vs 0.53 along the wire (0.01 vs 0.48 on the Y view). Its known weakness is the converse:
thousands of columns that carry nothing drown a weak signal, so the trained arm is compared with
references that are not a fit of another encoder (geometry alone, the raw tokenizer inputs).

Protocol, the same for every arm:

* rows are split by EVENT, never by row (rows of one event are not independent): the first
  :func:`split_events` events are fitted on, the rest scored. They come in that order, so the
  metric fits once the training events are in and scores each later event as it arrives,
  holding only its predictions;
* the fit sees at most :data:`FIT_ROWS` rows, an equal seeded sample of each training event
  (:func:`fit_rows`), so its memory and time do not grow with the split; every scored row is
  scored;
* inside the training events the last ~10% are held out for early stopping on the pooled
  within-group R^2 (:func:`group_r2`): selection by MSE halts a head that has only learned which
  (event, plane) a row belongs to, and almost all of ``u``'s variance is between groups;
* every input column -- features and ``extra`` (presence bits, geometry) alike -- is
  STANDARDISED from the training rows: one massive-activation channel (measured: 4836 against a
  next-largest 462) otherwise sets the input scale, and ``extra`` left in its own units next to
  thousands of unit-variance features is ignored (the random arm scored 0.005 with geometry
  columns that alone score 0.121). A column constant over the training rows is centred, not
  scaled, and no column is scaled by more than 1e3;
* the budget is in EPOCHS (at least :data:`MIN_STEPS` steps), so a larger split does not train
  the head less; seeded;
* every arm is fitted ``probe_seeds`` times on the same rows, seeds ``seed + k``. The numbers
  with their bootstrap interval are seed ``seed``'s; beside them ``seed_mean`` /
  ``seed_half_range``, (max - min) / 2 over the fits -- the probe's own variance, which an event
  bootstrap of one fitted probe leaves out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from helix.data.cells import FT, FW, unkey
from helix.model.cells_join import cell_inputs
from helix.eval.events import VIEWS, bootstrap

#: Fraction of the evaluated events the probe is fitted on; the rest are scored.
TRAIN_FRAC = 2 / 3
#: Rows one probe is fitted on, at most. A row holds n_bands x d float16 features (6 KB at
#: 4 x 768), so ~6 GB per probe; an event has ~9k patch rows (SBND campaign), so the budget binds
#: from ~110 training events on.
FIT_ROWS = 1_000_000
EPOCHS, MIN_STEPS, BATCH, LR = 6, 300, 4096, 1e-3
CHUNK = 16384                       # rows per inference / statistics chunk


@dataclass
class ProbeContext:
    """What the probe is built from: the run, the encoder it probes, how many leading events it
    fits on, the seeds, and how many fits per arm."""
    run: object
    model: object
    n_train: int
    seed: int = 0
    n_boot: int = 200
    probe_seeds: int = 2


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


def plane_group(e, g):
    """One id per (event, plane gid); :func:`group_event` inverts it."""
    return np.asarray(e, np.int64) * 16 + np.asarray(g, np.int64)


def group_event(k):
    return np.asarray(k, np.int64) // 16


def group_r2(p, y, group):
    """Per group: ``(keys, sse, sst)`` of prediction ``p`` against truth ``y`` once both are
    centred within the group, so a per-group offset costs nothing. The pooled within-group
    ``R^2 = 1 - sum(sse) / sum(sst)``; unlike a correlation it also charges a prediction of the
    wrong scale."""
    keys, inv = np.unique(np.asarray(group), return_inverse=True)
    inv, G = inv.ravel(), len(keys)
    n = np.bincount(inv, minlength=G)
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    dp = p - (np.bincount(inv, p, G) / n)[inv]
    dy = y - (np.bincount(inv, y, G) / n)[inv]
    return keys, np.bincount(inv, (dy - dp) ** 2, G), np.bincount(inv, dy * dy, G)


def seed_spread(points):
    """Over the probe seeds: ``{"seed_mean": {k: mean}, "seed_half_range": {k: (max - min) / 2}}``
    of each key's point value, one dict per seed -- the fit's own variance, which the event
    bootstrap (one fitted probe, resampled events) does not contain."""
    out = {"seed_mean": {}, "seed_half_range": {}}
    for k in points[0]:
        v = np.array([p[k] for p in points], np.float64)
        ok = v[np.isfinite(v)]
        out["seed_mean"][k] = round(float(ok.mean()), 4) if len(ok) else float("nan")
        out["seed_half_range"][k] = round(float(ok.max() - ok.min()) / 2, 4) if len(ok) else float("nan")
    return out


class Probe:
    """A fitted regression head: ``predict(X, extra) -> np.ndarray``. ``X`` (n, d) features (zero
    columns for a geometry-only arm) and ``extra`` (n, k): torch on any device, or numpy."""

    def __init__(self, net, mu, sd, device):
        self.net, self.mu, self.sd, self.device = net, mu, sd, device
        self.val_score = None

    def inputs(self, X, extra):
        import torch
        A = extra if isinstance(extra, torch.Tensor) else torch.from_numpy(np.asarray(extra, np.float32))
        return (torch.cat([X.to(self.device).float(), A.to(self.device).float()], 1) - self.mu) / self.sd

    def predict(self, X, extra):
        import torch
        with torch.no_grad():
            out = [self.net(self.inputs(X[s:s + CHUNK], extra[s:s + CHUNK])).float().squeeze(-1).cpu()
                   for s in range(0, len(extra), CHUNK)]
        return torch.cat(out).numpy() if out else np.zeros(0, np.float32)


def fit_probe(X, extra, y, event, *, group=None, seed=0, device=None):
    """Fit the probe head on training rows.

    ``X``: (n, d) features (torch, CPU; d = 0 for geometry alone); ``extra``: (n, k) float32
    appended to them; ``y``: (n,) target; ``event``: (n,) event index; ``group``: (n,) int --
    select on the within-group R^2 rather than validation MSE.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    event = np.asarray(event)
    y = np.asarray(y, np.float32)
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
                        nn.GELU(), nn.Linear(512, 1)).to(dev)
    probe = Probe(net, mu, sd, dev)

    evs = np.unique(event)
    if len(evs) < 2:
        raise ValueError("a probe fit needs rows from at least 2 events (one held out to stop on)")
    va = np.isin(event, evs[-max(1, int(round(0.1 * len(evs)))):])
    tri, vai = np.nonzero(~va)[0], np.nonzero(va)[0]
    Xg, Ag = X.to(dev), A.to(dev)
    yg = torch.from_numpy(y).to(dev)

    def score():
        p = probe.predict(Xg[vai], Ag[vai])
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
        loss = F.mse_loss(net(probe.inputs(Xg[b], Ag[b])).squeeze(-1), yg[b])
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
                         "variance, or a NaN loss)")
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
    (:func:`group_r2`), so the coarse between-event and between-plane structure
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
