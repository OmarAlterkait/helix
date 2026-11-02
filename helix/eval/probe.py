"""The probe: an MLP fitted in memory on frozen features. ONE implementation, used by both
``floor`` (charge per fine cell) and ``uprobe`` (the along-wire coordinate per patch).

Protocol, the same for every arm of every metric:

* rows are split by EVENT, never by row (rows of one event are not independent): the first
  :func:`split_events` events are fitted on, the rest scored. They come in that order, so a
  metric fits once the training events are in and scores each later event as it arrives,
  holding only its predictions;
* the fit sees at most :data:`FIT_ROWS` rows, an equal seeded sample of each training event
  (:func:`fit_rows`), so its memory and time do not grow with the split; every scored row is
  scored;
* inside the training events the last ~10% are held out for early stopping, scored by the
  metric the caller reports -- validation MSE, or with ``group`` the within-group correlation
  (:func:`helix.eval.windows.fisher_z`, the reported statistic). Selection by MSE halts a head
  that has only learned which (event, plane) a row belongs to when, as for the along-wire
  coordinate, almost all the target's variance is between groups;
* features are STANDARDISED per channel from the training rows: one massive-activation channel
  (measured: 4836 against a next-largest 462) otherwise sets the input scale and the score
  tracks it, not the representation (two seeds of one recipe: 0.82 and 0.68 on raw inputs).
  ``extra`` columns (presence bits, offsets, geometry) are appended unstandardised;
* the budget is in EPOCHS (at least :data:`MIN_STEPS` steps), so a larger split does not train
  the head less; seeded.
"""

from __future__ import annotations

import numpy as np

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
    """A fitted head: ``predict(X, extra) -> np.ndarray``. ``X`` (n, d) features (zero columns for
    a geometry-only arm) and ``extra`` (n, k): torch on any device, or numpy."""

    def __init__(self, net, mu, sd, device):
        self.net, self.mu, self.sd, self.device = net, mu, sd, device
        self.val_score = None

    def inputs(self, X, extra):
        import torch
        A = extra if isinstance(extra, torch.Tensor) else torch.from_numpy(np.asarray(extra, np.float32))
        return torch.cat([(X.to(self.device).float() - self.mu) / self.sd, A.to(self.device).float()], 1)

    def predict(self, X, extra):
        import torch
        with torch.no_grad():
            out = [self.net(self.inputs(X[s:s + CHUNK], extra[s:s + CHUNK])).squeeze(-1).float().cpu()
                   for s in range(0, len(extra), CHUNK)]
        return torch.cat(out).numpy() if out else np.zeros(0, np.float32)


def fit_probe(X, extra, y, event, *, group=None, seed=0, device=None):
    """Fit the probe head on training rows.

    ``X``: (n, d) features (torch, CPU; d = 0 for geometry alone); ``extra``: (n, k) float32
    appended unstandardised; ``y``: (n,) target; ``event``: (n,) event index; ``group``: (n,)
    int -- select on the within-group correlation rather than validation MSE.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from helix.eval.windows import fisher_z

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    y, event = np.asarray(y, np.float32), np.asarray(event)
    n = len(y)
    s1 = torch.zeros(X.shape[1], dtype=torch.float64, device=dev)
    s2 = torch.zeros_like(s1)
    for c in X.split(CHUNK):                                 # chunked: no float copy of the matrix
        c = c.to(dev).double()
        s1 += c.sum(0)
        s2 += (c * c).sum(0)
    mu = (s1 / n).float()
    sd = (s2 / n - (s1 / n) ** 2).clamp(min=0).sqrt().float().clamp(min=1e-3)
    net = nn.Sequential(nn.Linear(X.shape[1] + extra.shape[1], 1024), nn.GELU(), nn.Linear(1024, 512),
                        nn.GELU(), nn.Linear(512, 1)).to(dev)
    probe = Probe(net, mu, sd, dev)

    evs = np.unique(event)
    if len(evs) < 2:
        raise ValueError("a probe fit needs rows from at least 2 events (one held out to stop on)")
    va = np.isin(event, evs[-max(1, int(round(0.1 * len(evs)))):])
    tri, vai = np.nonzero(~va)[0], np.nonzero(va)[0]
    Xg, Ag = X.to(dev), torch.as_tensor(np.asarray(extra, np.float32)).to(dev)
    yg = torch.from_numpy(y).to(dev)

    def score():
        p = probe.predict(Xg[vai], Ag[vai])
        if group is None:
            return -float(np.mean((p - y[vai]) ** 2))
        z = list(fisher_z(p, y[vai], np.asarray(group)[vai]).values())
        return float(np.tanh(np.mean(z))) if z else float("nan")

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
        raise ValueError("probe: no validation score at any check (no validation group with "
                         "enough rows, or a NaN loss)")
    net.load_state_dict(best_sd)
    probe.val_score = float(best)
    return probe
