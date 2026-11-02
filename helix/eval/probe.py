"""The probe: an MLP fitted in memory on frozen features. ONE implementation, used by both
``floor`` (charge per fine cell) and ``uprobe`` (the along-wire coordinate per patch).

Protocol, the same for every arm of every metric:

* rows are split by EVENT, never by row (rows of one event are not independent): the caller
  fits on its training events and scores on the rest (:func:`split_events`). The training events
  come first, so a caller fits once they are in and scores each later event as it arrives,
  holding only its predictions;
* the fit sees at most :data:`FIT_ROWS` rows, an equal seeded sample of each training event
  (:func:`fit_rows`), so its memory and time do not grow with the split; every scored row is
  scored;
* inside the training events the last ~10% are held out for early stopping, scored by the
  metric the caller reports -- validation MSE, or with ``group`` the within-group correlation.
  Selection by MSE halts a head that has only learned which (event, plane) a row belongs to
  when, as for the along-wire coordinate, almost all the target's variance is between groups;
* features are STANDARDISED per channel from the training rows: one massive-activation channel
  (measured: 4836 against a next-largest 462) otherwise sets the input scale and the score
  tracks it, not the representation (two seeds of one recipe: 0.82 and 0.68 on raw inputs).
  ``extra`` columns (presence bits, offsets, geometry) are appended unstandardised;
* the budget is in EPOCHS, so a larger split does not train the head less; seeded.
"""

from __future__ import annotations

import numpy as np

#: Fraction of the evaluated events the probes are fitted on; the rest are scored.
TRAIN_FRAC = 2 / 3

#: Rows one probe is fitted on, at most. A row holds n_bands x d float16 features (6 KB at
#: 4 x 768), so ~6 GB per probe; an event has ~6k floor and ~9k uprobe rows (SBND campaign), so
#: the budget binds from ~110-170 training events on.
FIT_ROWS = 1_000_000


def split_events(n_events):
    """How many of ``n_events`` (in evaluation order) are probe-training events."""
    if n_events < 3:
        raise ValueError(f"{n_events} events: a probe needs at least 3 (fit, early stop, score)")
    return min(n_events - 1, max(2, int(round(TRAIN_FRAC * n_events))))


def fit_rows(n, n_train, rng):
    """The rows (sorted indices into ``n``) one of ``n_train`` training events gives the fit: all
    of them, or a uniform sample of ``ceil(FIT_ROWS / n_train)``."""
    k = -(-FIT_ROWS // n_train)
    return np.arange(n) if n <= k else np.sort(rng.choice(n, k, replace=False))


def gather(feats, idx):
    """Per row, the features of its covering token in each band, concatenated -> (n, nb * d)
    float16 on the CPU; zeros where a band has no token (``idx == -1``; the caller carries the
    absence as presence bits)."""
    import torch
    idx = np.asarray(idx, np.int64)
    n, nb = idx.shape
    d = feats.shape[1]
    X = torch.zeros((n, nb * d), dtype=torch.float16, device=feats.device)
    for b in range(nb):
        m = idx[:, b] >= 0
        if m.any():
            rows = torch.from_numpy(np.nonzero(m)[0]).to(feats.device)
            X[rows, b * d:(b + 1) * d] = feats[torch.from_numpy(idx[m, b]).to(feats.device)].half()
    return X.cpu()


def fisher_r(y, p, group, min_rows=21):
    """``tanh(mean(arctanh r))`` over the groups' Pearson r (see
    :func:`helix.eval.windows.fisher_z`), NaN when no group qualifies."""
    from helix.eval.windows import fisher_z
    z = list(fisher_z(np.asarray(p, np.float64), np.asarray(y, np.float64), np.asarray(group), min_rows).values())
    return float(np.tanh(np.mean(z))) if z else float("nan")


class Probe:
    """A fitted head: ``predict(X, extra) -> np.ndarray``."""

    def __init__(self, net, mu, sd, device, val_score):
        self.net, self.mu, self.sd, self.device, self.val_score = net, mu, sd, device, val_score

    def _inputs(self, X, extra, s, e):
        import torch
        parts = []
        if X is not None:
            parts.append((X[s:e].to(self.device).float() - self.mu) / self.sd)
        if extra is not None:
            parts.append(torch.as_tensor(np.asarray(extra[s:e], np.float32)).to(self.device))
        return torch.cat(parts, 1)

    def predict(self, X, extra=None, chunk=16384):
        import torch
        n = (X if X is not None else extra).shape[0]
        with torch.no_grad():
            out = [self.net(self._inputs(X, extra, s, s + chunk)).squeeze(-1).float().cpu()
                   for s in range(0, n, chunk)]
        return torch.cat(out).numpy() if out else np.zeros(0, np.float32)


def fit_probe(X, y, event, *, extra=None, group=None, epochs=6, batch=4096, lr=1e-3, seed=0, device=None):
    """Fit the probe head on training rows.

    ``X``: (n, d) features (torch, any dtype, CPU) or None; ``extra``: (n, k) float32 appended
    unstandardised, or None; ``y``: (n,) target; ``event``: (n,) event index; ``group``: (n,) int
    -- select on the within-group correlation (:func:`fisher_r`) rather than validation MSE.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    y = np.asarray(y, np.float32)
    event = np.asarray(event)
    n = len(y)
    if X is not None:
        s1 = torch.zeros(X.shape[1], dtype=torch.float64, device=dev); s2 = torch.zeros_like(s1)
        for c in X.split(65536):                             # chunked: no float copy of the matrix
            c = c.to(dev).double(); s1 += c.sum(0); s2 += (c * c).sum(0)
        mu = (s1 / n).float()
        sd = (s2 / n - (s1 / n) ** 2).clamp(min=0).sqrt().float().clamp(min=1e-3)
    else:
        mu = sd = None
    d_in = (X.shape[1] if X is not None else 0) + (extra.shape[1] if extra is not None else 0)
    net = nn.Sequential(nn.Linear(d_in, 1024), nn.GELU(), nn.Linear(1024, 512), nn.GELU(),
                        nn.Linear(512, 1)).to(dev)
    probe = Probe(net, mu, sd, dev, None)

    evs = np.unique(event)
    if len(evs) < 2:
        raise ValueError("a probe fit needs rows from at least 2 events (one held out to stop on)")
    va_ev = evs[-max(1, int(round(0.1 * len(evs)))):]
    va = np.isin(event, va_ev)
    tri, vai = np.nonzero(~va)[0], np.nonzero(va)[0]
    Xg = X.to(dev) if X is not None else None
    Ag = torch.as_tensor(np.asarray(extra, np.float32)).to(dev) if extra is not None else None
    yg = torch.from_numpy(y).to(dev)

    def inputs(rows):
        parts = []
        if Xg is not None:
            parts.append((Xg[rows].float() - mu) / sd)
        if Ag is not None:
            parts.append(Ag[rows])
        return torch.cat(parts, 1)

    def score():
        with torch.no_grad():
            p = torch.cat([net(inputs(torch.from_numpy(c).to(dev))).squeeze(-1).float()
                           for c in np.array_split(vai, max(1, len(vai) // 16384))]).cpu().numpy()
        if group is None:
            return -float(np.mean((p - y[vai]) ** 2))
        r = fisher_r(y[vai], p, np.asarray(group)[vai])
        return r if np.isfinite(r) else -1e9

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(6, epochs * len(tri) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    every = max(1, steps // 6)
    tri_d = torch.from_numpy(tri).to(dev)
    best, best_sd = -np.inf, None
    for s in range(steps):
        b = tri_d[torch.randint(len(tri_d), (batch,), device=dev)]
        loss = F.mse_loss(net(inputs(b)).squeeze(-1), yg[b])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (s + 1) % every == 0 or s + 1 == steps:
            v = score()
            if v > best:
                best, best_sd = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_sd)
    net.eval()
    probe.val_score = float(best)
    return probe
