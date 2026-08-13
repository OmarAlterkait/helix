"""The probe head, fitted correctly.

The reference (``probe_3d_mlp.fit_mlp``) trains a fixed 3000 steps x 8192 batch
with replacement and evaluates on one 75/25 event split. Three consequences,
all of which change the reported number without changing the representation:

1. **A fixed step budget is not a fixed amount of training.** 3000 x 8192 =
   24.6M samples is ~73 passes over research's ~337k patches (80 events) but
   only ~15 over ~1.63M (388 events). Running the identical probe on a larger
   holdout trains the head 5x less and scores lower for no scientific reason.
   Here the budget is in EPOCHS, so the measurement is invariant to holdout size.

2. **No early stopping.** ``probe_3d_rigor`` added it specifically — its header
   says it "kills the overfit-driven negative floor and the
   'harder-to-memorize ranks higher' confound" — but the probe that survived to
   the end does not have it. Without it an arm that memorises faster can score
   higher. Here every fold selects on a validation split by the probe's own
   metric.

3. **One 75/25 split.** Every number then rests on a quarter of the holdout with
   no averaging over which events land in eval. ``ridge_oof`` already used
   8-fold event-grouped CV. Here folds are event-grouped and every event gets an
   out-of-fold prediction, which is also what makes the per-(event, plane)
   metric well defined over the whole split.

Kept from the reference because they were already right: standardisation from
TRAIN-fold moments only, splitting by EVENT (never by row, which would leak
patches of the same event across the split), and averaging over seeds.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ProbeHead", "fit_probe"]


class ProbeHead:
    """Two-hidden-layer MLP, matching the reference's ``SmallMLP``."""

    def __init__(self, d_in, hidden=128, dropout=0.1):
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def __call__(self, x):
        return self.net(x).squeeze(-1)


def _event_folds(event, n_folds, seed):
    """Assign whole EVENTS to folds — never rows, which would leak."""
    uniq = np.unique(event)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    fold_of = {int(u): int(order[i] % n_folds) for i, u in enumerate(uniq)}
    return np.array([fold_of[int(e)] for e in event], np.int64)


def fit_probe(X, y, event, plane=None, *, n_folds=5, epochs=40, batch=8192,
              lr=2e-3, weight_decay=1e-2, seeds=(0, 1, 2), val_frac=0.2,
              patience=6, device=None, hidden=128, dropout=0.1, verbose=False):
    """Out-of-fold predictions for every row, averaged over seeds.

    Returns ``(oof, info)``. ``oof[i]`` is predicted by a head that never saw
    row ``i``'s EVENT during training or early-stopping selection.

    **Selection uses the probe's own metric when ``plane`` is given**, not
    validation MSE. This matters more than it sounds: 97.4% of ``u``'s variance
    is BETWEEN ``(event, plane)`` groups while ``fisher_r`` scores WITHIN them,
    so val MSE plateaus almost immediately — long before any within-group
    structure is learned — and MSE-based early stopping halts a head that has
    learned only which group a row belongs to. Measured: an MSE-stopped head
    scored 0.026 on inputs where a plain global least-squares fit reaches 0.189.
    ``probe_3d_rigor`` selects on "val per-event within-plane R^2" for the same
    reason; the probe that survived to the end lost that.
    """
    import torch

    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.float32)
    event = np.asarray(event)
    if X.shape[0] != y.shape[0] or X.shape[0] != event.shape[0]:
        raise ValueError(f"ragged inputs: X {X.shape}, y {y.shape}, event {event.shape}")
    if len(np.unique(event)) < n_folds:
        raise ValueError(
            f"{len(np.unique(event))} events cannot make {n_folds} event-grouped "
            f"folds — reduce n_folds or probe more events")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    folds = _event_folds(event, n_folds, seed=12345)
    oof = np.zeros((len(seeds), X.shape[0]), np.float64)
    stopped_at = []

    for si, seed in enumerate(seeds):
        for f in range(n_folds):
            te = folds == f
            tr_all = ~te
            # A validation slice for early stopping, also split by EVENT so the
            # stopping decision cannot be made on a training event's patches.
            tr_events = np.unique(event[tr_all])
            rng = np.random.default_rng(seed * 1000 + f)
            n_val = max(1, int(round(val_frac * len(tr_events))))
            val_events = set(rng.permutation(tr_events)[:n_val].tolist())
            va = tr_all & np.array([e in val_events for e in event])
            tr = tr_all & ~va
            if not tr.any() or not va.any():
                raise ValueError(f"fold {f}: empty train or val split")

            # Standardise from TRAIN-fold moments only.
            mu = X[tr].mean(0, keepdims=True)
            sd = X[tr].std(0, keepdims=True).clip(1e-6)
            ym = float(y[tr].mean())

            # Standardised copies stay on the CPU; only minibatches cross to the
            # GPU. The design is rows x dims and both grow: at 388 events and
            # 2048 feature dims it is 2.5M x 2064 x 4 B = 20.6 GB, which OOMs a
            # 40 GB card once the model and workspace are resident. A minibatch
            # is ~67 MB. Nothing about the fit changes — only where the array
            # lives between steps.
            def _t(mask):
                return (torch.from_numpy(((X[mask] - mu) / sd).astype(np.float32)),
                        torch.from_numpy((y[mask] - ym).astype(np.float32)))

            Xtr, ytr = _t(tr)
            Xva, yva = _t(va)
            Xte, _ = _t(te)
            ytr_d = ytr.to(dev)

            def _fwd(Xcpu, chunk=65536):
                """Forward a CPU-resident design in chunks."""
                outs = []
                for i in range(0, Xcpu.shape[0], chunk):
                    outs.append(head(Xcpu[i:i + chunk].to(dev, non_blocking=True)))
                return torch.cat(outs) if outs else torch.empty(0, device=dev)
            va_event = event[va]
            va_plane = None if plane is None else np.asarray(plane)[va]
            va_y = y[va]

            torch.manual_seed(seed)
            head = ProbeHead(X.shape[1], hidden, dropout)
            head.net.to(dev)
            opt = torch.optim.AdamW(head.net.parameters(), lr, weight_decay=weight_decay)

            n = Xtr.shape[0]
            steps_per_epoch = max(1, n // batch)
            best, best_state, bad, stop_ep = -np.inf, None, 0, epochs
            for ep in range(epochs):
                head.net.train()
                perm = torch.randperm(n)          # CPU: indexes a CPU design
                for k in range(steps_per_epoch):     # EPOCHS, not a fixed budget
                    idx = perm[k * batch:(k + 1) * batch]
                    xb = Xtr[idx.cpu()].to(dev, non_blocking=True)
                    loss = ((head(xb) - ytr_d[idx]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                head.net.eval()
                with torch.no_grad():
                    vpred = _fwd(Xva)
                    if va_plane is None:
                        score = -float(((vpred.cpu() - yva) ** 2).mean())  # MSE fallback
                    else:
                        from .metrics import fisher_r
                        r, _, _ = fisher_r(va_y, vpred.cpu().numpy() + ym,
                                           va_event, va_plane)
                        score = -1e9 if not np.isfinite(r) else r
                if score > best + 1e-6:
                    best, bad = score, 0
                    best_state = {k: v.detach().clone()
                                  for k, v in head.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        stop_ep = ep + 1
                        break
            if best_state is not None:
                head.net.load_state_dict(best_state)
            stopped_at.append(stop_ep)

            head.net.eval()
            with torch.no_grad():
                oof[si, te] = _fwd(Xte).cpu().numpy().astype(np.float64) + ym
            if verbose:
                print(f"  seed {seed} fold {f}: stopped at epoch {stop_ep}, "
                      f"val score {best:+.4f}", flush=True)

    return oof.mean(0), dict(selection="fisher_r" if plane is not None else "mse",
                             n_folds=n_folds, seeds=list(seeds), epochs=epochs,
                             batch=batch, mean_stop_epoch=float(np.mean(stopped_at)),
                             hit_epoch_cap=int(sum(s == epochs for s in stopped_at)),
                             n_rows=int(X.shape[0]),
                             n_events=int(len(np.unique(event))))
