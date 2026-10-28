"""The ``denoise`` metric: a denoiser's per-cell charge, scored on the floor's cells and windows.

The same scored events, map rows, windows, ``Q0`` and :func:`helix.eval.windows.scalars` as
``floor`` -- only the per-cell prediction comes from a trained denoiser instead of a probe on
frozen features, so the numbers are directly comparable with the floor's ``trained`` arm. Cells
no token covers are predicted 0.

The metric needs only a predictor, ``predict(event, keys) -> (y, presence, covered)``: ``y`` the
predicted ``log1p(q / Q0)`` per fine-cell key, ``presence`` a per-cell probability of charge or
None, ``covered`` whether a token covers the cell. :func:`load_denoiser` makes one from a
denoiser run (``configs/pimm/denoise.py``); anything else that maps an
:class:`helix.eval.events.Event` to per-cell charge plugs in the same way.

Reported per score: ``charge`` (window score = predicted charge, ``expm1(max(y, 0)) * Q0``) and,
for a model with a presence head, ``presence`` (window score = the probability); ``charge``
also carries ``val_mse`` on the map rows.
"""

from __future__ import annotations

import numpy as np

from helix.data.cells import Q0, cell_inputs, cell_neighbors, unkey
from helix.eval import windows as W
from helix.eval.floor import score_maps
from helix.eval.probe import split_events


def load_denoiser(path, device=None, weights="best"):
    """``(predict, tokenizer)`` from a denoiser run (``configs/pimm/denoise.py``): its run
    directory -- ``weights`` picks the checkpoint, by default the lowest validation loss -- or
    one checkpoint in it (:func:`helix.model.checkpoint.load_run`)."""
    import torch
    from helix.eval.recon import autocast
    from helix.model.checkpoint import load_run

    run = load_run(path, weights, device=device)
    model, cfg = run.model, run.tokenizer
    if not hasattr(model, "head"):
        raise ValueError(f"{path}: not a denoiser run (model type {run.config['model'].get('type')!r})")
    dev = next(model.parameters()).device

    def predict(ev, keys):
        idx, aux = cell_inputs(keys, ev.cell_key, ev.band_lengths, cfg)
        ok = (idx >= 0).any(1)
        y, pr = np.zeros(len(keys), np.float32), np.zeros(len(keys), np.float32)
        if ok.any():
            nbr = cell_neighbors(keys[ok], ev.cell_key, ev.band_lengths, cfg)
            t = lambda a: torch.as_tensor(a).to(dev)                    # noqa: E731
            with torch.no_grad(), autocast(dev):
                yc, logit = model.predict(ev.batch, t(idx[ok]), t(aux[ok]), t(nbr))
            y[ok], pr[ok] = yc.float().cpu().numpy(), torch.sigmoid(logit.float()).cpu().numpy()
        return y, pr, ok

    return predict, cfg


class Denoise:
    name = "denoise"

    def __init__(self, predict, n_events, tokenizer, *, seed=0, n_boot=200):
        self.predict, self.cfg, self.seed, self.n_boot = predict, tokenizer, seed, n_boot
        self.n_train = split_events(n_events)                 # the floor's scored events, no others
        self.c = dict(p=[], pr=[], y=[], e=[], pl=[], wp=[], wpr=[], wq=[])
        self.meta = []

    def add(self, ev):
        if ev.index < self.n_train:
            return
        tr = W.event_truth(ev.cells, ev.coeff_cells, np.random.default_rng([self.seed, ev.index]))
        p, pr, ok = self.predict(ev, tr["mrows"])
        c = self.c
        c["p"].append(p[ok]); c["y"].append(np.log1p(tr["mq"][ok] / Q0).astype(np.float32))
        c["e"].append(np.full(ok.sum(), ev.index)); c["pl"].append(unkey(tr["mrows"][ok])[0] % 3)
        if pr is not None:
            c["pr"].append(pr[ok])
        if len(tr["wkey"]):
            wp, wpr, _ = self.predict(ev, tr["wkey"])
            c["wp"].append(wp); c["wq"].append(tr["wq"])
            if wpr is not None:
                c["wpr"].append(wpr)
            self.meta += [dict(m, ev=ev.index) for m in tr["meta"]]

    def result(self):
        c = {k: np.concatenate(v) if v else np.zeros(0) for k, v in self.c.items()}
        out = dict(n_train_events=self.n_train, n_windows=len(self.meta))
        scores = [("charge", c["p"], np.expm1(np.maximum(c["wp"], 0)) * Q0)]
        if len(self.c["pr"]):
            scores.append(("presence", c["pr"], c["wpr"]))
        for name, p, ws in scores:
            r = score_maps(p, c["y"], c["e"].astype(np.int64), c["pl"].astype(np.int64), ws, c["wq"], self.meta,
                           self.cfg, self.n_boot, self.seed)
            if name == "charge":
                r["val_mse"] = float(np.mean((c["p"] - c["y"]) ** 2))
            out[name] = r
        return out
