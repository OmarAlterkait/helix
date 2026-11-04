"""The ``denoise`` metric: a denoiser's per-cell charge, scored on the floor's windows.

The same scored events, windows and numbers (:class:`helix.eval.floor.Scored`) as ``floor`` --
only the per-cell prediction comes from a trained denoiser instead of a probe on frozen
features, so the numbers are directly comparable with the floor's ``trained`` and ``amplitude``
arms. Cells no token covers are predicted 0.

The metric needs only a predictor, ``predict(event, keys) -> (y, presence)``: ``y`` the
predicted ``log1p(q / Q0)`` and ``presence`` the probability of charge per fine-cell key, both 0
where no token covers the cell. :func:`load_denoiser` makes one from a denoiser run
(``configs/pimm/denoise.py``).

Reported per score: ``charge`` (window score = predicted charge, ``expm1(max(y, 0)) * Q0``) and
``presence`` (window score = the probability).
"""

from __future__ import annotations

import numpy as np

from helix.data.cells import Q0, cell_inputs, cell_neighbors
from helix.eval.floor import Scored


def load_denoiser(path, device=None):
    """``(predict, tokenizer)`` from a denoiser run (``configs/pimm/denoise.py``): its run
    directory -- the checkpoint of the lowest validation loss -- or one checkpoint in it
    (:func:`helix.model.checkpoint.load_run`)."""
    import torch
    from helix.eval.recon import autocast
    from helix.model.checkpoint import load_run

    run = load_run(path, "best", device=device)
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
        return y, pr

    return predict, cfg


class Denoise:
    name = "denoise"

    def __init__(self, ctx):
        self.ctx = ctx
        self.scored = Scored()

    def add(self, ev):
        c = self.ctx
        if ev.index < c.n_train:
            return
        tr = ev.truth(c.seed, windows=True)
        y, pr = c.denoiser(ev, tr["wkey"])
        self.scored.add(ev, tr, {"charge": np.expm1(np.maximum(y, 0)) * Q0, "presence": pr})

    def result(self):
        c = self.ctx
        return dict(n_train_events=c.n_train, **self.scored.result(c.n_boot, c.seed))
