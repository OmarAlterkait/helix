"""MAE token masking — model-owned.

Extracted verbatim from ``research/coeff_foundation_model/fm/train.py``. It sat
in the training loop; the mask is part of the objective, and the one-argument
``model(batch)`` contract leaves nowhere to pass one in, so it moves with the
model.
"""
from __future__ import annotations

import torch


def make_mask(B, mode, ratio, n_planes, gen=None):
    """mode: random (frac of tokens) | plane (whole plane(s)) | block (wire-slab/plane)."""
    n = B["n_cells"]; dev = B["plane_id"].device
    rnd = (lambda *s: torch.rand(*s, generator=gen, device=dev)) if (gen is not None and gen.device.type == dev.type) \
        else (lambda *s: torch.rand(*s, device=dev))
    if mode == "random":
        return rnd(n) < ratio
    gid = B["plane_id"]
    if mode == "plane":                       # cross-plane: hide whole plane(s)
        gids = torch.unique(gid)
        # DELIBERATE DELTA from research/train.py, which calls randperm without
        # the generator and so draws from the GLOBAL rng: 8 draws with an
        # identical `gen` seed gave 5 distinct masks. Callers that pass `gen` are
        # asking for reproducibility (research's own perband_mse does, per batch,
        # and never got it). With gen=None this is byte-identical to before, and
        # training passes gen=None — so training dynamics are untouched.
        perm = gids[torch.randperm(len(gids), generator=gen, device=dev)]
        pick = perm[:n_planes]
        return torch.isin(gid, pick)
    if mode == "block":                       # contiguous wire-slab per plane (~ratio wide)
        wp = B["wire_pos"]; m = torch.zeros(n, dtype=torch.bool, device=dev)
        for g in torch.unique(gid):
            sel = gid == g
            w = wp[sel]; lo, hi = float(w.min()), float(w.max()) + 1
            win = (hi - lo) * ratio
            start = lo + float(rnd(1)) * max(hi - lo - win, 0.0)
            m[sel] = (w >= start) & (w < start + win)
        return m
    raise ValueError(mode)
