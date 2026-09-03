"""MAE token masking — model-owned.

Extracted verbatim from ``research/coeff_foundation_model/fm/train.py``. It sat
in the training loop; the mask is part of the objective, and the one-argument
``model(batch)`` contract leaves nowhere to pass one in, so it moves with the
model.
"""
from __future__ import annotations

import torch


#: Views (U, V, Y) per TPC volume. `plane_id` is the global plane gid, so
#: ``gid // VIEWS_PER_VOLUME`` is the volume and ``gid % VIEWS_PER_VOLUME`` the
#: view — the same decomposition tokenize and probe/designs already assume.
VIEWS_PER_VOLUME = 3


def make_mask(B, mode, ratio, n_planes, gen=None):
    """mode: random | plane | plane_any | block.

    * ``random``    — ``ratio`` of tokens, drawn independently.
    * ``plane``     — ``n_planes`` whole planes PER VOLUME; every volume punctured.
    * ``plane_any`` — ``n_planes`` whole planes from the event, volumes ignored.
      What research does and what every run to date trained on; use it to
      reproduce or compare against those, not for new runs.
    * ``block``     — a contiguous wire slab (~``ratio`` wide) in every plane.

    ``plane`` and ``plane_any`` differ in whether a fully sighted volume survives,
    which decides whether the model must triangulate or may interpolate — see the
    measurements in the body.
    """
    n = B["n_cells"]; dev = B["plane_id"].device
    rnd = (lambda *s: torch.rand(*s, generator=gen, device=dev)) if (gen is not None and gen.device.type == dev.type) \
        else (lambda *s: torch.rand(*s, device=dev))
    if mode == "random":
        return rnd(n) < ratio
    gid = B["plane_id"]
    if mode in ("plane", "plane_any"):        # cross-plane: hide whole plane(s)
        # Two selections, because they are two different TASKS and the difference
        # is not visible in the loss. Measured on a real R1 event (gids 0..5 =
        # 2 volumes x 3 views), 200 draws each:
        #
        #   plane_any, n=1   16.5% of cells   punctured volume keeps 2/3 views,
        #                                     the OTHER volume keeps all 3, always
        #   plane,     n=1   32.9% of cells   every volume down to 2/3 views
        #   plane,     n=2   66.1% of cells   every volume down to 1/3 views
        #
        # Two views already determine a 3D point, so under `plane_any` the model
        # can interpolate from an intact volume and is never forced to
        # triangulate — even n_planes=3 left a volume untouched in ~90% of draws.
        # `plane` punctures every volume, which is what the mode is FOR.
        #
        # `plane_any` is kept because it is what research/train.py does and what
        # every run to date trained on, so it is the only way to reproduce or
        # compare against them. New runs should prefer `plane`.
        #
        # DELIBERATE DELTA from research in both: research calls randperm without
        # the generator and so draws from the GLOBAL rng — 8 draws with an
        # identical `gen` seed gave 5 distinct masks. Callers that pass `gen` are
        # asking for reproducibility (research's own perband_mse does, per batch,
        # and never got it). At gen=None `plane_any` is byte-identical to it.
        gids = torch.unique(gid)
        shuffle = lambda g: g[torch.randperm(len(g), generator=gen, device=dev)]
        if mode == "plane_any":               # n_planes from the whole event
            return torch.isin(gid, shuffle(gids)[:n_planes])
        # `plane`: n_planes PER VOLUME, every volume punctured. volume =
        # gid // 3, view = gid % 3 is the convention already baked in across the
        # codebase (`toff[plane_gid % 3]` in tokenize's tick transform,
        # `vol, pl = plane // 3, plane % 3` in probe/designs).
        vol = torch.div(gids, VIEWS_PER_VOLUME, rounding_mode="floor")
        pick = torch.cat([shuffle(gids[vol == v])[:n_planes]
                          for v in torch.unique(vol)])
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
