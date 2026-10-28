"""MAE token masking — model-owned: the mask is part of the objective, and the
one-argument ``model(batch)`` contract leaves nowhere to pass one in.
"""
from __future__ import annotations

import torch


#: Views (U, V, Y) per TPC volume. `plane_id` is the global plane gid, so
#: ``gid // VIEWS_PER_VOLUME`` is the volume and ``gid % VIEWS_PER_VOLUME`` the
#: view — the same decomposition tokenize and probe/designs already assume.
VIEWS_PER_VOLUME = 3


def make_mask(B, mode, ratio, n_planes, gen=None):
    """mode: random | plane.

    * ``random`` — ``ratio`` of tokens, drawn independently.
    * ``plane``  — ``n_planes`` whole planes PER VOLUME; every volume punctured
      (``ratio`` is unused).
    """
    n = B["n_cells"]; dev = B["plane_id"].device
    rnd = (lambda *s: torch.rand(*s, generator=gen, device=dev)) if (gen is not None and gen.device.type == dev.type) \
        else (lambda *s: torch.rand(*s, device=dev))
    if mode == "random":
        return rnd(n) < ratio
    if mode != "plane":
        raise ValueError(f"mask mode must be 'random' or 'plane', got {mode!r}")
    # Cross-plane: hide whole planes. Puncturing EVERY volume, measured on a real
    # R1 event (gids 0..5 = 2 volumes x 3 views):
    #
    #   n=1   32.9% of cells   every volume down to 2/3 views
    #   n=2   66.1% of cells   every volume down to 1/3 views
    #
    # Difficulty is set by the views left in a punctured volume. Scored on a
    # cooldown checkpoint, 388-event probe split:
    #
    #   random, 0.75   masked   var_expl 0.703
    #   plane,  n=1    0.333    var_expl 0.506
    #   plane,  n=2    0.665    var_expl 0.015   <- one view left: collapse
    #
    # Hiding a third of the cells as whole planes is far harder than hiding 75%
    # at random: cross-plane is the hard axis, not the masked fraction.
    #
    # randperm takes the generator, so a fixed `gen` reproduces the mask (the
    # evaluator relies on that).
    gid = B["plane_id"]
    gids = torch.unique(gid)
    shuffle = lambda g: g[torch.randperm(len(g), generator=gen, device=dev)]
    # volume = gid // 3, view = gid % 3: the convention across the codebase
    # (`toff[plane_gid % 3]` in tokenize's tick transform, `vol, pl = plane // 3,
    # plane % 3` in probe/designs).
    vol = torch.div(gids, VIEWS_PER_VOLUME, rounding_mode="floor")
    pick = torch.cat([shuffle(gids[vol == v])[:n_planes]
                      for v in torch.unique(vol)])
    return torch.isin(gid, pick)
