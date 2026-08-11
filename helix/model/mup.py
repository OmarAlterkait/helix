"""muP plumbing that a trainer needs — pure, so it is testable without a trainer.

``FMModel.param_groups`` emits AdamW groups whose per-group ``lr`` encodes muP:
hidden weights get ``base_lr / m`` for width multiplier ``m = d / d_base``. That
encoding is fragile in exactly one place — the scheduler. Anything that assigns a
single LR to every group silently discards muP, and the run looks fine: the loss
still falls, just to a worse place, and only at another width would you notice.

These two helpers are what a trainer integration needs to avoid that, kept here
rather than in ``helix.integrations.pimm`` so they can be tested without pimm
installed.
"""

from __future__ import annotations


def param_group_ratios(param_groups, base_lr):
    """Per-group ``lr / base_lr`` — the muP ratios to preserve under a schedule.

    Capture these right after building the optimizer, before any scheduler has
    touched it. ``fm/mae_ddp.py`` does the same thing and reapplies them every
    step (``pg["lr"] = lr_at(step) * ratio[i]``).
    """
    if base_lr is None or base_lr == 0:
        raise ValueError(f"base_lr must be a non-zero number, got {base_lr!r}")
    out = []
    for i, g in enumerate(param_groups):
        lr = g.get("lr")
        if lr is None:
            raise ValueError(f"param group {i} has no 'lr'; muP ratios are undefined")
        out.append(lr / base_lr)
    return out


def expand_max_lr(max_lr, ratios):
    """A scheduler's ``max_lr`` -> a per-group list carrying the muP ratios.

    ``OneCycleLR`` accepts ``max_lr`` as a scalar OR a per-group list; the list
    is how layer-wise schedules are expressed. A scalar assigns every group the
    same peak, which flattens muP — so a scalar is expanded here rather than
    passed through.

    An already-correct list passes through unchanged, so a config may still
    specify peaks per group explicitly. A list of the wrong length is an error
    rather than something to pad or truncate.
    """
    if isinstance(max_lr, (list, tuple)):
        if len(max_lr) != len(ratios):
            raise ValueError(
                f"max_lr has {len(max_lr)} entries but the optimizer has "
                f"{len(ratios)} param groups — these must correspond one-to-one")
        return list(max_lr)
    return [max_lr * r for r in ratios]
