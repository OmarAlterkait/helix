"""muP plumbing that a trainer needs — pure, so it is testable without a trainer.

``FMModel.param_groups`` emits AdamW groups whose per-group ``lr`` encodes muP:
hidden weights get ``base_lr / m`` for width multiplier ``m = d / d_base``. That
encoding is fragile in exactly one place — the scheduler. Anything that assigns a
single LR to every group silently discards muP, and the run looks fine: the loss
still falls, just to a worse place, and only at another width would you notice.
The WSD schedules are ``LambdaLR``s, which scale each group's own ``lr`` and so
keep the ratios.

The helper is kept here rather than in ``helix.integrations.pimm`` so it can be
tested without pimm installed.
"""

from __future__ import annotations


def param_group_ratios(param_groups, base_lr):
    """Per-group ``lr / base_lr`` — the muP ratios to preserve under a schedule.

    Capture these right after building the optimizer, before any scheduler has
    touched it.
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

