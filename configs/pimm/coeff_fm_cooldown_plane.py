"""Cooldown for the raised-cross-plane arm — the delta from `coeff_fm_cooldown`.

Inherits the cooldown wholesale (WSDCooldownLR, warmup 0, floor 1e-3, the
three-run subset, 14,264 steps, model_best ON, ~100 evals) so the two arms are
annealed by an identical protocol and the comparison is protocol-free.

Three things differ, and only three:

  * `weight` points at the plane25 stable phase instead of the baseline's.
  * the masking policy is restated, because `_base_` reaches
    coeff_fm_train_8run -> coeff_fm_train, where plane_frac is 0.1. Without this
    the cooldown would anneal the plane25 weights under the BASELINE's policy —
    a silent protocol switch a third of the way through the experiment, visible
    nowhere in the loss.
  * `save_path`.

WARM START, not resume — see the base for why that distinction is load-bearing,
and why the RAW `model/last` is used rather than `model_ema.pth`.

Stable-phase A/B this anneals from (same evaluator, 388-event probe split, EMA
weights, both at 112,677 steps):

    task              base     plane25    delta
    random 0.75       0.6862   0.6788     -0.007
    plane     n=1     0.4329   0.6305     +0.198
    plane_any n=1     0.4169   0.5428     +0.126
"""

_base_ = ["./coeff_fm_cooldown.py"]

model = dict(plane_frac=0.25, plane_mode="plane", n_planes=1)

weight = "/sdf/data/neutrino/omara/exp/helix/coeff-fm-train-r1-8run-plane25/model/last"
save_path = "/sdf/data/neutrino/omara/exp/helix/coeff-fm-cooldown-r1-8run-plane25"
