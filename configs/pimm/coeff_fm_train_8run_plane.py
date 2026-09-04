"""8-run pretraining with the cross-plane task turned UP — one variable changed.

Everything else is `coeff_fm_train_8run.py` verbatim: same corpus, same eight
runs, same split, same STEPS/warmup/schedule, same optimizer, same architecture.
Only the masking POLICY moves, so a difference against that run is attributable.

WHY. Scored on the cooldown checkpoint, 388-event probe split, seeded masks:

    policy            masked   val     var_expl   bce     closure
    random 0.75       0.750    2.786   0.703      0.099   1.000
    plane_any n=1     0.169    3.177   0.510      0.115   1.000
    plane     n=1     0.333    3.276   0.506      0.154   1.073
    plane     n=2     0.665    4.832   0.015      0.264   0.523

Two readings drive this config:

1. Cross-plane is the HARD AXIS, and masked fraction is not. Hiding 17% of cells
   as whole planes costs more than hiding 75% at random (0.510 vs 0.703). Three
   runs have now trained the cross-plane task at plane_frac=0.1 and it is still
   the weakest thing the model does. That is the headroom.

2. Difficulty is set by views left in the PUNCTURED volume and by nothing else.
   `plane` and `plane_any` at n=1 score the same (0.506 vs 0.510) though `plane`
   masks twice as many cells, so the intact other volume contributes nothing —
   it is a different region of the detector. What `plane` buys is twice the
   plane-reconstruction examples per step at identical difficulty.

CHOICES.

`n_planes=1`, deliberately not 2. At n=2 one view is left, and one wire plane
gives wire + time — the 3D position the other views encode is underdetermined,
so var_expl 0.015 and closure 0.52 may be near an identifiability ceiling rather
than a training deficit. Training against it would mostly teach the conditional
mean. n=1 is demonstrably hard AND demonstrably learnable.

`plane_frac=0.25`, up from 0.1. With `plane` mode already doubling the examples
per plane step, this is ~5x the historical cross-plane exposure while leaving
three quarters of steps on the random task — which is what teaches within-plane
structure, and where the model is strong (0.703) and should stay strong. Going
further risks the failure the mixer exists to prevent: a run that only ever masks
planes never learns the within-plane task.

`mask_ratio` stays 0.75. It is the other obvious knob and it is deliberately NOT
touched: moving two things at once would make the result unattributable, and 0.75
is where the random task already works.

EVAL stays `mask_mode="random"` (the CoeffFMEvaluator default). val loss must
remain comparable to every previous run, and a plane-masked eval would make it a
mixture. Score the cross-plane number separately with scripts/eval_checkpoint.py
and `--options hooks.4.mask_mode=plane`, which is how the table above was made.
"""

_base_ = ["./coeff_fm_train_8run.py"]

# `plane_mode` is stated even though it is the default, for the reason the base
# states serial/rope_split: it leaves no trace in the weights, so a checkpoint
# cannot record which selection it trained on.
model = dict(plane_frac=0.25, plane_mode="plane", n_planes=1)

save_path = "/sdf/data/neutrino/omara/exp/helix/coeff-fm-train-r1-8run-plane25"
