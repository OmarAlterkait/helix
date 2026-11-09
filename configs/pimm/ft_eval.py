"""pimm config: the evaluation protocol -- a fixed-budget fine-tune of a pretrained FM.

The primary evaluation of a pretrained encoder (docs/DECISIONS.md, 2026-11-09): the denoiser
(``denoise.py``) trained on a fixed 1,024-event labelled subset (``subset_seed=0``, the same
events for every encoder) for 16 epochs, then scored on the probe split by

    python -m helix.eval $HELIX_EXP/<run> --out $HELIX_EXP/<run>/eval.json

Held fixed so that the encoder is what varies: the point charge head, conditional charge (charge
regressed on charged cells only), no linear-amplitude input. Vary ``model.fm`` (the encoder),
``seed`` (the fine-tune's own spread) and ``model.pretrained=False`` (the from-scratch floor);
``model.freeze_encoder=True`` is the frozen diagnostic, scored the same way.

The length is in epochs and the global batch is the rank count (one event per rank), so
1,024 x 16 / 4 = 4,096 steps only on 4 ranks (one node): any other world size is refused rather
than run at another length. A study is ``configs/sweeps/ft_eval.yaml``.

    scripts/submit.sh ft_eval --run.name ft_<encoder> --resources.nnodes 1 -- model.fm=<FM run>
"""
import os as _os

_base_ = ["./denoise.py"]

# WORLD_SIZE under torchrun, SLURM_NTASKS inside a job step; neither outside a job (a config
# loaded by hand, the launcher's preflight), where the four ranks are what will be asked for
_ws = _os.environ.get("WORLD_SIZE") or (_os.environ.get("SLURM_NTASKS") if "SLURM_STEP_ID" in _os.environ else None)
if _ws is not None and int(_ws) != 4:
    raise ValueError(f"ft_eval: WORLD_SIZE={_ws}; the protocol is 1024 events x 16 epochs at one event per rank "
                     "on exactly 4 ranks (4096 steps). Run it on one 4-GPU node.")

epoch = 16
model = dict(cond_charge=True, dist="point", lin_amp_scale=0.0)
data = dict(train=dict(n_events=1024, subset_seed=0))

del _os, _ws
