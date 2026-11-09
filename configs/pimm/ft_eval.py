"""pimm config: the evaluation protocol -- a fixed-budget fine-tune of a pretrained FM.

The primary evaluation of a pretrained encoder (docs/STATUS.md): the denoiser
(``denoise.py``) trained on a fixed 1,024-event labelled subset (``subset_seed=0``, the same
events for every encoder) for 16 epochs, then scored on the probe split by

    python -m helix.eval $HELIX_EXP/<run> --out $HELIX_EXP/<run>/eval.json

Held fixed so that the encoder is what varies: the point charge head, conditional charge (charge
regressed on charged cells only), no linear-amplitude input. Vary ``model.fm`` (the encoder),
``seed`` (the fine-tune's own spread) and ``model.pretrained=False`` (the from-scratch floor);
``model.freeze_encoder=True`` is the frozen diagnostic, scored the same way.

The length is in epochs and the global batch is the rank count (one event per rank), so
1,024 x 16 / 4 = 4,096 steps only on 4 ranks (one node): ``required_world_size`` makes the trainer
refuse any other world size rather than run at another length (configs read no world size). A study is ``configs/sweeps/ft_eval.yaml``.

    scripts/submit.sh ft_eval --run.name ft_<encoder> --resources.nnodes 1 -- model.fm=<FM run>
"""
_base_ = ["./denoise.py"]

required_world_size = 4                  # checked by FMTrainer (helix.integrations.pimm.trainer.check_world)
epoch = 16
model = dict(cond_charge=True, dist="point", lin_amp_scale=0.0)
data = dict(train=dict(n_events=1024, subset_seed=0))

