"""pimm config: the supervised denoiser -- a pretrained FM run plus a per-cell head.

Noisy coefficient tokens -> pre-response charge per 2-wire x 16-tick cell, with a presence head
(helix.model.denoise; labels: helix.data.denoise.DenoiseRows on the ``cells`` truth). The recipe
is M6d of docs/archive/DENOISE.md, and it is the default:

* the encoder is the FM run ``model.fm`` (default ``$HELIX_EXP/fm_cooldown``, the cooled FM): its architecture, its
  tokenizer (``CoeffTokenize(cfg="model")``, filled by FMTrainer from that run) and, with
  ``model.pretrained=True``, its EMA weights; ``pretrained=False`` is the from-scratch baseline
  on the same architecture; ``model.freeze_encoder=True`` keeps the encoder fixed;
* ``model.cond_charge``: False = M6d (detection, maps), True = M6dc (calorimetry: charge
  regressed on charged cells only);
* encoder LR 3e-3 (muP-scaled), head LR 1e-2 (``model.lr_head``), linear warmup ("auto") then a
  cosine to 2%; 4 epochs (~38k steps at 16 GPUs, the study's 40k);
* label sampling: ``DenoiseRows``' defaults (whole noise windows, near-activity cells and
  windows, support threshold 0 on the hits label); the val split draws with a fixed seed;
* the train/val split is the FM's identity-hash HOLDOUT; best checkpoint
  (``model/model_best.pth``) by validation loss.

Label efficiency: ``data.train.n_events=N`` trains on a fixed, nested random subset of the train
split (``data.train.subset_seed``); raise ``epoch`` with it, since the length is in epochs. Full
scoring is ``python -m helix.eval <FM run> --metrics denoise --denoiser <this run>``.

    LINKS=4 scripts/train.sh denoise [key=value ...]
    NAME=denoise_m6dc scripts/train.sh denoise model.cond_charge=True
"""
# helix is importable here because the launcher exports PYTHONPATH; the sys.path block and
# its HelixPathBootstrap hook belong to the base.
from helix.paths import root as _root                            # noqa: E402

_base_ = ["./fm.py"]

save_path = str(_root("HELIX_EXP") / "denoise")
epoch = 4

model = dict(
    _delete_=True,
    type="Coeff-Denoise",
    fm=str(_root("HELIX_EXP") / "fm_cooldown"),
    pretrained=True,
    freeze_encoder=False,
    cond_charge=False,
    lr_head=1e-2,
    compile_blocks=True,
)
optimizer = dict(lr=3e-3)
scheduler = dict(_delete_=True, type="WarmupCosineLR", warmup="auto", floor=0.02)

# ---------------------------------------------------------------------------- data
_tokenize = dict(type="CoeffTokenize", part="coeff", cfg="model")
_collect = dict(type="CoeffCollect", part="coeff")
_val = dict(modalities=("coeff", "cells"),
            transform=[_tokenize, dict(type="DenoiseRows", part="coeff", seed=0), _collect])
data = dict(
    train=dict(modalities=("coeff", "cells"), n_events=None, subset_seed=0,
               transform=[_tokenize, dict(type="DenoiseRows", part="coeff"), _collect]),
    val=_val,
    test=_val,
)

# ---------------------------------------------------------------------------- hooks
hooks = [
    dict(type="HelixPathBootstrap"),
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=1),
    dict(type="InformationWriter", log_frequency=50),
    dict(type="FiniteGuard", check_every=100, max_skips=20),
    dict(type="DenoiseEvaluator", every_n_steps="auto", max_batches=16),
    dict(type="CheckpointSaver", save_freq="auto", evaluator_every_n_steps="auto"),
]

del _root, _tokenize, _collect, _val
