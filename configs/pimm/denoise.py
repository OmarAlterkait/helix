"""pimm config: the supervised denoiser -- a pretrained FM run plus a per-cell head.

Noisy coefficient tokens -> pre-response charge per 2-wire x 16-tick cell, with a presence head
(helix.model.denoise; labels: helix.data.denoise.DenoiseRows on the ``cells`` truth). The recipe
is M6d of DENOISE at tag archive-2026-11, and it is the default:

* the encoder is the FM run ``model.fm`` (default ``$HELIX_EXP/fm_cooldown``, the cooled FM): its architecture, its
  tokenizer (``CoeffTokenize(cfg="model")``, filled by FMTrainer from that run) and, with
  ``model.pretrained=True``, its EMA weights; ``pretrained=False`` is the from-scratch baseline
  on the same architecture; ``model.freeze_encoder=True`` keeps the encoder fixed;
* ``model.cond_charge``: False = M6d (detection, maps), True = M6dc (calorimetry: charge
  regressed on charged cells only);
* ``model.lin_amp_scale``: 0 = the FM's input alone (asinh per slot); S > 0 (M10lin: 30) adds
  the linear amplitude sinh(inp) / S as input columns, zero at the start (docs/DECISIONS.md:
  0 until a d1m run confirms r1's gain);
* ``model.dist``: the charge head -- ``"point"`` (squared error, the default), ``"gauss"`` (a
  learned per-cell width, Gaussian likelihood) or ``"cat"`` (one categorical over no-charge +
  256 charge bins, in place of both heads; needs ``cond_charge=True``). One of the two will be
  chosen by a full-length comparison;
* presence = q > 25 e- (``_SUPPORT_MIN_Q``: the cells-3 truth's own cut, the evaluation's
  ``--presence-q``), shared by ``DenoiseRows`` and the model (cat's bin 0);
* encoder LR 3e-3 (muP-scaled), head LR 1e-2 (``model.lr_head``), linear warmup ("auto") then a
  cosine to 2%; 4 epochs (~38k steps at 16 GPUs, the study's 40k);
* label sampling: ``DenoiseRows``' defaults (whole noise windows, near-activity cells and
  windows); the val split draws with a fixed seed;
* the train/val split is the FM's identity-hash HOLDOUT; best checkpoint
  (``model/model_best.pth``) by validation loss (comparable within one ``dist`` only).

Label efficiency: ``data.train.n_events=N`` trains on a fixed, nested random subset of the train
split (``data.train.subset_seed``); raise ``epoch`` with it, since the length is in epochs. The
evaluation protocol (1,024 events, 16 epochs, 4 ranks) is ``ft_eval.py`` on this base. Scoring is
``python -m helix.eval <this run> --out e.json`` (its ``model/last``; ``--weights best`` for the
checkpoint of the lowest validation loss).

    scripts/submit.sh denoise --resources.nnodes 4 [-- key=value ...]
    scripts/submit.sh denoise --run.name denoise_m6dc -- model.cond_charge=True
    scripts/submit.sh denoise --run.name denoise_gauss -- model.cond_charge=True model.dist=gauss
    scripts/submit.sh denoise --run.name denoise_cat -- model.cond_charge=True model.dist=cat
"""
# helix is importable here because the launcher exports PYTHONPATH; the sys.path block and
# its HelixPathBootstrap hook belong to the base.
from helix.paths import root as _root                            # noqa: E402

_base_ = ["./fm.py"]

save_path = str(_root("HELIX_EXP") / "denoise")
epoch = 4

_SUPPORT_MIN_Q = 25.0                                            # e-: presence = q above this

model = dict(
    _delete_=True,
    type="Coeff-Denoise",
    fm=str(_root("HELIX_EXP") / "fm_cooldown"),
    pretrained=True,
    freeze_encoder=False,
    cond_charge=False,
    lr_head=1e-2,
    lin_amp_scale=0.0,
    dist="point",
    support_min_q=_SUPPORT_MIN_Q,
    compile_blocks=True,
)
optimizer = dict(lr=3e-3)
scheduler = dict(_delete_=True, type="WarmupCosineLR", warmup="auto", floor=0.02)

# ---------------------------------------------------------------------------- data
_tokenize = dict(type="CoeffTokenize", part="coeff", cfg="model")
_collect = dict(type="CoeffCollect", part="coeff")
_val = dict(modalities=("coeff", "cells"),
            transform=[_tokenize, dict(type="DenoiseRows", part="coeff", support_min_q=_SUPPORT_MIN_Q, seed=0),
                       _collect])
data = dict(
    train=dict(modalities=("coeff", "cells"), n_events=None, subset_seed=0,
               transform=[_tokenize, dict(type="DenoiseRows", part="coeff", support_min_q=_SUPPORT_MIN_Q),
                          _collect]),
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

del _root, _tokenize, _collect, _val, _SUPPORT_MIN_Q
