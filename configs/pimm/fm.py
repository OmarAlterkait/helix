"""pimm config: the coefficient FM, stable phase, over every run of the corpus.

A run is four choices, and only these differ between configs:

* the corpus: ``HELIX_CORPUS`` (a run dir; its parent is the corpus, ``_calib/RUNS.txt`` its runs);
* the model: ``model`` below; ``fm_smoke.py`` shrinks it;
* the phase: stable (this file), cooldown from a checkpoint (``fm_cooldown.py``);
* the length: ``epoch``.

Nothing here is a step count. Warmup, eval and checkpoint cadence are ``"auto"``: FMTrainer
derives them from the run's own step count (pimm's ``iters_per_epoch * epoch``, i.e. the
split, the corpus and the GPU count) through :mod:`helix.core.cadence`. Anything else is a
``key=value`` override after the config name, e.g. ``scripts/train.sh fm model.plane_frac=0.25``.

    LINKS=8 scripts/train.sh fm [key=value ...]
"""

# helix onto sys.path before custom_imports is read (pimm executes this file first). From
# HELIX_ROOT, not __file__: pimm executes a temp copy. insert(1): pimm's loader does
# insert(0, temp) -> import -> pop(0), which would remove an insert(0); an append would lose
# to site-packages.
import os as _os
import sys as _sys

_p = _os.environ.get("HELIX_ROOT")
if not _p or not _os.path.isdir(_p):
    raise SystemExit(
        f"HELIX_ROOT is {'not set' if not _p else f'not a directory: {_p}'}.\n"
        f"  Set it to your helix checkout (scripts/train.sh exports it from helix.paths).")
if _p not in _sys.path:
    _sys.path.insert(1, _p)
from helix.paths import root as _root                            # noqa: E402

# module objects must not stay in the namespace: pimm dumps it, and yapf rejects them
del _os, _sys, _p

custom_imports = dict(imports=["helix.integrations.pimm"], allow_failed_imports=False)

# ---------------------------------------------------------------------------- corpus
from helix.data.identity import corpus_runs as _corpus_runs     # noqa: E402
from helix.data.bins import reference_table as _reference_table  # noqa: E402

CORPUS = str(_root("HELIX_CORPUS").parent)
RUNS = _corpus_runs(CORPUS)
BINS = str(_reference_table())          # helix/data/data/reference_bins.json names the grid

# ---------------------------------------------------------------------------- run
weight = None
resume = False
evaluate = True
test_only = False
seed = 0
save_path = str(_root("HELIX_EXP") / "fm")

from helix.paths import world_size as _world_size               # noqa: E402
from helix.paths import scheduler as _scheduler                 # noqa: E402

_WORLD = _world_size()
batch_size = _WORLD                     # one event per rank: the FM has no event separation
batch_size_val = _WORLD
batch_size_test = _WORLD
num_worker = int(_scheduler("train", "workers_per_gpu") or 4) * _WORLD

epoch = 3
eval_epoch = 1
clip_grad = 1.0
sync_bn = False
enable_amp = True
amp_dtype = "bfloat16"
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
matmul_precision = "high"
prefetch_factor = None
detect_anomaly = False
mix_prob = 0
deterministic = False
param_dicts = None                      # FMTrainer takes muP's groups from the model
structured_logging = dict(enabled=False, trace_hooks=False, batch_stats_every=1,
                          max_file_size_mb=128, backup_count=3)

# ---------------------------------------------------------------------------- model
model = dict(
    type="Coeff-FM",
    bins=BINS,
    n_slot=128,                         # pw=16 * pt=8, fixed by the tokenizer
    n_band=4,                           # A4, D4, D3, D2 (D1 dropped)
    n_plane=6,
    d=768,
    blocks=12,
    dec_blocks=4,
    heads=12,                           # head dim 64 (fused_qk needs it)
    n_bins=128,                         # K=128 needs Ampere: logits ~2.4 GB at a full event
    mup=True,
    qk_norm=True,                       # without it the d768 stable phase went NaN
    d_base=128,
    rope_split=False,
    gp=1024,
    gd=2048,
    plane_frac=0.1,
    compile_blocks=True,
    fused_qk=True,
)
# The peak LR scales as sqrt(batch): 4.4e-3 at 16 events per step, the batch the recipe was
# measured at (2.2e-3 was the measured optimum at 4 events, docs/archive/SCIENCE.md sections 5, 9).
# muP transfers it across width only.
optimizer = dict(type="AdamW", lr=4.4e-3 * (_WORLD / 16) ** 0.5, weight_decay=0.05, betas=(0.9, 0.95),
                 fused=True)
scheduler = dict(type="WSDStableLR", warmup="auto")

# ---------------------------------------------------------------------------- data
transform = [
    dict(type="CoeffTokenize", part="coeff", cfg=dict(pw=16, pt=8)),
    dict(type="CoeffCollect", part="coeff"),
]
# The split hashes each event's identity, so it does not depend on the corpus's build,
# shard layout or run list.
HOLDOUT = dict(seed=0, fractions=dict(train=0.95, val=0.03, probe=0.02))
_common = dict(type="CoeffTPCDataset", data_root=CORPUS, split=RUNS, dataset_name="sim_wire",
               modalities=("coeff",), transform=transform, holdout=HOLDOUT)
data = dict(
    train=dict(**_common, split_role="train", max_event_size=600_000),
    val=dict(**_common, split_role="val"),
    test=dict(**_common, split_role="val"),
)

# ---------------------------------------------------------------------------- hooks
hooks = [
    dict(type="HelixPathBootstrap"),    # puts the sys.path block back into pimm's dumped config
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=1),
    dict(type="InformationWriter", log_frequency=200),
    dict(type="FiniteGuard", check_every=100, max_skips=20),
    dict(type="WeightEMA", decay=0.9999, save_freq="auto"),
    dict(type="CoeffFMEvaluator", every_n_steps="auto", max_batches=200),
    dict(type="CheckpointSaver", save_freq="auto"),
]
train = dict(type="FMTrainer")

del _root, _corpus_runs, _reference_table, _world_size, _scheduler, _WORLD
