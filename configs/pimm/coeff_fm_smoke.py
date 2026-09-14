"""pimm config: a few real FMTrainer steps on the corpus — the wiring smoke test.

Trains a SMALL model FROM SCRATCH over a handful of events. The point is not the
model; it is that pimm's own Trainer, hooks, optimizer, scheduler, evaluator and
checkpointing all run against helix's model and pimm-data's corpus.

Deliberately from scratch rather than from m113:

  * it exercises the ``bins=`` path (a categorical head with no checkpoint has no
    edges, and they are training-set statistics the model cannot invent)
  * m113 is d=512 and a full event is ~31-40k tokens; training it needs more than
    an 11 GB Turing card has, while the wiring is identical at any width

Run (Turing, 1 GPU)::

    srun --partition=turing --account=mli:cider-ml --gpus=1 --cpus-per-task=4 \\
         --mem=16384M --time=0:30:00 singularity exec --nv -B /sdf,/lscratch \\
         /sdf/data/neutrino/omara/images/helix-train.sif \\
         bash -lc 'python3 -m pimm.train --config-file .../coeff_fm_smoke.py'
"""

# Put helix AND the pimm-data checkout on sys.path BEFORE custom_imports is read.
# pimm's Config.fromfile executes this file first and only then processes
# `custom_imports` (pimm/utils/config.py:394-400), so this is enough to make
# `helix.integrations.pimm` importable without helix being installed.
#
# It matters because pimm's scripts/train.sh hard-sets PYTHONPATH to its own code
# directory in every branch, clobbering anything the caller exported — so a
# launch through `pimm submit` cannot see either package by environment alone.
# Doing it here keeps the config runnable under `pimm submit`, a bare `torchrun`,
# or a direct `python -m pimm.train`, with no image change and nothing to install.
#
# BOTH are needed, for different reasons. helix is absent from the pimm image
# entirely; pimm_data IS installed there but at 0.3.0, which predates the coeff
# corpus and has no CoeffTPCDataset — so the checkout has to WIN over
# site-packages, not merely be present.
#
# insert(1), which is the only position that works:
#   insert(0) is defeated by pimm's own loader — Config._file2dict does
#     sys.path.insert(0, temp_dir) -> import_module -> sys.path.pop(0), and this
#     file executes during that import, so an insert(0) here is what pop deletes.
#   append survives the pop but loses to site-packages' stale pimm_data 0.3.0
#     ("No module named 'pimm_data.coeff'").
# insert(1) sits just under pimm's temp dir: the pop removes the temp dir and
# leaves ours at the front.
#
# NOT derived from __file__: pimm copies the config into a temporary module
# before executing it (Config._file2dict), so __file__ points at the temp copy.
# HELIX_ROOT / PIMM_DATA_SRC override, so this is not pinned to one checkout.
import os as _os
import sys as _sys

for _v, _p in (("HELIX_ROOT", "/sdf/group/neutrino/omara/helix"),
               ("PIMM_DATA_SRC", "/sdf/group/neutrino/omara/pimm-data/src")):
    _p = _os.environ.get(_v) or _p
    # Cannot self-locate here: pimm copies the config to a temp file before
    # executing it, so __file__ is the copy, and helix is not importable yet --
    # putting it on the path is this block's whole job. So the default is a
    # literal, and the only defence is to REFUSE a path that is not there.
    # Without this, a wrong or absent checkout surfaces much later as a bare
    # ImportError from custom_imports with the real ModuleNotFoundError
    # swallowed by import_modules_from_strings -- after a job has queued and
    # started.
    if not _os.path.isdir(_p):
        raise SystemExit(
            f"{_v} does not exist: {_p}\n"
            f"  Set {_v} to your checkout. This config cannot derive it: pimm\n"
            f"  executes a temp copy, so __file__ points at the copy, and helix\n"
            f"  is not importable until this block puts it on sys.path.")
    if _p not in _sys.path:
        _sys.path.insert(1, _p)
# REQUIRED, not tidiness. Config._file2dict keeps every module-level name that
# does not start with `__` (pimm/utils/config.py:261-262), so these would enter
# the config dict as MODULE OBJECTS. Config.dump then renders
# `_os = <module 'os' ...>` and yapf rejects it —
# `YapfError: <unknown>:1:5: invalid syntax` — killing the run during setup,
# before step 1. Observed on the first launch after the bootstrap was added.
# helix is importable now (the block above put it on sys.path), so external
# paths come from the ONE table in helix/paths.py rather than from literals
# repeated across seven configs. Values stay plain strings: pimm dumps this
# config and re-reads the dump, and a PosixPath would not survive that.
from helix.paths import root as _root                      # noqa: E402

del _os, _sys, _v, _p

custom_imports = dict(
    imports=["helix.integrations.pimm"],
    allow_failed_imports=False,
)

# r1 (tau=0.05), like every other training config. coeff_fm_encode.py stays on
# the pre-tau vintage deliberately -- it restores m113, which trained there.
CORPUS = str(_root("HELIX_CORPUS"))
# K=32 rather than the corpus default K=128. Not a wiring choice — a memory one:
# the categorical head's logits are (n_cells, n_slot, K), so at a full 31-40k-cell
# event K=128 is ~2.4 GB of logits before gradients, which does not fit an 11 GB
# Turing card. The real training run wants K=128 on Ampere; this exercises the
# same code path at a size Turing can hold.
BINS = str(_root("HELIX_ARCHIVE") / "coeff_bins_K32_smoke.pt")

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
weight = None
resume = False
evaluate = True
test_only = False
seed = 0
# SHARED storage, not /lscratch: that is node-local, so a checkpoint written
# there by a compute node is gone the moment the job ends — the save succeeds and
# logs, and the artifact simply is not there afterwards.
save_path = str(_root("HELIX_EXP") / "coeff_fm_smoke")

# pimm's `batch_size` is the GLOBAL batch across all ranks — default_config_parser
# asserts `batch_size % world_size == 0` and derives batch_size_per_gpu from it.
# What the FM requires is batch_size_per_gpu == 1 (it has no event separation, so
# a rank holding two events would attend across them). So:
#
#     batch_size = number of GPUs        1 GPU -> 1,  2 GPUs -> 2,  8 GPUs -> 8
#
# which is exactly how the research trainer scaled: "each rank processes 1
# event/step; gradients all-reduced => global batch = world events".
# See MULTI_EVENT_BATCHING.md.
batch_size = 1
batch_size_val = 1
batch_size_test = 1
num_worker = 2

epoch = 1
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
param_dicts = None            # FMTrainer refuses this being set; muP comes from the model

# Read by pimm/train.py (the ENTRYPOINT), not by the Trainer — a config-contract
# test that only scans engines/train.py for `self.cfg.<name>` misses these.
structured_logging = dict(
    enabled=False,
    trace_hooks=False,
    batch_stats_every=1,
    max_file_size_mb=128,
    backup_count=3,
)

# ---------------------------------------------------------------------------
# model — small, from scratch, with edges derived from THIS corpus
# ---------------------------------------------------------------------------
model = dict(
    type="Coeff-FM",
    bins=BINS,
    n_slot=128,               # pw=16 * pt=8, fixed by the tokenizer
    n_band=4,                 # A4, D4, D3, D2 (D1 dropped)
    n_plane=6,
    d=256,                    # d/d_base = 2, so muP actually scales (at d=128, m=1)
    blocks=2,
    dec_blocks=1,
    heads=4,
    n_bins=32,
    dec_mode="cross",         # SerialFMModel's decoder is grouped-cross
    mup=True,
    d_base=128,
    # Stated, not inherited. serial/rope_split/gp/gd leave NO trace in the
    # weights, so a checkpoint written by this run cannot record what it used —
    # which is exactly how m113 ended up evaluable only by guessing. Each of the
    # four moves every golden digest. rope_split=False matches m113; flip it
    # deliberately if you mean to.
    serial=True,
    rope_split=False,
    gp=1024,
    gd=2048,
)

optimizer = dict(type="AdamW", lr=3e-4, weight_decay=0.05)
scheduler = dict(type="OneCycleLR", max_lr=3e-4, pct_start=0.25,
                 anneal_strategy="cos", div_factor=10.0, final_div_factor=1000.0)

# ---------------------------------------------------------------------------
# data — a handful of events, so the run is minutes not hours
# ---------------------------------------------------------------------------
transform = [
    # cell_t is REQUIRED — PatchConfig has no default, because a run that means
    # to be comparable with m113 must say which one it chose: the two differ on
    # ~94% of cells and nothing downstream reports it. ('centroid' is the other
    # option; research's 3D probe 0.60 vs 0.42 for it is inflated by leakage, see
    # coeff_fm_train.py.)
    dict(type="CoeffTokenize", part="coeff", clean_part="coeff_clean",
         cfg=dict(cell_t="grid_center"),   # as the long run (m113) did
         fm_names=True),
    dict(type="CoeffCollect", part="coeff"),
]

_common = dict(
    type="CoeffTPCDataset",
    data_root=CORPUS,
    dataset_name="sim_wire",
    modalities=("coeff", "coeff_clean"),
    transform=transform,
    # 32 events, so a 2-rank run still has 16 steps/rank: OneCycleLR's first
    # phase is `pct_start * total_steps - 1`, which is DEGENERATE (zero length,
    # ZeroDivisionError) when total_steps gets small — 8 events over 2 ranks
    # gives 4 steps and 0.25 * 4 - 1 = 0.
    max_len=32,
)

data = dict(train=dict(**_common), val=dict(**_common), test=dict(**_common))

hooks = [
    # MUST stay in this list. pimm dumps the RESOLVED config to
    # <save_path>/config.py, which keeps `custom_imports` but drops the sys.path
    # bootstrap at the top of this file (a statement, not a dict entry) — and
    # train.sh's resume branch loads THAT file, with PYTHONPATH set to pimm's own
    # code snapshot. Without this hook every chained/requeued job dies on a bare
    # ImportError after job 1 has spent its full allocation.
    dict(type="HelixPathBootstrap"),
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=1),
    dict(type="InformationWriter"),
    # every_n_steps so the smoke run actually exercises the eval -> model_best
    # path; with it unset the evaluator fires once, after_epoch, and the saver
    # never sees an eval step.
    dict(type="CoeffFMEvaluator", every_n_steps=5, max_batches=4),
    # `evaluator_every_n_steps` IS set here, unlike the real run: the smoke test
    # exists to prove the wiring works, and model_best is part of the wiring even
    # though coeff_fm_train deliberately declines to use it (a flat LR has no
    # best step — see there). Keeping it exercised means the choice not to use it
    # stays a choice rather than decaying into a broken path nobody would notice.
    dict(type="CheckpointSaver", save_freq=10, evaluator_every_n_steps=5),
]

train = dict(type="FMTrainer")

# _root is used above and must NOT survive into the config dict: Config._file2dict
# keeps every module-level name not starting with `__`, and Config.dump then
# renders it as a function repr that yapf rejects -- the same failure the
# `del _os, _sys` above exists for.
del _root
