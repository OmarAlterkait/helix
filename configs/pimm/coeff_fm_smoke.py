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
         /sdf/data/neutrino/youngsam/images/pimm-latest.sif \\
         bash -lc 'python3 -m pimm.train --config-file .../coeff_fm_smoke.py'
"""

custom_imports = dict(
    imports=["helix.integrations.pimm"],
    allow_failed_imports=False,
)

CORPUS = "/sdf/data/neutrino/omara/coeff_tpc/run_0027575715"
# K=32 rather than the corpus default K=128. Not a wiring choice — a memory one:
# the categorical head's logits are (n_cells, n_slot, K), so at a full 31-40k-cell
# event K=128 is ~2.4 GB of logits before gradients, which does not fit an 11 GB
# Turing card. The real training run wants K=128 on Ampere; this exercises the
# same code path at a size Turing can hold.
BINS = "/sdf/data/neutrino/omara/archive/coeff_bins_K32_smoke.pt"

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
save_path = "/sdf/data/neutrino/omara/exp/coeff_fm_smoke"

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
    # Pin cell_t explicitly. The default is 'centroid', which research measured
    # as the better probing representation (3D probe 0.60 vs 0.42) — but a run
    # that means to be comparable with m113 must say which one it chose, because
    # the two differ on ~94% of cells and nothing downstream reports it.
    dict(type="CoeffTokenize", part="coeff", clean_part="coeff_clean",
         cfg=dict(cell_t="centroid"),
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
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=1),
    dict(type="InformationWriter"),
    dict(type="CoeffFMEvaluator", max_batches=4),
    dict(type="CheckpointSaver", save_freq=None),
]

train = dict(type="FMTrainer")
