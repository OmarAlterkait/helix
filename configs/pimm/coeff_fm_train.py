"""pimm config: a real training run on the corpus, at the m113 architecture.

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
# Edges derived from THIS corpus (scripts/derive_coeff_bins.py). m113's came from
# the old white-noise cache and are mis-sized per band here — see NOISE_BANDS.md.
# K=128 needs Ampere: the logits are (n_cells, n_slot, K), ~2.4 GB at a full event.
BINS = "/sdf/data/neutrino/omara/archive/coeff_bins_run0027575715.pt"

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
save_path = "/sdf/data/neutrino/omara/exp/coeff_fm_train"

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
# ALL THREE are global and all three are asserted divisible by world_size
# (default_config_parser: `batch_size_val is None or batch_size_val % world_size
# == 0`). batch_size_val = 1 therefore aborts any multi-rank launch at setup,
# before a single step runs — found by the first real 4-GPU launch.
batch_size = 4                # 4 ranks x 1 event = m113's effective batch
batch_size_val = 4            # likewise 1 event per rank
batch_size_test = 4
num_worker = 4

epoch = 25                    # m113 saw each event ~25x; see STEPS below
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
    d=512,
    blocks=12,
    dec_blocks=4,
    heads=8,
    n_bins=128,
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
    # m113 masked whole planes on 10% of steps. Without it nothing ever forces
    # cross-plane triangulation — a different pretraining task, not a nudge.
    plane_frac=0.1,
)

# betas are NOT AdamW's default here. m113 used (0.9, 0.95) (mae_ddp.py:110);
# torch defaults to (0.9, 0.999), and passing only lr/weight_decay silently took
# the default. beta2 0.999 is a ~1000-step second-moment window against ~20 at
# 0.95, so one large gradient damps updates ~50x longer — the opposite of what a
# large-LR transformer recipe wants, and worst precisely when paired with a
# too-short warmup.
optimizer = dict(type="AdamW", lr=1.1e-3, weight_decay=0.05, betas=(0.9, 0.95))
# WSD stable phase, as the base run (m113) trained: linear warmup then FLAT.
#
#   m113    4,000 warmup steps, then constant 1.1e-3, no horizon baked in
#   (was)   OneCycleLR pct_start=0.25 -> 25% of steps warming, cosine to ~0
#
# Warmup is given in ABSOLUTE STEPS, not a rate. Trainer.build_scheduler
# (pimm engines/train.py:733) OVERWRITES cfg.scheduler.total_steps with
# iters_per_epoch * epoch, unconditionally — so `warmup_rate=4000/1_010_000`
# was reinterpreted against a 1,500-step run and became 5.94 steps of warmup:
# full 1.1e-3 by step 7 instead of step 4000, 571x research's LR at that point,
# on a cold 12-block d=512 transformer. WSDStableLR ignores total_steps (a flat
# phase needs no horizon) so the schedule cannot be rescaled by max_len, epoch
# or GPU count.
#
# For the cooldown, swap in WSDCooldownLR (1 - sqrt(p)), which is bit-identical
# to research's lr_mode="decay". pimm's PolyLR is a different curve.
# Every cadence below is DERIVED from the run length. Hard-coding m113's
# absolute constants onto a shorter run is the error this file already made
# twice: `warmup_rate` reinterpreted against a 1,500-step run (~6 steps), then
# `warmup=4000` on a 4,425-step run (90% of the run). m113's values are
# FRACTIONS of a 1,010,000-step run, so that is how they are expressed.
#
#   warmup   4,000 / 1,010,000 = 0.40%
#   eval    10,000 / 1,010,000 = 0.99%
#   save     2,000 / 1,010,000 = 0.20%
# Resolved from the identity split below, recorded in holdout.json beside the
# corpus. A literal, not a fraction of 19,999, so that changing HOLDOUT without
# re-resolving is a visible inconsistency rather than a silent rescale.
N_TRAIN_EVENTS = 19_034
STEPS = N_TRAIN_EVENTS * epoch // batch_size
WARMUP = max(100, round(0.0040 * STEPS))
EVAL_EVERY = max(50, round(0.0099 * STEPS))
SAVE_EVERY = max(50, round(0.0020 * STEPS))

scheduler = dict(type="WSDStableLR", warmup=WARMUP)

# ---------------------------------------------------------------------------
# data — a handful of events, so the run is minutes not hours
# ---------------------------------------------------------------------------
transform = [
    # grid_center, because that is what the long run used. m113's config does not
    # set `cellt`, so it took mae_ddp.py:54's default "canonical" — which is
    # helix's "grid_center" (identical formula; helix renamed it). The two differ
    # on ~94% of cells, so this is not a detail.
    #
    # PatchConfig's default is "centroid", which research measured as the better
    # probing representation (3D probe 0.60 vs 0.42). That makes centroid worth
    # an ablation once probes exist — but it is not what the reference run did,
    # so it is not the default here.
    dict(type="CoeffTokenize", part="coeff", clean_part="coeff_clean",
         cfg=dict(cell_t="grid_center"),
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
)

# A REAL split, keyed on event IDENTITY rather than position.
#
# This was `event_range=(0, N_TRAIN)` etc. A positional slice is a valid split
# exactly once: the joint index is built over a GLOB of shards, so adding a
# shard renumbers everything after it and a probe holdout defined as
# (17700, 18000) silently becomes a set of trained-on events. This corpus is
# explicitly built to grow — one run now, eight planned — so that is the failure
# that would actually happen, and nothing would report it.
#
# blake2b(run/source_file) + event is the SIMULATION event's identity, so the
# split survives shard add/remove/reorder and even a corpus rebuild with a
# different basis or noise model. Fractions cover [0,1) with no gap, so no event
# lands in nothing — the positional scheme left 999 events (5%) unassigned.
#
# Resolved once and written to holdout.json beside the corpus: reproducible is
# not the same as auditable, and a probe holdout that exists only as code cannot
# be inspected, diffed or cited.
#
#   train 19,034   val 577   probe 388   (= 19,999, the whole corpus)
HOLDOUT = dict(seed=0, fractions=dict(train=0.95, val=0.03, probe=0.02))

data = dict(
    train=dict(**_common, holdout=HOLDOUT, split_role="train"),
    val=dict(**_common, holdout=HOLDOUT, split_role="val"),
    test=dict(**_common, holdout=HOLDOUT, split_role="val"),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=1),
    # Research logs every 200 steps; log_frequency=1 would emit ~1M console
    # lines and TB rows on a full run.
    dict(type="InformationWriter", log_frequency=200),
    # EMA is not polish for a WSD run: the stable phase is flat by design, so the
    # raw weights sit at full LR noise for the whole run and the EMA is what
    # stands in for an annealed model until a cooldown is actually run.
    # save_freq matches CheckpointSaver so the EMA and the weights land within a
    # step of each other; the hook logs a warning if they diverge on resume.
    dict(type="WeightEMA", decay=0.9999, save_freq=SAVE_EVERY),
    # Was every_n_steps unset -> after_epoch only -> exactly ONE eval, after
    # training. No training curve, and model_best selection was vacuous.
    dict(type="CoeffFMEvaluator", every_n_steps=EVAL_EVERY, max_batches=200),
    # Was save_freq=None -> CheckpointSaver.after_step returns early and only
    # after_train saves. On a preemptible partition a long run could never make
    # progress. Research saves every 2000 ("preemption loses <= this many").
    dict(type="CheckpointSaver", save_freq=SAVE_EVERY),
]

train = dict(type="FMTrainer")
