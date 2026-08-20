"""pimm config: the stable phase over the FULL eight-run R1 corpus.

`coeff_fm_train.py` trained on one run (19,034 events, 25 epochs, 118,962 steps)
and is kept as-is because it is the record of a completed run. This is its
eight-run counterpart, derived from it so the model, tokenizer, optimizer,
schedule shape and BINS cannot drift.

Corpus: 790 shards over 8 runs, 344 GB, ~158k events. Seven runs have 100 source
files; run_0027670361 has 90 upstream, so it contributes 17,123 train events
rather than ~19,000 — a fact about the simulation, not a build failure (verify it
with `--source-root`, which distinguishes the two).

THE BINS ARE NOT RE-DERIVED, and that is the considered choice. Measured on the
built corpus: bins derived on run 2 alone and on run 1 alone, both at 480 events
so both converged, disagree by 0.129% of a band's span — 0.16 bin widths, which
is SMALLER than the sampling spread of the 120-event table actually shipped
(0.159%). Run-to-run over sampling = 0.81x. The eight runs are one distribution,
so a table re-derived over all of them would move less than the noise already in
the shipped one, while costing value-CE comparability with every earlier run and
invalidating every trained K=128 head. `cent_ratio` agrees to 0.38% median.

THE SPLIT IS NOT CHANGED EITHER. It hashes simulation identity, so growing the
corpus reassigns nothing: run 1's counts are 19,034 / 577 / 388 with one run
built and 19,034 / 577 / 388 with all eight. Holding out a whole RUN was
considered and rejected — the runs are the same distribution (measured above), so
it would test nothing extra while coupling the split to run composition.
"""

_base_ = ["./coeff_fm_train.py"]

# ---------------------------------------------------------------------------
# corpus: the root, with the runs named as the split
# ---------------------------------------------------------------------------
# `split=[run, ...]` is the reader's multi-run form (readers/coeff_tpc.py
# _find_files): it globs each run's directory in turn and records which run each
# shard came from. The base points `data_root` at ONE run directory, which is why
# this cannot be expressed as a bare data_root change.
CORPUS_ROOT = "/sdf/data/neutrino/omara/coeff_tpc_r1"
RUNS = [
    "run_0027575715", "run_0027587651", "run_0027651463", "run_0027654870",
    "run_0027663748", "run_0027668746", "run_0027670361", "run_0027719646",
]

_over = dict(data_root=CORPUS_ROOT, split=RUNS)
data = dict(train=dict(**_over), val=dict(**_over), test=dict(**_over))

# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------
# RESOLVED from the identity split over all eight runs, not scaled from 19,034.
# The base's comment applies with more force here: a literal means that changing
# HOLDOUT or the run list without re-resolving is a visible inconsistency rather
# than a silent rescale, and STEPS is derived from this — a stale value would run
# a job that looks fine and is 8x too short.
#
#   train 150,239   val 4,641   probe 3,111   (= 157,991)
N_TRAIN_EVENTS = 150_239

# epoch=3, NOT the base's 25. This holds the COMPUTE fixed and spends the extra
# corpus on unique events instead of repeats:
#
#   base   19,034 x 25 / 4 = 118,962 steps, each event seen ~25x
#   here  150,239 x  3 / 4 = 112,679 steps, each event seen ~3x
#
# Same step budget to within 5%, so this run is directly comparable to the
# completed one and the difference is attributable to data diversity rather than
# to length. For masked pretraining more unique data at fixed compute is the
# better trade, and holding steps fixed is what makes that claim testable.
#
# To instead hold EPOCHS fixed, set epoch=25: ~939k steps, ~8x the wall clock
# (the base run took ~11 h on 4 GPUs, so ~3.6 days). That is a different
# experiment and wants its own config, not a flag flipped here.
epoch = 3
batch_size = 4
STEPS = N_TRAIN_EVENTS * epoch // batch_size

WARMUP = max(100, round(0.0040 * STEPS))
EVAL_EVERY = max(50, round(0.0099 * STEPS))
SAVE_EVERY = max(50, round(0.0020 * STEPS))
scheduler = dict(type="WSDStableLR", warmup=WARMUP)

save_path = "/sdf/data/neutrino/omara/exp/helix/coeff-fm-train-r1-8run"

# The hook list is re-stated because the cadences above are new values, and a
# `_base_` merge would otherwise keep the base's numbers inside the hook dicts.
# model_best stays OFF for the same reason as the base: the schedule is flat, so
# there is no best step. See coeff_fm_train.
hooks = [
    dict(type="HelixPathBootstrap"),
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=1),
    dict(type="InformationWriter"),
    dict(type="WeightEMA", decay=0.9999, save_freq=SAVE_EVERY),
    dict(type="CoeffFMEvaluator", every_n_steps=EVAL_EVERY, max_batches=200),
    dict(type="CheckpointSaver", save_freq=SAVE_EVERY),
]
