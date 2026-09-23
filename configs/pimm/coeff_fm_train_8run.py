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

# A derived config runs BEFORE pimm processes `_base_`, so the base's sys.path
# bootstrap has not happened yet. It does NOT get its own: a config that touches
# sys.path must also register HelixPathBootstrap so a RESUMED job can still
# import helix (tests/test_pimm_config_contract.py pins that pairing), and the
# hook belongs to the base. So this relies on helix already being importable --
# which it is, because the launcher exports PYTHONPATH -- and fails loudly if not.
from helix.paths import root as _root                          # noqa: E402

_base_ = ["./coeff_fm_train.py"]

# ---------------------------------------------------------------------------
# corpus: the root, with the runs named as the split
# ---------------------------------------------------------------------------
# `split=[run, ...]` is the reader's multi-run form (readers/coeff_tpc.py
# _find_files): it globs each run's directory in turn and records which run each
# shard came from. The base points `data_root` at ONE run directory, which is why
# this cannot be expressed as a bare data_root change.
CORPUS_ROOT = str(_root("HELIX_CORPUS").parent)
# READ from the corpus's own `_calib/RUNS.txt` rather than retyped here. That
# file is hand-written and is step zero of a build (docs/RUNBOOK.md §1) -- both
# build phases abort without it -- so it is the closest thing the corpus has to
# a declaration of what it contains, and the build side already reads it
# (submit_coeff_corpus.sh, calibrate_norm_sigma.sh). Only the training configs
# retyped it, which meant a corpus copy carrying a DIFFERENT subset would train
# happily on whatever the literal named.
from helix.data.identity import corpus_runs as _corpus_runs    # noqa: E402
RUNS = _corpus_runs(CORPUS_ROOT)

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
#: The corpus this was resolved against: 8 runs, 157,991 events. Recorded so the
#: literal above can be CHECKED rather than merely asserted in prose.
#:
#: Why the literal is still the value: resolving the split exactly means reading
#: `n_events` from every shard header (790 of them), which is seconds -- fine
#: once, far too slow on every config load, and `pimm submit` loads this config
#: on the login node during preflight as well as in the job. So only the cheap
#: invariant below is checked. NOTHING verifies the literal itself: an earlier
#: version of this comment said tests/test_pimm_config_contract.py did, and no
#: test ever has. If the holdout or the run list changes, re-resolve it with
#: scripts/write_holdout.py.
N_EVENTS_TOTAL = 157_991
if len(RUNS) != 8:
    raise SystemExit(
        f"this config's budget was resolved over 8 runs ({N_EVENTS_TOTAL:,} events) "
        f"but _calib/RUNS.txt names {len(RUNS)}: {', '.join(RUNS)}.\n"
        f"  N_TRAIN_EVENTS and therefore STEPS would be wrong -- a job that looks\n"
        f"  fine and is the wrong length. Re-resolve the split for this corpus\n"
        f"  (scripts/write_holdout.py) and update N_TRAIN_EVENTS/N_EVENTS_TOTAL.")

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
# Inherited from the base, which derives it from WORLD_SIZE. NOT restated here:
# the only correct global batch is the rank count (one event per rank), so a
# literal would be right at exactly one GPU count and wrong at every other --
# which is precisely what a batch-size/LR scaling sweep varies.
#
# STEPS below needs the same number, and `batch_size` is NOT in scope yet: a
# derived config executes BEFORE pimm merges `_base_`, so the base's value does
# not exist here. Take it from the same source the base does.
from helix.paths import world_size as _world_size               # noqa: E402
_WORLD = _world_size()
STEPS = N_TRAIN_EVENTS * epoch // _WORLD

# The cadences come from helix.core.cadence -- the ONE place the rule lives.
# They used to be three lines here and three more in the base config, and the
# duplicate silently shadowed a fix applied to the base.
#
# Note what is NOT here: WARMUP and `scheduler`. Warmup is driven by model
# WIDTH, and `model` is not in scope in a derived config (same reason
# `batch_size` is not, two comments above). Since this config does not change
# the model, the base's width-scaled warmup flows through the merge unchanged --
# which is correct, and the only way to avoid restating the width here.
from helix.core.cadence import eval_every as _eval_every         # noqa: E402
from helix.core.cadence import save_every as _save_every         # noqa: E402

EVAL_EVERY = _eval_every(STEPS)
SAVE_EVERY = _save_every(STEPS)
del _eval_every, _save_every

save_path = str(_root("HELIX_EXP") / "coeff-fm-train-r1-8run")

# The hook list is re-stated because the cadences above are new values, and a
# `_base_` merge would otherwise keep the base's numbers inside the hook dicts.
# model_best stays OFF for the same reason as the base: the schedule is flat, so
# there is no best step. See coeff_fm_train.
hooks = [
    dict(type="HelixPathBootstrap"),
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=1),
    # log_frequency is restated because this list replaces the base's wholesale,
    # and dropping it here once meant a log line -- and a device sync -- every
    # step. It is also how often InformationWriter reads device scalars back:
    # every step is still recorded, in one transfer per interval.
    dict(type="InformationWriter", log_frequency=200),
    dict(type="WeightEMA", decay=0.9999, save_freq=SAVE_EVERY),
    dict(type="CoeffFMEvaluator", every_n_steps=EVAL_EVERY, max_batches=200),
    dict(type="CheckpointSaver", save_freq=SAVE_EVERY),
]

# _corpus_runs too: a single leading underscore does not keep a name out of the
# dumped config; only `__` does.
del _root, _corpus_runs, _WORLD, _world_size
