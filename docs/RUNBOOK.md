# Runbook

The commands that actually work, in the order you need them. Every path resolves
through `helix/paths.py`, so the environment variables named here are the only
things that change between machines.

Everything below runs in ONE image:

    IMG=/sdf/data/neutrino/omara/images/helix-train.sif
    PY="apptainer exec --nv -B /sdf $IMG /opt/pimm/.venv/bin/python"

See `docs/ARCHITECTURE.md` §4 for what is in it and how to rebuild it.

---

## 0. First: does this environment work?

    python -m helix.paths

Prints every root, whether it came from the environment or a default, and
whether it exists. **Run this before anything else in a new environment.** The
defaults are where things live on the machine helix was developed on; they are
defaults, not truths.

Then the suites, which need no data:

    cd <your helix checkout>     && $PY -m pytest -q
    cd <your pimm-data checkout> && $PY -m pytest -q

**Test counts are deliberately not quoted.** They depend on whether pimm is
importable -- ~40 tests are `@pimm_importable` and SKIP without it -- and
quoting a number turns every change into a documentation edit. Three commits in
one session existed only to bump one, and four documents still disagreed
afterwards. What matters: the suite is green, and you ran it with pimm on the
path so those ~40 actually execute. Seven failures hid in that gap once.

If you see fewer passes and more skips, something optional is missing and the
skip reasons say which (`pytest -q -rs`). If you see a COLLECTION ERROR, the
environment is wrong, not the code.

### The one environment rule

helix and pimm-data must move together. The forward-model transforms
(`AddNoise`, `Digitize`) are registered by helix now and were registered by
pimm-data before; `pimm_data/_registry.py` raises `KeyError` on a duplicate. So:

**rebuild the image whenever pimm-data changes.** The build asserts this and
will fail rather than produce a mismatched image.

---

## 0b. Smoke the whole path with NO data

Before your own simulation exists, `pimm_data.testing.make_jaxtpc_sample` writes
a schema-conformant JAXTPC v3 dataset you can drive the whole pipeline with.
This is the check to run first on a new cluster: it exercises the code, the
container and the plumbing without a byte of production data.

    $PY -c "from pimm_data.testing import make_jaxtpc_sample; \
             make_jaxtpc_sample('$W/sim', dataset_name='sim_wire', \
                                n_events=200, readout_type='wire')"

    $PY scripts/build_coeff_corpus.py --shard $W/sim/sensor/sim_wire_sensor_0000.h5 \
        --out $W/corpus/run_synth --dataset-name sim_wire --mode serial --backend torch
    $PY -m helix.data.coeff_verify $W/corpus/run_synth --dataset-name sim_wire
    $PY scripts/write_holdout.py --corpus $W/corpus/run_synth --dataset-name sim_wire \
        --train 0.6 --val 0.2 --probe 0.2
    $PY scripts/derive_coeff_bins.py --corpus $W/corpus/run_synth \
        --dataset-name sim_wire --out $W/bins.pt --events 100 --K 128
    $PY scripts/smoke_train_fm.py --corpus $W/corpus/run_synth \
        --bins-from $W/bins.pt --events 4 --steps 8

All five pass on a 100-event synthetic corpus (verified from clean clones,
2026-09-17). `coeff_verify` prints `corpus OK`; `smoke_train_fm` prints a
DECREASING loss from about ln(128) = 4.85, which is where an untrained
categorical head starts.

Two things to know, both real:

* **The default holdout fractions do not work at this scale.** 95/3/2 is tuned
  for 158k events; the split is a hash of event identity, so a 2% part over 100
  events can legitimately select ZERO and `write_holdout` refuses. That refusal
  is correct -- pass smoke fractions as above.
* **`dump_probe_truth` cannot run on synthetic input, by design.** The fixture
  is schema-conformant, not physics-conformant: its `hits`/`step` are random
  draws with no relationship to the `sensor` waveforms it ships beside. So the
  pixel->cell join does not correspond, and the coverage guard says so --
  "band-0 charge-weighted coverage 0.673 < 0.9 ... a mismatched event scores
  ~0.37, a good one ~0.999". That is the guard doing its job. The probe stages
  need real simulation; everything before them does not.

---

## 1. Build a coefficient corpus

Input: doraemon sensor shards (`HELIX_SENSOR_ROOT`).
Output: sharded HDF5 under `HELIX_CORPUS_ROOT/<run>/`.

### The order, which is not optional

Six steps, and the first two cannot be swapped. `norm_sigma` is frozen BEFORE any
shard is written, because `CoeffTPCReader` refuses to open a corpus whose shards
disagree on it — so it must be computed once, from a sample of every run, and
then handed to every build job. Building first and calibrating after produces a
corpus that cannot be opened.

| # | step | command | produces |
|---|---|---|---|
| 0 | name the runs | write `<corpus root>/_calib/RUNS.txt` **by hand** | the run list |
| 1 | freeze `norm_sigma` | `scripts/calibrate_norm_sigma.sh [events_per_run]` | `_calib/norm_sigma_global.npy` |
| 2 | build | 8 × `RUN_INDEX=$i sbatch scripts/submit_coeff_corpus.sh` | the shards |
| 3 | verify | `$PY -m helix.data.coeff_verify <run dir> --dataset-name sim_wire` | pass/fail |
| 4 | write the split | `$PY scripts/write_holdout.py --corpus <run dir>` | `holdout.json` |
| 5 | derive the grid | `$PY scripts/derive_coeff_bins.py --corpus <run dir> --out bins.pt` | `bins.pt` |

**Step 0 is a real step.** `_calib/RUNS.txt` is one run name per line and nothing
generates it. Both phase 1 and phase 2 abort without it, and phase 2's message
says "run phase 1 first" — which is misleading, because phase 1 needs it too.

**`KGATE` must match between steps 1 and 2.** The frozen table is calibrated at a
particular gate, and `submit_coeff_corpus.sh` defaults `KGATE` to empty, meaning
`DetectorConfig`'s 3.0. Calibrating at one gate and building at another
mis-normalises the whole corpus with nothing to say so.

**Each corpus root carries its OWN `_calib`.** `norm_sigma` is computed from the
GATED coefficients, so the r1 and pre-tau generations have genuinely different
tables. Borrowing one root's `_calib` for another silently mis-normalises
everything built under it.

**Step 4 is a free check on the whole rebuild.** The split is keyed on
`blake2b(run/source_file) + event` — the SIMULATION event's identity — so it is
independent of basis, noise model, shard size and shard order. A corpus rebuilt
with a different gate or wavelet gets the *same* split. So
`write_holdout.py --compare <production holdout.json>` reproducing byte for byte
is evidence the rebuild preserved event identity end to end. Use it; it costs
nothing.

**Step 5 checks itself.** A plain derive compares its result against the grid
declared in `helix/data/data/reference_bins.json` and says whether it matches the
one the released models were trained against. See "The norm_sigma table and the
bin grid" below.

    # Phase 2 is EIGHT submissions, not one: this cluster's MaxArraySize is 100,
    # so an 0-799 array is rejected outright. RUN_INDEX picks the run.
    for i in 0 1 2 3 4 5 6 7; do
      RUN_INDEX=$i sbatch --account=<facility>:<repo> \
        --export=ALL,RUN_INDEX,SRC_ROOT=<sensor shards>,OUT_ROOT=<corpus parent> \
        scripts/submit_coeff_corpus.sh
    done

**The account is not the same as the training job's.** Accounts are authorised
per partition. The corpus build pins `--partition=turing` (the DSP is
architecture-sensitive — the same shard gives 58,411,720 surviving coefficients
on a 2080 Ti and 58,421,269 on an A100, and a corpus built across both is
incoherent), and on S3DF `mli:cider-ml` does NOT cover turing while `mli:default`
does. Training runs on ampere, where `mli:cider-ml` is correct.
`sacctmgr -n show assoc user=$USER format=Account,Partition` lists yours.

**Some tasks may report a missing source file, and that is normal.** The array is
0-99 but the simulator does not guarantee 100 contiguous files — `run_0027670361`
is missing indices 51-56 and 94-97, so it yields 180 shards and 17,999 events,
in production too. Those tasks skip with a message and exit 0. A task that FAILS
is a real failure.

That is the production path: a Slurm job array, `--backend torch` (the default —
measured 10.6 ms/plane, vs 176 ms on numpy), one GPU per task, `turing`
partition. `--backend jax` exists as an override; it is not what built the
current corpus.

Single-shard, interactively, to check the chain end to end:

    $PY scripts/build_coeff_corpus.py --help

The builder is a BUILD-TIME composer: it is the one place that imports both
helix and pimm-data. The read path never does.

**Counting events.** Do NOT sum each shard's `n_events`: the builder writes
TWO coeff shards per source file, so that double-counts. Count distinct
`(ident/source_file, ident/event)` pairs — 19,999 per run, 157,991 over the 8
runs, matching the split in `configs/pimm/coeff_fm_train_8run.py`.

**What a corpus records.** Every shard carries `basis_digest`, `removal_json`
(including `tau`), `sigma_norm`, `noise_json`, `provenance_json`, and
`ident/{event,run,source_file,noise_seed}`. This is not decoration — the next two
stages check it, and a corpus without it cannot be used (§5).

**Verify before using:**

    $PY -m helix.data.coeff_verify <corpus dir> --dataset-name sim_wire

(`verify_corpus(data_root, dataset_name, *, split=...)` takes the dataset name
as a required second argument; calling it with the directory alone is a
TypeError.)

It fails a corpus that records no builder, that mixes helix versions across
shards, or that was built from a dirty working tree.

### The norm_sigma table and the bin grid

Both are TRAINING-SET STATISTICS and both must come from the corpus you will
actually train on. They sit at OPPOSITE ends of the build, which is the thing to
get right:

    # BEFORE the build — frozen, then handed to every build job (step 1)
    scripts/calibrate_norm_sigma.sh          # one table the whole corpus shares

    # AFTER the build — derived from the finished shards (step 5)
    $PY scripts/derive_coeff_bins.py --corpus <run dir> --out bins.pt

`norm_sigma` is an INPUT to the shards and is recorded in every one of them;
the bin grid is computed FROM them. Reversing that is the mistake the order
table above exists to prevent.

The bins shipped with `m113` were derived from an old cache built with a
different (white) noise model. Do not reuse bins across corpus generations —
and note that a derive now tells you whether the grid it produced is the one the
released models were trained against, so a mistake here is reported rather than
inherited.

`derive_coeff_bins.py` also takes `--like <bins.pt>` to reuse the parameters an
existing grid records, and `--verify <bins.pt>` to rederive and compare without
writing. The derivation is importable as `helix.data.bins` if you need it from
code.

---

## 2. Train

ONE launcher. `launch/` was the predecessor and is retired -- `scripts/` ran
every current result (the 8-run's 16 links, 8run_plane's 12, and both cooldowns)
and carries the config-fingerprint resume guard, the QOS knob and the helix code
snapshot.

    scripts/chain_coeff_fm_train.sh <N> [config]           # a chain of links
    sbatch --export=ALL,CFG=<config> scripts/submit_coeff_fm_train.sh   # one link

Default config is `configs/pimm/coeff_fm_train_8run.py`. Knobs are environment
variables, not `#SBATCH` edits: `CFG`, `ACCOUNT`, `QOS`, `EXCLUDE`, `LOGDIR`.

**Every run snapshots helix into `<save_path>/helix-code/`.** `provenance.json`
records the commit and the dirty flag, and that was not enough: the 8-run's
eleven links all recorded commit `87248d88` on branch `extraction`, and that
hash resolves in no repository today -- the 2026-09-14 consolidation rewrote
every commit, so the recorded identity ceased to exist. A hash is a pointer into
a history someone may rewrite; a copy is not. (pimm's `train.sh` has always
snapshotted ITS code into `<save_path>/code`; helix, which defines the model and
the tokenizer, had no equivalent until now.)

**Why a chain and not `--requeue`.** This cluster runs `PreemptMode=CANCEL`
(QoS preemptable = "within,cancel"), so a preempted job is CANCELLED and never
returns to the queue, whatever `--requeue` says. This was confirmed the hard
way: a run reached epoch 11/25 (step 51,170, 3h20m) and simply vanished. Each
chain link starts when the previous one ENDS for any reason
(`--dependency=afterany`) and carries `ALLOW_RESUME=1`. `--requeue` is kept
because it still covers node failure, which IS requeued.

**Resume works**, which is what makes the chain safe: pimm's loader put the saved
RNG state on the GPU and `torch.set_rng_state` rejects a CUDA tensor, so
`resume=True` used to die before step 1.
`helix.integrations.pimm._patch_rng_restore_to_cpu` moves it back. An eviction
now costs at most `SAVE_EVERY=238` steps.

**`checkpoint_format=legacy` is not a preference.** pimm defaults a multi-GPU run
to DCP, whose save planner calls `dist.gather_object` on the default process
group, which pimm initialises NCCL-only. NCCL has no gather: the first save died
with "NCCL Error 1: unhandled cuda error" and every rank hung until the 600 s
watchdog fired.

The wrapper REFUSES to start if `EXP_DIR/model` already holds a checkpoint on a
first attempt — otherwise a reused `RUN_NAME` silently continues someone else's
model. Set `ALLOW_RESUME=1` deliberately.

Configs live in `configs/pimm/` and are reached as `-c helix/<name>`; the
launcher creates the link into pimm's `configs/` from the checkout it is running
(see `docs/ARCHITECTURE.md` §7).

---

## 3. Evaluate a checkpoint

### Two kinds of checkpoint. Read this once; it saves an afternoon.

A run leaves behind files that look interchangeable and are not.

| | RESUME state | EVAL state |
|---|---|---|
| what | `<save_path>/model/last/`, `iter_N.pth`, `model_ema.pth` | a `pimm export` dir, a helix eval artifact |
| holds | weights **+ optimizer, scheduler, RNG, sampler, step** | one weight set + architecture + operating point |
| for | continuing training bit-identically | producing a number that is reproducible and attributable |
| owner | pimm's. Nothing in helix reads it. | helix's |

Pointing an eval tool at resume state is the single most common way to waste a
GPU hour here. It no longer fails obscurely — `helix.model.artifact` recognises
each shape by name and the error says what to run — but the fix is always the
same, and it is two steps, not one:

    # 1. bundle the weights with the architecture and tokenizer they trained with
    pimm export --run-dir <save_path> model_ema.pth /tmp/exp

    # 2. promote it, so it can say WHICH weights, WHICH corpus, WHICH code
    $PY scripts/export_artifact.py /tmp/exp --weights ema \
        --corpus <corpus> -o <run>/artifact

Step 2 is not ceremony. pimm's `_sanitize_config` nulls the `weight` key on
every export, and the exported file is always named `model.safetensors`
whatever it was exported from — so an export cannot say whether it holds the
EMA or the raw weights, and every probe row written before this carries
`weights_are_ema: null`. On a flat-LR WSD run the raw weights sit at full LR
noise for the whole stable phase, which is the entire reason the EMA exists, so
comparing an unattributed arm against an EMA arm compares two noisy draws
rather than two models. `--weights` is required and unguessable on purpose: you
are asserting what you just exported, once, while you still know.

An artifact is a directory of two files — `weights.safetensors` and
`artifact.json` — and it is portable by construction: nothing in it names a
path that has to exist.

    $PY scripts/eval_checkpoint.py --config configs/pimm/coeff_fm_eval_probe.py \
        --options weight=<run>/model/model_ema.pth save_path=<out>

Scores a FROZEN checkpoint. This exists because `CoeffFMEvaluator` hooks
`after_step`/`after_epoch` only, so every trainer-driven route to a number takes
an optimizer step first and reports weights that have already moved. Comparing
two checkpoints that way scores something near the artifact rather than the
artifact.

---

## 4. The 3D probe

Two stages. Stage 1 is expensive and reusable; stage 2 is cheap and per-checkpoint.

    # Stage 1 -- per-pixel truth for a held-out split, written beside the corpus
    $PY scripts/dump_probe_truth.py --corpus <dir> --source $HELIX_SENSOR_ROOT

    # Stage 2 -- EXPORT the run. Not optional, and easy to get wrong.
    #
    # run_probe refuses a raw pimm checkpoint: model_ema.pth holds weights and
    # nothing else, so neither the architecture nor the tokenizer is recoverable
    # from it, and scoring a model on a tokenizer it never trained with gives a
    # plausible WRONG number (cell_t alone moves 94% of cells).
    #
    # The run directory already has both beside the weights -- config.py carries
    # n_slot/n_band/n_plane/d/blocks/heads and cell_t -- so `pimm export` bundles
    # them into a portable directory. Nothing needs inventing.
    pimm export --run-dir <save_path> model_ema.pth <export_dir>

    # Stage 3 -- PROMOTE the export into an eval artifact.
    #
    # An export is portable but cannot describe itself in three ways that each
    # change the number. `_sanitize_config` nulls `weight`, so nothing records
    # WHICH weight set it holds -- every probe row so far carries
    # weights_are_ema=null, and an EMA arm and a raw arm compare in silence.
    # Nothing records the CORPUS the weights trained on, so a coeff_tpc model
    # scored against coeff_tpc_r1 produces a plausible number. And nothing
    # records the helix commit that defined the tokenizer.
    #
    # --weights is required and unguessable on purpose: you are asserting what
    # you just exported, once, at the moment you know it.
    $PY scripts/export_artifact.py <export_dir> --weights ema \
        --corpus <dir> -o <artifact_dir>

    # Stage 4 -- probe the artifact (an export dir still works; it just cannot
    # attribute itself, and run_probe will say so)
    $PY scripts/run_probe.py --checkpoint <artifact_dir> --corpus <dir> \
        --tag <name> --out probe_results.jsonl \
        --cache-dir $SCRATCH/probe_cache

`--cache-dir` makes the extraction loop resumable: 2.5M patches over 388 events,
**12m28s measured** (job 38395161, from the cache chunk mtimes), which a
preemption used to discard before a single probe was fitted. Chunks are keyed by a hash of
everything that changes the features (both weight digests, layer, tokenizer,
corpus, truth, and the storage precision), so a stale cache cannot be read by
mistake. The cost is disk: **44 GB measured** for the 388-event split (2,503,852
patches at feature dim 2048), less than the naive
`n_patch x feat_dim x 3 arms x 4 B` because the `raw` arm is narrower than the
two feature arms.

Either `$SCRATCH` (the documented home for large temporary data, 100 GB, so
44 GB fits but not comfortably) or a directory beside the run (more room, and
what the validation used). Reload of the full 44 GB takes a few minutes and is
CPU-bound, not I/O-bound — measured on job 38444314 at 98% CPU — so placement is
a space decision, not a speed one. Delete the cache once the number is in.

`--cache-dir` also resumes the FITS, and that is the bigger half by far.
Measured on the same run: `geo` 36 min, `trained` 58 min, and `random` had run
past 1h20m when it was preempted. A row is written only once all four finish -- so a job that hits its time limit three arms in used to
lose all three. Each arm is now persisted the moment it completes, keyed by the
fit parameters (`folds`/`epochs`/`seeds`/`random_seed`) rather than by the
feature hash, so changing a fit parameter refits while still reusing the
extraction that cost the GPU hour. Budget four hours for a 388-event probe with
both probe types, or expect to resume once.

Full precision is the default, and it was earned. At float16 a resumed run
reproduced three arms exactly but moved `raw` from +0.0044 to +0.0045 -- 1e-4,
some 20x under the probe's own seed-to-seed sigma of 0.0021, and still wrong,
because it made `cache_resumed_events` a field that moves the number. Verified
end to end: a resumed run now reproduces all four arms, their per-group std and
their stop epochs identically. `--cache-half` halves the disk at that measured
cost, keyed apart so the two caches can never be read as one another's.

The two resumes compose, and both have now been exercised by a real preemption
rather than by a test. Job 38395161 was preempted on ampere at 1:47 having banked
the extraction and two fitted arms; resubmitting the identical script reported

    cache .../probe_cache/1391900325c983a6: 388 packs, resuming at event 388
    cache: 2 arm(s) already fitted, reusing
      [mlp] geo      fisher_r=+0.1514 (cached)
      [mlp] trained  fisher_r=+0.8403 (cached)

and restarted at the third arm. Its predecessor 38360259, with none of this, hit
its 4h wall one arm in and produced nothing at all.

`--requeue` is NOT the mechanism and is not worth setting: 38395161 had it and
did not come back, because preemption here cancels rather than requeues.
Resubmit by hand; the cache is what makes that cheap.

`--cell-t` matters and run_probe will refuse rather than guess. An eval artifact
always carries its tokenizer, so you will not need the flag for one; a bare
`pimm export` carries `cell_t` only if the run's config stated it, and for
anything that does not you must pass the value the run trained with
(`grep cell_t <run>/config.py`). The old
fallback silently chose `centroid` where every helix config trains
`grid_center`, and they differ on 94% of cells.

Stage 1 depends on the corpus only for WHICH events to dump, never for their
content, so the artifact survives a corpus rebuild.

Stage 1 reads compressed doraemon `hits`/`step` shards and needs `hdf5plugin` —
present in the image. Without it h5py raises "can't open directory
/usr/local/hdf5/lib/plugin".

Other analyses:

    scripts/feats_rank.py       # per-layer feature rank (RankMe)
    scripts/probe_xattn.py      # decoder cross-attention maps
    scripts/viz_mask_recon.py   # masked reconstructions
    scripts/plot_train_progress.py

---

## 5. The guard that will stop you

`helix/data/identity.py` compares a checkpoint's recorded corpus identity
(`basis_digest`, `removal_json`, `sigma_norm`) against the corpus being read. It
**refuses** on mismatch and **warns** when a corpus is unstamped.

This is not paranoia. `coeff_tpc` and `coeff_tpc_r1` share a run name and differ
only in the coherent gate (r1 records `tau=0.05`) — same wavelet, bands, gids,
sigma_norm, noise model, different surviving coefficients. Each corpus is
internally consistent, so a reader pointed at the wrong one is perfectly happy
and the failure is a plausible NUMBER, not an error.

If the guard refuses, do not work around it. Find the corpus the checkpoint was
trained on.

Corpora currently on disk:

| corpus | gate | use |
|---|---|---|
| `coeff_tpc_r1` (8 runs, 157,991 events, 345 GB) | `tau=0.05` | **current** |
| `coeff_tpc` | pre-tau | `m113` only (`HELIX_LEGACY_CORPUS`) |

---

## 6. Where things are

| what | where |
|---|---|
| corpora | `/sdf/data/neutrino/omara/` (`HELIX_CORPUS_ROOT`) |
| run outputs | `/sdf/data/neutrino/omara/exp/helix` (`HELIX_EXP`) |
| bin tables, converted checkpoints | `/sdf/data/neutrino/omara/archive` (`HELIX_ARCHIVE`) |
| retired research checkpoints | `.../archive/fm_research_ckpts/` (413 files, 165 GB) |
| retired caches (record only) | `.../archive/retirement-backups/` |
| the image | `/sdf/data/neutrino/omara/images/helix-train.sif` |

Nothing large belongs on `/sdf/group` — it is a 10 TB quota that was at 100%
until 742 GB of dead cache and stray checkpoints were cleared off it.

### The old research checkpoints

The 394 checkpoints other than `m113` are NOT self-contained and mostly cannot be
evaluated. The research trainer kept bin edges in a separate `tier1_bins.pt`
named only on the command line, and nothing in the checkpoint records which one;
the edges are training-set statistics and are not recoverable from the weights.

There is no way back any more, and that is deliberate. `tools/convert_fm_ckpt.py`
inferred architecture from the tensors and inlined the bins; it was run exactly
once, on `m113`, and has been retired. m113 is now
`archive/fm_m113_artifact` — a self-describing eval artifact carrying the
weights, the operating point, the pre-tau `basis_digest`, and the converter's
whole provenance record (the research checkpoint, its train config, its bins
file, the research-side name for its `cell_t`).

`archive/fm_m113_converted.pt` is still on disk as lineage but nothing reads it;
pointing a tool at it prints where the artifact is. For the other 394
checkpoints you would need the matching bins file, and for most it no longer
exists — the converter is in git history if that ever changes, but re-exporting
a run beats reviving it.

**If the m113 artifact is ever lost**, it cannot be rebuilt from `main`: the
reader that understands a converted blob was retired. The recipe is a detached
worktree at `eb216b7` — the last commit that could read one — plus the two
lineage hunks from `922aa52` (`_tok_extra` in `helix/model/artifact.py`, and
`source_provenance` in `scripts/export_artifact.py`), which is the combination
that never existed as a single commit. The artifact's own `provenance.note`
says this too, and its `provenance.helix.git_dirty` is `true` for exactly that
reason — the tree that built it was not a clean commit, and recording that
honestly is worth more than a tidy field.

Check it is intact without loading a single tensor:

    $PY -c "from helix.model.artifact import inspect; \
             p = inspect('$HELIX_ARCHIVE/fm_m113_artifact').provenance; \
             print(p['weights_digest'])"    # 7d795cc3ab49f90a79f98927647028b5
