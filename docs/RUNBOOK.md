# Runbook

Procedures as commands. Run them from the helix checkout. Every path, the container
and the scheduler come from `helix/sites/<site>.yaml` (`docs/ARCHITECTURE.md` §10);
the S3DF values are given where a command needs them. Work heavier than a quick check
goes to Slurm: `corpus.sh` and `train.sh` submit their own jobs; for the rest an
`sbatch --wrap` form is shown.

---

## 1. Host environment

The submitting host needs a Python >= 3.8 with PyYAML (to read the site profile) and
`sbatch`; jobs run in the container.

    scripts/make_host_env.sh               # once: $SCRATCH/helix-host-venv (HELIX_HOST_VENV, PYTHON override)
    source scripts/helix_env.sh            # exports HELIX_*, HELIX_CONTAINER_*, HELIX_SLURM_*, HELIX_PYTHON
    $HELIX_PYTHON -m helix.paths           # site, each root and its source, container, scheduler

`helix_env.sh` finds the venv without being told (it tries `HELIX_PYTHON`, the venv,
then system pythons). Any `HELIX_*` set in the environment wins over the profile, e.g.
`HELIX_CORPUS=<root>/<run>` to point training at another corpus. `HELIX_SITE` selects the
profile; set it in batch jobs.

Run a command in the container:

    scripts/helix_run.sh python -m helix.paths
    HELIX_FORWARD_ENV="CUDA_LAUNCH_BLOCKING" scripts/helix_run.sh python ...   # forward extra variables

---

## 2. Tests

    scripts/helix_run.sh python -m pytest tests -q            # a few minutes; or on a CPU node:
    sbatch --account=mli:cider-ml --partition=milano --cpus-per-task=8 --mem=32G --time=01:00:00 \
        --wrap="$HELIX_ROOT/scripts/helix_run.sh python -m pytest -q -p no:cacheprovider tests"

(`SBND-data-helix/jobs/tests.sh` is the same job with `HELIX_SRC=<checkout>`.)

Read the skips, not the count. Tests needing pimm skip unless pimm is importable
(`helix_run.sh` puts `HELIX_PIMM_ROOT` on `PYTHONPATH`); tests needing real data skip
unless the site's data roots resolve; CUDA tests skip without a GPU; the JAXTPC
reconciliation tests need `HELIX_JAXTPC_ROOT`. For a run that is meant to show that
everything works, turn silent skips into errors:

    HELIX_REQUIRE_PIMM=1 HELIX_REQUIRE_DATA=1 scripts/helix_run.sh python -m pytest tests -q -rs

A collection error means the environment is wrong, not the code. `helix_run.sh` sets
`PYTHONNOUSERSITE=1`, so packages under `~/.local` cannot leak into a run.

**Testing a pimm-data change.** The container installs pimm-data, so a plain pytest
tests the installed copy. Put the working tree first:

    HELIX_FORWARD_ENV=PYTHONPATH PYTHONPATH=<pimm-data>/src scripts/helix_run.sh python -m pytest -q
    ... python -c "import pimm_data; print(pimm_data.__file__)"     # check which copy ran

`tests/test_cross_repo_duplication.py` needs the same `PYTHONPATH` to see your edits.
helix itself is not installed in the image, so its suite always tests the checkout.

**Clean room** (before a handover): fresh clones of helix, pimm-data and pimm, nothing
else reachable:

    HELIX_ROOT=$CR/helix HELIX_PIMM_ROOT=$CR/pimm HELIX_FORWARD_ENV=PYTHONPATH \
      PYTHONPATH=$CR/pimm-data/src HELIX_REQUIRE_PIMM=1 HELIX_REQUIRE_DATA=1 \
      $CR/helix/scripts/helix_run.sh python -m pytest tests -q

---

## 3. Build a corpus

    scripts/corpus.sh plan      SOURCE OUT RUN_DIR...   # OUT/_calib/CHUNKS.txt, RUNS.txt, RUN_DIRS.txt
    scripts/corpus.sh calibrate SOURCE OUT              # one job: norm_sigma per run, mean -> norm_sigma_global.npy
    scripts/corpus.sh build     SOURCE OUT              # job arrays over CHUNKS.txt
    scripts/corpus.sh all       SOURCE OUT RUN_DIR...   # the three, build after calibration (afterok)

`SOURCE` is `sim`, `sbnd` or the path of a build script ending in `helix.build.main`.
(goop is optical only and has no `norm_sigma` to calibrate; build it with
`scripts/build_goop.py` directly.) Examples (S3DF):

    scripts/corpus.sh all sim  $HELIX_CORPUS_ROOT/coeff_tpc_new $HELIX_SENSOR_ROOT/run_*
    scripts/corpus.sh all sbnd /sdf/data/neutrino/omara/sbnd_data/coeff_sbnd_new \
        /sdf/data/neutrino/omara/sbnd_data/extracted/0182*

Environment: `EVENTS_PER_TASK` (500; sim: one sensor file per task), `EVENTS_PER_SHARD`
(sim 200, else 100), `CAL_EVENTS` (100 per run), `CONFIG` (chain overrides, json or
yaml), `CONC` (16 tasks at once per array), `ARRAY_MAX` (100), `LOGS` (`OUT/_logs`).
A task writes `PREFIX.summary_NNNN.json` beside its shards.

* **Order.** Every shard embeds the one global `norm_sigma`, so calibration finishes
  before the build; `build` refuses to start without `_calib/norm_sigma_global.npy`
  unless a calibration job is pending in the same `all`.
* **Resume.** Rerun `scripts/corpus.sh build SOURCE OUT`. Shards appear only when
  complete (`.part` then rename) and each task runs with `--skip-complete`, so only
  missing shards are rebuilt. At S3DF preempted tasks are also requeued (`--requeue`).
* **One GPU type per corpus.** turing and A100 disagree on 0.016% of kept coefficients;
  the site pins `scheduler.corpus.partition` (S3DF: turing, account `mli:default`).
* **A clean tree.** Every shard records the helix commit and dirty flag; `coeff_verify`
  rejects a corpus built from a dirty tree or from more than one commit. Do not edit the
  checkout while a build runs.
* **Upstream gaps are not failures.** `run_0027670361` has 90 of 100 sensor files
  (missing 51-56, 94-97); the plan only lists files that exist.
* `CONFIG` must be the same for `calibrate` and `build` (the table depends on the gate).

One shard by hand (a GPU node), to check the chain:

    scripts/helix_run.sh python scripts/build_sim.py --input RUN/sim_wire_sensor_0000.h5 \
        --output OUT/RUN/sim_wire --device cuda --count 200 --events-per-shard 200
    scripts/helix_run.sh python scripts/build_sbnd.py --input RUN/*.h5 --output OUT/RUN/sbnd \
        --device cuda --count 100

All options: `helix/build.py`'s docstring (`--first`, `--spread`, `--shard-start`,
`--norm-sigma`, `--calibrate`, `--config`, `--optical-config`, `--no-optical`, `--workers`,
`--deflate`, `--events-log`, `--profile`, `--skip-complete`).

---

## 4. Verify a corpus

    scripts/helix_run.sh python -m helix.data.coeff_verify OUT                       # every run, against _calib/CHUNKS.txt
    scripts/helix_run.sh python -m helix.data.coeff_verify OUT/RUN --dataset-name sim_wire \
        [--expect N] [--modalities coeff coeff_clean] [--source-root SENSOR_RUN_DIR]
    scripts/helix_run.sh python -m helix.data.coeff_verify compare A.h5 B.h5 [A2.h5 B2.h5 ...]

Prints `corpus OK` or the problems (exit 1). The whole-corpus check reads every
shard's metadata: for a full corpus submit it (`sbatch ... --wrap="$HELIX_ROOT/scripts/helix_run.sh
python -m helix.data.coeff_verify OUT"`). `--source-root` tells an upstream gap (source
absent: fine) from a failed task (source present: error). `compare` ignores only
provenance attributes (commit, host, time).

**Count events** as distinct `(ident/source_file, ident/event)`, not by summing
`n_events` (noisy and clean shards both carry them).

---

## 5. Split and bin grid

    scripts/helix_run.sh python scripts/write_holdout.py --corpus OUT/RUN [--compare OTHER/holdout.json]
    scripts/helix_run.sh python scripts/derive_coeff_bins.py --corpus OUT/RUN --out bins.pt
    scripts/helix_run.sh python scripts/derive_coeff_bins.py --verify $HELIX_ARCHIVE/coeff_bins_r2_run0027575715.pt \
        --corpus OUT/RUN

`write_holdout.py` writes `OUT/RUN/holdout.json` (default fractions 0.95 / 0.03 / 0.02;
pass e.g. `--train 0.6 --val 0.2 --probe 0.2` on a small corpus, where a 2% part can be
empty and the script refuses). The split hashes the simulation event's identity, so a
rebuilt corpus reproduces it; `--compare` checks that.

The bin grid is derived after the build from finished shards; `norm_sigma` before it.
The production grid is named in `helix/data/data/reference_bins.json`
(`coeff_bins_r2_run0027575715`, read from `HELIX_ARCHIVE`); a derive reports whether its
result matches it. `--like bins.pt` reuses the parameters a grid records.

---

## 6. Train

    scripts/train.sh CONFIG [key=value ...]

`CONFIG` is a name in `configs/pimm/` or a path. Each of `LINKS` jobs (default 1) runs
`python -m pimm.train --config-file CONFIG` in the container as `GPUS` Slurm tasks (`srun`;
pimm takes each rank from its task), starting
when the previous one ends (`--dependency=afterany`), resuming from the run's latest
complete checkpoint (`resume=True weight=<it>`) or starting the run if there is none.
The run lives in `$HELIX_EXP/NAME` (`NAME` defaults to the config name; logs in
`$HELIX_EXP/NAME/slurm`). The same `NAME` with a different config or options is refused.

Environment: `LINKS` (1), `NAME`, `GPUS` (4, one node), `TIME` (24:00:00), `MEM`
(50G per GPU where the site asks for memory), `LOGS`. Scheduler facts:
`scheduler.train` (S3DF: `mli:cider-ml`, ampere, QOS normal).

    LINKS=8 scripts/train.sh fm                        # the stable phase over HELIX_CORPUS's corpus
    scripts/train.sh fm model.plane_frac=0.25          # any other variant: key=value (pimm --options)
    NAME=fm_pf25 scripts/train.sh fm model.plane_frac=0.25   # ... as its own run
    scripts/train.sh fm_cooldown                       # WSD cooldown from $HELIX_EXP/fm/model/last
    GPUS=1 scripts/train.sh fm_smoke                   # wiring check: small model, 32 events
    LINKS=4 scripts/train.sh denoise                   # the denoiser (M6d) on $HELIX_EXP/fm; needs the cells truth
    NAME=denoise_m6dc scripts/train.sh denoise model.cond_charge=True        # M6dc: calorimetry
    NAME=dn_scratch_N1024 scripts/train.sh denoise model.pretrained=False data.train.n_events=1024 epoch=16   # le3: 4k steps x 4 GPUs

* **The corpus** is `HELIX_CORPUS`'s parent, with its runs from `_calib/RUNS.txt`
  (S3DF default: `coeff_tpc_r2`). `HELIX_CORPUS=<root>/<run> scripts/train.sh fm` trains
  on another; the variable reaches the jobs.
* **No step counts.** Warmup and the eval / checkpoint / EMA-save cadence are `"auto"`
  and resolved by `FMTrainer` from `iters_per_epoch * epoch`; the log line `cadence over
  N steps: ...` shows them. Change the length with `epoch=`.
* **Batch = GPUs.** `batch_size` is the world size (one event per rank:
  `helix.paths.world_size()`, Slurm's task count inside a job), in every config.
* **Cooldown** is a separate short run warm-started with `weight=` (never `resume=`): it
  anneals over its own steps on the first 3 runs for one epoch. It reads the stable run at
  `$HELIX_EXP/fm`; for another stable run pass `weight=$HELIX_EXP/<name>/model/last`.
* **Preemption** costs at most one checkpoint interval: the next link resumes. S3DF
  cancels preempted jobs, NERSC requeues them; `afterany` continues either way.
* fm_smoke reads its bins from `$HELIX_ARCHIVE/coeff_bins_K32_smoke.pt`.
* **The denoiser** (`configs/pimm/denoise.py`) is an FM run plus a head: `model.fm=<run dir>`
  gives the encoder's architecture, tokenizer and (`model.pretrained`, default True) EMA
  weights; `model.freeze_encoder=True` trains the head only. It reads the `cells` truth
  (section 7: `scripts/build_truth.py` first). Its length is epochs like the FM's: a
  label subset (`data.train.n_events`) needs a larger `epoch` for the same step count. The
  best checkpoint by validation loss is `model/model_best.pth`.

Plot progress: `scripts/helix_run.sh python scripts/plot_train_progress.py label=$HELIX_EXP/fm --out fig.png`.

---

## 7. Evaluate

Truth first, once per corpus run (simulation `hits`/`step` from the parent of
`HELIX_SENSOR_ROOT`; writes `{name}_cells_<k>.h5` beside each coefficient shard):

    scripts/helix_run.sh python scripts/build_truth.py --corpus <run dir>

Then score a run directly from its pimm directory -- no export (GPU: the varlen attention
needs flash attention, sm80+):

    sbatch --account=$HELIX_SLURM_TRAIN_ACCOUNT --partition=$HELIX_SLURM_TRAIN_PARTITION \
        --qos=$HELIX_SLURM_TRAIN_QOS --gpus=1 --cpus-per-task=8 --mem=100G --time=08:00:00 \
        --wrap="HELIX_INTERACTIVE=0 $HELIX_ROOT/scripts/helix_run.sh python -m helix.eval \
                $HELIX_EXP/fm --metrics recon,floor,uprobe --out $HELIX_EXP/fm/eval.json"

`RUN` is the run directory (`--weights ema` default, or `last`) or one checkpoint inside it
(`model/iter_N.pth`). Events: the run's own corpus, `probe` role of its holdout
(`--split`, `--max-events` for a quick look). Metrics: `recon` (the training evaluator's
masked reconstruction), `floor` (detection floor / charge map from frozen features; trained
and random-init arms, rankme), `uprobe` (along-wire coordinate; geo, trained, random arms),
`denoise` (`--denoiser <denoise run>`, its best checkpoint, on the floor's windows). The probes fit
on the first two thirds of the events and score the rest, with [16, 84]% event-bootstrap
intervals. See `helix/eval/__init__.py`.

**The identity guard.** `helix.data.identity` refuses a checkpoint whose recorded
`basis_digest` differs from the corpus being read. Do not work around it; find the
corpus the checkpoint trained on.

---

## 8. Migrate a corpus built before the format stamp

Readers require `/config` `format = "helix-coeff-2"`. A corpus built before the stamp
is migrated in place, metadata only (r2 and SBND v1 have been: their old attributes are in
each root's `_calib/format1_attrs.json`, and v1's run directories are now `018251`, ...):

    scripts/helix_run.sh python scripts/migrate_format.py SOURCE ROOT RUN_DIR... [--events-per-task 500] [--dry-run]

`RUN_DIR...` are the input runs the corpus was built from, in its order; each corpus run
directory must be named for one of them (`<run>` or `run<run>`). Per coeff and
coeff_clean shard it stamps `format`, rewrites `removal_json` and `basis_digest` to what
the current chain records (the gate's readout-groups schema), drops `mode` / `backend`
from `noise_json`, and refuses if any other basis field differs (that corpus needs a
rebuild). Before writing it saves every shard's old attributes to
`_calib/format1_attrs.json` and an existing `CHUNKS.txt` to `CHUNKS.txt.format1`;
afterwards it renames `run<run>` directories to `<run>` and writes the plan files as
`corpus.sh plan` would. Stamped shards are skipped, so a rerun finishes an interrupted
migration.

It touches every shard, so submit it (S3DF; the CPU suffices). As it was run for r2 and v1:

    R=/sdf/data/neutrino/omara/coeff_tpc_r2
    IN=$(for d in $R/run_*; do echo $HELIX_SENSOR_ROOT/${d##*/}; done)
    sbatch --account=mli:cider-ml --partition=milano --cpus-per-task=4 --mem=16G --time=04:00:00 \
        --wrap="$HELIX_ROOT/scripts/helix_run.sh python scripts/migrate_format.py sim $R $(echo $IN) --dry-run"
    # read the log, then the same without --dry-run, then
    #   scripts/helix_run.sh python -m helix.data.coeff_verify $R

    R=/sdf/data/neutrino/omara/sbnd_data/coeff_sbnd_v1           # (before the rename: run dirs run018251, ...)
    IN=$(for d in $R/run*; do echo /sdf/data/neutrino/omara/sbnd_data/extracted/${d##*/run}; done)
    # ... migrate_format.py sbnd $R $IN, as above

Older corpora (pre-tau `coeff_tpc`, r1) differ beyond the gate and are not readable.

---

## 9. Regression check

`SBND-data-helix/jobs/regress.sh` rebuilds, with a given checkout on a 2080 Ti (the corpora's
GPU type), one SBND shard (TPC + optical, 100 events of run 018251) and one simulation shard
(noisy + clean, 200 events of `run_0027575715`), and compares them byte for byte through
`coeff_verify compare`: the SBND shard with `coeff_sbnd_v1/018251` itself, the simulation
shard with the reference build in
`/sdf/data/neutrino/omara/sbnd_data/SBND-data-helix/work/regress/reference/sim` (r2's own
coefficients predate the board-grouped gate).

    HELIX_SRC=$HELIX_ROOT sbatch /sdf/group/neutrino/omara/SBND-data-helix/jobs/regress.sh
    # log: /sdf/data/neutrino/omara/sbnd_data/SBND-data-helix/work/logs/regress-<job>.out,
    # ending REGRESS_IDENTICAL or REGRESS_DIFFERENT

Run it after any change meant to be output-neutral.

---

## 10. Where things are

    $HELIX_PYTHON -m helix.paths

| what | root |
|---|---|
| corpora | `HELIX_CORPUS_ROOT` |
| the training corpus (one run dir of it) | `HELIX_CORPUS` |
| run outputs | `HELIX_EXP` |
| bin tables, artifacts | `HELIX_ARCHIVE` |
| simulation sensor files | `HELIX_SENSOR_ROOT` |
| pimm checkout | `HELIX_PIMM_ROOT` |
| the container | `container:` in the site profile |

| corpus | where (S3DF) | |
|---|---|---|
| simulation r2 | `/sdf/data/neutrino/omara/coeff_tpc_r2` | production; the bin grid's corpus |
| SBND v1 | `/sdf/data/neutrino/omara/sbnd_data/coeff_sbnd_v1` | production |
| `coeff_tpc` (pre-tau), `coeff_tpc_r1` | | retired, not readable |
