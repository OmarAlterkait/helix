# Runbook

Procedures as commands. Run them from the helix checkout. Every path, the container
and the scheduler come from `helix/sites/<site>.yaml` (`docs/ARCHITECTURE.md` §10), so
the same commands run at S3DF and NERSC; the S3DF values are given where a command needs
them, and section 11 lists what differs at NERSC. Work heavier than a quick check goes
to Slurm: `corpus.sh` and `train.sh` submit their own jobs; for the rest an
`sbatch --wrap` form is shown.

The order for a new corpus: build (§3), verify (§4), truth (§7), bin grid (§5),
pretrain (§6), denoiser (§6), evaluate (§7).

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

    scripts/corpus.sh plan      SOURCE OUT RUN_DIR...   # OUT/_calib/CHUNKS.txt, RUNS.txt
    scripts/corpus.sh calibrate SOURCE OUT              # one job: norm_sigma per run, mean -> norm_sigma_global.npy
    scripts/corpus.sh build     SOURCE OUT              # job arrays over CHUNKS.txt
    scripts/corpus.sh all       SOURCE OUT RUN_DIR...   # the three, build after calibration (afterok)

`SOURCE` is `sim`, `sbnd` or the path of a build script ending in `helix.build.main`.
(goop is optical only and has no `norm_sigma` to calibrate; build it with
`scripts/build_goop.py` directly.) Examples (S3DF):

    scripts/corpus.sh all sim  $HELIX_CORPUS_ROOT/coeff_tpc_new $HELIX_SIM_ROOT/j00*
    scripts/corpus.sh all sbnd /sdf/data/neutrino/omara/sbnd_data/coeff_sbnd_new \
        /sdf/data/neutrino/omara/sbnd_data/extracted/0182*

Environment: `EVENTS_PER_TASK` (500; sim: one sensor file per task), `EVENTS_PER_SHARD`
(sim 200, else 100), `CAL_EVENTS` (100 per run), `CONFIG` (chain overrides, json or
yaml), `CONC` (16 tasks at once per array), `ARRAY_MAX` (100), `LOGS` (`OUT/_logs`).
A task writes `PREFIX.summary_NNNN.json` beside its shards. A run's calibration reads the
inputs its tasks in `CHUNKS.txt` read.

**The simulation corpus** is built from the JAXTPC wire campaign `doraemon_2026_1M_lartpc`
(`HELIX_SIM_ROOT`, its `data/jaxtpc_wire`; layout and identity: ARCHITECTURE section 4).
`RUN_DIR` is a campaign group `j00X000` (200 tasks, 200,000 events), so the corpus has five
runs `j000000 .. j004000`; a build task is one task's sensor file (1000 events, 5 shards);
shard k holds campaign events `k*200 .. k*200+199`, i.e. generator job k. `EVENTS_PER_SHARD`
must divide 1000. Planning a group reads each sensor file's `/dprod/events` only (seconds).
Measured on one task (j000085-000089, 2080 Ti): 1.27 events/s, 13 min per task, 1.13 MB per
event (~226 MB per 200-event shard), 1.4 GB of GPU memory; the whole campaign is ~220
GPU-hours and ~1.1 TB of coefficients, plus its cells truth.

**Clean shards are off.** A build writes only the noisy `*_coeff_*` (and optical)
shards; nothing trains on noise-free values. `--keep-clean` on a build script also
writes `PREFIX_coeff_clean_k.h5` from the simulation's noise-free planes, to inspect the
DSP; through `corpus.sh`, set `BUILD_ARGS=--keep-clean`.

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
* **Upstream gaps are not failures.** The plan only lists input files that exist, and
  `coeff_verify` judges a gap against the plan (the campaign has all 1000 tasks).
* `CONFIG` must be the same for `calibrate` and `build` (the table depends on the gate).

One shard by hand (a GPU node), to check the chain:

    T=$HELIX_SIM_ROOT/j000000/jaxtpc_wire_j000085-000089      # jobs 85-89: shards 85-89
    scripts/helix_run.sh python scripts/build_sim.py --input $T/sensor/sim_wire_sensor_${T##*/}.h5 \
        --output OUT/j000000/sim_wire --device cuda --count 200 --events-per-shard 200 --shard-start 85 [--keep-clean]
    scripts/helix_run.sh python scripts/build_sbnd.py --input RUN/*.h5 --output OUT/RUN/sbnd \
        --device cuda --count 100

All options: `helix/build.py`'s docstring (`--first`, `--spread`, `--shard-start`,
`--norm-sigma`, `--calibrate`, `--config`, `--optical-config`, `--no-optical`, `--keep-clean`,
`--workers`, `--deflate`, `--events-log`, `--profile`, `--skip-complete`).

---

## 4. Verify a corpus

    scripts/helix_run.sh python -m helix.data.coeff_verify OUT                       # every run, against _calib/CHUNKS.txt
    scripts/helix_run.sh python -m helix.data.coeff_verify OUT/RUN --dataset-name sim_wire \
        --modalities coeff [--expect N]
    scripts/helix_run.sh python -m helix.data.coeff_verify compare A.h5 B.h5 [A2.h5 B2.h5 ...]

Prints `corpus OK` or the problems (exit 1). On one run directory pass
`--modalities coeff` for a corpus built without `--keep-clean`: the default,
`coeff coeff_clean`, also requires a clean shard beside every noisy one (the whole-corpus
form checks clean pairs only where clean shards exist). The whole-corpus check reads every
shard's metadata: for a full corpus submit it (`sbatch ... --wrap="$HELIX_ROOT/scripts/helix_run.sh
python -m helix.data.coeff_verify OUT"`). Against the plan, a shard gap the plan never wrote
(an input absent upstream) is fine and a planned shard absent is an error; one run directory
alone reports every gap. `compare` ignores only
provenance attributes (commit, host, time).

**Count events** as distinct `(ident/source_file, ident/event)`, not by summing
`n_events` (noisy, clean and cells shards all carry them).

---

## 5. Bin grid (and the split)

The FM's categorical head bins the NOISY coefficients (uniform in asinh on each side of
zero, one bin over the sub-threshold gap). The production grid is named in
`helix/data/data/reference_bins.json`: `coeff_bins_d1m_j000000_noisy`, derived from the
campaign corpus's first run (`coeff_tpc_d1m/j000000`) with K = 128 over 120 events, read by
every config from `$HELIX_ARCHIVE/<name>.pt` (derived 2026-10-29; its digests are recorded). To derive a grid
(CPU; milano is AMD, so numpy's results are the canonical ones, and the script disables
AVX-512 itself elsewhere):

    sbatch --account=mli:cider-ml --partition=milano --cpus-per-task=8 --mem=32G --time=01:00:00 \
        --wrap="$HELIX_ROOT/scripts/helix_run.sh python scripts/derive_coeff_bins.py \
                --corpus $HELIX_CORPUS --out $HELIX_ARCHIVE/coeff_bins_d1m_j000000_noisy.pt"

For a new reference (a new name in the JSON, `digests: null`), the log ends with `records no
digests yet ... record these` and a JSON object of three digests (`edges`, `cent_asinh`,
`cent_ratio`): paste it as the value of `"digests"` and commit. Any other site checks its copy of the corpus against the record instead
of copying the file:

    scripts/helix_run.sh python scripts/derive_coeff_bins.py \
        --verify $HELIX_ARCHIVE/coeff_bins_d1m_j000000_noisy.pt --corpus $HELIX_CORPUS

(`VERIFIED` or `MISMATCH`; `--like bins.pt` reuses the parameters a grid records.) A
derive on another corpus reports whether its result matches the reference. A grid is
derived after the build from finished shards; `norm_sigma` before it.

**The split** is not a file: `CoeffTPCDataset(holdout=...)` hashes each event's identity
(blake2b(run/source_file) + event; fm.py's fractions 0.95 / 0.03 / 0.02 for train / val /
probe), so a rebuilt corpus reproduces it. `scripts/write_holdout.py --corpus OUT/RUN
[--compare OTHER/holdout.json]` writes it out as `OUT/RUN/holdout.json` for inspection;
nothing reads that file.

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

    LINKS=8 scripts/train.sh fm                        # pretrain: the stable phase over HELIX_CORPUS's corpus
    scripts/train.sh fm_cooldown                       # then the WSD cooldown from $HELIX_EXP/fm/model/last
    NAME=fm_pf25 scripts/train.sh fm model.plane_frac=0.25   # a variant: key=value (pimm --options), as its own run
    GPUS=1 scripts/train.sh fm_smoke                   # wiring check: small model, 32 events
    LINKS=4 scripts/train.sh denoise model.fm=$HELIX_EXP/fm_cooldown       # the denoiser (M6d) on an FM run; needs the cells truth
    NAME=denoise_m6dc scripts/train.sh denoise model.fm=$HELIX_EXP/fm_cooldown model.cond_charge=True   # M6dc: calorimetry
    NAME=dn_scratch_N1024 scripts/train.sh denoise model.pretrained=False data.train.n_events=1024 epoch=16   # from scratch, 1,024 labelled events

* **The corpus** is `HELIX_CORPUS`'s parent, with its runs from `_calib/RUNS.txt`
  (S3DF default: `coeff_tpc_d1m`). `HELIX_CORPUS=<root>/<run> scripts/train.sh fm` trains
  on another; the variable reaches the jobs.
* **No step counts.** Warmup and the eval / checkpoint / EMA-save cadence are `"auto"`
  and resolved by `FMTrainer` from `iters_per_epoch * epoch`; the log line `cadence over
  N steps: ...` shows them. Change the length with `epoch=`.
* **Batch = GPUs.** `batch_size` is the world size (one event per rank:
  `helix.paths.world_size()`, Slurm's task count inside a job), in every config. One node
  per run (`GPUS` up to 4 at S3DF): `train.sh` submits `--nodes=1`.
* **GPU type.** The varlen attention needs sm80+ (A100 or newer): `ampere` at S3DF,
  never `turing`.
* **The recipe** is `nz_pw16` (`configs/pimm/fm.py`: d768, LR 4.4e-3), whose LR was tuned
  at 29.2k steps on 16 GPUs; see `docs/DECISIONS.md` before changing the length much.
* **Cooldown** is a separate short run warm-started with `weight=` (never `resume=`): it
  anneals over its own steps on the first 3 runs for one epoch. It reads the stable run at
  `$HELIX_EXP/fm`; for another stable run pass `weight=$HELIX_EXP/<name>/model/last`.
* **Preemption** costs at most one checkpoint interval: the next link resumes. S3DF
  cancels preempted jobs, NERSC requeues them; `afterany` continues either way.
* fm_smoke reads its bins from `$HELIX_ARCHIVE/coeff_bins_K32_smoke.pt`.
* **The denoiser** (`configs/pimm/denoise.py`) is an FM run plus a head: `model.fm=<run dir>`
  (default `$HELIX_EXP/fm`, the stable phase) gives the encoder's architecture, tokenizer
  and (`model.pretrained`, default True) EMA weights; `model.freeze_encoder=True` trains the head only. It reads the `cells` truth
  (section 7: `scripts/build_truth.py` first). Its length is epochs like the FM's: a
  label subset (`data.train.n_events`) needs a larger `epoch` for the same step count. The
  best checkpoint by validation loss is `model/model_best.pth`.

Plot progress: `scripts/helix_run.sh python scripts/plot_train_progress.py label=$HELIX_EXP/fm --out fig.png`.

---

## 7. Evaluate

Truth first, once per corpus run (each event's `hits`/`step` from the campaign at
`HELIX_SIM_ROOT`: the task its `/ident` source file names, the entry its campaign
`event_id` names; writes `{name}_cells_<k>.h5` beside each coefficient shard;
existing files are skipped, so a rerun resumes). It is CPU work over every shard, one
process per core; submit it per run:

    for d in $HELIX_CORPUS_ROOT/coeff_tpc_new/j00*; do
      sbatch --account=mli:cider-ml --partition=milano --cpus-per-task=16 --mem=64G --time=12:00:00 \
          --wrap="$HELIX_ROOT/scripts/helix_run.sh python scripts/build_truth.py --corpus $d"
    done

Then score a run directly from its pimm directory -- no export (GPU: the varlen attention
needs flash attention, sm80+):

    sbatch --account=$HELIX_SLURM_TRAIN_ACCOUNT --partition=$HELIX_SLURM_TRAIN_PARTITION \
        --qos=$HELIX_SLURM_TRAIN_QOS --gpus=1 --cpus-per-task=8 --mem=64G --time=03:00:00 \
        --wrap="HELIX_INTERACTIVE=0 $HELIX_ROOT/scripts/helix_run.sh python -m helix.eval \
                $HELIX_EXP/fm_cooldown --metrics recon,floor,uprobe,labels --out $HELIX_EXP/fm_cooldown/eval.json"
    # a denoiser: RUN is its FM run, --denoiser the denoiser run
    sbatch ... --wrap="HELIX_INTERACTIVE=0 $HELIX_ROOT/scripts/helix_run.sh python -m helix.eval \
                $HELIX_EXP/fm_cooldown --metrics denoise --denoiser $HELIX_EXP/denoise --out $HELIX_EXP/denoise/eval.json"

`RUN` is the run directory (`--weights ema` default, or `last`) or one checkpoint inside it
(`model/iter_N.pth`). Events: the run's own corpus, `probe` role of its holdout
(`--max-events` for a quick look). Metrics: `recon` (the training evaluator's
masked reconstruction), `floor` (detection floor from frozen features, per-view 1% FPR, by
energy; trained against the coefficient amplitude), `uprobe` (along-wire coordinate,
within-group R^2; trained against geometry), `labels` (motif / pid per signal cell, balanced
accuracy; trained against the token layout), `denoise` (`--denoiser <denoise run>`, its best
checkpoint, on the floor's windows). The probes fit
on the first two thirds of the events (at most `FIT_ROWS` = 1M rows each, sampled evenly over
them, so memory stays ~25 GB at any split) and score the rest as they arrive, with [16, 84]%
event-bootstrap intervals; definitions in docs/DECISIONS.md ("The evaluation's definitions").
See `helix/eval/__init__.py`.

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
    IN=$(for d in $R/run_*; do echo /sdf/data/neutrino/doraemon/wire_test_00_00_02/sensor/${d##*/}; done)
    sbatch --account=mli:cider-ml --partition=milano --cpus-per-task=4 --mem=16G --time=04:00:00 \
        --wrap="$HELIX_ROOT/scripts/helix_run.sh python scripts/migrate_format.py sim $R $(echo $IN) --dry-run"
    # read the log, then the same without --dry-run, then
    #   scripts/helix_run.sh python -m helix.data.coeff_verify $R

    R=/sdf/data/neutrino/omara/sbnd_data/coeff_sbnd_v1           # (before the rename: run dirs run018251, ...)
    IN=$(for d in $R/run*; do echo /sdf/data/neutrino/omara/sbnd_data/extracted/${d##*/run}; done)
    # ... migrate_format.py sbnd $R $IN, as above

Older corpora (pre-tau `coeff_tpc`, r1) differ beyond the gate and are not readable. The
`sim` example is historical: the build no longer reads `wire_test_00_00_02` (format 1), so a
wire_test-built corpus can no longer be migrated or rebuilt; r2 already is migrated.

---

## 9. Regression check

`SBND-data-helix/jobs/regress.sh` rebuilds, with a given checkout on a 2080 Ti (the corpora's
GPU type), one SBND shard (TPC + optical, 100 events of run 018251) and one simulation shard
(noisy, 200 events of `run_0027575715`), and compares them byte for byte through
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
| bin tables | `HELIX_ARCHIVE` |
| the simulation campaign (sensor, step, hits) | `HELIX_SIM_ROOT` |
| pimm checkout | `HELIX_PIMM_ROOT` |
| the container | `container:` in the site profile |

| corpus | where (S3DF) | |
|---|---|---|
| simulation d1m | `/sdf/data/neutrino/omara/coeff_tpc_d1m` | production: campaign group j000000 (200k events), with cells; the bin grid's corpus |
| simulation r2 | `/sdf/data/neutrino/omara/coeff_tpc_r2` | superseded (wire_test_00_00_02); readable, not rebuildable |
| SBND v1 | `/sdf/data/neutrino/omara/sbnd_data/coeff_sbnd_v1` | production |
| `coeff_tpc` (pre-tau), `coeff_tpc_r1` | | retired, not readable |

---

## 11. NERSC

`helix/sites/nersc.yaml` is picked up by itself (`$NERSC_HOST`); the commands above are
the same. What differs:

* **Host Python.** The login nodes' `python3` is 3.6: run `scripts/make_host_env.sh` once
  (or set `HELIX_PYTHON`), then `source scripts/helix_env.sh`.
* **Container.** shifter in batch (`train.sh`, `corpus.sh` and `HELIX_INTERACTIVE=0`
  force it), podman-hpc on an interactive TTY; `HDF5_USE_FILE_LOCKING=FALSE` comes from
  the profile. The profile's image tag predates the removal of jax: rebuild and repoint it
  before relying on it.
* **Corpus.** Training reads `$SCRATCH/coeff_tpc_d1m` (Lustre, purged on inactivity), the
  durable copy is `$CFS/m5238/users/oalter/coeff_tpc_d1m` (`HELIX_CORPUS_DURABLE`). It is
  not staged there yet (TODO); once it is, re-stage after a purge with
  `rsync -a $HELIX_CORPUS_DURABLE/ $SCRATCH/coeff_tpc_d1m/`. `corpus.sh` has not been run
  here yet (it takes account, QOS and constraint from the profile; NERSC sets no partition).
* **Bin grid.** Not copied: verify it from the staged corpus (§5, `--verify`), on a GPU
  node (no CPU allocation) through `scripts/helix_run.sh`.
* **Training** (`m5238_g`, `-C gpu`, QOS `preempt` with `--time-min 02:00:00`; preempted
  jobs are requeued and the `afterany` chain continues):

      LINKS=8 GPUS=4 scripts/train.sh fm
      scripts/train.sh fm_cooldown
      LINKS=4 scripts/train.sh denoise model.fm=$HELIX_EXP/fm_cooldown

  One node per run; multi-node training is deferred (TODO).
* **Truth and evaluation** run on GPU nodes (there is no usable CPU allocation):

      sbatch -A $HELIX_SLURM_TRAIN_ACCOUNT -C gpu -q shared --gpus=1 -c 32 --mem=56G --time=08:00:00 \
          --wrap="HELIX_INTERACTIVE=0 $HELIX_ROOT/scripts/helix_run.sh python scripts/build_truth.py --corpus $HELIX_CORPUS_DURABLE/j000000"
      sbatch -A $HELIX_SLURM_TRAIN_ACCOUNT -C gpu -q shared --gpus=1 -c 32 --mem=56G --time=08:00:00 \
          --wrap="HELIX_INTERACTIVE=0 $HELIX_ROOT/scripts/helix_run.sh python -m helix.eval \
                  $HELIX_EXP/fm_cooldown --metrics recon,floor,uprobe,labels --out $HELIX_EXP/fm_cooldown/eval.json"

  `shared` caps a GPU's share at 32 cores and 56 GB; the floor holds its features in host
  memory (10-40 GB over the probe split), so one evaluation per job. Truth needs the
  simulation campaign (`HELIX_SIM_ROOT`) at this site; NERSC has none. Write it into
  the durable copy, as above, and re-stage: `$SCRATCH` is purged.
