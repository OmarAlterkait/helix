# Runbook

Procedures as commands, run from the helix checkout. Paths, container and scheduler come from
`helix/sites/<site>.yaml` (`docs/ARCHITECTURE.md` §11), so the commands are the same at S3DF
and NERSC; differences are marked **S3DF:** / **NERSC:**. Anything heavier than a quick check
goes to Slurm: `corpus.sh`, `submit.sh` and `sweep.py` submit their own jobs.

Order for a new corpus: host env (§1), build (§2), verify (§3), truth (§4), bin grid (§5),
train (§6), evaluate (§7, §8).

## 1. Host environment and tests

The submitting host needs a Python with PyYAML and `sbatch`; jobs run in the container.

    scripts/make_host_env.sh           # once: $SCRATCH/helix-host-venv (helix base + pimm's launch deps, no torch)
    source scripts/helix_env.sh        # exports HELIX_*, HELIX_CONTAINER_*, HELIX_SLURM_<KIND>_*, HELIX_PYTHON
    $HELIX_PYTHON -m helix.paths       # site, every root and where it came from, container, scheduler

Any `HELIX_*` in the environment wins over the profile (e.g. `HELIX_CORPUS=<root>/<run>`).
Set `HELIX_SITE` in batch jobs. **NERSC:** the system `python3` is 3.6, so the venv is needed.

    scripts/helix_run.sh python -m pytest tests -q
    HELIX_REQUIRE_PIMM=1 HELIX_REQUIRE_DATA=1 scripts/helix_run.sh python -m pytest tests -q -rs   # skips become errors

Read the skips: pimm, real data, CUDA and `HELIX_JAXTPC_ROOT` tests skip when absent. A
collection error means the environment is wrong. `HELIX_FORWARD_ENV="VAR ..."` forwards more
variables into the container. To test a pimm-data working tree instead of the image's copy:
`HELIX_FORWARD_ENV=PYTHONPATH PYTHONPATH=<pimm-data>/src scripts/helix_run.sh python -m pytest -q`.

**S3DF**, on a CPU node:

    sbatch --account=mli:cider-ml --partition=milano -c 8 --mem=32G -t 01:00:00 \
        --wrap="$HELIX_ROOT/scripts/helix_run.sh python -m pytest -q -p no:cacheprovider tests"

**NERSC** has no usable CPU allocation; use a shared GPU:

    sbatch -A m5238_g -C gpu -q shared --gpus=1 -c 32 --mem=56G -t 01:00:00 \
        --wrap="HELIX_INTERACTIVE=0 $HELIX_ROOT/scripts/helix_run.sh python -m pytest -q tests"

## 2. Build a corpus

    scripts/corpus.sh plan      SOURCE OUT RUN_DIR...   # OUT/_calib/CHUNKS.txt and RUNS.txt
    scripts/corpus.sh calibrate SOURCE OUT              # one job: norm_sigma per run; mean -> norm_sigma_global.npy
    scripts/corpus.sh build     SOURCE OUT              # job arrays over CHUNKS.txt
    scripts/corpus.sh all       SOURCE OUT RUN_DIR...   # the three; build waits on calibrate (afterok)

    scripts/corpus.sh all sim $HELIX_CORPUS_ROOT/coeff_tpc_new $HELIX_SIM_ROOT/j00*

`SOURCE` is `sim`, `sbnd`, `module:Class` or a build script ending in `helix.build.main(Source)`.
For `sim`, a `RUN_DIR` is a campaign group (200 tasks, 200k events); a task is one sensor file
(1000 events, 5 shards of 200). Environment: `EVENTS_PER_TASK` (500), `EVENTS_PER_SHARD` (sim
200, must divide 1000; else 100), `CAL_EVENTS` (100 per run), `CONFIG` (chain overrides; the same
for calibrate and build), `BUILD_ARGS` (e.g. `--no-optical`), `CONC` (16), `ARRAY_MAX` (100),
`LOGS` (`OUT/_logs`). One task on a 2080 Ti: ~13 min, ~226 MB per shard.

- **Order:** every shard embeds the global `norm_sigma`, so `build` refuses to start without it
  unless a calibration job is pending in the same `all`.
- **Resume:** rerun `scripts/corpus.sh build SOURCE OUT`; tasks run with `--skip-complete`.
- **A clean, unchanging tree:** every shard records the helix commit and dirty flag, and the
  verifier refuses a dirty tree or mixed commits. Build from a frozen worktree.
- **One GPU type:** the profile pins `scheduler.corpus` (**S3DF:** turing, `mli:default`,
  preemptable with `--requeue`). **NERSC:** A100s only, so a NERSC build is a different corpus
  (new digest, grid and truth); `corpus.sh` has not been run there. The campaign is not at NERSC
  (`HELIX_SIM_ROOT: null`).
- `goop` is optical only: build it with `python -m helix.build goop` directly.

One shard by hand on a GPU node:

    T=$HELIX_SIM_ROOT/j000000/jaxtpc_wire_j000085-000089
    scripts/helix_run.sh python -m helix.build sim --input $T/sensor/sim_wire_sensor_${T##*/}.h5 \
        --output OUT/j000000/sim_wire --device cuda --count 200 --events-per-shard 200 --shard-start 85

All options: `helix/build.py`'s docstring.

## 3. Verify

    scripts/helix_run.sh python -m helix.data.verify OUT                    # every run, against _calib/CHUNKS.txt
    scripts/helix_run.sh python -m helix.data.verify OUT/RUN --dataset-name sim_wire [--expect N]
    scripts/helix_run.sh python -m helix.data.verify compare A.h5 B.h5 [A2.h5 B2.h5 ...]

Prints `corpus OK` or the problems (exit 1). Against the plan, a gap the plan never wrote is
fine and a planned shard absent is an error; a run directory alone reports every gap. The
whole-corpus check reads every shard's metadata, so submit it for a full corpus. Count events
as distinct identities, not by summing `n_events`.

## 4. Truth

`scripts/build_truth.py` writes `{name}_cells_<k>.h5` beside each coeff shard from the campaign's
`hits` / `step` / `labl` at `HELIX_SIM_ROOT`; existing files are skipped, so a rerun resumes.
CPU work, one process per core, per run directory. **S3DF only** (NERSC has no campaign; copy
the cells shards with the corpus):

    for d in $HELIX_CORPUS_ROOT/coeff_tpc_new/j00*; do
      sbatch --account=mli:cider-ml --partition=milano -c 16 --mem=64G -t 12:00:00 \
          --wrap="$HELIX_ROOT/scripts/helix_run.sh python scripts/build_truth.py --corpus $d"
    done

**NERSC:** `$SCRATCH` is purged on inactivity. Re-stage from the durable copy:
`rsync -a $CFS/m5238/users/oalter/coeff_tpc_d1m/ $SCRATCH/coeff_tpc_d1m/`.

## 5. Bin grid

The production grid is named in `helix/data/data/reference_bins.json` and read from
`$HELIX_ARCHIVE/<name>.pt`. A site with the corpus verifies its copy instead of carrying it:

    scripts/helix_run.sh python -m helix.data.bins \
        --verify $HELIX_ARCHIVE/coeff_bins_d1m_j000000_noisy.pt --corpus $HELIX_CORPUS

(`VERIFIED` or `MISMATCH`.) Derive a grid, for a new corpus only, after the build:

    scripts/helix_run.sh python -m helix.data.bins --corpus $HELIX_CORPUS --out $HELIX_ARCHIVE/<name>.pt

A new reference (a new name in the JSON with `digests: null`) logs three digests to paste into
`"digests"` and commit. `--like bins.pt` reuses a grid's recorded parameters. Do not re-derive
a grid for a grown corpus: it changes every trained head.

## 6. Train

    scripts/submit.sh CONFIG [--recipe preempt|interactive|PATH] [--run.name NAME] [pimm submit flags] [-- key=value ...]

`CONFIG` is a name in `configs/pimm/` or a path. `submit.sh` runs `pimm submit` on the host with
the site's facts as flags; submitit chains the links (`--chain.jobs`, default 8) and each resumes
from `$HELIX_EXP/NAME/model/last`. The same `NAME` with another config, overrides or node count
is refused. `--resources.nnodes N` gives 4 GPUs per node; `--dry-run` renders only. Recipes:
`preempt` (default, 24 h links) and `interactive` (4 h; **NERSC** only, two jobs per user).
**NERSC:** `submit.sh` refuses a pimm checkout not at the pinned `pimm_rev` or dirty
(`HELIX_PIMM_UNPINNED=1` overrides). **NERSC:** QOS preempt with a 2 h floor; a preempted link is
requeued under the same job id. **S3DF:** QOS normal (not preempted); links end on walltime.

    scripts/submit.sh fm --resources.nnodes 4                      # stable phase over HELIX_CORPUS's corpus
    scripts/submit.sh fm_cooldown --resources.nnodes 4             # WSD cooldown from $HELIX_EXP/fm/model/last
    scripts/submit.sh fm --run.name fm_pf25 -- model.plane_frac=0.25
    scripts/submit.sh fm_smoke --recipe interactive --resources.nnodes 2 --chain.jobs 1
    scripts/submit.sh ft_eval --run.name ft_cool --resources.nnodes 1 -- model.fm=$HELIX_EXP/fm_cooldown
    scripts/submit.sh denoise --resources.nnodes 4 -- model.fm=$HELIX_EXP/fm_cooldown
    scripts/submit.sh denoise --run.name dn_gauss -- model.cond_charge=True model.dist=gauss

- **No step counts.** Warmup and save cadence are `"auto"` (log line `cadence over N steps`);
  change the length with `epoch=`. A label subset (`data.train.n_events=N`) needs more epochs.
- **Batch = GPUs.** `submit.sh` appends `batch_size=W ... num_worker=...` for W = nodes x
  GPUs; `FMTrainer` refuses anything else and scales the FM's peak LR by sqrt(W / 16) (the
  denoiser's LR is absolute). By hand (`python -m pimm.train`) on several ranks, pass the batch.
- **Warm starts** are `model.init=<run>` (the cooldown) or `model.fm=<run or checkpoint>` (the
  denoiser), never pimm's `weight=`. `model.pretrained=False` is from scratch;
  `model.freeze_encoder=True` trains the head only.
- **ft_eval** must run on exactly 4 ranks (one node): its length is the protocol.
- **A hung rank** dumps its stacks and exits after 300 s without a step (`StepWatchdog`); the
  NCCL flight recorder dumps on the collective timeout. The chain stops at a failure: resubmit
  the same command, which resumes.
- **GPU type:** sm80+ (A100) only for training and evaluation; never turing.

**A study** is a file: `$HELIX_PYTHON scripts/sweep.py configs/sweeps/<study>.yaml [--dry-run]
[--only A,B]` submits each arm through `submit.sh`, an optional second stage per arm (`then:`,
e.g. the cooldown), and evaluation jobs (`after_each`, `after_all`) on the site's `eval` job
kind, held by `afterok`. Studies: `d1m_lr.yaml` (LR and seed, with cooldowns and uprobe),
`ft_eval.yaml` (encoders through the fine-tune, two seeds each, plus from-scratch),
`d1m_denoise.yaml` (denoiser heads at full length).

## 7. Evaluate an encoder: the fine-tune

Fine-tune it (`ft_eval`, §6, or the `ft_eval.yaml` study), then score the fine-tuned run on a
GPU node:

    EVAL="-A $HELIX_SLURM_EVAL_ACCOUNT -q $HELIX_SLURM_EVAL_QOS --gpus=1 -c 32 --mem=56G -t 04:00:00"
    # S3DF: add -p ampere.  NERSC: add -C gpu.
    sbatch $EVAL --wrap="HELIX_INTERACTIVE=0 $HELIX_ROOT/scripts/helix_run.sh python -m helix.eval \
        $HELIX_EXP/ft_cool --out $HELIX_EXP/ft_cool/eval.json"

`RUN` is a run directory (`--weights last`, the default; `ema` / `best` for runs from before
2026-11-09) or one checkpoint in it. The events are the probe split of the run's corpus
(`--max-events N` for a quick look). A run trained at another site finds its corpus and FM run
under this site's roots, or takes `--corpus` / `--fm`; a different `basis_digest` is refused (do
not work around it). Output: `eval.json` and `eval.windows.npz`. Read
`metrics.denoise` for the clear and crowded@near efficiencies per energy and view, against
`amplitude` on the same windows (definitions: `docs/ARCHITECTURE.md` §10).

Compare two encoders on identical windows, paired, seed by seed:

    scripts/helix_run.sh python -m helix.eval compare \
        --a $HELIX_EXP/ft_X/eval.windows.npz $HELIX_EXP/ft_X_s1/eval.windows.npz \
        --b $HELIX_EXP/ft_Y/eval.windows.npz $HELIX_EXP/ft_Y_s1/eval.windows.npz --out X_vs_Y.json

Without `--b`, `--arm presence --arm-b amplitude` gives what the fine-tune adds over the corpus
gate. Read `views`: a difference carried by one view is not a win.

## 8. uprobe

The frozen along-wire probe of a pretrained FM run (the second key):

    sbatch $EVAL --wrap="HELIX_INTERACTIVE=0 $HELIX_ROOT/scripts/helix_run.sh python -m helix.eval \
        --uprobe $HELIX_EXP/fm_cooldown --out $HELIX_EXP/fm_cooldown/uprobe.json"

It holds up to 1M rows of features per fit in host memory (~6 GB each, `--probe-seeds` fits per
arm), so run one evaluation per job.

## 9. Where things are

| what | root | NERSC | S3DF |
|---|---|---|---|
| corpora | `HELIX_CORPUS_ROOT` | `$SCRATCH` | `/sdf/data/neutrino/omara` |
| the training run dir | `HELIX_CORPUS` | `$SCRATCH/coeff_tpc_d1m/j000000` | `.../coeff_tpc_d1m/j000000` |
| runs | `HELIX_EXP` | `$CFS/m5238/users/oalter/exp/helix` (S3DF runs: `.../exp_s3df`) | `/sdf/data/neutrino/omara/exp/helix` |
| bin tables | `HELIX_ARCHIVE` | `$CFS/m5238/users/oalter/archive` | `/sdf/data/neutrino/omara/archive` |
| campaign | `HELIX_SIM_ROOT` | none | `.../doraemon_2026_1M_lartpc/data/jaxtpc_wire` |
| pimm | `HELIX_PIMM_ROOT` | `$HOME/nu/pimm-pin` (pinned) | `/sdf/group/neutrino/omara/pimm-fm` |

**NERSC containers:** shifter in batch (submit.sh jobs, `corpus.sh`, `HELIX_INTERACTIVE=0`),
podman-hpc on an interactive TTY; `HDF5_USE_FILE_LOCKING=FALSE` comes from the profile. The
image tag `helix-nersc:jax-cuda12` is stale: rebuild and repoint before relying on a new
dependency.
