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

    cd <your helix checkout>     && $PY -m pytest -q   # 395 passed, 50 skipped
    cd <your pimm-data checkout> && $PY -m pytest -q   # 360 passed,  8 skipped

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

## 1. Build a coefficient corpus

Input: doraemon sensor shards (`HELIX_SENSOR_ROOT`).
Output: sharded HDF5 under `HELIX_CORPUS_ROOT/<run>/`.

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
actually train on:

    scripts/calibrate_norm_sigma.sh          # one frozen table the corpus shares
    $PY scripts/derive_coeff_bins.py --corpus <dir> --out bins.pt [--events 120]

The bins shipped with `m113` were derived from an old cache built with a
different (white) noise model. Do not reuse bins across corpus generations.

---

## 2. Train

    sbatch launch/coeff_fm_train.sbatch                    # one job
    ./launch/chain_submit.sh <n_jobs> <run_name>           # a chain

**Set these on the command line** — `#SBATCH` directives cannot read shell
variables, so the defaults in the file point at one person's allocation and log
directory:

    sbatch --account=<facility>:<repo> --output=<your logs>/coeff-fm-%j.out \
           launch/coeff_fm_train.sbatch

`chain_submit.sh` forwards `"$@"` to every link.

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

    # NOT tools/convert_fm_ckpt.py. It is frozen as the one-time rescue of the
    # historical m113 checkpoint, which could not describe itself, and it cannot
    # read a pimm checkpoint at all (it wants 'model'/'ema'; pimm writes
    # 'state_dict'). helix/model/checkpoint.py says so: "Nothing trained from
    # here should go through it."

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
        --tag <name> --out probe_results.jsonl

`--cell-t` matters and run_probe will refuse rather than guess: an artifact and a
converted checkpoint carry their tokenizer, but for one that does not you must
pass the value the run trained with (`grep cell_t <run>/config.py`). The old
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

### Reviving an old research checkpoint

The 394 checkpoints other than `m113` are NOT self-contained and mostly cannot be
evaluated. The research trainer kept bin edges in a separate `tier1_bins.pt`
named only on the command line, and nothing in the checkpoint records which one;
the edges are training-set statistics and are not recoverable from the weights.

`tools/convert_fm_ckpt.py` is the way back: it infers architecture from the
tensors, cross-checks it against the checkpoint's metadata, strips the DDP
prefix, and inlines the bins into one file that stands alone. It has been run
once, producing `archive/fm_m113_converted.pt`. For the others you must supply
the matching bins file, and for most it no longer exists.
