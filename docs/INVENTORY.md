# Inventory: everything that is not in git

Companion to `HANDOVER.md`. Measured **2026-09-18** on S3DF. Sizes are what `du`
reported, not estimates.

The three repositories total about 17 MiB and carry no data. Everything below is
what has to be copied, rebuilt, or deliberately left behind.

Paths are given relative to the roots in `helix/paths.py`, so that a receiving
site can set the env var rather than recreate the directory layout.

---

## Summary

| tier | contents | size |
|---|---|---|
| **required** | r1 corpus, bin table, container | **354 GB** |
| **recommended** | + eval weights, cooldown branch points, m113 artifact | **~356 GB** |
| **optional** | + legacy corpus, full pretraining runs | **580 GB** |
| **leave behind** | retired research checkpoints, retirement backups | 160 GB |

Almost all of it is the corpus. Everything else required is **9.2 GB of
container and 8 KB of bin table**, and the container can be rebuilt instead of
copied. Continuing the training — as opposed to only reproducing its numbers —
costs about **1.5 GB of selected checkpoints**, not the 112 GB the pretraining
directories occupy.

---

## 1. Required

Without these, nothing trains.

### The corpus — 345 GB

    $HELIX_CORPUS_ROOT/coeff_tpc_r1/

Eight run directories, 202 files each: `sim_wire_coeff_NNNN.h5` shards plus a
`holdout.json` recording the train/eval split. The `_calib` subdirectory holds
the norm_sigma calibration.

`HELIX_CORPUS` must name **one run directory**, not the root — for example
`coeff_tpc_r1/run_0027575715`. Pointing it at the root is a plausible-looking
mistake that fails later and unhelpfully.

This generation applies the occupancy gate `tau = 0.05`. Its `basis_digest` is
`8c4542b6…`, and that digest is what every downstream consumer checks against.

### The bin table — 8.3 KB

    $HELIX_ARCHIVE/coeff_bins_r1_tau05_run0027575715_v2.pt

The categorical bin grid the model's objective is defined over. Derived from
training-set statistics by `scripts/derive_coeff_bins.py`, and not recoverable
from a trained checkpoint.

**It is, however, rederivable from the corpus — verified bit-identical on
2026-09-18.** The derivation is deterministic: it pools the *first* `--events`
events in dataset order, with no sampling, seed or RNG anywhere in the script.
The file records its own parameters, so nothing has to be guessed:

```
corpus: .../coeff_tpc_r1/run_0027575715     events: 120     K: 128     n_bands: 4
```

Rerunning with exactly those (and the default `--lo-pct 0.05 --hi-pct 99.95`)
reproduced `edges`, `cent_asinh` and `cent_ratio` with `max|diff| = 0`:

```bash
python scripts/derive_coeff_bins.py \
  --corpus $HELIX_CORPUS_ROOT/coeff_tpc_r1/run_0027575715 \
  --events 120 --K 128 --n-bands 4 --out bins.pt
```

Its other input, the `norm_sigma` table, lives in the corpus under `_calib/`
and therefore travels with it.

Two conditions on that guarantee. It must be rederived from **`run_0027575715`
specifically** — a table from any of the other seven runs would be
self-consistent but incompatible with the existing checkpoint. And the shard set
must be identical, since "first 120 events" is defined by dataset order.

So copy it — eight kilobytes removes a class of mistake — but it is not a single
point of failure, and losing it does not strand the corpus.

Four other bin tables sit beside it and are not interchangeable:

| file | belongs to |
|---|---|
| `coeff_bins_r1_tau05_run0027575715_v2.pt` | **the production r1 model** |
| `coeff_bins_r1_tau05_run0027575715.pt` | superseded r1 revision |
| `coeff_bins_run0027575715.pt` | pre-tau corpus |
| `coeff_bins_k4_run0027575715.pt` | pre-tau, k=4 variant |
| `coeff_bins_K32_smoke.pt` | smoke fixture |

Copy all five if the cost of deciding exceeds 40 KB.

### The container — 9.17 GB

    $HELIX_IMAGE   →   /sdf/data/neutrino/omara/images/helix-train.sif

Or rebuild from `container/helix-train.def` in about eleven minutes — see
`HANDOVER.md` §3. Rebuilding is preferable to copying if the receiving site has
apptainer, because the def pins its base by digest and the result is
byte-reproducible in the ways that matter.

Currently baked: `pimm_data 2b20573c`, jax `cuda12`, torch `2.10.0+cu126`,
PyWavelets 1.8.0, base `ghcr.io/deeplearnphysics/pimm v0.5.1`.

---

## 2. Recommended

Not needed to train from scratch; needed to continue, compare, or reproduce a
published number.

### Checkpoints — what each is actually for

The production checkpoint is a **result**, not an input. The cooldown config
reads

```python
weight = '.../coeff-fm-train-r1-8run/model/last'
resume = False
```

so training continues from the *pretraining* run, and `resume = False` means
**weights only** — no optimizer, no scheduler, no step count. That is what makes
the continuation cheap: any single `iter_NNNNN.pth` serves as a branch point
just as well as the DCP directory.

| purpose | what you need | size |
|---|---|---|
| evaluate, or reproduce the published numbers | `coeff-fm-cooldown-r1-8run/model/model_ema.pth` + `provenance.json` + `config.py` | **226 MB** |
| rerun the cooldown as it was run | `coeff-fm-train-r1-8run/model/last/` | **679 MB** |
| a longer or earlier cooldown — the top-ranked experiment | a few `iter_*.pth` near the plateau | **226 MB each** |
| retrain from scratch | nothing | — |

Pretraining ran 112,500 steps over 3 epochs — 37,500 steps/epoch, checkpointed
every 225 steps — so the plateau at **epoch 1.78** sits near **step 66,750**.
Branch candidates run from roughly `iter_59175` (epoch 1.58) upward. Three or
four of those is about 1 GB, against 113 GB for all 500.

### The production run directory — 5.8 GB

    $HELIX_EXP/coeff-fm-cooldown-r1-8run/

The 8-run model after cooldown. Contains:

| file | what it is |
|---|---|
| `model/model_ema.pth` | 226 MB — the EMA weights, the ones to evaluate |
| `model/model_best.pth` | 226 MB — best-by-metric |
| `model/last/` | rank-sharded DCP resume state (`trainer.dcp/__0_0.distcp` …) |
| `model/iter_*.pth` | periodic checkpoints, 226 MB each |
| `provenance.json` | what attributes the weights to a corpus and a commit |
| `config.py`, `resolved_config.json` | the configuration actually used |
| `events.out.tfevents.*` | six tensorboard logs across preemption boundaries |

Resume needs **both** `weight=` and `resume=`; `model/last` is a directory, not
a file. `docs/RUNBOOK.md` §2 has the incantation.

Dropping the `iter_*.pth` files brings this under 1 GB if space is tight, at the
cost of being unable to restart from a mid-run point.

### The m113 eval artifact — 226 MB

    $HELIX_ARCHIVE/fm_m113_artifact/
      artifact.json          49 KB
      weights.safetensors   226 MB

The one surviving research checkpoint, converted into a self-describing eval
artifact: weights, operating point, the **pre-tau** `basis_digest`, and the
converter's full provenance record. Used for the cross-corpus comparison in
`docs/SCIENCE.md`.

It is pre-tau, so evaluating it needs the legacy corpus below. That dependency
is the only reason to move 54 GB of superseded data.

---

## 3. Optional

### The legacy corpus — 54 GB

    $HELIX_LEGACY_CORPUS   →   coeff_tpc/run_0027575715

Pre-tau, `basis_digest` `7f954a84…`. Required **only** to evaluate m113. Never
mix it with r1 — `helix/data/identity.py` exists to refuse exactly that, and the
refusal is a feature.

### The pretraining runs — 112 GB each

    $HELIX_EXP/coeff-fm-train-r1-8run/            112 GB
    $HELIX_EXP/coeff-fm-train-r1-8run-plane25/    112 GB
    $HELIX_EXP/coeff-fm-train-r1/                  16 GB
    $HELIX_EXP/coeff-fm-cooldown-r1-8run-plane25/ 5.8 GB

The runs the cooldown continued from, plus the plane-masking arm. They are large
only because 500 periodic checkpoints are retained at 226 MB each. The parts
that matter are small:

    coeff-fm-train-r1-8run/model/last/            679 MB   DCP resume state
    coeff-fm-train-r1-8run/model/model_ema.pth    226 MB   pretraining end weights
    coeff-fm-train-r1-8run/model/iter_*.pth       226 MB   each, 500 of them

Take `model/last/` plus a handful of `iter_*.pth` around the plateau and leave
the other ~490 behind. Copying all of both 112 GB directories is only warranted
for auditing the loss curve checkpoint-by-checkpoint rather than through the
tensorboard logs, which are already in the run directory.

`$HELIX_EXP` totals **293 GB** across all run directories.

### Source simulation

    $HELIX_SENSOR_ROOT   →   /sdf/data/neutrino/doraemon/wire_test_00_00_02/sensor

46 run directories of simulator output. Needed only to **build a new corpus** or
to regenerate probe truth (`scripts/dump_probe_truth.py` takes `--source`).

This lives in a different group's directory, not in the helix owner's space. If
the receiving site intends to build corpora rather than only train on the
existing one, arranging access to equivalent simulation is a prerequisite, not a
detail.

---

## 4. Leave behind

### Retired research checkpoints — 160 GB, 397 files

    $HELIX_ARCHIVE/fm_research_ckpts/

Do not copy. These are not self-contained: the research trainer kept bin edges
in a separate file named only on the command line, and nothing in the checkpoint
records which one. The edges are training-set statistics and cannot be recovered
from the weights, and for most of these the matching bins file no longer exists.

`tools/convert_fm_ckpt.py` — which inferred architecture from the tensors and
inlined the bins — was run exactly once, on m113, and has been retired. It
remains in git history. Re-exporting a run beats reviving one.

`$HELIX_ARCHIVE/fm_m113_converted.pt` (226 MB) is kept as lineage; nothing reads
it, and pointing a tool at it prints where the artifact is instead.

*(`docs/RUNBOOK.md` §6 records 413 files / 165 GB for this directory. The
measured figure today is 397 files / 160 GB.)*

### Retirement backups — 149 MB

    $HELIX_ARCHIVE/retirement-backups/

Record of what was deleted and why, including `HAZARDS.md`. Small enough to
bring for provenance, useless for running anything.

---

## 5. Regenerated, never copied

These are outputs. Copying them across sites invites a stale one being trusted.

| what | produced by | notes |
|---|---|---|
| probe truth | `scripts/dump_probe_truth.py` | needs `--corpus` and `--source`; resumable |
| probe feature cache | `scripts/run_probe.py` | 44 GB measured for the full probe set; keyed by a config hash, resumes per event |
| norm_sigma table | `scripts/calibrate_norm_sigma.sh` | lives in the corpus `_calib/`; resume is keyed on a per-run gate stamp |
| eval artifacts | `scripts/export_artifact.py` | promotes a `pimm export` into something attributable |

The feature cache is the one worth knowing about before you start: the full
probe run is long, and it resumes at event granularity, so an interrupted run
costs one event rather than the whole pass.

---

## 6. Disk notes for the receiving site

- Nothing large belongs on a `/sdf/group`-style software quota. The original
  site's 10 TB group area was at 100% until 742 GB of dead cache and stray
  checkpoints were cleared off it.
- The corpus is read heavily during training. Node-local scratch for the probe
  feature cache is worth arranging.
- Home directories are typically small and snapshotted — never stage builds,
  package caches, or container images there. The container build in particular
  needs `APPTAINER_TMPDIR` and `APPTAINER_CACHEDIR` pointed at scratch, or it
  will fill whatever it defaults to.
