# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

HELIX covers the whole path from raw LArTPC wire data to a trained foundation
model: coherent-noise removal and wavelet sparsification, a coefficient corpus,
a masked autoencoder over those coefficients, and a 3D probe that asks whether
the learned representation knows where charge is.

| package | may import pimm-data / pimm | what it is |
|---|---|---|
| `helix.core` | **no** | wavelet transform, robust statistics, hysteresis, coeff IO, provenance |
| `helix.tpc` | **no** | LArTPC physics: TPC chain, coherent gate, forward model (noise + digitize), readouts, readers |
| `helix.build` | **no** | the build pipeline (sources are `scripts/build_{sim,sbnd,goop}.py`) |
| `helix.model` | no (defers one import) | the FM: tokenizer, blocks, masking, loss, muP |
| `helix.eval` | no (events read through `helix.data`) | evaluating a run: `python -m helix.eval` (recon, floor, uprobe, denoise) |
| `helix.optical` | no | the optical codec — written beside the TPC shards, not yet fed to the FM |
| `helix.data` | **pimm-data** | coeff corpus reader/dataset/verifier, registered transforms, identity guard |
| `helix.integrations.pimm` | **pimm** | adapters into the training framework |

## The invariant

`helix.core` and `helix.tpc` must NEVER import `pimm_data`. `helix.data` and
`helix.integrations` may, and do. This is not style: pimm-data requires
`torch>=2.5` and `hdf5plugin` unconditionally, while helix's base install is
numpy/h5py/PyWavelets/scipy. `tests/test_boundary.py` enforces it with five
complementary checks — read its docstring before moving code between
subpackages. One of them runs the other way: `helix.data` MAY import pimm-data,
but `bins` and `identity` inside it must not, or a login node cannot load the
training config.

Only two directories import an external package. `helix/data/*` reaches
pimm-data through a few public symbols (`ShardReaderBase`, `ShardEventDataset`,
the registries, `Compose`); `helix/integrations/*` reaches pimm through its
registries. Nothing else imports either (`helix.model.tokenize` defers its one
registration import).

## Commands

Everything runs in one container, and NOTHING names it. The runtime, image,
in-image interpreter and binds are site facts declared in
`helix/sites/<site>.yaml`; `scripts/helix_run.sh` reads them.

```bash
source scripts/helix_env.sh && $HELIX_PYTHON -m helix.paths   # FIRST: site, roots, container, scheduler
scripts/helix_run.sh python -m pytest tests -q
scripts/helix_run.sh python scripts/build_sim.py --input RUN/sim_wire_sensor_0000.h5 --output OUT/RUN/sim_wire --device cuda
```

`helix.paths` needs only os/pathlib/yaml, so it runs outside the container too —
run it first, always. It picks the site from `HELIX_SITE` or auto-detects
(`$NERSC_HOST`, `/sdf`). `source scripts/helix_env.sh` exports the same values
into a shell (roots, `HELIX_CONTAINER_*`, `HELIX_SLURM_<KIND>_*`), which is how
`corpus.sh` and `train.sh` get their sbatch flags. On S3DF the system python3 is
3.6 without PyYAML: `scripts/make_host_env.sh` builds `$SCRATCH/helix-host-venv`,
which `helix_env.sh` finds by itself.

`helix_run.sh` handles what used to be spelled out per call site:
`PYTHONNOUSERSITE=1` (a bound home makes `~/.local` packages visible inside the
image, so a green run can be yours alone), `PYTHONPATH=pimm:helix`, the site's
own environment, and an explicit bind of both checkouts.

When editing pimm-data, put its working tree ahead of the installed copy with
`PYTHONPATH=<pimm-data>/src` — the image INSTALLS pimm-data, so a plain pytest
tests the baked copy, not your edits.

**Two runtimes at NERSC, and the choice is not cosmetic.** podman-hpc is
interactive and SINGLE-NODE: it does not inject NERSC's NCCL plugin at all, and
measured 1.23 GiB/s across nodes against shifter's 4.9. shifter is the batch
runtime. `helix_run.sh` picks podman-hpc when interactive; pass
`HELIX_INTERACTIVE=0` to force shifter (`corpus.sh` and `train.sh` do).

**Submitting does NOT happen in the container**, because sbatch is not in it.
`scripts/corpus.sh` (corpus: plan, calibrate, build arrays) and `scripts/train.sh
CONFIG [key=value ...]` (training: a chain of `LINKS` jobs, `--dependency=afterany`,
each running `python -m pimm.train` in the container and resuming from the latest
checkpoint) run on the host and need only the host Python above. pimm's own launch
layer (`pimm submit`, launch recipes) is not used; helix needs only pimm's training
engine.

See `docs/RUNBOOK.md` for host env → corpus → verify → train → eval → probe, and
`docs/ARCHITECTURE.md` for the modules and the shard format.

## Structural patterns worth knowing

**One backend: torch, any device.** The DSP (TPC chain, optical encoder) is torch on
the GPU or the CPU; there is no numpy twin to keep in step. torch is still never
imported at `import helix` time: corpus readers, the tokenizer and the optical
decoder (`unhaar`) are numpy, and `helix.core.wavelet` imports torch when it runs.

**Values are raw; normalisation is a table.** The corpus stores unnormalised
coefficients plus a `norm_sigma` table (in every shard's `/config`) applied at
tokenize — deliberately, so the corpus stays reversible and tokenization stays
changeable.

**Every shard says its format.** `/config` `format = "helix-coeff-2"` is written by
`helix.core.coeff_io` and required by every reader; there are no legacy read paths.
An unstamped corpus is migrated with `scripts/migrate_format.py` or rebuilt.

**Configs state intent.** `configs/pimm/fm.py` plus `fm_cooldown`,
`fm_smoke`; no config carries a step count (warmup and cadence are `"auto"`, filled
by `FMTrainer` through `helix.core.cadence`). A variant is a `key=value` override,
not a new file.

**The corpus is compression; the tokenizer is interpretation.** Changing the
wavelet, the gate or `tau` costs a full corpus rebuild (r2: 262 GB) and a new `basis_digest`,
invalidating comparability with every existing checkpoint. Changing `n_bands`,
cell geometry or masking costs a config change and a retrain. Vary
interpretation freely; change compression only with a measured reason.

**Reading a run has ONE owner (`helix.model.checkpoint.load_run`).** Downstream
code takes pimm's run directory (`save_path`) or one checkpoint in it, never an
exported copy: `resolved_config.json` gives the architecture, the train split's
`CoeffTokenize` cfg and the corpus; `model/model_ema.pth` (WeightEMA) or
`model/last` the weights, with the bins as persistent buffers; `provenance.json`
the code and corpus identity. Starting a training run from a checkpoint is pimm's
`weight=`. Evaluation is `python -m helix.eval RUN --metrics ... --out FILE`
(`helix/eval/`); truth is the `cells` modality (`scripts/build_truth.py`).

**`plane_id` currently serves FOUR roles** — FiLM conditioning, RoPE projection
axis, masking group, and plane identity. `make_mask` groups by that LABEL, not
by token position, so the token set is already order-free (nothing assumes
planes are contiguous). For a second modality those four roles separate; see
`docs/ARCHITECTURE.md` §8.

## Traps that have cost real time

- **Provenance refuses a dirty tree.** Build a corpus from an uncommitted
  checkout and every shard records `git_dirty: True`; `coeff_verify` then
  refuses it. This cost a 344 GB (r1) rebuild once.
- **`#SBATCH` directives cannot read shell variables**, so `--account`,
  `--output` and `--partition` are literal. That is why scheduler facts live in
  the site profile and reach `sbatch` as CLI flags (which override `#SBATCH`),
  via `scripts/helix_env.sh`. The corpus build and training may need DIFFERENT
  accounts — they do at S3DF, because authorisation is per partition — which is
  why `scheduler:` is split by job kind. `sacctmgr -n show assoc user=$USER
  format=Account,Partition,QOS` lists yours.
- **A root a site does not have is `null`, not a wrong path.** `helix.paths`
  refuses an unconfigured root rather than inventing one, so a config that needs
  it fails at import with a named cause. That is correct for RUNNING it, and it
  means a test that merely LOADS configs must tell "absent at this site" apart
  from "broken" — catch `SiteError` and skip. Two tests already do.
- **The global batch IS the rank count.** The FM takes one event per rank (no
  event separation — `docs/archive/MULTI_EVENT_BATCHING.md`), and `FMTrainer`
  raises unless `batch_size // world_size == 1`. It is derived from `WORLD_SIZE`, never written
  down; a literal is right at exactly one GPU count.
- **GPFS does not support the locks HDF5 and uv take.** Every real-shard read at
  NERSC fails `OSError: [Errno 524] unable to lock file` without
  `HDF5_USE_FILE_LOCKING=FALSE`, which the site profile's `env:` sets. `uv` hits
  the same thing on its cache; point `UV_CACHE_DIR` at node-local `/tmp`.
- **A corpus run may have gaps.** `run_0027670361` is missing source files
  51-56 and 94-97; 180 shards / 17,999 events is complete for it, not a failure.
- **The floor's features live on the host.** `helix.eval` floor holds every map
  row's float16 features in memory, ~10-40 GB over the 388-event probe split;
  several evaluations on one node can exhaust host memory silently.
- **`save_path` must resolve through `helix.paths`**, never relative to the
  checkout — a relative one wrote a run INTO the repo and four files were
  committed.
