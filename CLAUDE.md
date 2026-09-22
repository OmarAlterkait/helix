# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

HELIX covers the whole path from raw LArTPC wire data to a trained foundation
model: coherent-noise removal and wavelet sparsification, a coefficient corpus,
a masked autoencoder over those coefficients, and a 3D probe that asks whether
the learned representation knows where charge is.

| package | may import pimm-data / pimm | what it is |
|---|---|---|
| `helix.core` | **no** | detector-agnostic wavelets, coeff IO, provenance, lazy backend dispatch |
| `helix.tpc` | **no** | LArTPC physics: coherent gate, forward model (noise + digitize), geometry, corpus builder |
| `helix.model` | no (defers one import) | the FM: tokenizer, blocks, masking, loss, muP |
| `helix.probe` | no | the 3D deconvolution probe |
| `helix.optical` | no | PMT light path — a SEPARATE pipeline, not part of the wire FM |
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
pimm-data through 9 public symbols; `helix/integrations/pimm/*` reaches pimm
through its registries. Nothing else imports either.

## Commands

Everything runs in one container, and NOTHING names it. The runtime, image,
in-image interpreter and binds are site facts declared in
`helix/sites/<site>.yaml`; `scripts/helix_run.sh` reads them.

```bash
python -m helix.paths              # FIRST: site, every root, its source, whether it exists
scripts/helix_run.sh python -m pytest tests -q
scripts/helix_run.sh python scripts/build_coeff_corpus.py --shard ... --out ...
```

`helix.paths` needs only os/pathlib/yaml, so it runs outside the container too —
run it first, always. It picks the site from `HELIX_SITE` or auto-detects
(`$NERSC_HOST`, `/sdf`). `source scripts/helix_env.sh` exports the same values
into a shell, for `#SBATCH` and for pimm's launcher, neither of which can read
them any other way.

`helix_run.sh` handles what used to be spelled out per call site:
`PYTHONNOUSERSITE=1` (a bound home makes `~/.local` packages visible inside the
image, so a green run can be yours alone), `PYTHONPATH=pimm:helix`, the site's
own environment, and an explicit bind of both checkouts.

When editing pimm-data, put its working tree ahead of the installed copy with
`PYTHONPATH=<pimm-data>/src` — the image INSTALLS pimm-data, so a plain pytest
tests the baked copy, not your edits.

**Two runtimes at NERSC, and the choice is not cosmetic.** podman-hpc is
interactive and SINGLE-NODE: it does not inject NERSC's NCCL plugin at all, and
measured 1.23 GiB/s across nodes against shifter's 4.9. shifter is the
multi-node runtime and the one pimm's launcher drives. `helix_run.sh` picks
podman-hpc when interactive; pass `HELIX_INTERACTIVE=0` to force shifter.

**Submitting does NOT happen in the container**, because sbatch is not in it.
`scripts/make_launcher_env.sh` builds the small host-side environment pimm calls
launcher-only — its dependency list minus pimm-data, plus helix's base install —
and `scripts/submit_helix.sh` finds it with no flag. Two things keep that
environment sufficient: `helix/data/__init__.py` resolves the reader and the
dataset lazily, so reading the bin table does not pull torch, and pimm's launch
preflight loads the training config with `import_custom_modules=False`. The
second is a LOCAL change in the pimm checkout; a fresh clone needs it again.

See `docs/RUNBOOK.md` for corpus → train → eval → probe, and
`configs/launch/README.md` for getting a run onto the scheduler.

## Structural patterns worth knowing

**Lazy multi-backend dispatch (`helix.core.backend`).** jax/torch are imported
only when their backend is selected, never at `import helix` time. Each op
family ships `<family>_<backend>.py`; `backend.ops(...)` imports only the active
one. Adding an op means adding it to every backend module, or it silently works
on one.

**Values are raw; normalisation is a sidecar.** The corpus stores unnormalised
coefficients plus a `norm_sigma` table applied at tokenize — deliberately, so
the corpus stays reversible and tokenization stays changeable.

**The corpus is compression; the tokenizer is interpretation.** Changing the
wavelet, the gate or `tau` costs a 344 GB rebuild and a new `basis_digest`,
invalidating comparability with every existing checkpoint. Changing `n_bands`,
cell geometry or masking costs a config change and a retrain. Vary
interpretation freely; change compression only with a measured reason.

**Reading a checkpoint has ONE owner (`helix.model.artifact`).** Eight shapes
are in circulation and eight loaders used to read them, each knowing a subset —
a `pimm export` directory was taught to `load_probe_model` and not to
`patch_config_from_checkpoint`, so probing a pimm-trained model died in
`torch.load`. `detect`/`inspect`/`load`/`build` are the whole surface;
`tests/test_artifact_formats.py` is the matrix, and a ninth shape without a row
there fails. The split is by PURPOSE: RESUME state (`<save_path>/model/last/`,
`iter_N.pth`) is pimm's and nothing here reads it; EVAL state is a weight set
plus the operating point that makes its number reproducible. The one caller
deliberately left alone is `hooks.py:234` — EMA resume, training hot path.

**A `pimm export` cannot attribute itself; promote it.** `_sanitize_config`
nulls `weight`, so an export cannot say whether it holds EMA or raw weights, and
it names neither the corpus nor the helix commit. `scripts/export_artifact.py
<export_dir> --weights ema --corpus <dir> -o <artifact_dir>` records all three.
Probe the artifact, not the export, or the row is unattributed.

**`plane_id` currently serves FOUR roles** — FiLM conditioning, RoPE projection
axis, masking group, and plane identity. `make_mask` groups by that LABEL, not
by token position, so the token set is already order-free (nothing assumes
planes are contiguous). For a second modality those four roles separate; see
`docs/ARCHITECTURE.md` §8.

## Traps that have cost real time

- **Provenance refuses a dirty tree.** Build a corpus from an uncommitted
  checkout and every shard records `git_dirty: True`; `coeff_verify` then
  refuses it. This cost a 344 GB rebuild once.
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
  event separation — `MULTI_EVENT_BATCHING.md`), and the trainer raises if
  `batch_size != world_size`. It is derived from `WORLD_SIZE`, never written
  down; a literal is right at exactly one GPU count.
- **GPFS does not support the locks HDF5 and uv take.** Every real-shard read at
  NERSC fails `OSError: [Errno 524] unable to lock file` without
  `HDF5_USE_FILE_LOCKING=FALSE`, which the site profile's `env:` sets. `uv` hits
  the same thing on its cache; point `UV_CACHE_DIR` at node-local `/tmp`.
- **A corpus run may have gaps.** `run_0027670361` is missing source files
  51-56 and 94-97; 180 shards / 17,999 events is complete for it, not a failure.
- **The probe's extraction loop is the long pole.** 2.5M patches over 388
  events before a single probe is fitted; a preemption used to discard all of
  it. Pass `--cache-dir $SCRATCH/probe_cache` — chunked, keyed by everything
  that changes the features, resumable.
- **`save_path` must resolve through `helix.paths`**, never relative to the
  checkout — a relative one wrote a run INTO the repo and four files were
  committed.
