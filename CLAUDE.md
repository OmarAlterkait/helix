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
numpy/h5py/PyWavelets/scipy. `tests/test_boundary.py` enforces it with three
complementary checks — read its docstring before moving code between
subpackages.

Only two directories import an external package. `helix/data/*` reaches
pimm-data through 9 public symbols; `helix/integrations/pimm/*` reaches pimm
through its registries. Nothing else imports either.

## Commands

Everything runs in one container (`docs/ARCHITECTURE.md` §4):

```bash
IMG=${HELIX_IMAGE:-/sdf/data/neutrino/omara/images/helix-train.sif}
PY="apptainer exec -B /sdf,/lscratch $IMG /opt/pimm/.venv/bin/python"

$PY -m helix.paths                 # FIRST: every external path, its source, whether it exists
$PY -m pytest -q                   # expect 440 passed / 54 skipped
```

`env PYTHONNOUSERSITE=1` is worth adding: `-B /sdf` remounts home, so anything
pip-installed under `~/.local` is visible inside the container and a green run
may be yours alone. When editing pimm-data, put its working tree ahead of the
installed copy with `PYTHONPATH=<pimm-data>/src` — the image INSTALLS pimm-data,
so a plain pytest there tests the baked copy, not your edits.

See `docs/RUNBOOK.md` for corpus → train → eval → probe.

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
  `--output` and `--partition` are literal. The corpus build (turing) and
  training (ampere) need DIFFERENT accounts — `sacctmgr -n show assoc
  user=$USER format=Account,Partition,QOS` lists yours.
- **A corpus run may have gaps.** `run_0027670361` is missing source files
  51-56 and 94-97; 180 shards / 17,999 events is complete for it, not a failure.
- **`save_path` must resolve through `helix.paths`**, never relative to the
  checkout — a relative one wrote a run INTO the repo and four files were
  committed.
