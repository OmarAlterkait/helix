# Architecture

Two packages, one boundary. This document says where the boundary is, why it is
there, and which of the choices behind it were measured rather than assumed.

Everything here was verified against the code on 2026-09-12 (helix 0.3.0,
pimm-data 0.4.0). Where a number appears, the way it was measured is named.

---

## 1. The two packages

**helix** — LArTPC signal processing, the coefficient corpus, the foundation
model, and the 3D probe. Base install is `numpy`, `h5py`, `PyWavelets`, `scipy`.
torch and jax are OPTIONAL extras; `pimm-data` is the optional `[pimm]` extra.

**pimm-data** — a generic data layer between sharded HDF5 detector output and
PyTorch training loops, serving four detector families (jaxtpc, lucid, optical,
coeff). Requires `torch>=2.5` and `hdf5plugin` UNCONDITIONALLY.

That asymmetry is the whole reason a boundary exists rather than one package.

| | helix | pimm-data |
|---|---|---|
| torch | optional extra | hard requirement |
| PyWavelets | hard requirement | not a dependency |
| scope | one detector's physics | any detector family |

### The invariant

    helix.core and helix.tpc must NEVER import pimm_data.
    helix.data and helix.integrations may, and do.

This is not style. It is what makes the two-container split possible (§4).
`tests/test_boundary.py` enforces it; see §6 for why that test has three parts
instead of one.

---

## 2. helix, subpackage by subpackage

| subpackage | modules | may import pimm_data | what it is |
|---|---|---|---|
| `helix.core` | 10 | **no** | wavelet transforms, coefficient IO, backend dispatch, provenance |
| `helix.tpc` | 19 | **no** | LArTPC physics: coherent removal, the noise forward model, digitization, geometry, the corpus builder's plumbing |
| `helix.model` | 9 | no (see below) | the foundation model: tokenizer, blocks, masking, loss, muP |
| `helix.probe` | 8 | no | the 3D deconvolution probe |
| `helix.optical` | 6 | no | PMT light path (separate from the wire path) |
| `helix.data` | 6 | **yes** | corpus dataset/reader/verifier, the registered transforms, the identity guard |
| `helix.integrations` | 9 | **yes, in submodules** | adapters into a training framework (pimm) |

Two subtleties that matter and are easy to get wrong:

**`helix.model` is clean, deliberately.** `tokenize.py` defers its one
`pimm_data.transform` import into a function body. The tokenizer's protocol is
duck-typed — a callable taking `data` and returning `data`, carrying a `scope`
attribute — so registration is the consumer's one-liner rather than a dependency.

**`helix.integrations`'s `__init__` is an empty namespace, also deliberately.**
Its docstring states the rule: "each module in this package imports a
THIRD-PARTY framework at module scope, so importing one is an explicit act by a
consumer that already has that framework installed." So the PACKAGE imports
clean and only its SUBMODULES are heavy. That is what keeps `import helix` free
of torch, and `test_boundary.py` locks it in — an `__init__` that started
re-exporting its submodules would break it silently.

---

## 3. Where the forward model lives, and why

The LArTPC forward model — intrinsic noise, coherent noise, ADC digitization —
is helix's. Going-to-dense — scattering sparse rows into per-plane grids — is
pimm-data's. The rule: *every detector family wants densify; only one wants this
particular detector's physics.*

| | lives in | |
|---|---|---|
| `densify`, `offset2batch` | pimm-data | generic scatter, any family |
| `dense_ops_jax.densify_plane_jax` | pimm-data | still densification, despite the filename |
| `Densify` transform | pimm-data | |
| `noise.py`, `noise_jax.py` | **helix** | LArTPC physics |
| `add_intrinsic_noise`, `digitize` | **helix** | |
| `AddNoise`, `Digitize` transforms | **helix** (`helix/data/transforms.py`) | |
| `configs/jaxtpc/sensor_dense_gpu.py` | **helix** | a recipe applying helix physics to jaxtpc shards |

### Why the transforms are in `helix.data`, not `helix.tpc`

Measured, not assumed: **importing `pimm_data.transform` to reach
`@TRANSFORMS.register_module` pulls 36 pimm_data modules and torch.**
`helix.tpc` today imports with neither.

So the split is: kernels in `helix.tpc`, registered wrappers in `helix.data`.
A registered transform is inherently a pimm-data object; a noise kernel is not.

### This move is LOCKSTEP

`pimm_data/_registry.py:171` raises `KeyError` on duplicate registration
(verified by direct test). Therefore:

* old pimm-data + new helix → both register `AddNoise` → **KeyError**
* new pimm-data + old helix → nobody registers it → **missing transform**

No environment may straddle it. Both containers must be rebuilt together.
This is why `pimm-fm`'s lock had to be repointed: it pinned pimm-data at
`aea39aff`, 26 commits behind, a revision that still defines `AddNoise`.

---

## 4. The container

**One image: `/sdf/data/neutrino/omara/images/helix-train.sif`**, built by
`container/helix-train.def`. It does corpus building AND training.

This section used to say the two-container split was correct and not friction.
That was wrong, and the way it was wrong is worth recording, because the
reasoning looked sound.

The split was real but it was an artifact of using two other people's images:

| | `develop.sif` | `pimm-latest.sif` |
|---|---|---|
| jax | 0.5.3 | absent |
| torch | 2.5.1+cu121 | 2.10.0+cu126 |
| pywt | 1.8.0 | **absent** |
| pimm_data | **absent** | **0.3.0 — stale** |
| hdf5plugin | **absent** | present |

Neither could run the whole workflow, so the split looked structural. Three
things were actually wrong:

1. **`pimm-latest.sif` baked pimm_data 0.3.0, which still REGISTERS AddNoise and
   Digitize.** helix registers both now, so the lockstep conflict of §3 was live
   inside the training image. It is also the reason
   `configs/pimm/coeff_fm_train.py` carries an elaborate `sys.path.insert(1)`
   dance whose comment says the checkout "has to WIN over site-packages, not
   merely be present" — a workaround for a broken image, living in a config.

2. **`develop.sif` has no pimm_data at all**, and the corpus builder needs it
   (`scripts/build_coeff_corpus.py` imports `densify` from `pimm_data`). It only
   ever appeared to work because of an editable `.pth` in ONE developer's home
   (`__editable__.pimm_data-0.1.0.pth` -> that person's checkout), which the
   container picks up because `-B /sdf` remounts home. A colleague gets nothing.
   Test results measured that way were partly an artifact of whose shell ran them.

3. **`develop.sif` has no hdf5plugin**, so reading real doraemon shards failed
   with an HDF5 filter error.

The corrected image fixes all three. It is built FROM `pimm-latest.sif` to keep
the CUDA work worth keeping — torch 2.10+cu126 and the prebuilt
torch_scatter / torch_sparse / spconv / flash-attn wheels that cannot be compiled
on a login node with no CUDA toolkit — and then:

* uninstalls pimm_data 0.3.0 and installs the current one (replaced, not
  shadowed, so which copy wins cannot depend on `sys.path` order)
* adds PyWavelets, helix's only dependency the base image lacked
* adds pytest, because the first thing anyone does with a new environment is ask
  whether it works
* adds **jax CPU, deliberately not `jax[cuda12]`**. The corpus builder DEFAULTS
  to `--backend torch` ("measured 10.6 ms") and `submit_coeff_corpus.sh` passes
  it, so jax is an override rather than the production path. A CUDA jax would
  install its own `nvidia-*` wheels beside torch's, which is how a pinned CUDA
  stack gets broken. CPU jax costs nothing and lets 19 jax-parametrised tests
  actually run instead of skipping.

**pimm and helix are NOT installed in the image.** They are the code under
development and arrive by bind mount / PYTHONPATH, so editing either does not
require a rebuild. Only dependencies are baked.

The `%post` ends with an assertion that fails the BUILD if pimm_data is stale,
has lost `Densify`, or still registers `AddNoise`/`Digitize` — so a lockstep
violation cannot ship in an image again.

Verified in that image with user site-packages disabled, so no home directory can
help: **helix 386 passed / 50 skipped, pimm-data 360 passed / 8 skipped.**

### Rebuilding

    APPTAINER_TMPDIR=$LSCRATCH apptainer build --fakeroot \
      /sdf/data/neutrino/omara/images/helix-train.sif container/helix-train.def

Two traps the definition already handles, both found the hard way:

* uv installs by hardlinking from its cache, and in the build sandbox those links
  do not materialise — a cached package "installs" in 99 ms and then imports as
  `ModuleNotFoundError: No module named 'pywt.version'`. The FIRST build passes
  (nothing cached yet) and the REBUILD fails.
* `UV_LINK_MODE=copy` alone was not enough, because by then the cache held
  half-written entries and copying a corrupt entry reproduces the corruption.
  The build sets `UV_NO_CACHE=1` and does not depend on cache state at all.

Rebuild whenever pimm-data changes: the image is part of the lockstep pair.

## 5. Data flow, end to end

```
  doraemon sensor shards  (HDF5, raw wire ADC)
            |
            |  helix.tpc: coherent gate -> DWT -> threshold
            |  scripts/build_coeff_corpus.py          [develop.sif]
            v
  coefficient corpus      (HDF5 shards: value + coord/{band,plane_gid,wire,tau}
                           + ident/{event,run,source_file,noise_seed}
                           + config attrs incl. basis_digest, removal_json)
            |
            |  helix.data: reader -> dataset -> CoeffTokenize
            |  pimm training loop via helix.integrations.pimm   [pimm-latest.sif]
            v
  foundation model        (masked autoencoding over wavelet coefficients)
            |
            v
  3D probe / evaluators   (helix.probe)
```

Each stage stamps provenance, and the next stage checks it. `helix/data/identity.py`
compares a checkpoint's recorded corpus identity (`basis_digest`, `removal_json`,
`sigma_norm`) against the corpus actually being read: it REFUSES on mismatch and
WARNS when a corpus is unstamped. That guard is why the retired 540 GB
`fm_cache_tpc` could never have been used again — it carried no stamps at all.

---

## 6. Why the boundary test has three parts

`tests/test_boundary.py` looks redundant and is not. Each part covers a hole the
others leave, and each was mutation-tested by actually adding
`import pimm_data` to `helix/tpc/noise.py` and confirming a failure.

1. **Package-level import probes.** `helix`, `helix.core`, `helix.tpc`,
   `helix.model.tokenize`, `helix.integrations` must each import in a clean
   subprocess with pimm_data, torch and jax all absent.

2. **A source scan.** A grep alone false-positives: `helix/tpc/dense_ops.py`,
   `geometry.py`, `io.py` and `noise.py` all NAME pimm_data in comments, and
   `tokenize.py` documents a registration example. This part proves the files
   that mention it do not import it.

3. **A per-submodule sweep.** The necessary one. `import helix.tpc` executes
   only its `__init__`, which pulls **5** submodules — `noise`, `dense_ops`,
   `geometry` and `noise_jax` are NOT among them, so part 1 cannot see a
   violation in any of those files. Part 3 imports each submodule alone, and
   catches violations *transitively*: the mutation in `noise.py` also failed
   `dense_ops` and `noise_jax`, which import it.

Only `pimm_data` is forbidden in part 3. torch and jax are not — the backend
modules (`wavelet_ops_torch`, `coherent_ops_jax`, `dense_ops`) import them at
module scope by design and are declared optional extras. The invariant this file
defends is about pimm-data, not about weight of imports.

---

## 7. Paths

`helix/paths.py` is the single resolution point for every external path. Each is
an environment variable with an S3DF default, and the defaults are "where these
live on the machine helix was developed on; they are defaults, not truths."

`python -m helix.paths` reports every root, whether it came from the environment
or the default, and whether it exists. That is the first command to run in a new
environment.

`repo()` locates the checkout from `__file__`, never from a name — so a renamed
or relocated checkout still finds itself.

Two things paths.py deliberately does NOT cover:

* **`#SBATCH` directives.** They cannot read shell variables, so `--account`,
  `--output` and `--partition` in `launch/coeff_fm_train.sbatch` stay literal and
  point into one person's allocation and log directory. They are overridable on
  the command line and `chain_submit.sh` forwards `"$@"`; the script header says
  so with an example.
* **`pimm-fm/configs/helix`.** pimm's `train.sh` resolves `-c helix/<cfg>` under
  its OWN `configs/` with no absolute-path option, so a link must exist there.
  `launch/coeff_fm_train.sbatch` now creates it from the running checkout
  (creating, repointing a stale one, or failing loudly on a real directory).

---

## 8. Open items

**`cubic_wireplane_geometry.json` is duplicated.** Both repos ship a
byte-identical 153 KB copy (`md5 06f18b9c646303b8c21c2ffad8919f91`) and a
`load_plane_registry` each. pimm-data's `Densify` genuinely needs a plane
registry — `n_wires`/`n_ticks`/`pedestal` — to size each plane's grid, and
helix's DSP needs the same file.

Arguments for leaving it: the file is an *exported artifact* from JAXTPC
(`scripts/export_plane_geometry.py`), authored by neither repo; in both, the
bundled copy is only a fallback (`_resolve` takes any path); and the two
`canonical_plane_id` functions are genuinely different — helix's is wire-only,
pimm-data's also handles `volume_N_Pixel`, which is why
`test_canonical_plane_id_stable` did not move across the boundary.

The argument against: it is the same mirror pattern that `test_forward_mirror.py`
was written to police, and that test was deleted when its subject stopped being
mirrored. A duplicated data file drifts as silently as duplicated code.

Not resolved here because it is a design call, not a packaging fix. If it is
resolved, the likely shape is: the detector description belongs to whoever owns
the detector (helix), and pimm-data's `Densify` receives `geom=` from the caller
rather than shipping its own copy.

**The `pimm-fm` pin is a path source.** `{ path = "../pimm-data", editable = true }`
works because the two checkouts are siblings. At handover, once pimm-data is
pushed, it flips back to a git source WITH an explicit `rev` — the original had
no `rev`, so `uv lock` silently followed the default branch and the lock file was
the only record of what got resolved.
