# Handover

Everything needed to take over the helix foundation-model work, as of
**2026-09-18**. It assumes the coefficient corpus has already been copied to the
receiving site; `docs/INVENTORY.md` lists every other file that is not in git.

Read this once, end to end, before running anything. It is short. The depth
lives in the four documents already in the repo, and this one only tells you
which to open and in what order.

---

## 1. The three repositories

All three are required. helix alone does not train; pimm alone does not know
what a coefficient is.

| repo | remote | branch | revision | role |
|---|---|---|---|---|
| **helix** | `github.com/OmarAlterkait/helix` | `main` | `6518f55` | LArTPC physics: DSP, corpus, the FM, the probe |
| **pimm-data** | `github.com/OmarAlterkait/pimm-data` | `master` | `2b20573c` | generic HDF5 → torch data layer |
| **pimm** | `github.com/DeepLearnPhysics/pimm-private` | `coeff-fm` | `39512a9` | the trainer |

Sizes are small — 10.6 MiB, 418 KiB and 6.25 MiB packed. Nothing here is a
large clone.

**What each one owns.** helix owns anything that knows the detector is a liquid
argon TPC. pimm-data owns anything detector-agnostic. pimm is an upstream
training framework that neither of the other two forks — `coeff-fm` carries a
single commit on top of `main`, and that commit only pins a dependency.

`helix.core` and `helix.tpc` must never import `pimm_data`; `helix.data` and
`helix.integrations` may. A test enforces this, which is what lets the DSP half
run in environments where torch and jax do not exist.

### The `coeff-fm` branch is not a fork

It is `origin/main` plus one commit pinning `pimm-data` to an explicit
revision. It holds no Python.

The FM training configs live in **helix**, under `configs/pimm/`, and are passed
to the trainer by absolute path — `scripts/submit_coeff_fm_train.sh` invokes
`python -m pimm.train --config-file "$CFG"` directly rather than going through
pimm's `scripts/train.sh`, whose `-c` flag resolves only beneath `configs/`.

That matters because it means **helix never writes into the pimm checkout**. An
earlier arrangement created a `configs/helix` symlink inside pimm to work around
the relative-only `-c`; the consolidated launcher removed the need for it. If
you find such a symlink in a pimm working tree, it is a leftover — delete it.

Keep the branch rebased on `origin/main` rather than letting it drift. It was
ten commits behind when this handover was prepared, and picking up upstream's
W&B forking/rewind and slurm `time_min` work cost nothing.

---

## 2. The lockstep rule

**This is the one constraint that silently breaks everything, so it comes before
the instructions.**

The LArTPC forward model — `AddNoise`, `Digitize`, the coherent-noise oracle —
used to live in pimm-data and now lives in helix. Both register their transforms
into the same registry by string, and `pimm_data/_registry.py` raises `KeyError`
on a duplicate registration.

So a pimm-data from **before** the move, plus a current helix, does not
misbehave subtly — it fails to import at all. There is no partial-credit state.
The pair must move together.

The revision appears in **four places**, and all four move or none do:

| file | field |
|---|---|
| `helix/container/helix-train.def` | `PIMM_DATA_REV` |
| `helix/pyproject.toml` | `[tool.uv.sources] pimm-data.rev` |
| `helix/uv.lock` | two entries |
| `pimm/pyproject.toml` (branch `coeff-fm`) | `[tool.uv.sources] pimm-data.rev` |

`helix/tests/test_lockstep_pin.py` asserts the def and helix's pyproject agree.
It does not see pimm's copy — check that one by hand.

**Never pin pimm-data without a `rev`.** An unpinned git dependency follows
whatever the default branch holds, `uv lock` records the result in a file nobody
reads, and the environment straddles the boundary. That is exactly how the
previous pin drifted to a pre-move revision.

After changing the revision, **rebuild the image**. An environment assembled
from the new pin and the old image is the straddle by another route.

---

## 3. The container

One image covers both halves of the workflow — DSP needs PyWavelets, training
needs pimm-data, and the two used to be separate images that were each missing
what the other had.

    /sdf/data/neutrino/omara/images/helix-train.sif      9.17 GB

Built from `container/helix-train.def` on a pinned base digest, so it is
reproducible on any cluster with apptainer:

```bash
export APPTAINER_TMPDIR=$LSCRATCH/aptmp APPTAINER_CACHEDIR=$LSCRATCH/apcache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
cd helix
apptainer build --fakeroot helix-train.sif container/helix-train.def
```

Roughly 11 minutes on 16 CPUs. The base is
`ghcr.io/deeplearnphysics/pimm v0.5.1`, pinned by digest
`sha256:c0dc5a64…a776bf`, so the build does not drift when that tag moves.

**jax build variant.** The def takes `%arguments JAX_EXTRA`, default `cuda12`.
For a CPU-only site:

    apptainer build --fakeroot --build-arg JAX_EXTRA=cpu helix-train.sif container/helix-train.def

The build refuses to finish if installing jax *moves* any nvidia wheel torch is
using — added wheels are fine, moved ones are not, because that silently
reshuffles torch's CUDA stack.

---

## 4. First hour on a new machine

Genuinely sequential — each step depends on the one before.

**1. Clone all three, side by side.**

```bash
git clone git@github.com:OmarAlterkait/helix.git
git clone git@github.com:OmarAlterkait/pimm-data.git
git clone -b coeff-fm git@github.com:DeepLearnPhysics/pimm-private.git pimm
```

**2. Point the environment at your paths.** Every external path is an env var
with a default naming the machine helix was developed on. Set the ones that
differ; see §5.

**3. Ask the code where it thinks everything is.**

```bash
python -m helix.paths
```

It prints all twelve roots, whether each came from the environment or a default,
and whether it exists. Anything marked missing will fail later in a way that
looks like a code bug. Fix this first.

**4. Build the image** (§3), or point `HELIX_IMAGE` at an existing one.

**5. Run the tests.**

```bash
apptainer exec -B /sdf,/lscratch $HELIX_IMAGE \
  env PYTHONNOUSERSITE=1 PYTHONPATH=$PWD \
  /opt/pimm/.venv/bin/python -m pytest tests -q
```

Expect **492 passed, 54 skipped**. The skips are real-data and GPU tests; a skip
because data is absent looks identical to a skip because the machine has no GPU,
which is why `tests/_paths.py` exists — read its docstring if the count differs.

**6. Smoke the whole path with no data at all.** `docs/RUNBOOK.md` §0b builds a
synthetic corpus from `pimm_data.testing` and runs corpus → train → probe end to
end. This is the step that proves the handover worked, and it needs no
simulation output. It is also why pimm-data `2b20573c` matters: earlier
revisions did not stamp `config/{num_wires, volume_ranges}` into the fixture,
and the probe stage died on a missing key.

Only after §0b passes is it worth pointing anything at real data.

---

## 5. Environment

Twelve roots, all overridable, all listed by `python -m helix.paths`. The ones
that matter most:

| variable | what it points at |
|---|---|
| `HELIX_CORPUS_ROOT` | parent of the built coefficient corpora |
| `HELIX_CORPUS` | **one** corpus — a run dir, not the root |
| `HELIX_SENSOR_ROOT` | simulator output the corpus is built from |
| `HELIX_ARCHIVE` | bin tables, eval artifacts |
| `HELIX_EXP` | where runs write |
| `HELIX_IMAGE` | the container |
| `HELIX_PIMM_ROOT` | the pimm checkout |
| `HELIX_PIMM_DATA_SRC` | a pimm-data checkout's `src/` |
| `HELIX_LEGACY_CORPUS` | the pre-tau corpus — m113 only |

Two defaults will not resolve anywhere but the original site and should be set
explicitly: `HELIX_SENSOR_ROOT` points into `/sdf/data/neutrino/doraemon/`,
which belongs to a different group, and `HELIX_OPTICAL_DATA` points into
another user's home directory.

**One environment rule.** Do not put a pimm-data checkout on `PYTHONPATH` next
to the image's installed copy. Two resolvable copies make which one wins depend
on `sys.path` order, which is the fragility the single image exists to end. The
image's baked pimm-data is the authority; override it only deliberately, and
never while also trusting the lockstep guarantee.

---

## 6. What must move

Covered in full by **`docs/INVENTORY.md`** — every file not in git, with sizes
and whether it is required, optional, or lineage only.

The short version, assuming the corpus is already across, is that **very little
else is large**:

| | what | size |
|---|---|---|
| required | the bin table `coeff_bins_r1_tau05_run0027575715_v2.pt` | 8 KB |
| required | the image — or rebuild it from the def, which is equivalent | 9.17 GB |
| to evaluate | `coeff-fm-cooldown-r1-8run/model/model_ema.pth` + `provenance.json` + `config.py` | 226 MB |
| to continue | `coeff-fm-train-r1-8run/model/last/`, plus a few `iter_*.pth` near the plateau | ~1.5 GB |
| optional | the m113 artifact, for the cross-corpus comparison | 226 MB |
| do not move | 397 retired research checkpoints — not self-contained, see §8 | 160 GB |

Two things that look like separate items but live **inside** the corpus tree and
travel with it — provided it was copied whole rather than filtered on
`sim_wire_coeff_*.h5`:

- `coeff_tpc_r1/_calib/` (40 KB) — the norm_sigma tables
- `coeff_tpc_r1/run_0027575715/truth/probe_truth_probe.h5` (1.1 GB) — the probe
  truth, which exists for that one run only

Check both landed, along with each run's `holdout.json`. Regenerating the probe
truth requires the source simulation, which is the one input that belongs to a
different group.

The bin table is **rederivable bit-identically** from the corpus (§1 of the
inventory has the command and the verification). Copy it anyway — it is eight
kilobytes — but it is not a single point of failure.

---

## 7. State of the work

**Verified on 2026-09-18**, by execution, on the revisions above:

| check | result |
|---|---|
| pimm-data suite | 363 passed, 7 skipped |
| helix suite, inside the new image | 492 passed, 54 skipped |
| lockstep guard | 4 passed |
| image bakes the pinned rev | `2b20573c`, forward model absent from pimm-data and registered by helix |
| `python -m helix.paths` | all twelve roots resolve |

The helix suite was run against the image's own baked pimm-data with no
`PYTHONPATH` override — the deliverable validating itself rather than a
stand-in.

### The science, in one paragraph

The model is a masked autoencoder over wavelet coefficients; the 3D probe asks
whether its representation knows where charge is. Masking whole planes rather
than random cells is worth **+0.160** on the cross-plane task and takes the
probe from **0.53 to 0.85**. The production run plateaued at **epoch 1.78 of 3**
— the last 1.2 epochs bought +0.0037 — and the cooldown then added **+0.041 for
12.7% more compute** and was still rising when it ended. Train/eval gap is
+0.040 / +0.038 at 59,150,848 parameters, so nothing is overfitting.

`docs/SCIENCE.md` has the measurements, including a hypothesis that measurement
refuted. Ranked by expected return, the next experiments are: **a longer and
earlier cooldown** (strongest — it was still climbing), then a bigger model,
then an objective sweep, then more data (weakest — 8× data bought representation,
not reconstruction).

---

## 8. Known gaps

Things that are true, that you would otherwise discover the hard way.

**No eval artifact for the production checkpoint.** `m113` has one; the 8-run
cooldown does not. Its weights are attributable through the run directory's
`provenance.json`, but not self-describing the way an artifact is. Creating one
is a single `scripts/export_artifact.py` invocation and is the one real gap in
the handover manifest.

**The 394 other research checkpoints cannot be evaluated.** The research trainer
kept bin edges in a separate file named only on the command line, and nothing in
the checkpoint records which. The edges are training-set statistics and are not
recoverable from the weights. The converter was run once, on m113, and retired.
Re-exporting a run beats reviving one.

**Two corpus generations exist and must not be mixed.** `coeff_tpc_r1` applies
the occupancy gate `tau = 0.05` — 3 wires of 64 — and `coeff_tpc` predates it.
Measured over 1,200 (event, plane) pairs the gate gives 5.32× lower stripe
residual and 2.61× fewer off-signal pixels above 5 ADC. They carry different
`basis_digest` values (`8c4542b6…` and `7f954a84…`) and `helix/data/identity.py`
exists to refuse the mismatch. m113 is pre-tau, which is why the legacy corpus is
still listed.

**`cubic_wireplane_geometry.json` is duplicated** byte-identically across helix
and pimm-data. A design call, not a packaging bug — `docs/ARCHITECTURE.md` §8
argues both sides.

**`plane_id` carries four meanings at once** in `helix.model`: FiLM conditioning,
RoPE axis, masking group, and plane identity. One modality made that free; a
second will need them split. What is *not* a problem, despite looking like one:
masking selects planes by per-token label, not token position, so the token set
is already order-free.

**`helix/optical` is live but out of scope.** A separate PMT-light pipeline
sharing only `helix.core`. It is not part of the FM path and the handover
validation does not exercise it.

**Superseded images remain on disk.** `helix-train-NEW.sif` and
`helix-train-cuda.sif` sit beside the canonical image; `helix-train-prepin.sif`
is the pre-pin build kept as a rollback. All three are deletable once the new
image has been exercised.

---

## 9. Where to read next

| you want to | read |
|---|---|
| run something | `docs/RUNBOOK.md` |
| know why it is shaped this way | `docs/ARCHITECTURE.md` |
| know what was measured | `docs/SCIENCE.md` |
| run or extend the tests | `TESTING.md` |
| know what is not in git | `docs/INVENTORY.md` |
| understand the corpus format | `COEFF_CORPUS_DESIGN.md` |
| see what was decided and rejected | `DECISIONS.md` |

`docs/RUNBOOK.md` §3 — "Two kinds of checkpoint. Read this once; it saves an
afternoon." — is the single highest-value section in the repository for someone
picking this up.
