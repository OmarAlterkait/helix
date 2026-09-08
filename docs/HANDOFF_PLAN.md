# Handoff plan — making helix + pimm-data transferable

**Goal.** A colleague takes ownership of the whole codebase and continues the
training runs, on infrastructure that is not this cluster.

**Status.** Revised after an assumption audit (31 assumptions, 6 clusters) that
refuted several of the first draft's premises. Corrections are marked ⚠ below.

**Pinned to.** Every claim here was checked against:

| repo | commit |
|---|---|
| helix | `72e9866` |
| pimm-data | `9d7b6ca` (branch `consolidation-boundary`) |
| pimm | `bccdd298` |

The audit's sharpest finding was that the first draft cited line numbers without
SHAs, and the tree moved under it — the coeff registration path, the pimm import
requirement, and the verifier's checks all changed shape mid-audit. Any step
below whose evidence is a bare line number should be re-derived before it runs.

**Already done** (was phase 1 in the first draft): `helix/paths.py` is live and
`tests/_paths.py` resolves through it. `HELIX_CORPUS` had two defaults with two
different meanings; it now has one. See `72e9866`.

---

## 0. What "done" means

The colleague can, without asking us anything:

1. clone helix and pimm-data at a commit that exists on a remote;
2. install, with dependencies that resolve;
3. run `python -m helix.paths` and be told exactly what external data they need;
4. follow one runbook from corpus to trained model to probe number;
5. read what the science established, and which numbers compare to which;
6. modify the code without discovering that two repos duplicate each other.

Six statements, each currently false.

---

## 1. Inventory — measured, not estimated

### Hardcoded external paths

| area | files | lines | in scope |
|---|---|---|---|
| `configs/` | 9 | 31 | yes |
| `scripts/` | 22 | 53 | yes |
| `tools/` | 1 | 3 | yes |
| `tests/` | 2 | 6 | yes (already partly env-driven) |
| `helix/` | 1 | 11 | yes |
| `research/` | 138 | 277 | **policy decision, see §6** |

Five categories, each wanting a different answer: corpora and derived data;
simulator source shards; sibling code checkouts; containers; and one external
JAXTPC asset (`config/noise_spectrum.npz`).

### Documents, by whether they help or mislead

| file | state |
|---|---|
| `README.md` | **misleads** — describes a DSP library; no mention of the FM, corpus, or training |
| `CONSOLIDATION_PLAN.md` | **misleads** — a *different* consolidation, different goal, finished job |
| `RESEARCH_EXTRACTION_MAP.md` | needs audit — may describe a completed migration |
| `MULTI_EVENT_BATCHING.md` | needs audit — oldest file here |
| `TODO.md` | needs audit |
| `TESTING.md`, `DECISIONS.md`, `COEFF_CORPUS_DESIGN.md`, `NOISE_BANDS.md` | current |
| runbook, science record, architecture | **absent** |

### Known defects blocking handoff

* helix declares no pimm-data dependency, though `helix.data` imports it.
* `pimm-fm/uv.lock` pins pimm-data at a rev predating both the coeff family and
  `fd70064` ("split on event IDENTITY, not position") — a stale install produces
  **wrong holdout splits, silently**.
* `legacy-corpus-repro` exists only on this machine; three refusal messages in
  live scripts point at it.
* Nothing is pushed: ~38 commits across two repos.
* The forward model is still duplicated between the repos.

---

## 2. Phases, in dependency order

Each phase states blast radius, verification, and whether it can be undone.

### Phase 1 — Portability (independent, reversible)

Route every external path through `helix/paths.py`. **37 files, 115 lines** —
the first draft said 35/104 and missed `launch/` (2 files, 12 lines: a sbatch
template and a yaml carrying a hardcoded driver venv). `research/` is excluded
and is being retired, not ported.

⚠ **Two path categories the first draft did not name:** a personal driver venv
(`launch/coeff_fm_train.yaml`) and assets under another user's home
(`scripts/optical/*` → `/sdf/home/y/youngsam/...`, 5 files). Neither is
obtainable by a recipient. And three helix checkouts exist side by side
(`helix`, `helix-consolidate`, `helix-extraction`) — `tests/_paths.py` defaults
`HELIX_RESEARCH_ROOT` into a *different* one than we develop in. Decide which is
canonical before the sweep, or the sweep bakes in the ambiguity.

* **Breaks if wrong:** a config resolves to the wrong corpus and a job trains on
  the wrong data — silently. This is the highest-risk *silent* failure in the plan.
* **Verification:** `python -m helix.paths` reports all roots present; both
  suites green; and a build/train/eval dry-run whose resolved config is diffed
  against today's resolved config — **byte-identical resolved paths on S3DF** is
  the acceptance test. Portability that changes what resolves here is a bug.
* **Reversible:** yes, purely mechanical.

### Phase 2 — Packaging and pins (independent, reversible)

Declare helix's pimm-data dependency; relock pimm-fm past `fd70064`; bump
versions; regenerate stale local build artifacts.

⚠ **The pywt story in the first draft was wrong.** It claimed training pulled
pywt in through read-path validation (`coeff_io.py:83 → validate →
derive_band_lengths`), and proposed removing that to make training pywt-free.
Measured inside `pimm-latest.sif` (which has no pywt): `CoeffTPCReader.read_event`
over 19,999 real events with `pywt` never entering `sys.modules`. **Training is
already pywt-free**; that validation belongs to a separate offline codec used by
scripts and tests, not by the training reader. Nothing to remove.

Two real pywt facts to carry instead:
* `helix/core/wavelet_ops_torch.py:26` imports pywt at module scope — latent,
  reached only if `HELIX_BACKEND=torch`.
* `helix/data/coeff_reader.py:397` imports it inside `write_coeff_shard` —
  writer only.

⚠ **The two-container split is correct and should stay.** `develop.sif` (pywt +
torch, no pimm) builds corpora; `pimm-latest.sif` (pimm's locked env, no pywt)
trains on finished coefficients. Two jobs, two environments. The first draft
treated the split as friction to remove and proposed adding pywt to pimm's
lockfile — solving a symptom in someone else's repo.

⚠ `pimm-latest.sif` is **not self-contained**: `/opt/pimm/src` is empty and the
image ships only the locked dependency environment (`uv sync --no-install-project`).
pimm becomes importable only via a bind-mounted checkout. Its provenance IS
confirmed — image labels give revision `972bcd37` = "Release pimm v0.5.1", and
that commit's Dockerfile diffs identical to the working tree's. So the recipe is
real and version-controlled; the image just isn't standalone.

* **Breaks if wrong:** an install that pulls a different pimm-data than the one
  tested.

⚠ **The "silent contaminated holdout" claim was overstated.** Every entry point
in this project runs with `PYTHONPATH` pointing at working trees, which shadow
the image's pinned `pimm_data` 0.3.0. A misconfiguration now surfaces as an
ImportError on `from pimm_data import read_shard_meta, ShardReaderBase`
(`coeff_reader.py:36`) — a crash, not a wrong split. The relock is still correct
hygiene for anyone installing normally; it is not the emergency the first draft
made it.
* **Verification:** fresh install in a clean prefix; `import helix.data` works;
  probe split identity matches a known-good manifest.
* **Reversible:** yes.

### Phase 3 — Boundary completion (LOCKSTEP, hard to undo)

Forward-model kernels stay in `helix.tpc`; registered transform wrappers go in a
new `helix.data.transforms`; pimm-data deletes its copies; the generic keyed
scatter is extracted into pimm-data.

* **Why lockstep:** `pimm_data/_registry.py:171` raises `KeyError` on duplicate
  registration. Old-pimm-data + new-helix double-registers; new-pimm-data +
  old-helix has no `Densify` at all. Both repos must flip in one coordinated pair
  of commits, and no environment may straddle it.
* **Breaks if wrong:** registry-name resolution fails at **config-build time
  inside a batch job**, hours in — not at import.
* **Irreversible part:** once pimm-data's copies are deleted, before/after
  bit-comparison is impossible. **Capture a golden batch first** (fixed seeds,
  through both implementations) as phase 3's first action.
* **Prerequisite:** `helix/tpc/noise_jax.py` and `dense_ops_jax.py` do not exist;
  `pimm_data.dense_ops_jax` has no test coverage anywhere. Write parity tests
  before copying untested code.
* **Also here:** repoint `test_legacy_parity.py:82` and
  `test_corpus_acceptance.py:288` at `helix.tpc.*` **first** — otherwise the
  whole numeric safety net soft-skips at the exact moment its subject changes.

### Phase 4 — Documentation (independent)

Write the four documents; audit and delete or rewrite the misleading ones.

* **Breaks if wrong:** nothing executable. The risk is a document that is
  confidently wrong, which is worse than none — every claim gets checked against
  the code before it ships.
* **Reversible:** yes.

### Phase 5 — Publish (irreversible in practice)

Push helix, pimm-data, and `legacy-corpus-repro`; tag the handoff commit.

* **Irreversible:** history becomes public and other people may branch from it.
* **Precondition:** phases 1–4 complete and both suites green, so the first thing
  the colleague sees is coherent.

---

## 3. Ordering constraints

```
Phase 2 (relock)  ─┐
                   ├─▶ before any further probe/eval numbers
Phase 1            ─┘

Phase 3 requires: golden batch captured, jax parity tests written,
                  legacy-parity tests repointed
Phase 5 requires: 1, 2, 3, 4 all done
Phase 4 can run in parallel with 1 and 2; its architecture doc
        must be written AFTER 3, or it documents a shape that changed
```

Phases 1, 2 and 4 are parallel-safe. Phase 3 is the only one that must be
serialised, and it is the only one that can fail silently in a batch job.

---

## 4. Risk register

| risk | severity | mitigation |
|---|---|---|
| Eval against a corpus the checkpoint did not train on | **high, silent** | assert checkpoint↔corpus `basis_digest` at eval; ingredients exist, nothing joins them |
| Stale pimm-data pin | medium, LOUD (see ⚠ phase 2) | relock as hygiene, not as an emergency |
| Path indirection changes what resolves on S3DF | high, silent | resolved-config diff as the acceptance test |
| Registry name resolution fails in a batch job | high, delayed | lockstep commit pair; grep every `type=` string before flipping |
| Numeric drift across the transform move | medium, permanent | golden batch captured before deletion |
| Parity tests soft-skip at the moment of change | medium, silent | repoint before deleting |
| A doc ships a confidently wrong claim | medium | every claim checked against code; cite file:line |
| Containers are not obtainable off-cluster | **unresolved, see §5** | — |

---

## 4b. The one guard worth building

Nothing ties a checkpoint to the corpus it trained on. Each corpus is internally
consistent, so a reader pointed at the wrong one is satisfied;
`eval_checkpoint.py` validates only that the split name exists. Both halves of
the machinery already exist — `coeff_verify` parses corpus provenance, and
`_bootstrap.provenance()` records helix/pimm-data/pimm commit + dirty flag into
every run. Joining them is one assertion: compare the corpus `basis_digest`
against the value recorded in the run's provenance, and refuse on mismatch.

This is the guard that would have caught the twin-`HELIX_CORPUS` defect as a
crash instead of a plausible number, and it is worth building before the
portability sweep rather than after.

## 5. Open questions this plan cannot answer

1. **Containers — mostly ANSWERED.** `pimm-latest.sif` is built from
   `pimm-fm/.github/docker/Dockerfile`, confirmed by image labels (revision
   `972bcd37`, "Release pimm v0.5.1") whose Dockerfile diffs identical to the
   working tree. So it is reproducible from a version-controlled recipe. Two
   caveats remain: the image is not standalone (needs a bound pimm checkout),
   and `develop.sif` has no recipe found — its provenance is still open.
   Separately: inside `develop.sif`, `pimm_data` is importable only via a
   personal `~/.local` editable install, so a recipient running the identical
   command gets a different module set. Runbook commands must set PYTHONPATH
   explicitly rather than relying on ambient state.
2. **Corpus transfer.** The corpus is ~800 files of derived data; the runs are
   meaningless without it. Rebuilding needs the JAXTPC sensor shards *and* the
   simulator's noise spectrum. This is a data-logistics answer, not a code change.
3. **`research/` — DECIDED: retire it, with two preconditions.** The parity
   tests do not read this tree at all: `tests/_paths.py:28` points
   `HELIX_RESEARCH_ROOT` at a *different, unversioned* checkout
   (`…/omara/helix/research/…`), so if that vanished the tests would skip and
   stay green. Before deleting:
   * `scripts/viz_2x2_corpus.py:317` inserts `research/coherent_coeffs` on
     `sys.path` and imports `induction` under `--removal de2`. Port it, drop the
     flag, or declare it out of scope — it is a live consumer.
   * **Capture goldens first.** `test_training_parity` (training-step parity) and
     `test_legacy_parity` (legacy-DSP parity) are the only witnesses to those
     properties; `test_model_fm`'s frozen golden is committed and survives. Freeze
     what the parity tests assert before removing what they assert it against.
4. **pimm.** It is a fork of someone else's framework. Do we hand over our fork,
   or does the colleague track upstream? The trainer hook for the dense path
   lives here, unwritten.

---

## 6. What this plan deliberately does NOT do

* Does not write pimm's `batch_transform` hook. It is systems-level code in a
  repo whose contribution rules require explicit approval, and the dense path
  runs per-event without it.
* Does not port `research/` until §5.3 is decided.
* Does not touch the science. The masking result stands; this is packaging.
