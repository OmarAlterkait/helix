# Handoff plan — making helix + pimm-data transferable

**Goal.** A colleague takes ownership of the whole codebase and continues the
training runs, on infrastructure that is not this cluster.

**Status.** PLAN ONLY. Nothing in phases 1–5 has been executed. One artifact
exists and is wired to nothing: `helix/paths.py` (phase 1's keystone). Delete it
if this plan is rejected.

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

Route every external path through `helix/paths.py`. 35 files, 104 lines.

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

* **Breaks if wrong:** an install that pulls a different pimm-data than the one
  tested. **Do the relock before any further probe numbers** — the current pin
  can silently contaminate a holdout split.
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
| Stale pimm-data pin contaminates holdout splits | **high, silent, active now** | phase 2 first; verify split identity against a manifest |
| Path indirection changes what resolves on S3DF | high, silent | resolved-config diff as the acceptance test |
| Registry name resolution fails in a batch job | high, delayed | lockstep commit pair; grep every `type=` string before flipping |
| Numeric drift across the transform move | medium, permanent | golden batch captured before deletion |
| Parity tests soft-skip at the moment of change | medium, silent | repoint before deleting |
| A doc ships a confidently wrong claim | medium | every claim checked against code; cite file:line |
| Containers are not obtainable off-cluster | **unresolved, see §5** | — |

---

## 5. Open questions this plan cannot answer

1. **Containers.** `develop.sif` and `pimm-latest.sif` are prebuilt images on
   S3DF, one in another user's directory. There is no recipe in either repo. A
   colleague elsewhere cannot obtain them. Genuine portability needs a
   buildable definition, which we do not have and may need to ask for.
2. **Corpus transfer.** The corpus is ~800 files of derived data; the runs are
   meaningless without it. Rebuilding needs the JAXTPC sensor shards *and* the
   simulator's noise spectrum. This is a data-logistics answer, not a code change.
3. **`research/` policy.** 138 files, 277 hardcoded paths, a vendored copy of the
   original research code that parity tests validate against. Options: ship as-is
   and mark historical; port it too; or drop it and lose the parity tests. It is
   the single largest block of non-portable content.
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
