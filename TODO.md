# Deferred work

Standing decisions (what was chosen and the measurement behind it) are in
`DECISIONS.md`. This file is what is NOT done.


Things consciously left undone, with enough context to pick up cold. Ordered by
what blocks what, not by size.

Larger records get their own file: `MULTI_EVENT_BATCHING.md` (why the FM is one
event per forward), `NOISE_BANDS.md` (what the noise model does to each band, and
why m113 is out-of-distribution here).

**Running anything.** ONE container now — see `CLAUDE.md` and
`docs/ARCHITECTURE.md` §4. The two-container instructions that were here named
`develop.sif` and `pimm-latest.sif` and were both wrong: one had no pimm-data at
all, the other baked a pimm-data predating the boundary move.

GPU accounts: `neutrino:default@ampere` carries ONLY the `preemptable` QOS, and
preemption here CANCELS rather than requeues, so anything longer than the
preemption window cannot finish on it. Use an account that has `normal`
(`neutrino:cider-nu`, `mli:cider-ml`, `mli:nu-ml-dev`) and pass `--qos=normal`;
`sacctmgr -n show assoc user=$USER format=Account,Partition,QOS` lists yours.

---

## 1. Retire the research bundle — DONE (from the REPO)

`git ls-files research` returns nothing: the 13-module bundle and the DSP
golden that `research/goldens/capture.py` anchored are out of the tree
(f825057). The directory still exists on the machine helix was developed on,
gitignored, which is why the citations below still resolve for one reader and
nobody else.

What is NOT done: ~15 docstrings across `helix/model/mask.py`, `tokenize.py`,
`helix/tpc/coherent_gate*.py` and four scripts cite `research/...` paths for
provenance. They are historical references, not imports — nothing breaks — but
they point at a directory a handover target does not have. Quote the fact
inline, or drop the citation.

## 3. Probes and eval — the 3D probe is DONE

`scripts/run_probe.py` + `scripts/dump_probe_truth.py` are the two stages, and
`scripts/export_artifact.py` promotes a run into a scoreable, attributable
artifact. Measured on the handover validation run: trained +0.8403 against
random +0.1161, geo +0.1514, raw +0.0616.

Still missing: charge closure and per-band variance explained as EVALUATORS
inside a run, rather than as scripts after it. pimm has
`EventProbeSuiteEvaluator` and `hooks/eval/pretrain/probes/` to build them into
— and the `eval-contract` branch of pimm-private is 26 commits of exactly that
groundwork, unpushed at the time of writing.

## 4. A real pretraining run — DONE

112,677 steps via a chain of links on the 8-run corpus; eval `var_expl 0.6844`
against production's 0.7030. What is still open is the COOLDOWN: the chain
exhausted at step 6,417 of 14,264 with all three links preempted, on an account
whose only QOS is preemptable (see above).

## 5. Corpus scale — DONE

All 8 runs built: 790 shards, 344 GB, ~158k events (train 150,239 / val 4,641 /
probe 3,111). The bins were NOT re-derived — see DECISIONS.md for why, and for
the yardstick that would overturn it.

## 6. `--backend jax` still imports pimm-data

`scripts/build_coeff_corpus.py` imports `pimm_data` on EVERY backend (the
reader/dataset layer is pimm-data's), so the jax path is not the special case
this entry used to claim. `tests/test_forward_mirror.py`, cited here as the pin,
was deleted with the forward model. torch is the production backend
(`--backend torch`, measured 10.6 ms), so jax remains an override.

## 7. Housekeeping

* ~~`tests/test_coeff_dataset.py` pins the cross-repo golden to
  `helix-consolidate`~~ — FIXED: it reads `HELIX_ROOT`, defaulting to
  a git worktree of it.
* ~~Version split: `pyproject.toml` 0.1.0 vs `helix/__init__.py` 0.2.0~~ — FIXED (0362328): pyproject reads the module via `[tool.setuptools.dynamic]`.
* ~~The 10 back-compat flat shims~~ — REMOVED, along with the 6 import-broken
  scripts that were their only non-test users.

## 8. Courtesy report to pimm's author

`engines/train.py::run_step` does
`if "offset" in input_dict: input_dict["coord"].shape[0]` — a batch with an
offset and no coord raises KeyError AFTER the forward. The sibling accounting at
line ~454 is guarded; this is not. One line. We do not hit it (`CoeffCollect`
emits no offset, asserted by a test), but it affects any non-point-cloud model.
