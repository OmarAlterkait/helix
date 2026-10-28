# Decisions

Standing decisions live here; the measurements behind the older ones are in
`docs/archive/SCIENCE.md`, and the shape of the code they apply to is in
`docs/ARCHITECTURE.md`. The newest round is first (§0); the sections after it are
older and name the corpora and scripts of their time (r1, the retired launchers);
where a later entry supersedes one, it says so. Two documents referenced by older entries --
`RESEARCH_EXTRACTION_MAP.md` (the research-tree extraction map) and
`CONSOLIDATION_PLAN.md` -- described work that is now finished and were retired
on 2026-09-12; they remain in git history and in
`$HELIX_ARCHIVE/retirement-backups/superseded-docs_2026-09-12/`.

Standing decisions for the coefficient FM, each with the evidence behind it and
what would overturn it. The commit log has the narrative; this is the state.

A decision without a measurement next to it is a preference, and several entries
here started as preferences that the measurement then contradicted. Those are
marked **RETRACTED** rather than deleted — knowing which way a question was
answered wrongly is worth as much as the answer.

---

## 0. One of each (2026-10-24 .. 26)

**One build entry point; sources are scripts.** `helix/build.py` holds the event loop,
the TPC and optical chains, the shard writer, calibration and the task plan, and knows
no source; `scripts/build_sim.py`, `build_sbnd.py`, `build_goop.py` each define one
reader class and end with `helix.build.main(Source)`. *Why:* three builders
(`build_coeff_corpus.py`, `helix.tpc.sbnd_run`, `helix.optical.goop_run`) each carried
their own copy of everything after the read. A new source is now a new script and
nothing else. Shards are written `.part` and renamed, so `--skip-complete` lets a
resubmitted task skip finished work; `scripts/corpus.sh` replaced
`submit_coeff_corpus.sh` and `calibrate_norm_sigma.sh`. Checked output-neutral: rebuilt
simulation and SBND shards were byte-identical to r2 and v1.

**One noise model, torch.** `helix.tpc.noise` builds the simulation corpus and is the
load-time `AddNoise` / `Digitize`. *Why:* the numpy and torch implementations
(`dense_ops.py`, the `coherent_numpy` switch) had to be kept in step by hand. The numpy
port survives only as the test reference `tests/_noise_reference.py`. Output-neutral.

**One coherent gate, on the readout's channel groups.** `helix.tpc.coherent_gate` takes
group labels from the readout: SBND front-end boards across planes, the simulation's
64-wire noise blocks per plane. `coherent.groups` names the grouping (null: none);
`group_size` lives only in the readout. *Why:* the contiguous-block gate and the labeled
gate were two implementations of one operation, and the block gate silently ignored the
`sigc_scope: group` its config recorded (simulation now records `global`, what it always
did). Not output-neutral for simulation: the trailing partial block of a 1969 / 1443-wire
plane is reduced over a padded group (ulp-level float32 changes; run_0027575715 shard 0:
23 of 200 events differ, 2.4e-5 of the kept set). SBND coefficients byte-identical.

**One corpus verifier.** `python -m helix.data.coeff_verify` checks a run, or a whole
corpus against its plan (`_calib/CHUNKS.txt`), and `compare`s shards byte for byte.
*Why:* SBND-data-helix's `verify_corpora.py`, `readback.py` and `compare_shards.py`
duplicated it outside the repo that owns the format.

**A format stamp, and no legacy read paths.** Every coeff shard carries `/config`
`format = "helix-coeff-2"`, and every reader requires it. The fallbacks for shards
without `/ident`, without `coord_digest`, and the flat JAXTPC sensor schema are removed.
*Why:* each fallback was a branch that only old data exercised, and a reader that
accepts anything cannot say what it read. Corpora built before the stamp are migrated in
place (`scripts/migrate_format.py`: metadata only, old attributes backed up); r2 and SBND
v1, the production corpora, were migrated so and verify against their plans; pre-tau `coeff_tpc` and r1 are retired and unreadable.

**Configs state intent; step counts are derived at run time.** `configs/pimm/fm.py` is
the whole run (corpus = `HELIX_CORPUS`'s parent and its `_calib/RUNS.txt`, the m113
architecture, the WSD stable phase); `fm_cooldown`, `fm_smoke` override only
what differs, and anything else is `key=value`. Warmup and eval / save cadence are
`"auto"`, filled by `FMTrainer` from pimm's `iters_per_epoch * epoch` through
`helix.core.cadence`. *Why:* nine configs carried `N_TRAIN_EVENTS` tables and step
counts copied between them; a fix applied to one was shadowed by another. The run's
length now follows from the corpus, the split and the GPU count, which is what it
depends on. `HELIX_LEGACY_CORPUS` and `resolve_cooldown_subsets.py` went with them.

**An sbatch chain instead of `pimm submit`.** `scripts/train.sh CONFIG [key=value ...]`
submits `LINKS` jobs chained `--dependency=afterany`; each runs `python -m pimm.train`
in the site container and resumes from the latest complete checkpoint. *Why:* `pimm
submit` could not run from S3DF (its preflight imports the training stack on the host,
and its job script finds configs only inside pimm's own `configs/`). helix needs pimm's
training engine, not its launch layer, so `configs/launch/`, the launcher venv
(`make_launcher_env.sh`) and the S3DF-only `submit_coeff_fm_train.sh` /
`chain_coeff_fm_train.sh` went; the host needs only PyYAML (`scripts/make_host_env.sh`).
`afterany` continues whether a preempted job is requeued (NERSC) or cancelled (S3DF).
Checked: a 3-link `fm_smoke` chain whose first link timed out resumed with loss and LR
continuous.

**The production bin grid is r2's** (`coeff_bins_r2_run0027575715`, derived from r2's
first run with r1's parameters: 120 events, K = 128). *Why:* r2's inner edges sit a
median 0.35 / 0.46 / 0.64 / 0.98 local bin widths (A4 / D4 / D3 / D2, max 2.0) from r1's,
against 0.16 between r1 runs, so by the yardstick of the freeze decision below the r1
grid does not describe r2. Models trained on r1's grid are not value-comparable with
models trained on this one.

**Sites are profiles; the target is NERSC.** Paths, container and scheduler live in
`helix/sites/<site>.yaml` and reach shell scripts through `scripts/helix_env.sh` and
`scripts/helix_run.sh`, so moving to NERSC changes a profile, not the scripts.

---

## Corpus

**Ship R1 (`gate_tau=0.05`), not k30.** R1 wins on denoising (SSE 0.714,
off-signal 80k→67k), support cleanliness (band-0 zero-clean 4.81%→1.90%), and
ties or wins cross-plane. Both corpora live at
`/sdf/data/neutrino/omara/coeff_tpc{,_r1}`.

**Build all 8 runs; the corpus is 790 shards / 344 GB / ~158k events.** Seven
runs have 100 source files, `run_0027670361` has 90 upstream — complete at 90, not
a build failure. Identity split: **train 150,239 / val 4,641 / probe 3,111**.

**FREEZE the bin grid. Do not re-derive on the pooled corpus.** Bins derived on
run 2 alone vs run 1 alone, both at 480 events so both converged, differ by
**0.129% of a band's span = 0.16 bin widths**, which is *smaller* than the
sampling spread of the 120-event table actually shipped (0.159%). Run-to-run over
sampling = **0.81×**. `cent_ratio` agrees to 0.38% median.

Re-deriving would move the grid less than the noise already in it, while costing
value-CE comparability with every earlier run and invalidating every trained
K=128 head. Ship `coeff_bins_r1_tau05_run0027575715_v2.pt`.

*Since:* the grid was re-derived at the second site, because the corpus there is
not byte-identical to the one v2 fingerprints (same run, same `basis_digest`,
different arrays; cause not established). The DECISION above is unchanged --
bins are not re-derived when the corpus GROWS -- but the shipped name is no
longer a literal. It is declared in `helix/data/data/reference_bins.json` and
read by `helix.data.bins.reference_table()`. A number produced against v3 is not
comparable with one against v2; see that file's `_rerecord_note`.

*Would overturn it:* a run whose bins differ from run 1's by materially more than
the 0.16-bin-width yardstick — i.e. evidence the runs are not one distribution.

*Superseded for r2* (§0): r2 is a different corpus, not a grown one, and its grid
was derived anew.

**The frozen grid costs 1.75% worst case, not 11.5%.** Applying run 1's grid to
run 2's data reclassifies 7.8% of coefficients carrying 11.5% of |charge| — but
reclassification is *not* error: adjacent bins of a K=128 partition have
near-equal centroids. Measured charge error is **1.75%** worst case, ~0.4% by the
centroid comparison. Do not quote 7.8/11.5% as an error.

**RETRACTED — hold out a whole RUN for validation.** Proposed as "a materially
stronger generalisation test"; the measurement above kills it. The runs are the
same distribution, so a held-out run tests nothing extra, and it would couple the
split to run composition. The existing identity hash is better and was verified,
not assumed: run 1's counts are 19,034 / 577 / 388 with one run built and
**identical** with all eight. Growing the corpus reassigns nothing.

---

## Model, metrics, artifacts

**Centroids are never optional.** `set_bins` derives any table it is not given,
so no centroid buffer is ever NaN. They used to be optional, so consumers asked
"are these present?" — three files asked, one forgot, and `var_expl` /
`charge_closure` / `charge_bias` were absent from **every** eval log ever
produced. Centroid args are keyword-only.

`checkpoint.apply_bins` is the caller on the RESUME path (and in
`scripts/smoke_train_fm.py`), not the sole caller — several test files call
`model.set_bins(edges)` directly. That distinction matters: the guarantee here comes from `set_bins`
DERIVING any table it is not given, not from a single choke point upstream of it.
A reader who believed the choke point existed could add a caller and expect the
invariant to hold for free.

**`cent_lin` and `bin_cent_measured` are deleted — both were tables nothing
read.** `cent_lin` also carried a real bias (planes pooled across 22%-different
sigmas). `bin_cent_measured` was added *by this work* two commits after removing
`cent_lin`, committing the same sin. Old checkpoints carrying either still load
(`_STALE_BUFFERS`).

**Validation scores the whole val set.** Every rank evaluates its shard and the
sums all-reduce. Previously rank 0 evaluated alone with a `DistributedSampler`
loader, i.e. **1/world_size** of the set reported as the number — the
`batches=145` in the 4-GPU logs. Costs nothing: the other ranks were blocked at a
barrier anyway.

**`model_best` OFF for the stable phase, ON for the cooldown.** A flat WSD
schedule has no best step — raw val loss is a plateau plus noise whose max is the
luckiest eval, and it would select *raw* weights while everything downstream
reads the EMA. An annealing schedule has a real minimum and its raw weights *are*
the annealed model. Confirmed: the completed 1-run stable phase recorded
`best_metric_value: -inf` after 118,950 steps; the cooldown writes monotonically
improving `Best validation` lines.

**The cooldown is a separate short run, never a tail on the stable phase.** WSD's
stable phase commits to no horizon, so only the cooldown config knows
`total_steps`, which `1 − √p` needs. Warm-start with `weight=`, **never**
`resume=` — resume restores the step counter and would put the run past `p = 1`,
at the floor from step one, which looks fine (loss falls) and anneals nothing.

*Verified live:* `Lr(423)/Lr(1)` observed 0.8349 vs 0.8348 predicted for `1 − √p`
— 0.01%.

**Cooldown length uses a RUN subset, not `max_len`.** `epoch` is an integer and
one epoch over all 8 runs is 33% of the stable phase. Three runs = 14,264 steps =
12.7%, inside the usual 10–20% band. `max_len` would take the first N events of a
run-ordered index and anneal almost entirely on run 1. Sound only because the
runs are one distribution — see above.

---

## Structure

**`helix/integrations/pimm` is a package, not a 1089-line module.** Split by
concern: `_compat` (import-time pimm patches, must import first), `data`,
`model`, `trainer`, `hooks` (what a run *writes*), `eval` (what a run
*measures*). The module path is unchanged, so `imports=["helix.integrations.pimm"]`
and every symbol import still work.

**Eval stays in helix**, against the extraction map's "EVAL is pimm's
job". That line was written when eval meant downstream probes and baselines,
which do belong there. This is the *training* metric and it reaches into helix
internals no framework should know about — `core.raw_heads`, the bin buffers,
`losses_cat`. Moving it inverts the one-way dependency the integration exists to
preserve.

**No top-level shims.** The nine re-export modules are gone; import from
`helix.core.*` / `helix.tpc.*`. (`helix.tpc.wavelet`, once the `(image,
DetectorConfig)` adapter, no longer exists; the TPC chain calls `helix.core.wavelet`
directly.)

**Tests exercise shipped code, not copies.** `bucketize_bins` and
`rank_from_gram` exist because the tests previously pinned hand-copies of logic
that lived elsewhere.

---

## Known-complex, deliberately unchanged

**The ±1e18 edge sentinels are gratuitous** — every consumer slices them off
(`edges[b, 1:-1]`), and `_close_open_edges` exists only to undo them. Not
removed: `edges` is `(n_band, K+1)` in every checkpoint on disk including m113,
the frozen goldens and the trained run. Churning a persisted format for tidiness
would break all three. Delete when the format next changes for a real reason.

**Deriving centroids from edges** is unreachable for anything current (the v2
sidecar measures both; live checkpoints carry both). Kept only for pre-fix blobs
like m113, which has no measured `cent_ratio` and never will. (The research
bundle has since retired; this is the remaining reason to keep the fallback.)

---

## Operational

Details and incantations live in `docs/RUNBOOK.md`; `TODO.md` is what is NOT
done. There is one launcher, `scripts/train.sh` (§0). The decisions are:

- **Corpus builds on `turing` only.** The DSP is architecture-sensitive: turing
  and A100 disagree on 0.016% of surviving coefficients and are not
  bit-comparable. Run 1 was built entirely on turing.
- **Training on `ampere`.** K=128 at full event size makes the logits ~2.4 GB for
  one event before the backward doubles it.
- **Never edit the helix tree mid-build.** `coeff_verify` fails a corpus whose
  shards carry different `code.git`; it caught exactly this and 7 shards were
  rebuilt.
- **`coeff_verify --source-root`** distinguishes an upstream gap from a failed
  job. A gap whose source exists is still an error.
- **Arrays of at most 100 tasks** — `MaxArraySize` is 100 here, so the old
  `#SBATCH --array=0-799` was unsubmittable; `corpus.sh build` splits the plan into
  arrays of `ARRAY_MAX` (100).
- **Chained links**, each resuming iff a complete checkpoint exists. Training
  runs at QOS `normal` (`helix/sites/s3df.yaml`): preemptable is priority 1
  against normal's 10000, so a long job waits rather than runs, and S3DF
  preemption cancels rather than requeues. Resume needs `weight=` **and** `resume=`, which
  `scripts/train.sh` passes from pimm's latest-checkpoint resolver.
