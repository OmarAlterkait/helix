# Decisions

Standing decisions live here; the measurements behind them are in
`docs/SCIENCE.md`, and the shape of the code they apply to is in
`docs/ARCHITECTURE.md`. Two documents referenced by older entries --
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

`checkpoint.apply_bins` is the caller on the RESUME path, not the sole caller —
`scripts/smoke_train_fm.py:88` calls `model.set_bins(edges)` directly, as do five
test files. That distinction matters: the guarantee here comes from `set_bins`
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
`helix.core.*` / `helix.tpc.*`. `helix.tpc.wavelet` is **not** a shim despite
having been labelled one — it is the `(image, DetectorConfig)` adapter and the
only implementation of that signature.

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
done. There are still TWO launcher scripts (`scripts/submit_coeff_fm_train.sh`
is the production one, `launch/coeff_fm_train.sbatch` the older) and that is one
too many — see RUNBOOK §2. The decisions are:

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
- **One run per array submission** — `MaxArraySize` is 100 here, so the old
  `#SBATCH --array=0-799` was unsubmittable.
- **Short chained links on preemptable QoS**, each resuming iff a complete
  checkpoint exists. Preemptable is priority 1 against normal's 10000, so a long
  job waits rather than runs. Resume needs `weight=` **and** `resume=`; the
  artifact is `model/last`, a *directory*. See
  `memory/pimm-resume-and-slurm.md`.
