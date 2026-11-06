# Is the classical MicroBooNE (Wire-Cell) benchmark a valid comparison?

Adversarial review of `baselines/uboone_sp/`, 2026-10-08. Earlier results (`ladder.md`,
`near.md`, `roi*.md`) are kept unchanged. The new code is in `v2/`. The new numbers are in
`v2/results/benchmark.json`, `v2/checks.json` and `v2/results/aspects.json`.

Labels:
- **[V]** verified: I ran it, or I read the code or paper line that is cited.
- **[A]** assumed: not checked.

## Verdict

**The old headline was a straw man.** The MicroBooNE default scored 0.116, and the
"without the fake cut" row scored 0.141. That benchmark was faithful to Wire-Cell, but it
was not a valid comparison, for three separate reasons:

1. **Parameters tuned for a MIP-scale detector.** The absolute fake-signal cut is commented
   in the C++ as "electrons, about 1/2 of MIP per tick" (`ROI_refinement.cxx:1294`) [V]. It
   kills every 0.1-0.2 MeV ROI on V and Y. The ROI thresholds were never tuned for this
   regime either.
2. **One pooled threshold over three planes with different noise scales.** U's large
   fake-ROI charges set the pooled 1% threshold, and that hides Y. The same fake-cut-free
   chain scores 0.141 with one pooled threshold. It scores 0.338 with a 1% threshold per
   plane [V].
3. **The window statistic was fixed to ROI-masked charge.** On the induction planes the best
   classical statistic is a significance map without ROIs.

**What changed.** I tuned the chain on 96 non-test events (ev000-095, their own noise),
froze it, and applied it to the 128 test events. The near-activity efficiency at 0.1-0.2 MeV
rises from 0.116 to **0.516 [0.502-0.533]**. A per-plane 1% threshold is used for every
method. With the eval's one pooled threshold the number is 0.514 for the all-significance
variant.

**What still holds.** The model M4n (presence) scores **0.579 [0.566-0.597]** with per-plane
thresholds and 0.558 [0.549-0.577] with the eval's pooled threshold. So the model still
wins, by ~0.06, not by ~0.44:
- The 64-label model (le3_ft_N64, 0.429) is **below** the tuned classical chain.
- On the collection plane the tuned classical chain is **ahead** near activity (Y 0.92 vs
  0.90).
- The model's whole margin is on the induction planes: U 0.50 vs 0.37, V 0.38 vs 0.30.
- **Far from activity** the model's margin stays large: 0.85 vs 0.50 pooled, 0.83 / 0.78 vs
  0.36 / 0.25 on U / V. Part of the model's far-threshold efficiency is the near-activity
  haze it is credited for, and the far noise windows favour the classical chain. Both are
  explained under "Threats found in this review".

**What remains unvalidated.** Wire-Cell itself could not be run (no installation here), so
the port has been validated step by step, not end to end.

*Correction from the follow-up:* the sim's exact response is available (public JAXTPC, production commit 1ea2282) and reproduces the stored sensor. The fitted kernel matches the exact effective kernel to 1-4%, and the benchmark with the exact kernel is unchanged (0.507 vs 0.516). See `exact_kernel.md`.

## Corrected benchmark (test events ev260-387, all 128, event bootstrap 200x, 16-84%)

The metric is the eval's 1%-FPR window efficiency, where a window scores the max over its
2x16 cells:
- "near" sets the threshold on the 7,680 near-activity charge-free windows.
- "far" sets it on the 7,680 far charge-free windows.
- "pp" uses a per-plane threshold, so every plane runs at 1% FPR.
- "pooled" uses one threshold over all planes, exactly as `eval_denoise` does. It is
  meaningful only when the planes' scores share a scale.

| method | tuned on | near pp 0.1-0.2 MeV | near pooled (eval) | far pp | far pooled (eval) | near U / V / Y (0.1-0.2) | near <0.1 / 0.2-0.5 (pp) | near SNR 5-7 / 7-10 / 10-15 (pp) |
|---|---|---|---|---|---|---|---|---|
| classical, MicroBooNE default (faithful port) | - | 0.081 [0.067-0.095] | 0.116 [0.106-0.129] | 0.081 | 0.096 | 0.21 / 0.02 / 0.01 | 0.02 / 0.41 | 0.04 / 0.16 / 0.26 |
| classical, default without the fake cut (old 0.141 row) | - | 0.338 [0.319-0.355] | 0.141 [0.112-0.157] | 0.348 | 0.101 | 0.21 / 0.02 / 0.85 | 0.07 / 0.84 | 0.08 / 0.43 / 0.86 |
| classical, tuned Wire-Cell ROI charge (`tuned_q`) | ev000-095 | 0.436 [0.421-0.453] | 0.438 [0.428-0.452] | 0.428 | 0.441 | 0.33 / 0.11 / 0.92 | 0.09 / 0.90 | 0.19 / 0.65 / 0.97 |
| **classical, tuned best statistic (`tuned_any`)** | ev000-095 | **0.516 [0.502-0.533]** | (0.296: mixed scales) | 0.499 | - | **0.37 / 0.30 / 0.92** | 0.09 / 0.96 | 0.30 / 0.80 / 0.98 |
| classical, tuned, significance only (`tuned_noroi`) | ev000-095 | 0.515 [0.500-0.533] | **0.514 [0.502-0.534]** | 0.494 | 0.493 | 0.37 / 0.30 / 0.92 | 0.09 / 0.96 | 0.30 / 0.79 / 0.98 |
| model, 64 labelled events (le3_ft_N64), presence | 64 labels | 0.429 [0.414-0.443] | 0.428 [0.415-0.445] | 0.633 | 0.624 | 0.28 / 0.23 / 0.82 | 0.08 / 0.94 | 0.14 / 0.61 / 0.95 |
| model, 64 labelled events, charge | 64 labels | 0.260 [0.227-0.297] | 0.278 [0.237-0.311] | 0.650 | 0.651 | 0.24 / 0.17 / 0.38 | 0.01 / 0.93 | 0.08 / 0.39 / 0.72 |
| **model M4n, presence** | ~150k labels | **0.579 [0.566-0.597]** | **0.558 [0.549-0.577]** | 0.851 | 0.827 | **0.50 / 0.38 / 0.90** | 0.07 / 0.98 | 0.34 / 0.82 / 0.98 |
| model M4n, charge | ~150k labels | 0.385 [0.317-0.431] | 0.420 [0.377-0.473] | 0.811 | 0.796 | 0.48 / 0.31 / 0.37 | 0.02 / 0.97 | 0.28 / 0.53 / 0.74 |
| (reference) Gaus_wide charge, oracle ROI from truth | truth | 0.999 | 0.999 | 0.999 | 0.999 | - | 0.99 / 1.00 | not a detector |

How the rows were picked (on the tuning split only; picks frozen on test):
- **`tuned_q`**: U `q_s1.25_f0.25`, V `q_s1.5_f0`, Y `q_s0.67_f0`. `sX` = thresholds × X; `fY` = fake-signal cut × Y; f0 = cut off.
- **`tuned_any`**: U `zlf06`, V `zor06a2`, Y `q_s0.67_f0`.
- **`tuned_noroi`**: U `zlf06`, V `zor06a2`, Y `zlf` (on Y every LF variant equals the plain Wiener significance).

Ties on the tuning objective are broken by name. Y's `q_s0.67_f0` and `q_s0.83_f0` tie
exactly (J = 0.9337); on test they give 0.924 vs 0.928.

The statistics in those picks:
- **`zlf06`**: Wiener_tight × LF(τ = 0.06 MHz), divided by the per-wire RMS, cell max.
- **`zor06a2`**: the max of the tighter-LF (0.1 MHz) and LF-0.06 significances, after averaging the cell's two wires.
- **`zlf04`**: as `zlf06`, with τ = 0.04 MHz.
- **`zlf`**: as `zlf06`, with τ = 0.02 MHz, Wire-Cell's ROI_tight_lf. On Y it is the plain Wiener_tight / RMS.

Per-plane efficiencies with all energy bins, the SNR bins and the far-threshold columns are
in `v2/results/benchmark.json` (`test.<row>.boot`). The eval-standard keys (`std`) are there
for every charge row and the model rows. Figure: `figs/validity_summary.png`.

How the classical chain and the model compare on the remaining keys:
- **SNR bins (near threshold, pooled per-plane decisions, clean deposits).** The tuned
  classical chain matches M4n bin by bin: 0.30 vs 0.34 at 5-7, 0.80 vs 0.82 at 7-10, 0.98 vs
  0.98 at 10-15.
- **The model's per-plane advantage near activity** comes from deposits with foreign charge
  nearby, which the SNR bins exclude.
- **Far threshold.** The model is far ahead at every SNR below 10: 0.66 vs 0.30 at 5-7.
- **Charge-free deposits.** The model scores 0.31 at the far threshold on deposits whose
  clean sensor holds no signal. That is the haze (see `near.md`), not sensitivity.

## Threats, tests, outcomes, fixes

### 1. Faithfulness of the port to Wire-Cell

**Threat.** The port (`sp/wirecell_sp.py`) had never been run against Wire-Cell.

**Could I run Wire-Cell?** No [V]:
- `command -v wire-cell` returns nothing.
- `module spider wire-cell` / `wirecell` find nothing.
- The only Wire-Cell-related shifter images (`hepcce2/larwire-kokkos`, `zhihuadong/larwire-sycl`) have no `wire-cell` executable.

Even with a binary, running OmnibusSigProc on our sim would need our 2D response converted
into a Wire-Cell field-response file plus a trivial electronics response. I did not attempt
that conversion.

**Step-level checks I could do** [V unless noted]:

- **Filters.** I fetched `HfFilter.cxx`, `LfFilter.cxx` and `Response.cxx` at the same commit
  (b7bd2a1a) into `reference/wirecell_extra/`.
  - `hf_filter = exp(-0.5 (f/σ)^p)`, zero at f = 0 when `flag` is set; `lf_filter = 1 - exp(-(f/τ)^2)` (`Response.cxx:435-444`).
  - Frequencies are `i/N · 2 · max_freq`, folded above `max_freq` (`use_negative_freqs` defaults to true).
  - The time filters therefore equal `|rfftfreq(N, 0.5 µs)|` in MHz. The wire filters (`max_freq = 1`) equal `2|fftfreq(N_w)|`.
  - The port uses exactly these, with the MicroBooNE `sp-filters.jsonnet` values.
- **Units of the fake-signal cut.** It is in electrons per tick. The code comment reads "electrons, about 1/2 of MIP per tick" (`ROI_refinement.cxx:1293-1294`).
- **Tight ROI formation.** `find_ROI_by_decon_itself` (`ROI_formation.cxx:407-549`) was compared line by line with `find_tight`.
  - Same thresholds (signed `th·rms + 1`; the tight-LF RMS is used for both the tight and tighter outputs) and the same `cal_RMS`.
  - One harmless difference: C++ jumps one sample past a run's end, and that sample is below threshold by construction.
- **Regression against the earlier port.** The refactored v2 runner (`v2/classical.py`: ROI chain run once per threshold scale, fake cut applied to copies) reproduces the earlier sweep exactly on ev300 (max |diff| = 0). It was checked for `q_s1_f0` = `wc_nofake`, the oracle-ROI charge, and the Y default `wc_s1`. On U, `q_s1_f1` differs from `wc_s1` as intended: v2 applies the fake cut on U, while master's U branch never removes anything, so the U default is `q_s1_f0`.
- **Point source (paper I §5.2.1).** A 10k e- point charge was put through our fitted kernel and the port, with no noise (`v2/checks.json`).
  - The closest wire gets 0.77 (U/V) and 0.945 (Y) of the charge. The paper says ">60% … if the point source … is close to the wire". Consistent.
  - Time smearing is 2.83 ticks for a delta and 3.45 ticks with 2 ticks of diffusion. The paper quotes 2.7 / 2.3 ticks including ~2 ticks of 1 m diffusion. **Not reproduced.**
  - Master's charge filter `Gaus_wide` (σ_f = 0.1114 MHz) alone gives σ_t = 1/(2πσ_f) = 2.86 ticks. The paper's numbers therefore imply a narrower charge filter at the time [A].
  - This affects charge smearing within a 16-tick cell, not detection, and no detection statistic chosen here uses `Gaus_wide`.
- **Deconvolved noise level vs the paper.** Our per-wire ROI RMS is 48 / 69 / 34 e-/tick (U / V / Y). At the paper's factors (3.5 induction, 5 collection) the thresholds would be 167 / 241 / 168 e-/tick. Paper I quotes "about 350 / 500 / 300 e-/tick" (`extraction.tex:561-563`).
  - Ours are ~0.5× MicroBooNE's with the same plane ratios (V/U 1.44 vs 1.43). That is consistent with a lower noise-to-gain ratio in our sim [A].
  - The sentence is ambiguous: it could quote the RMS, not the threshold.
- **Not reproduced.** The paper's line-charge bias / resolution / inefficiency vs θxz (Fig. 26) exists only as figures. I did not digitise it.

**Outcome.** The filter layer, the thresholds and the units are verified. The ROI-refinement
logic (break/shrink/check/cleanup) remains the previous agent's port with its file:line
citations. I spot-checked it but did not re-audit it line by line [A]. That logic is
irrelevant to the tuned headline: its picks on U/V are ROI-free significance maps, and its Y
pick is a tight-ROI chain with no refinement beyond the 3σ re-check.

### 2. Detector-appropriate tuning

**Threat.** MicroBooNE's parameters were tuned for MicroBooNE's response, noise and MIP-scale
physics. The earlier "best setting" rows were chosen on the test events.

**Test** (`v2/classical.py`, `v2/classical_extra.py`, `v2/score.py`).
- **Tuning split.** truth_v2 ev000-095: probe-train events, never scored. Each has its own
  noise realisation (seed 1000 + ev): 4,857 iso, 5,760 far and 5,760 near noise windows, ~400
  iso per plane at 0.1-0.2 MeV.
- **Grid, per plane type.** Planes are independent in Wire-Cell, and MicroBooNE configures
  them per plane.
  - ROI threshold scale (`th_factor_ind` 3, `th_factor_col` 5, loose `l_factor` 3.5, all × s): s ∈ {0.67, 0.83, 1, 1.25, 1.5} on induction, {0.4, 0.5, 0.67, 0.83, 1, 1.25, 1.5} on collection.
  - × fake-signal cut scale f ∈ {0, 0.25, 0.5, 1}.
  - × statistic: ROI charge or ROI-gated significance.
  - Plus 19 ROI-free statistics from the Wire-Cell filter family (threat 3).
  - 59 candidates on induction, 75 on collection.
- **Objective.** J = mean(far, near) per-plane 1%-FPR efficiency at 0.1-0.2 MeV. The near-only
  and wide-energy objectives were also run.
- **Freeze, then apply to all 128 test events.**

**Outcome** [V].
- **Selection is robust.** All three objectives pick the same U/V statistics and the same Y
  family. Their test efficiencies agree to within 0.003 (near pp 0.514-0.517).
- **Tuning-to-test shrinkage is small.** On the tuning split `tuned_any` scores near pp 0.537
  / far 0.517. On test it scores 0.516 / 0.499.
- **The fake-signal cut is what kills the default.** Every top pick has f = 0 or 0.25
  (≤ 125/250 e-).
- **Y prefers a lower collection threshold:** 0.67 × 5σ.

**Fix.** Report the tuned chain as the classical number. Keep the default as the faithful
reference row.

**Caveat on how this search went.** The ROI grid and round-1 statistics were chosen before
looking at any test number. After seeing the preliminary **tuning-split** table I added
round-2 statistics: 2/4-wire averaging and LF 0.06 combinations, because V gained from both
on the tuning split. I also looked at a preliminary test table of the ROI-grid picks before
round 2. No pick was changed on the basis of a test number.

### 3. Score function

**Threat.** The eval scores a window by its max cell charge. That is natural for the model
but not necessarily best for a classical chain.

**Test.** I gave the classical chain every window statistic that is a monotone cell map,
selected on tuning:
- ROI charge;
- ROI-gated significance;
- Gaus_wide cell-sum significance, 1×1 and 3×3 cells, with or without an LF cut;
- Wiener_tight / RMS;
- Wiener × LF(τ) / RMS for τ ∈ {0.02, 0.04, 0.06, 0.1} MHz, their OR, and 2- and 4-wire averages.

**Outcome** [V].
- **Induction.** The ROI-free LF-filtered Wiener significance beats every ROI statistic: V
  0.30 vs 0.11 for the best ROI charge (test, near). The Wire-Cell ROI machinery adds no
  detection power at 0.1-0.2 MeV.
- **Gaus_wide cell sums are poor on induction** (≤ 0.04): its near-DC wander dominates (cell
  std 1,660 / 3,200 e- on U / V).
- **Collection.** ROI charge at 0.67 × threshold with no fake cut (0.924) ≈ Wiener
  significance (0.922).

**Fix.** The `tuned_any` / `tuned_noroi` rows.

### 4. Alignment

**Test.**
- **Noise-free decon vs truth.** Port Gaus_wide output of the stored clean sensor, 4 tuning
  events, peak pixel cross-correlation lag vs the hits truth: |Δt| < 0.01 tick on every
  plane, Δw = 0. The best fine-cell lag is (0, 0) on all planes (`v2/checks.json`).
- **Signed centroid offsets** (prediction - truth) on test iso windows ≥ 0.2 MeV, for the
  charge outputs, U / V / Y:
  - `tuned_q`: +0.01±0.02 / +0.02±0.06 / +0.03±0.03 wires and +0.05±0.09 / -0.04±0.29 / +0.02±0.14 ticks.
  - The model charge heads are equally centred: |Δw| ≤ 0.06, |Δt| ≤ 0.4.
  - The significance statistics show -0.2 to -0.3 wire on U/V. That is the weighting of a
    significance map, not a charge offset, and it is irrelevant for detection.

**Outcome.** No systematic offset [V]. No correction was needed.

### 5. Response kernel

**Superseded by `exact_kernel.md` (follow-up, same day).** The sim's exact response IS available.
- **Source.** The simulator is public (JAXTPC). The data's production commit 1ea2282 was exported read-only.
- **Perfect knowledge** [V]. JAXTPC's own functions reproduce the stored clean sensor of ev260-263 from the step files to 0.6-0.9%. That is the floor set by the step files' 0.3 mm position quantisation; total charge agrees to 0.9998.
- **The exact kernel equals the fit** [V]. The exact effective kernel for per-wire hits truth (least squares over the deposit population, through JAXTPC's code) differs from our fitted kernel by 1.3 / 3.8 / 1.9% (U / V / Y). Its held-out forward-fold residual is the same: exact 0.030 / 0.031 / 0.021 vs fitted 0.027 / 0.035 / 0.017.
- **No gain from drift dependence** [V]. A drift-dependent family given oracle drift does not improve on that (0.031 / 0.031 / 0.020).
- **Benchmark unchanged** [V]. The tuned classical chain, re-run end to end with the exact kernel, gives 0.507 [0.494-0.528] with the frozen picks, vs 0.516.
- **The example kernel is explained** [V]. It is the same JAXTPC kernel averaged over the pitch, without JAXTPC's linear sub-tick interpolation (a half-tick shift) and without diffusion.

The original text of this section follows. Its "not available" statement is wrong.

**Tests on the least-squares kernel** (`response/fit_*.npz`) [V]:
- **Support.** Energy in the outermost wire columns is 0.008% / 0.005% / 0.002% (U / V / Y).
  The first and last 10 ticks hold < 0.003%. The 21 × 211 support is large enough.
- **Drift dependence.** The held-out residual ||clean - store(hits ⊛ R)|| / ||clean|| is
  flat over the readout, in thirds of the drift window, using a zero-padded linear forward
  model:
  - U 0.019 / 0.016 / 0.021;
  - V 0.034 / 0.031 / 0.031;
  - Y 0.021 / 0.020 / 0.014.
  There is no drift-time dependence. A first, cyclic version showed 0.27 on the last third
  of U; that was the wrap of the induction precursor, not physics.
- **Sub-wire position dependence** remains the 2-4% residual the earlier fit reported. A
  translation-invariant per-wire kernel cannot express it.
- **Example kernel.** The same frozen picks were run with the example's `avg_response`
  instead of the fit; see the variants table at the end of §5.

**Kernel and information variants.** All are test events, 0.1-0.2 MeV, near-threshold pp
efficiency, with picks frozen from the gate-input tuning unless stated
(`v2/results/benchmark.json`, rows `<family>@exk` and `<family>@corpus`):

| classical input / kernel | near pp 0.1-0.2 | U / V / Y | far pp |
|---|---|---|---|
| gated waveform, fitted kernel (`tuned_noroi`) | 0.515 [0.500-0.533] | 0.37 / 0.30 / 0.92 | 0.494 |
| gated waveform, **example kernel** (`tuned_noroi@exk`) | 0.499 [0.482-0.515] | 0.35 / 0.27 / 0.92 | 0.478 |
| MicroBooNE default, example kernel (`default@exk`) | 0.059 | 0.14 / 0.03 / 0.01 | 0.052 |
| tuned ROI charge, example kernel (`tuned_q@exk`; earlier scoring run in which Y was ×0.83) | 0.398 | 0.24 / 0.09 / 0.92 | 0.387 |
| **corpus-like input** (gate → κ=1 threshold → no D1), re-tuned on ev000-047 of it (`corpus_retuned@corpus`) | 0.307 [0.296-0.320] | 0.06 / 0.06 / 0.86 | 0.313 |
| corpus-like input, gate-input picks (`tuned_noroi@corpus`) | 0.108 | 0.07 / 0.06 / 0.20 | 0.100 |

[V] What the variants show:
- **Kernel errors cost the tuned detector 0.02** (induction only). They cost the ROI-charge
  chain 0.04, mostly on U (0.33 → 0.24). The example kernel is 10-24% off the fit in norm.
- **The classical chain degrades much more than the model on the model's information.**
  - The re-tuned classical detector drops from 0.515 to 0.307 on a corpus-like input. On
    induction it collapses (0.06), because the per-wire RMS normalisation of its
    significance statistics is computed on mostly-zero waveforms.
  - The selected statistics were not designed for sparse input, and I did not re-engineer
    them. Treat 0.307 as a lower bound for a classical chain on that input [A].
  - On Y the plain Gaus_wide cell charge still works (0.86).
  - The model reads that input and gets 0.579.

### 6. Induction-plane processing beyond 2D decon + ROIs

**Threat.** MicroBooNE "later used additional induction processing (e.g. L1SP)."

**Checked** [V]:
- **L1SP** in MicroBooNE's configuration (`cfg/pgrapher/experiment/uboone/sp.jsonnet:50-110`)
  is a follow-on filter applied ONLY to channels 3566-4305, "subject to the shorted wire
  region fields". It fits unipolar + bipolar responses where shorted wires distort the
  field. It is not a general induction improvement, and our sim has no shorted regions.
  Implementing it would not help.
- **Multi-plane protection** (MP2/MP3, `OmnibusSigProc.h:273`) is off for MicroBooNE.
- **MicroBooNE's later induction ROI improvement** was DNN-based ROI finding, i.e. learned
  [A, from memory: Yu et al., JINST 16 P01036 (2021)].
- **The model reads all three planes in one token set**, so it can use cross-plane context.
  The classical chain, as MicroBooNE configures it, cannot. That is an unquantified
  advantage of the model [A].

**Quantified instead** [V]:
- **Weak planes.** The tuned classical chain is weakest on V (0.30 near) and U (0.37). A
  known-location ideal matched filter reaches 0.85 / 0.74 / 0.98 at 0.1-0.2 MeV
  (`docs/DENOISE.md`) [A, not re-run].
- **Oracle-ROI rungs set no ceiling.** Truth ROIs make every noise window score 0, so they
  measure charge survival, not detection. The relevant ceiling is the known-location bound
  above.
- **Gap to the model.** The model is 0.13 (U) and 0.08 (V) ahead near activity. The ideal
  bound is ~0.4-0.5 above both.

### 7. Label and information parity

- **Labels.**
  - The classical chain uses truth only through the fitted kernel (8 events of
    (hits, clean) pairs) and now through tuning: 96 events, about 60 parameters selected
    by grid search.
  - The 64-label model **loses** to the tuned classical chain near activity: 0.429 vs 0.516
    pp, and 0.428 vs 0.514 pooled-eval. That holds on every plane: U 0.28 vs 0.37, V 0.23 vs
    0.30, Y 0.82 vs 0.92.
  - It wins far from activity: 0.63 vs 0.50.
  - With ~150k labels the model wins near activity, by 0.06.
- **Information.**
  - The model reads the sparsified corpus: per-band hard threshold 3.35-3.9σ, D1 dropped. The
    classical chain reads full waveforms.
  - `docs/DENOISE.md`'s stage replay [V, doc]: a known-location matched filter on the corpus
    input keeps 0.19 / 0.48 / 0.86 at SNR 3-5 / 5-7 / 7-10, against 0.91 / 0.99 / 1.00 on
    the gated waveform.
  - Here I re-ran the classical detector on a corpus-like input:
    - the gate, then the corpus threshold (κ = 1, universal per band), then D1 dropped, then
      inverse DWT;
    - ROI-free statistics only, because the ROI chain on sparse input ran ~10× slower;
    - re-tuned on ev000-047 of that input.
    The result is in the variants table at the end of §5: 0.307 vs 0.515 on the full
    waveform.
  - The noise realisation is not the corpus's own: same generator, different draw [V]. Per
    event the model and the classical chain see different noise; the statistics are the same.

### 8. Statistics

- All 128 test events; event bootstrap with 200 resamples (16-84%), as in `eval_denoise` [V].
- Per-plane numbers for every row are in the json.
- Model reproduction [V]. `v2/model_dump.py` replays `eval_denoise`'s prediction on GPU: same
  tokens, `cell_inputs`, bf16 autocast, encoder once per event, chunked head. It reproduces
  **every window score** of `dn_near_{M4n_ft_full_near, le3_ft_N64}[_pres]_trained_windows.json`
  exactly (max |diff| = 0, 26,674 windows).
  - A first version disagreed on 13 windows because I had cropped the cell maps at the readout
    edge. Near-activity windows reach fine ticks -3 to 285, and `eval_denoise` predicts those
    cells. The fixed maps extend past the readout on both sides.

## Threats found in this review (not in the brief)

9. **Pooling across planes** (biggest straw-man effect after the fake cut). One 1%
   threshold over U+V+Y lets the noisiest plane's score scale set it:
   - default without the fake cut: 0.141 pooled vs 0.338 per-plane;
   - `tuned_any` mixes charge and significance: 0.296 pooled vs 0.516 per-plane.
   - The model is nearly scale-consistent across planes: 0.558 vs 0.579.
   Fix: report per-plane-threshold pooling ("pp") for every method. Use the eval's pooled key
   only for single-scale outputs.
10. **Far noise windows are drawn at corpus-kept coefficients** (`event_truth`: bg windows
    are centred on `coeff_cells` from the shard; `scripts/dump_resolution_truth.py:73-78`)
    [V].
    - Those are cells where the corpus's own noise survived the threshold, so the model's far
      threshold is set on windows selected for high noise in its own input.
    - For the classical chain, with an independent noise draw, they are effectively random
      windows.
    - This biases the far metric in the classical chain's favour [V mechanism; size not
      measured].
    - The near-activity windows are placed from truth only and carry no such bias, which
      makes the near threshold the fair headline. That agrees with DENOISE.md's choice.
11. **Edge cells.** The model predicts cells past the readout edges (fine ticks < 0 or > 270),
    and some near windows include them. The classical chain outputs 0 there. The effect is
    small: 13 windows changed when those cells were dropped [V].
12. **The model's far-threshold efficiency is inflated by haze.** On deposits with no clean
    signal at all it scores 0.31 at the far threshold [V], and 15% of its near-activity noise
    windows exceed its far threshold.
    - The classical chain's near_fpr_at_far1pct is 0.01: it has no haze. Its near and far
      thresholds are almost the same.
    - Comparisons at the far threshold therefore overstate the model's advantage.
13. **The oracle-ROI rungs of the old ladder are not detectors** (score 0 on every noise
    window) [V, earlier]. They are kept only as a charge-survival reference.

## What is verified, what is assumed

**Verified [V]:**
- the test split and windows;
- the model reproduction;
- Q0;
- that the selection used only ev000-095;
- the filters and units against the C++;
- the regression of v2 against v1;
- alignment;
- kernel support and the absence of drift dependence;
- L1SP's scope in MicroBooNE's config;
- per-plane and per-bin numbers with bootstrap errors;
- the far-window placement mechanism.

**Assumed [A]:**
- the ROI-refinement port beyond spot checks;
- that the 2018 paper used a narrower charge filter;
- the ideal-bound and stage-replay numbers quoted from `docs/DENOISE.md` (not re-run);
- that DNN-ROI was MicroBooNE's later induction step;
- that cross-plane context helps the model;
- the size of the far-window bias.

**Not validated:**
- end-to-end Wire-Cell execution;
- the sim's exact response (now verified; see `exact_kernel.md`).

## Reproduce (compute node, container)

```bash
H=baselines/uboone_sp/v2; S=/pscratch/sd/o/oalter/helix_work/uboone_sp/v2
scripts/helix_run.sh python $H/checks.py --events 0-3 --out $H/checks.json
scripts/helix_run.sh python $H/classical.py --events 0-95,260-387 --workers 108 --out $S/cls
scripts/helix_run.sh python $H/classical_extra.py --events 0-95,260-387 --out $S/cls            # round 1
scripts/helix_run.sh python $H/classical_extra.py --round 2 --events 0-95,260-387 --out $S/cls  # round 2
CUDA_VISIBLE_DEVICES=k scripts/helix_run.sh python $H/model_dump.py --events A-B --out $S/model \
    --ckpt M4n=/pscratch/sd/o/oalter/helix_work/denoise/M4n_ft_full_near/final.pt \
    --ckpt N64=/pscratch/sd/o/oalter/helix_work/denoise/le3_ft_N64/best.pt      # 4 GPUs, ranges of 0-95 and 260-387
# kernel / information variants (test, plus ev000-047 for re-tuning on the corpus-like input)
scripts/helix_run.sh python $H/classical.py --events 260-387 --kernel example --settings s1_f0,s1_f1,s1.25_f0.25,s1.25_f0,s1.5_f0,s0.83_f0 --out $S/cls_exk
scripts/helix_run.sh python $H/classical.py --events 0-47,260-387 --input corpus --settings NONE --out $S/cls_corpus2
#   + classical_extra.py rounds 1 and 2 with the same --kernel / --input on those dirs
scripts/helix_run.sh python $H/score.py --cls $S/cls --model $S/model --out $H/results \
    --extra exk=$S/cls_exk,corpus=$S/cls_corpus2 --corpus-tune $S/cls_corpus2   # = the final benchmark.json
scripts/helix_run.sh python $H/aspects.py --cls-full $S/cls_full --cls $S/cls --model $S/model --bench $H/results/benchmark.json --out $H/results
#   cls_full = classical.py --full ... on 260-387 (induction: --planes 0,1,3,4, Y: --planes 2,5; see aspects.md)
scripts/helix_run.sh python $H/pairs.py --out $H/results/pairs.json && scripts/helix_run.sh python $H/plot_aspects.py
python $H/tables.py      # markdown tables
```
