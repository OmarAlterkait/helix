# EXECUTION PLAN — operator-pinning phase (M0–M5)

The concrete, executable version of `reviews/review_synthesis.md`. Supersedes
the test battery in `mechanism_tests.md`. Code lives in
`research/coeff_foundation_model/` (promoted to a helix package once the
operator is pinned).

## Settled inputs (user decisions, 2026-06-11)

1. **Clean coefficients first.** No forward optical noise model this phase.
   Support on doraemon = production threshold with a **nominal σ = 2.6 ADC**
   (t_j = 1.2·σ_nom·√(2 ln N_j), A kept): "clean-support" convention —
   the signal coefficients that *would* survive production thresholding.
   light_output (real noise) remains the noisy-statistics reference; noise
   modeling returns when the operator phase ends.
2. **Unification only.** No paired charge+light work; no pairing
   verification. Fusion deferred.
3. **Primary decision metric = equal-weight per-band asinh-MSE** (mean over
   bands of per-band MSE in asinh(c/σ_b) space, bands weighted equally),
   computed on held-out events. Secondary diagnostics reported, never gating
   alone: signal-space L2, per-band tables, survivor-weighted MSE, probe.
4. **research/ scripts; execution starts now** with the no-training items.

## Global experiment protocol (applies to every trained arm)

- Splits by **event** (never chunk/coefficient). 3 event folds; ≥3 seeds for
  the ceiling and for any arm within 2ε of a decision boundary.
- ε := 1·σ_ceiling (std of the ceiling's primary metric across seeds × folds).
- Per-arm 3-point LR sweep {0.3×, 1×, 3×} around a common base, best-of;
  identical step budget; loss-plateau check before any comparison.
- Stratified read-outs always reported: per-band; orphaned vs parented
  coefficients; core-load percentile (top 1% / p95 / rest).
- Secondary criterion: frozen-feature linear probe on truth (`pe_counts`;
  TPC: `tpc_de` when applicable). An arm within ε on the primary is NOT
  adopted if its probe deficit exceeds δ := 1·σ_ceiling(probe).
- Quantized-input column (12-bit survivors) on the adopted arm only.
- Every decision logged in `DECISIONS.md` with the numbers that made it.

---

## M0 — doraemon optical loader + clean-support statistics  [CRITICAL PATH]

Data: `/sdf/data/neutrino/doraemon/optical_test_00_00_02/sensor/` — 100 files
× 200 events, label_N schema (NOT readable by helix.optical.io), noise-free,
164 ch, SER 5 µs, pedestal 29490.3, per-interaction truth.

- **T0.1 loader** `doraemon_optical.py`: iterate `event_NNN/label_K` groups →
  per-chunk float32 (pedestal-subtracted), `pmt_id`, `t0_ns`, `label`,
  per-label truth (`pe_counts`, `tpc_de`, ...). Resolve and document:
  pmt_id range/global-vs-side encoding; whether same-PMT chunks from
  different labels overlap in time (per-label splitting); chunk↔truth
  association granularity.
- **T0.2 clean-support stats** `measure_coeffs_doraemon.py` (reuses
  `measure_coeffs_optical.Acc`/`process_chunk` with σ_nom): per-band
  survival/counts, tree lift + Markov + shift sweep, values, coarse columns,
  per-event totals; dump a typical-event npz (the M2/M3 substrate).
- **T0.3 truth-anchored noise statement**: identify whether zero-PE chunks
  exist in noise-free stitching (likely not — no noise → no noise-only
  stitches); restate the "D1 signal-bearing" claim with truth instead of the
  contaminated |x|max>50 class.
- **Gate G0:** loader round-trips; clean-support per-band table produced;
  differences vs light_output table understood as (noise removal + SER 5 vs
  10 µs), not bugs.

## M1 — cost microbench (no training)

`microbench_treeops.py`, on `artifacts/typical_event_coeffs_smart.npz` (TPC),
`artifacts/typical_event_coeffs_optical.npz`, + a doraemon batch.

- **T1.1** index-map builders for the packed coefficient format: parent /
  ordered-children / ancestor-chain / dilated-Δ maps via per-band dense int32
  grids. Time map construction and gather→MLP→scatter per topology
  {C0, C1, C2, C3} at d_s ∈ {32, 64, 128}.
- **T1.2** per-band **dense vs gather** conv timings (optical bands at their
  true per-band occupancies 36%→0.05%) → gates the dense-conv choice
  (skeptic S6: 62.5% was cell-union, coefficient-level is ~2.7%).
- **T1.3** batched-stem amortization: 24 separate vs band-type-batched calls
  (TPC), per-band vs batched (optical).
- **Gate G1:** ms + bytes per event per topology and per conv mode; retires
  handoff opens #1/#2; provides the cost column for M3's decision rule.

## M2 — data-only information audit, in bits (no training)

`info_audit.py` → `artifacts/info_audit.json` + registered predictions.

- **T2.1 activity-CMI, exact** (contingency tables): I(child; ancestor_Δ |
  parent[, grandparent]) per band, both corpora, **within-chunk stratified**
  (kills the Simpson/envelope inflation). **Controls** (skeptic S2): gain
  from same-band neighbor at equal physical distance; gain from shifted
  non-parent at same level. Tree edges are hard-coded only if they beat both.
  Report bits/slot AND bits/event (population-mass weighted).
- **T2.2 value-CMI**: I(asinh value_child ; ancestor values | parent value)
  via Gaussian-copula + coarse-bin plug-in cross-check; conditioning sets ≤3;
  censoring handled by reporting active-ancestor subpopulation vs full
  population separately (the difference IS the censoring effect).
- **T2.3 orphan census**: per band, P(parent inactive | child active) and
  availability of the nearest active ancestor; defines the orphan stratum
  used in M3.
- **T2.4 alignment closure**: analytic coif3 per-level group delay δ_ℓ baked
  into the index maps; value-level R²(|child| ~ |parent|) vs shift, optical
  + **TPC** (never swept); extend coarse-level sweeps past the turnover.
  Then alignment closes for good.
- **T2.5 optical anchor sweep**: occupancy/fan-out/saturation at cell =
  2^10/2^9/2^8 on doraemon clean support; attach the trunk-cost column
  (flat-attention feasibility per anchor) so the anchor decision internalizes
  its trunk consequence.
- **T2.6 bit-count predictions, registered before M3**: source bits/event;
  token-stage capacity ratio; d_s arithmetic (p95 fan-out × bits ÷ 8) →
  predicted d_s knee per anchor; predicted N-flat/d-knee behavior of the
  bottleneck. M3/M5 then *verify* these one-point predictions.
- **Gate G2:** edge-reach justified in bits/event (or not) per Δ; anchor
  level chosen; d_s sized; predictions on file. Statistical hygiene: event
  bootstrap CIs everywhere; average precision (not AUC) for fine bands.

## M3 — REVISED (2026-06-11): short-run large-effect screening

The original 9-run × 3000-step star was stopped (one ceiling reference banked:
AE primary 3.49 @3000 steps). Pre-full-optimization tests only need to detect
LARGE effects — anything inside short-run noise is deferred to the actual
scaling phase, not measured with seeds and folds. Revised protocol, two
instruments:

- **Test A — routing (operator question), no bottleneck confound:**
  masked-coefficient prediction (30% masked, learned mask token, predict
  asinh values; per-band masked MSE + variance baseline + orphan strata).
  Fast-converging, closest proxy to the eventual MAE objective, directly
  tests the value-context propagation that M2 identified as the tree op's
  only defensible role. Arms: ceiling / rounds0 / c0w7 / c0w0, 500 steps.
- **Test B — bottleneck (P1/P2 verification):** the production-geometry AE,
  one arm, d_tok ∈ {256, 64}, 500 steps — does the knee move as the bit
  count predicts.
- **Decision filter:** act only on differences ≳10–20% of the metric;
  smaller → "indistinguishable pre-optimization, defer."

Original star spec kept below for reference (arms/strata definitions).

## M3 (original spec) — the star design (optical chunks, clean targets)

Substrate `star_model.py`: point transformer over a chunk's active
coefficients; inputs (asinh value, physical-time PE, log-scale PE, band
embedding); **ceiling** = full attention, per-level parameters. Every arm =
same substrate + attention masks + weight ties:

| arm | topology | weights |
|---|---|---|
| ceiling | full attention | per-level (generously tuned; envelope with C2 arm) |
| C2+W7 | hierarchical up + ancestor-chain attention down | per-level |
| C0+W7 | adjacent V-cycle (1 and 2 rounds) | per-level |
| C0+W2 | adjacent | shared + FiLM(scale) |
| C0+W0 | adjacent | fully shared |
| A-arm | adopted topology | A-band: root-partner vs plain-level vs excluded |

Bottleneck pinned to production geometry: anchor from T2.5, d_s from T2.6
(±2× bracket), learned pooling, two-head decode (occupancy + values),
skips off through the bottleneck.

- **Decision rule (registered):** adopt the cheapest (W, C) cell whose
  primary metric is within ε of the best cell AND whose probe deficit ≤ δ;
  cost from G1 ms/event. Report the full table; never decide marginally.
- **Contingencies (run only on trigger):** crossed cell W2+C2 if any arm is
  ambiguous; W6 (grouped) bisection iff W2 fails while W7 passes;
  gradient-reach then telegraph iff C0's fine-band or orphan-stratum gap > ε;
  "instantiate inactive interior tree nodes on active cones" variant iff the
  orphan stratum drives the C0–C2 gap.
- **Gate G3:** (W, C, rounds, A-band treatment, d_s) pinned for optical;
  orphan and core strata explain (or don't) the residual gap; bit-count
  predictions confirmed/refuted.

## M4 — transfer evaluations (no new training)

- Zero-shot M3 winner across depth: optical→TPC band range and reverse,
  under (a) physical-scale conditioning vs (b) free per-(modality, level)
  embeddings (the honest baseline — not level-index).
- Held-out-level interpolation (train without level 9, eval on it) and
  held-out-end extrapolation (train without 9–10) — extrapolation is what
  TPC transfer actually requires.
- SER 5 ↔ 10 µs pair (doraemon vs light_output); light_output is noisy →
  restrict to high-SNR coefficients and label the residual confound.
- **Gate G4:** scale-conditioned sharing transfers or it doesn't — the
  "one model" claim gets its number.

## M5 — TPC one-point verification (parallel with M3; production pipeline)

TPC stays on the production basis (noisy + smart removal kgate=4 + per-band
threshold) — machinery exists, measured statistics are on that basis.

- Pre-attention stack at defaults (per-level weights, C0), **rounds 0 vs 1
  control** (the one TPC mechanism question worth a run), tokenizer at 8×4
  with d from T2.6, AE stratified by core saturation; packed-batch masking
  equivalence check (no cross-event attention).
- **Gate G5:** TPC pre-attention validated end-to-end at one point; core
  stratum behaves as the bit-count predicts.

## Sequencing (5 GPUs available; most of M0–M2 is CPU)

```
day 1-2   M0 loader + clean-support stats          [G0]
day 2-3   M1 microbench                            [G1]
day 2-5   M2 info audit (parallel with M1)         [G2 + registered predictions]
week 2    M3 star design (arms parallel on GPUs)   [G3]
week 2-3  M5 TPC verification (parallel with M3)   [G5]
+days     M4 transfer evals (cheap, after M3)      [G4]
then      DECISIONS.md consolidation → promote operator to a helix package →
          trunk/SSL phase planning (Group D returns here)
```

## PHASE 2 (user-set, 2026-06-11): TOKENIZER-COMPLETE → tokenizer-only AE

**No trunk work at all.** Finish every tokenizer variation thoroughly, then
train an autoencoder using ONLY the tokenizer to see performance without
overcomplicating. No downstream probes. **Noisy input from the start**:
TPC = production pipeline (noise → smart removal kgate=4 → per-band κ=1);
optical = white σ=2.6 forward noise → per-chunk db1-MAD σ → κ=1.2 (A kept).

**Corpus reality (measured):** TPC = 29 runs × 100 files × ~200 ev ≈ **580k
events** on disk (not 20k). Optical = 20k doraemon events. Corpus risk at
pilot/mid scale: retired for TPC.

**The headline task: noisy→clean DENOISING autoencoder.** Both corpora have
clean truth, so the AE target = clean coefficient values on the noisy
production support. The meaningful performance number is MSE-to-clean vs the
**classical baseline** (the post-threshold noisy coefficients themselves =
what production keeps today), at the token-budget compression ratio.

**Tokenizer variation matrix** (short runs, large-effect filter, one factor
off a default config = {±1 gathers, K=2 blocks, C0 on, shared+FiLM, d_s=64,
attention-pool, anchor 1024 / 8×4, d_tok=256}):

| axis | variations | prior knowledge |
|---|---|---|
| within-band reach | ±1 / ±2 / ±4 | data-only CMI audit first (only ±1 audited so far) |
| blocks K | 1 / 2 / 3 | — |
| cross-level | off / C0 | measured −6.6% masked (D-08); re-check on noisy denoising task |
| conditioning | FiLM / shared | ~3% (D-08) — spot-check only |
| d_s | 32 / 64 / 128 | cost flat (M1) |
| pooling | attention-pool ×1 query / ×4 queries / linear-sum | mean rejected (design) |
| anchor / patch | 1024 vs 512 (opt); 8×4 vs 16×8 (TPC) | token cost table (compute_budget) |
| d_tok | 64 / 256 | +4% at 500 steps (D-09); decisive only near convergence |
| decoder | slot-MLP vs per-band heads | instrument-side, one check |

**Order:** (1) noisy+clean coefficient datasets, optical then TPC (the TPC
dump = the production GPU pipeline, doubles as M5); (2) reach CMI audit
(data-only); (3) the variation matrix as short denoising-AE runs; (4) one
longer convergence run of the chosen config vs the classical baseline,
core-stratified — the phase deliverable.

## Out of scope this phase (explicitly)

Forward optical noise model; charge+light fusion/pairing; trunk architecture
sweeps (within:across, latent-vs-structured); SSL objective comparisons
(Group D); FM-scale corpus generation plan (flagged to user, unowned);
κ choices (settled).
