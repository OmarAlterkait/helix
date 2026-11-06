# M7 diagnostics: bright charge, faint bins and distribution fidelity of M7dc (vs M6dc / M7d / M6d)

2026-10-12. Post-hoc diagnostics only: no training, no new loss, no new head. The only new inference is bf16 maps of M7d / M7dc on the test events (the eval's own `model_dump2.py`) and an fp32 variant of the same (`bright_dump.py --precision fp32`). Every fitted quantity (products parameters, the Q1b pred-binned correction, the spectrum slope, the Q1c token-phase factors, the model + Tikhonov blend weights) comes from **calibration events**; nothing is tuned on test events. The baseline for method, floors and caveats is `bright_budget.md` (the M6 version of question 1).

**Data.**
- Test: all 128 test events (ev260-387), every plane. Calibration: ev000-063 minus ev041 (63 events). Floors: ev260-307 (48 events, `bright/floor`), plus new corpus-input floors on the calibration events (`m7diag/cal_floor`) for the deployable blend check only.
- Truth: **complete truth Tf** from `bright_build.rebuild_attr` (imported unchanged). It is identical to the corpus complete truth (`denoise_truth_complete`) that the products were fitted against: 0 difference at every cell with Tf > 1 k e- and 100% agreement of the presence label Tf > 25 e- on ev260 and ev300 [V].
- Maps: M6d / M6dc = `dense3/model`, M7d / M7dc = new `products/model_test_m7` (both `model_dump2.py`, bf16). Products (p_cal, calo, clean) applied with `params_{M}.json` exactly as `scripts/denoise_products.py` (its `products()` is called).
- Sanity [V]: the M6 numbers reproduce `bright_budget.md` to the last digit (M6dc gated U −1.35% (0.0425), Y −1.36% (0.0349)), and M6d calo 102-500 e- truth/pred = 1.306 as in `dense/results/products_stored.md`.

**Conventions** as `bright_budget.md`: r = pred / truth − 1 (clipped to ±1); median and hw68 (half the 16-84% spread). "calo" is the products' calorimetry map; for bright cells clean = calo (p_cal ≈ 1). Bright = Tf ≥ 16 k e- per 2-wire × 16-tick cell; tiles with Tf ≥ 64 k e-; particles with E ≥ 1 MeV.

**Sources.** Code: `causes/m7diag_dump.sh`, `m7diag_fp32.sh`, `m7diag_build.py` (records), `m7diag_analyze.py`, `m7diag_extra.py`, `m7diag_calfloor.sh`, `m7diag_post.py`. Numbers: `results/m7_diagnostics.json` (+ `_extra.json`, `_post.json`; all three merged in `m7_diagnostics_all.json`). Intermediates: `/global/cfs/cdirs/m5238/users/oalter/helix_runs/m7diag/`.

Labels: [V] verified by running; [A] assumed / interpretation.

---

## Q1. Bright and top-end charge

### Q1a. Headline per cell (Tf ≥ 16 k e-; 0.43 / 0.43 / 0.49 M cells on U / V / Y) [V]

Median r (hw68), complete truth:

| plane | M6d calo | M6dc calo | M7d calo | **M7dc calo** | M7dc raw = gated | M7dc calo, fp32 inference | hits truth vs complete |
|---|---|---|---|---|---|---|---|
| U | −0.53% (0.0519) | −1.23% (0.0425) | −0.60% (0.0515) | **−0.46% (0.0431)** | −0.58% (0.0431) | −0.02% (0.0419) | −1.27% (0.0092) |
| V | −0.69% (0.0544) | −1.49% (0.0447) | −0.45% (0.0542) | **−0.46% (0.0453)** | −0.59% (0.0453) | −0.09% (0.0443) | −1.27% (0.0092) |
| Y | −0.26% (0.0436) | −1.24% (0.0350) | −0.42% (0.0432) | **−0.20% (0.0355)** | −0.32% (0.0355) | +0.23% (0.0339) | −1.26% (0.0093) |

- The calo map adds +0.12-0.18 pt to the raw / gated charge (its Jensen factor exp(σ²/2) at large y) and changes no spread [V].
- M7dc removes the truth-definition deficit of M6dc (−1.2 to −1.5% → −0.2 to −0.5%) at the same spread: hw68 0.0431 / 0.0453 / 0.0355 vs M6dc 0.0425 / 0.0447 / 0.0350 (+1.4% relative, i.e. M7dc is not more precise per cell) [V].
- fp32 inference moves the M7dc bias by +0.4 pt (to −0.02 / −0.09 / +0.23%) and removes 4-9% of the variance, as it did for M6 [V].

**Regions (tiles with Tf ≥ 64 k e-), calo, median (hw68), all 128 events** [V]; floors from `bright_budget.json` (48 floor events):

| plane | tile | M6d | M6dc | M7d | **M7dc** | M7dc gated | floor: corpus, oracle / best Tikhonov | floor: noisy oracle |
|---|---|---|---|---|---|---|---|---|
| U | 2 × 2 | −0.60% (0.0326) | −1.29% (0.0270) | −0.59% (0.0319) | **−0.42% (0.0271)** | −0.56% | 0.043 / 0.063 | 0.041 |
| U | 4 × 4 | −0.76% (0.0257) | −1.47% (0.0218) | −0.61% (0.0246) | **−0.49% (0.0215)** | −0.69% | 0.039 / 0.100 | 0.036 |
| U | 16 × 8 | −1.01% (0.0222) | −1.89% (0.0221) | −0.72% (0.0204) | **−0.78% (0.0199)** | −1.06% | 0.038 | 0.033 |
| V | 4 × 4 | −0.95% (0.0282) | −1.77% (0.0238) | −0.52% (0.0271) | **−0.56% (0.0232)** | −0.74% | 0.045 / 0.118 | 0.041 |
| V | 16 × 8 | −1.26% (0.0247) | −2.25% (0.0243) | −0.73% (0.0229) | **−0.87% (0.0216)** | −1.12% | 0.041 | 0.036 |
| Y | 2 × 2 | −0.35% (0.0294) | −1.32% (0.0239) | −0.47% (0.0289) | **−0.21% (0.0243)** | −0.35% | 0.027 / 0.019 | 0.023 |
| Y | 4 × 4 | −0.46% (0.0233) | −1.44% (0.0196) | −0.44% (0.0224) | **−0.22% (0.0192)** | −0.41% | 0.022 / 0.021 | 0.019 |
| Y | 16 × 8 | −0.66% (0.0194) | −1.68% (0.0174) | −0.49% (0.0176) | **−0.29% (0.0154)** | −0.65% | 0.020 / 0.023 | 0.015 |

- At region level calo recovers the faint-cell gating loss (gated → calo: +0.2-0.4 pt at 16 × 8) [V].
- On Y, M7dc now beats every corpus-input linear inversion at 4 × 4 (0.0192 vs 0.021-0.022) and 16 × 8 (0.0154 vs 0.020-0.023), and equals the **noisy**-input oracle-support inversion at 16 × 8 (0.0154 vs 0.0151). M6dc was level or slightly worse there [V].

**Particles (E ≥ 1 MeV; ~121 k per plane), calo, share-weighted** [V]:

| plane | M6d | M6dc | M7d | **M7dc** | M7dc fp32 | hits truth vs complete |
|---|---|---|---|---|---|---|
| U | −0.72% (0.0320) | −1.35% (0.0266) | −0.58% (0.0317) | **−0.36% (0.0266)** | +0.07% (0.0256) | −1.47% (0.0079) |
| V | −0.97% (0.0346) | −1.65% (0.0286) | −0.45% (0.0344) | **−0.44% (0.0286)** | −0.05% (0.0278) | −1.47% (0.0079) |
| Y | −0.48% (0.0297) | −1.42% (0.0241) | −0.44% (0.0295) | **−0.14% (0.0242)** | +0.34% (0.0230) | −1.47% (0.0078) |

**The key question: does the Y-plane inefficiency vs the linear Tikhonov inversion of the corpus input persist? Yes** [V].

| Y, bright cells, 48 floor events | hw68 | share of the model's variance above the floor, 1 − hw_floor² / hw_model² |
|---|---|---|
| corpus input, best Tikhonov (λ = 1e-3) | 0.0316 | — |
| noisy input, best Tikhonov | 0.0292 | — |
| M6dc calo | 0.0349 | 18% |
| **M7dc calo** | **0.0356** | **21%** |
| M7dc calo, fp32 | 0.0340 | 14% |
| M6d / M7d calo | 0.0436 / 0.0433 | 47% / 47% |

Where it sits — hw68 per truth bin, same cells [V]:

| plane | bin | n | M7dc calo | corpus Tikhonov | noisy Tikhonov |
|---|---|---|---|---|---|
| Y | 16-64 k | 117,982 | **0.0367** | 0.0402 | 0.0368 |
| Y | 64-256 k | 55,803 | 0.0334 | **0.0175** | 0.0167 |
| Y | ≥ 256 k | 2,107 | 0.0386 | **0.0086** | 0.0085 |
| U | 64-256 k | 55,309 | **0.0377** | 0.0415 | 0.0360 |
| U | ≥ 256 k | 3,565 | 0.0419 | **0.0167** | 0.0157 |
| V | 64-256 k | 55,540 | **0.0395** | 0.0650 | 0.0609 |
| V | ≥ 256 k | 3,692 | 0.0432 | **0.0341** | 0.0319 |

- **The model's relative error does not fall with charge** above 16 k e-: 0.033-0.046 on every plane from 16 k to > 256 k. A noise-limited inversion falls roughly as 1/q (Y: 0.040 → 0.018 → 0.009). On Y the whole inefficiency is above 64 k e-; at 16-64 k the model is better than the corpus inversion [V].
- The model and the inversion are **complementary**: their per-cell errors correlate only 0.21 (Y, corpus). Their plain average has hw68 0.0278 on Y, better than either (0.0356, 0.0316). A blend with plane / charge-bin weights fitted on the calibration floors is in **Post-hoc checks** below [V].

### Q1b. Truth-binned linearity at the top end [V]

Median of pred / truth (hw68, n) per **complete-truth** bin, M7dc calo; saturated cells (clean sensor at the 12-bit limit within ±1 cell, the `bright_build` flag) separately:

| plane | 16-64 k | 64-256 k | 256 k-1 M | 1-2 M | > 2 M |
|---|---|---|---|---|---|
| U unsat | 0.996 (0.047) | 0.995 (0.038) | 0.997 (0.041, 9,648) | **0.985** (0.044, 41) | 0.815 (1 cell) |
| U sat | 0.928 (0.28, 34) | 1.007 (0.16, 64) | 1.002 (0.11, 82) | **0.958** (0.063, 62) | 0.835 (5 cells) |
| V unsat | 0.997 (0.049) | 0.994 (0.039) | 0.993 (0.042, 10,044) | **0.974** (0.055, 96) | 0.867 (6 cells) |
| V sat | — (1 cell) | 1.007 (0.13, 19) | 0.968 (0.090, 28) | **0.951** (0.019, 10) | 0.868 (5 cells) |
| Y unsat | 0.998 (0.037) | 0.998 (0.033) | 0.998 (0.039, 5,829) | **0.968** (0.065, 12) | — |
| Y sat | 0.915 (7 cells) | 0.908 (0.23, 17) | 0.992 (0.088, 35) | **0.955** (0.099, 16) | 0.675 (2 cells) |

The same bins for the other models, unsaturated 1-2 M: M6dc 0.995 / 0.979 / 0.960, M7d 0.966 / 0.951 / 0.958, M6d 0.987 / 0.964 / 0.976 (U / V / Y). For M7dc fp32: 0.985 / 0.968 / 0.994.

Findings:
- **Up to 1 M e- there is no top-end deficit**: 0.993-0.998 in every unsaturated bin (M6dc 0.984-0.995) [V].
- **Above 1 M e- the cells are few, and on U mostly saturated.** In 128 events there are 149 unsaturated + 88 saturated cells at 1-2 M (U: 41 + 62), and 7 + 12 cells above 2 M. The M6 budget's "U 103 cells at 1-2 M" were 60% saturated [V].
- **1-2 M e-: both populations are low.** Unsaturated −1.5 / −2.6 / −3.2% (U / V / Y); saturated −4.2 / −4.9 / −4.5%. Saturation adds about 2-3 pt on top of a deficit that is already present without it [V].
- **Above 2 M e- (19 cells): −13 to −33%, saturated or not**, for every model [V].

Mechanism checks, M7dc:

| candidate | test | result |
|---|---|---|
| hard cap / compression of y | max predicted y vs truth; log-space slopes, unsaturated cells ≥ 256 k | **not a cap.** The predicted y reaches 7.03-7.09 (≈ 2.3 M e-), above every truth y in the 1-2 M bins (max 6.74). Slope of y_pred on y_truth is 0.990-0.997, and of y_truth on y_pred 0.998-1.007 [V] |
| training support | complete-truth y of charged cells on the calibration events (stand-in for the training distribution) | **sparse.** Max 7.30; 99.99th percentile 5.95 (≈ 780 k e-). Only 4.2e-5 of charged cells are > 1 M (113 in 63 events; 7 above 2 M) [V] |
| bf16 step | grid occupancy; fp32 vs bf16 per truth bin | **not the cause.** Every bf16 y above 4 sits on the bf16 grid (step 0.031). The bf16 − fp32 shift in log q is −0.004 to −0.007 in every bin (rms 0.016), flat with charge, and fp32 keeps the 1-2 M deficit (U 0.985, V 0.968) [V] |
| Jensen / calo factor | calo / raw at the top; p_cal | **not the cause.** calo / q = 1.003-1.006 and p_cal ≥ 0.9999 above 1 M, so calo is about 0.4% **above** raw there [V] |
| saturation | sat vs unsat (table) | **adds 2-3 pt at 1-2 M**, but unsaturated cells are already low [V] |
| posterior shrinkage (a calibrated posterior mean is low when binned in truth on a falling spectrum) | predicted truth-binned shift −s σ²: spectrum slope s = −dlnN/dlnT and σ = hw68 of ln(Tf / calo) per predicted bin, both from calibration events | it explains only −0.5 to −0.7% at 1-2 M (s = 3.3-3.4, σ = 0.041-0.044), against −1.5 to −3.2% measured [V numbers; A model]. **Pred-binned** truth / calo is also > 1 at the top: 1.016 / 1.022 / 0.986 at 1-2 M, 1.20-1.23 above 2 M (3-6 cells). So the model is low both ways, which is not shrinkage alone [V] |
| deployable pred-binned correction fitted on calibration events | `bright_analyze.calibrate` on calo, applied to test | removes the global offset (16-256 k: 0.995 → 0.998-1.000). It cannot fix the top: there are only 22-32 calibration cells per plane at 1-2 M (the factors there are extrapolated), and the test medians change by −1 to +2 pt with no consistent direction [V] |

**Mechanism [A, consistent with every check above]:** the model **compresses the last ~0.5 in y (above ≈ 1 M e-)**, where the training support is thin (≈ 2 cells per event above 1 M). This is not a hard cap, not bf16 and not the calo factor. Saturation adds 2-3 pt where it occurs. The deficit is real, but it concerns 0.006-0.03% of bright cells and 0.05-0.06% of their MSE. Saturated cells carry 0.09-0.7% of it [V].

### Q1c. Error correlation structure [V]

Correlation of the relative error r between bright, unsaturated neighbours (M7dc calo; same numbers within ±0.035 for M6dc, M7d and fp32):

| plane | along wire, lag 1 / 2 / 4 / 8 / 16 cells | along time, lag 1 / 2 / 4 / 8 / 16 cells | diagonal (1,1) | 1/e length wire / time |
|---|---|---|---|---|
| U | **−0.055** / 0.028 / 0.036 / **0.051** / **0.043** | **0.183** / 0.144 / 0.076 / 0.054 / 0.023 | −0.066 | 0.60 / 0.77 cell |
| V | **−0.063** / 0.035 / 0.023 / **0.036** / 0.021 | **0.227** / 0.170 / 0.091 / 0.059 / 0.017 | −0.048 | 0.59 / 0.82 cell |
| Y | **0.188** / 0.090 / 0.054 / **0.084** / **0.066** | **−0.049** / 0.011 / 0.019 / 0.028 / 0.028 | 0.002 | 0.78 / 0.60 cell |

- **The per-cell errors are nearly independent.** Every |ρ| at lag 1 is ≤ 0.23, and every correlation length is below one cell.
  - In 2 × 2 tiles with ≥ 3 bright cells, the tile-mean share of var(r) is 30.8%. Independent cells would give 29.6%, so the shared component is about 1% of the variance [V].
- **The structure is anisotropic and plane-specific** [V]:
  - Induction (U / V): anti-correlated across adjacent wire pairs (−0.06), positively correlated along drift time with a slow tail (0.18-0.23 at 16 ticks, still 0.05 at 128 ticks).
  - Collection (Y): the reverse: positive across wires (0.19), anti-correlated between adjacent time cells (−0.05).
  - Interpretation [A]: charge is mis-assigned between neighbours along the poorly resolved axis (anti-correlation), plus a smooth low-frequency component along the other axis. Neither is a shared local gain.
- **A token-grid component** [V]:
  - Along the wire, the correlation has peaks at lags 8, 16 and 24 cells on all planes (Y 0.084 / 0.066 / 0.055 against 0.04 / 0.03 / 0.02 between them). 8 cells is 16 wires, the token width.
  - The median r depends on the cell's position inside its token (fw mod 8). It runs from +0.3% (U) / +0.5% (Y) at position 1 down to −1.0 to −1.1% at positions 6-7 on every plane.
  - hw68 is 3-9% larger at the token's edge cells (positions 0 and 7).
- **The `bright_budget` "positive correlation in 2 × 2 tiles" was partly the global bias.** Its cancellation metric on absolute errors is, for M6dc, −15 / −15 / −28% (U / V / Y), and −12 / −7 / −24% once the median bias is removed. For M7dc it is −4.5 / −7.9 / −24%. Only Y keeps a positive within-tile correlation, the across-wire 0.19 [V].
- **Nothing measurable at deploy time explains the shared part.** η² of the 2 × 2 tile-mean error over deciles: local predicted charge 0.38%; kept-coefficient count in the tile 0.16%, in the 5 × 5 box 0.10%, per unit predicted charge 0.21%; plane 0.10%; time position 0.01%; predicted density (pdens) 0.11%. Truth-side drift 0.02%, dE/dx 0.20%. All Spearman |ρ| ≤ 0.06 [V].
  - For M6dc the largest term was drift (η² 3.4%, ρ = −0.20): the hits-truth deficit that grows with drift. Complete-truth training removed it [V].

**What the residual per-cell spread is** [V numbers; A interpretation]:
- It is not local-gain-like noise: there is no isotropic positive correlation, and no deploy-time predictor.
- It is also not noise-limited above 64 k e-: the relative error stays at 3.3-4.6% while a linear inversion's falls as 1/q (Q1a). The model has a roughly **charge-independent relative-error floor of ~3.5%** per cell. That floor is mostly independent from cell to cell, apart from a weak along-axis smoothness and a token-position bias.

---

## Q2. The faintest predicted bins: decomposition of truth / pred [V]

Exact, binning-robust identity (`m7diag_extra.py`), with ĉ the conditional charge estimate (hurdle: Q0 (e^(y+σ²/2) − 1); non-hurdle: calo / p_cal) and "charged" meaning Tf > 25 e- (the M7 presence label):

**truth / pred = [Σ 1[ch] ĉ / Σ p_cal ĉ] (presence) × [Σ_ch Tf / Σ_ch ĉ] (conditional charge on truly charged cells) × [1 + Σ_(0 < Tf ≤ 25) Tf / Σ_ch Tf] (sub-25 e- charge)**

| model | calo bin | plane | truth / pred | presence | conditional charge | sub-25 e- | frac charged / mean p_cal / mean raw p |
|---|---|---|---|---|---|---|---|
| **M7dc** | 102-500 | all | **1.128** | **1.118** | 1.004 | 1.004 | 0.495 / 0.462 / 0.635 |
| M7dc | 102-500 | U | 1.106 | 1.088 | 1.013 | 1.004 | 0.449 / 0.423 / 0.590 |
| M7dc | 102-500 | V | 1.187 | 1.156 | 1.022 | 1.004 | 0.421 / 0.385 / 0.570 |
| M7dc | 102-500 | Y | 1.074 | 1.105 | 0.968 | 1.004 | 0.692 / 0.653 / 0.816 |
| **M7dc** | 500-1 k | all | **1.038** | **1.052** | 0.986 | 1.000 | 0.894 / 0.865 / 0.931 |
| M7dc | 500-1 k | U / V / Y | 1.035 / 1.079 / 1.005 | 1.035 / 1.078 / 1.048 | 1.000 / 1.001 / 0.958 | 1.000 | |
| M7d | 102-500 | all | 1.148 | 1.082 | **1.056** | 1.004 | 0.485 / 0.461 / 0.630 |
| M7d | 500-1 k | all | 1.059 | 1.043 | 1.015 | 1.000 | |
| M6dc | 102-500 | all | 1.327 | 1.277 | 1.035 | 1.003 | 0.512 / 0.414 / 0.605 |
| M6dc | 500-1 k | all | 1.186 | 1.090 | 1.088 | 1.000 | |
| M6d | 102-500 | all | 1.306 | 1.226 | 1.061 | 1.003 | 0.502 / 0.417 / 0.604 |

**By the products' pdens stratum (M7dc, 102-500 e-), truth / pred:**

| pdens stratum | truth / pred | n cells |
|---|---|---|
| 10 k-100 k | 1.231 | 74,821 |
| 100 k-1 M | 1.203 | 474,212 |
| 1 M-4 M | 1.101 | 884,441 |
| > 4 M | 1.073 | 404,578 |

Unweighted presence factor over the same strata: 1.117, 1.097, 1.062 and 1.045. The deficit is largest in the near-activity strata.

**Reliability of p_cal inside the M7dc 102-500 bin:**

| mean p_cal | 0.16 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.98 |
|---|---|---|---|---|---|---|---|---|---|
| fraction charged | 0.22 | 0.28 | 0.39 | 0.49 | 0.59 | 0.69 | 0.78 | 0.87 | 0.99 |

p_cal is under-confident by 0.03-0.05 across the whole range, not only at one end.

**The same on calibration events** (M7dc 102-500: 1.123 = 1.117 × 1.001 × 1.004; M6dc 1.326 = 1.276 × 1.035 × 1.003). This is a property of the products, not a test-set fluctuation [V].

Which assumption of `denoise_products` breaks there:
1. **The prior correction, for every model** [V].
   - In the faint bin, the raw presence over-states the charged fraction (0.635 vs 0.495) and the prior-corrected one under-states it (0.462).
   - One logit offset per (plane, pdens) stratum is right on average over the stratum's covered cells, as fitted. It is wrong conditionally, for the borderline cells that make up the faint predicted bins.
   - This is 90% of the M7dc deficit (1.118 of 1.128). Mechanism [A]: the correction assumes pure label-dependent (case-control) sampling. The training rows also draw "near" and "coverage" cells by position relative to charge (`DenoiseEvents.rows`), so the sampling depends on features as well. A constant odds offset then cannot be exact, and the error is largest where near-sampling dominates (pdens 10 k-1 M).
2. **The log-normal Jensen factor holds for the hurdle head, and breaks for the non-hurdle head** [V].
   - M7dc on truly charged cells in the bin: mean log residual −0.002, variance 0.029. The Jensen factor applied is 1.079 and the one needed is 1.084, so the conditional charge is right to 0.4% (U +1.3%, V +2.2%, Y −3.2%).
   - M7d's conditional estimate qJ / p is 5.6% low on truly charged cells. Its y regresses the zero-inflated mixture (all cells), and dividing by p in linear space is not the conditional mean of a mixture in log space.
3. **The support threshold is not the cause** [V]. Charge on 0 < Tf ≤ 25 e- cells adds 0.4% (102-500) and 0.04% (500-1 k), though 21% of the cells in the bin hold such charge.
4. M6 additionally carries the label mismatch [V]. Their presence was trained and calibrated on hits truth: own-label fraction 0.449 vs complete q > 25 fraction 0.512 for M6dc. That is why their presence factor is 1.23-1.28 against 1.08-1.12 for M7.

---

## Q3. Distribution fidelity

### Q3a. Law of total variance (all token-covered cells; per predicted calo bin) [V]

var(T) = var(C) + var(T − C) + 2 cov(C, T − C); a calibrated linear-space posterior mean has cov ≈ 0.

| map | cells, overall: var C / var T, var res / var T, 2 cov / var T (corr) | within-bin corr(C, res), < 256 k bins | ≥ 256 k bin | particles: var C / var T, corr(C, res) |
|---|---|---|---|---|
| **M7dc calo** | 0.989, 0.003, +0.008 (+0.07) | \|ρ\| ≤ 0.027 | +0.20 | 0.997, +0.10 (U 0.14, V 0.16, Y −0.01) |
| M7d calo | 0.979, 0.005, +0.016 (+0.12) | ≤ 0.038 | +0.22 | 0.993, +0.22 |
| M6dc calo | 0.979, 0.003, +0.018 (+0.16) | ≤ 0.062 | +0.14 | 0.979, **+0.59** |
| M6d calo | — | — | — | 0.995, +0.14 |

- **M7dc calo is a near-calibrated conditional mean in linear space.** Within every predicted bin below 256 k e-, its residual is uncorrelated with the prediction [V].
- The variance it "loses" overall is 1.1% of var(Tf), because var(Tf) is dominated by bright cells where calo is a measurement. Within the faint bins the narrowing is large by construction: var(calo) / var(Tf) is 0.045 at 102-1 k e-, 0.13 at 1-2 k and 0.59 at 4-8 k [V].
- The positive residual-prediction correlation above 256 k e- is the Q1b top-end compression [V].
- M6dc's +0.59 at particle level is its −1.4% scale deficit [V].

### Q3b. Per-cell charge distributions, cells above 1 k e- [V]

Each map is taken on its own cells ≥ 1 k e-. Wasserstein-1 is in log10 charge; quantiles are of log10 q.

| plane | map | n / n(Tf ≥ 1 k) | W1 (log10) | std log10 | 5% / 16% / 50% / 84% / 99% |
|---|---|---|---|---|---|
| U | **Tf** | 1 (955 k) | — | 0.705 | 3.07 / 3.22 / 4.03 / 4.83 / 5.41 |
| U | M7dc raw (conditional) | 1.034 | 0.050 | 0.730 | 3.03 / 3.14 / 3.96 / 4.82 / 5.41 |
| U | M7dc gated | 0.932 | 0.050 | 0.700 | 3.07 / 3.24 / 4.16 / 4.85 / 5.42 |
| U | **M7dc calo** | **0.867** | 0.124 | 0.670 | 3.11 / 3.36 / 4.27 / 4.87 / 5.44 |
| U | M7dc clean | 0.865 | 0.127 | 0.669 | 3.11 / 3.37 / 4.28 / 4.87 / 5.44 |
| V | M7dc calo | 0.839 | 0.154 | 0.663 (Tf 0.707) | 3.12 / 3.40 / 4.31 / 4.88 / 5.44 |
| Y | M7dc calo | 0.945 | 0.052 | 0.665 (Tf 0.682) | 3.10 / 3.30 / 4.20 / 4.81 / 5.34 (Tf 3.07 / 3.24 / 4.11 / 4.80 / 5.34) |
| U / V / Y | M6dc calo | 0.848 / 0.820 / 0.924 | 0.138 / 0.167 / 0.063 | | |

- The posterior-mean narrowing is a **faint-end** effect.
  - calo has 13% (U), 16% (V) and 5.5% (Y) fewer cells above 1 k e- than the truth. p_cal < 1 pulls uncertain 1-3 k e- cells below 1 k.
  - Above the 84th percentile (≈ 70 k e-), the calo and truth quantiles agree to 0.01-0.04 dex [V].
- The hurdle model's raw (conditional) charge reproduces the truth histogram best: W1 0.050 / 0.058 / 0.017 dex. It over-counts by 3-4%, because it is conditional on presence [V].

### Q3c. dE/dx-like distributions [V]

The sample is single-particle MIP-like charge:
- cells with majority share ≥ 0.9;
- tracks with median dE/dx 1.5-2.5 MeV/cm, Bragg fraction < 5%, length ≥ 10 cm, angle to the drift axis < 60°, overlap < 20% and no saturation;
- per track, **2-wire column sums** (the per-wire dQ/dx measurement), track end columns dropped;
- every column normalised by the track's median truth column.

| plane | columns / tracks | truth hw68 / p95 / p99 | **M7dc calo** hw68 / p95 / p99 | width ratio | extra width in quadrature | per-column calo/truth hw68 | per-track 60% truncated mean, calo/truth: median (hw68) |
|---|---|---|---|---|---|---|---|
| U | 3,534 / 106 | 0.052 / 1.234 / 2.04 | 0.058 / 1.233 / 2.04 | 1.13 | 0.027 | 0.027 | 0.993 (0.011) |
| V | 3,677 / 79 | 0.047 / 1.203 / 1.68 | 0.058 / 1.207 / 1.64 | 1.22 | 0.033 | 0.028 | 0.995 (0.011) |
| Y | 6,250 / 124 | 0.038 / 1.185 / 1.74 | 0.046 / 1.184 / 1.76 | 1.22 | 0.027 | 0.024 | 0.996 (0.009) |

- **In the dE/dx regime calo is not narrower than the truth: it is 13-22% wider.**
  - At ≥ 5 k e- per column calo behaves as a measurement with a 2.4-2.8% per-column error. That error adds in quadrature to the intrinsic (Landau plus diffusion) spread of 3.8-5.2%.
  - The tails (95th / 99th percentile) agree within the sample noise [V].
- For comparison, M6dc truncated means are 0.983-0.986, the hits-truth label deficit. M7d and M6d are 0.992-0.993 with hw68 0.011-0.016 [V].
- Per cell, normalised by the track's median cell, the spread (hw68 0.62) is dominated by how the track crosses the 2 × 16 cells. Truth and calo agree to 0.5% (0.619 vs 0.616) and carry no information on the narrowing [V].
- **For a spectrum** [A interpretation]:
  - A truncated-mean dE/dx from M7dc calo is unbiased to 0.4-0.7%, with a 1% per-track scatter. That is negligible against the 32-36% track-to-track spread (hw68) of the cell-based truncated mean in this sample.
  - Per-column likelihoods (Landau templates) should be built from calo, not from truth: the 2.5-3% per-column measurement error broadens the core by 13-22%.
  - The posterior-mean narrowing matters only below ~4 k e- per cell, i.e. for faint (sub-MIP) deposits and the low edge of charge histograms, not for MIP or Bragg dE/dx.

---

## Post-hoc checks (deployable; fitted on calibration events, scored on test) [V]

Two deployable checks, every parameter fitted on the 63 calibration events (`m7diag_post.py`; floors for the calibration events computed with the unchanged `floor.py --inputs corpus`; their truth equals the records' Tf exactly).

**1. Token-phase correction.** One factor per plane and per position inside the 16-wire token (fw mod 8), from the median truth / calo over predicted-bright calibration cells. The calibration factors reproduce the test pattern: U 0.996-1.010, Y 0.994-1.011.

| plane | M7dc calo before: median (hw68) | after |
|---|---|---|
| U | −0.46% (0.0431) | −0.10% (0.0430) |
| V | −0.46% (0.0453) | −0.14% (0.0452) |
| Y | −0.20% (0.0355) | −0.08% (0.0352) |

- It removes the phase-dependent median, and with it most of the global bias. Variance changes by −0.6% to +0.5%.
- So the token-position bias is a **bias, not a spread term** [V]. It is the same for M6dc: the medians go to −0.06 to −0.12% with the variance unchanged.

**2. Model + linear-inversion blend.** Per plane and predicted-charge bin: calo_b = w · calo + (1 − w) · Tikhonov(corpus input).
- λ is chosen on the calibration floors: 1e-3 on every plane.
- w is chosen on the calibration events (16-64 k / 64-256 k / ≥ 256 k predicted): U 0.9 / 0.6 / 0.2, V 0.95 / 0.8 / 0.35, Y 0.55 / 0.1 / 0.0.
- It is scored on the 48 test floor events, bright cells, hw68 (median):

| plane | M7dc calo | corpus Tikhonov | **blend** | blend by truth bin 16-64 k / 64-256 k / ≥ 256 k (model) |
|---|---|---|---|---|
| U | 0.0429 (−0.43%) | 0.0772 | **0.0394** (−0.55%) | 0.045 / 0.032 / 0.025 (0.046 / 0.038 / 0.042) |
| V | 0.0458 (−0.44%) | 0.1112 | **0.0440** (−0.75%) | 0.050 / 0.037 / 0.029 (0.050 / 0.040 / 0.043) |
| Y | 0.0356 (−0.18%) | 0.0316 | **0.0263** (−0.52%) | 0.031 / 0.017 / 0.009 (0.037 / 0.033 / 0.039) |

- On Y a calibration-fitted blend cuts the per-cell hw68 by 26% (variance −45%). Above 64 k e- it reaches the linear-inversion floor (0.017, 0.009).
- On U / V it gains 4-8% overall, and 15-40% above 64 k e-.
- The blend costs 0.1-0.3 pt of bias. The Tikhonov map carries its own small scale offset, which was not calibrated here.
- **This is the measurable proof that the high-charge error is model-limited**: the corpus input carries information the model does not use [V]. It is not a proposed product: it needs the exact-kernel inversion at deploy time, and the weights were fitted on only 63 events.


---

## What this implies

| finding | size | class | evidence |
|---|---|---|---|
| Truth-definition bias (M6) | −1.2 to −1.5% per bright cell / particle | **fixed by the M7 label change** | M7dc −0.2 to −0.5% (cells), −0.1 to −0.4% (particles); the drift-dependent shared error of M6dc (η² 3.4%) is gone [V] |
| bf16 inference | −0.4 pt bias, 4-9% of the cell variance | **fixable post hoc** (fp32 inference; no retraining) | M7dc fp32: −0.02 / −0.09 / +0.23% (cells), hw68 −3 to −4% [V] |
| Faint-bin calo deficit (102-500 e-: truth / pred 1.13; 500-1 k: 1.04) | 90% presence calibration | **fixable post hoc** by a conditional presence calibration on calibration events (e.g. per-stratum isotonic / Platt in p instead of one logit offset) [A that it fixes it; the miscalibration is reproduced identically on calibration events, V] | Q2 [V] |
| Non-hurdle conditional charge (M7d: −5.6% on truly charged faint cells) | | the hurdle head (M7dc) already removes it — a **model choice**, not a post-hoc fix | Q2 [V] |
| Top-end compression (≥ 1 M e-: −1.5 to −3% unsaturated, −13 to −33% above 2 M) | 0.006-0.03% of bright cells and 0.05-0.06% of the bright MSE | **needs training changes** (more weight / support above y ≈ 6.5); calibration events have too few such cells for a post-hoc fit | Q1b [V numbers; A mechanism] |
| ADC saturation | +2-3 pt deficit at 1-2 M; 0.02-0.06% of bright cells | information-limited at the input (clipped samples) plus the top-end compression | Q1b [V] |
| Charge-independent relative error floor (~3.5% per cell above 64 k e-) | Y: 21% of the cell variance vs a linear inversion; above 256 k on every plane | **model-limited, not information-limited**: the corpus carries the information (linear inversion reaches 0.9-1.7% on Y / U above 256 k). A calibration-fitted model + Tikhonov blend shows it is recoverable (Y hw68 0.0356 → 0.0263). Properly a **training** item (precision / weighting at high charge) | Q1a, Post-hoc checks [V] |
| Token-position bias (−1.1 to +0.5% across the 16-wire token) and lag-8 correlations | 0.1-0.4 pt of the median; no variance | **fixable post hoc**: a calibration-fitted per-phase factor takes the bright median to −0.08 to −0.14%. Also fixable by training (the head's within-token position encoding) [V size; A cause] | Q1c [V] |
| Residual per-cell spread at 16-64 k e- | hw68 0.037-0.050 | **information-limited** (noise): the model beats every linear inversion of the noisy / corpus input there | Q1a [V] |
| Posterior-mean narrowing | only below ~4 k e- per cell (13-16% fewer calo cells above 1 k on U/V); dE/dx spectra unaffected (calo 13-22% **wider** per column, truncated-mean scale −0.4 to −0.7%) | **inherent to a posterior mean** (information-limited by construction); for histograms of faint charge use the raw conditional charge of the hurdle head, or draws, not calo | Q3 [V] |


## Caveats

- The floors are least-squares or Tikhonov inversions with the exact per-wire kernel (`floor.py`), not information bounds. On test floor events the λ was chosen per plane on those same events (as in `bright_budget`). The blend in **Post-hoc checks** instead picks λ and the weights on the calibration floors.
- The per-cell token phase (fw mod 8) assumes a token spans 16 wires = 8 fine cells, aligned at wire 0 [A]. The lag-8/16/24 peaks are consistent with that.
- Above 1 M e- the statistics are tiny (149 unsaturated cells at 1-2 M, 7 above 2 M). The top-end medians carry ±1-2 pt sampling error [A, from the hw68 / √n].
- The shrinkage estimate assumes a locally power-law spectrum and a Gaussian log error [A].
- The saturation flag maps clipped sensor samples to cells with the tokenizer's offset ± 1 cell, as in `bright_budget` [A].
- Particle attribution, directions and the dE/dx sample use the complete-truth shares and PCA directions of `bright_build` [A for curved tracks / showers].
- M6 test maps come from `dense3` and M7 maps from the new `model_test_m7`. Both use the same `model_dump2.py` and the same token-covered cell set [V: M6 numbers reproduce `bright_budget`].

## Reproduce

```bash
H=baselines/uboone_sp/causes; R=/global/cfs/cdirs/m5238/users/oalter/helix_runs
# inference (one 4-GPU node; srun --jobid=<J> -N1 -n1 --overlap -c 120 --gpus-per-node=4 ...)
bash $H/m7diag_dump.sh          # M7d / M7dc bf16 maps, ev260-387 -> $R/products/model_test_m7
bash $H/m7diag_fp32.sh          # M7d / M7dc fp32 maps, ev260-387 -> $R/m7diag/fp32
bash $H/m7diag_calfloor.sh      # corpus-input floors on the calibration events -> $R/m7diag/cal_floor
# records (CPU)
JAX_PLATFORMS=cpu HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE scripts/helix_run.sh python $H/m7diag_build.py --events 0-63 --skip 41 --cal --workers 16 --out $R/m7diag/cal_cells
JAX_PLATFORMS=cpu HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE scripts/helix_run.sh python $H/m7diag_build.py --events 260-387 --workers 28 --out $R/m7diag/cells
# analysis
HELIX_INTERACTIVE=0 scripts/helix_run.sh python $H/m7diag_analyze.py --out $H/results/m7_diagnostics.json
HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE scripts/helix_run.sh python $H/m7diag_extra.py --out $H/results/m7_diagnostics_extra.json
HELIX_INTERACTIVE=0 scripts/helix_run.sh python $H/m7diag_post.py --out $H/results/m7_diagnostics_post.json
```
