# Causes of the denoiser's failings: hypotheses F, B, C

2026-10-10. The model is M6d (`M6d_ft_full_dec/final.pt`); M6dc (`M6dc_ft_full_dec_cond`) is shown where it differs. Its product is charge gated by presence > 0.5.

All numbers are on the 128 test events (ev260-387, every plane) and the 2-wire × 16-tick fine cells. They use the dense3 per-cell maps (CFS) and the classical tuned maps (`dense/cls_fit`; two planes on OST 61 come from `dense2/cls_fix`).

The density strata are those of `dense/analyze.py`: hits-truth charge in the ±8-cell box, cell excluded; iso = 0, ≤ 10 k = (0, 10 k], and so on.

Sources:
- code: `causes/build.py` (per-cell evidence) and `causes/analyze.py` (all tables);
- per-event records: `/global/cfs/cdirs/m5238/users/oalter/helix_runs/causes/cells/`;
- numbers: `causes/results/causes_FBC.json`.

Labels: [V] verified by running; [A] assumed.

## The machinery [V]

**Complete truth (Tf).** I rebuilt it with JAXTPC 1ea2282's own hits function (`wires.prepare_deposit_with_diffusion`) from every deposit in the step files: attenuated, Gaussian-diffused with σ(t_drift), integer wires × ticks, **no thresholds**.
- The production instead drops every per-group × wire × tick sample ≤ 25 e-.
- Deposit times are not in the step files. They are recovered per deposit group from the hits (as in `v2/resim.py`). A group without hits takes its track's t0, which places 2.0% of the deposit charge.
- 28.5% of the deposit charge belongs to tracks with no hits at all. It has no time and is left out. These are deposits outside the readout window: dropping them reproduced the stored sensor's total to 0.9998 (`exact_kernel.md` §2) [V for the sum, A per deposit].

**Machinery check.** The same rebuild with the production's 25 e- per-group cut (Tthr) reproduces the evaluation's hits truth Th.
- Per-plane cell-level relative L2: median 0.38%, 95th percentile 0.78%.
- Sum Tthr / Th = 1.00008.

**Completeness.** Tf / Th = **1.016** summed over all planes: the hits truth misses 1.6% of the in-window charge.

**Coefficient evidence.**
- **Kmax:** the largest kept corpus coefficient mapped to the cell, in units of the event's per-band noise σ. These are exactly the coefficients the model reads: bands A4-D2, placed with `tick_of_tau`, threshold 3.35-3.9σ.
- **Smax:** the same for the noise-free signal, i.e. the DWT of the stored clean sensor. That is what the corpus threshold acts on, without noise.
- **Kmax3 / Smax3:** their 3 × 3-cell maxima.

## F. Remaining haze in dense regions: is it a truth artefact?

**The failing.** After gating, about 10% of truth-empty cells in > 4 M neighbourhoods still predict > 100 e-: M6d 0.102, M6dc 0.101, classical 0.122. At 100 k-4 M the figure is 1.9-3.1%.

**Competing explanations.**
1. A truth artefact: the cells hold real sub-threshold charge.
2. Real over-prediction: blur or spill of neighbouring bright charge, or a context prior.

**Decisive test.** For every false cell (Th = 0, prediction > 100 e-), does the complete truth hold charge, and how much of the predicted charge does it account for?

| density | M6d false fraction (Th) → vs complete truth | false cells holding Tf ≥ 25 e- / ≥ 100 e- (all empty cells: Tf ≥ 25) | median Tf vs median prediction (e-) | share of the false charge that is real: Σmin(pred, Tf) / Σpred | classical: false fraction → vs complete; holding ≥ 25 |
|---|---|---|---|---|---|
| ≤ 10 k | 0.0116 → 0.0112 | 0.03 / 0.00 (0.003) | 0 vs 269 | 0.01 | 0.0020 → 0.0019; 0.02 |
| 10 k-100 k | 0.0243 → 0.0228 | 0.07 / 0.01 (0.008) | 0 vs 283 | 0.02 | 0.0057 → 0.0054; 0.07 |
| 100 k-1 M | 0.0190 → 0.0164 | 0.15 / 0.02 (0.016) | 0 vs 295 | 0.03 | 0.0250 → 0.0226; 0.11 |
| 1-4 M | 0.0301 → 0.0232 | 0.26 / 0.05 (0.038) | 2 vs 313 | 0.05 | 0.0680 → 0.0588; 0.17 |
| **> 4 M** | **0.1017 → 0.0790** | **0.29 / 0.06** (0.083) | 5 vs 348 | **0.06** | 0.1223 → 0.1052; 0.21 |

M6dc is the same to within 0.002 in every cell of this table. The re-scored rate counts only cells with Tf < 25 e- as empty.

**Re-scored region metrics** (32 × 128 regions).
- Faint-region linearity, sum ratio M6dc / M6d / classical:
  - 1-4 k regions: 0.91 / 0.67 / 0.36 against hits truth; 0.83 / 0.61 / 0.33 against complete truth;
  - 4-16 k regions: 0.98 / 0.87 / 0.56 against hits truth; 0.92 / 0.82 / 0.52 against complete truth.
- The complete truth holds 1.106× and 1.059× the hits charge in those regions.
- Empty regions above 1 Q0: 0.35% / 0.80% / 0.16% → 0.34% / 0.78% / 0.15%.

| explanation | share of the dense haze it accounts for | evidence |
|---|---|---|
| truth artefact: the false cells hold real sub-25 e- charge | **about 1/4 of the false-cell count in > 1 M** (22% at > 4 M, 23% at 1-4 M); 14% at 100 k-1 M; 3-6% when sparse. **About 5% of the false charge** (6% at > 4 M, 1-3% below 1 M) | 29% of > 4 M false cells hold ≥ 25 e-, against 8.3% of all empty cells there. The model does respond to real sub-threshold charge (3.5× enrichment), but where it does, the true amount is a few e- (median 5 e- at > 4 M) and it predicts 350 e- [V] |
| real over-prediction (spill / blur of neighbouring bright charge, context) | **about 3/4 of the false cells and about 95% of the false charge** in dense regions | after re-scoring against complete truth, 7.9% of truly empty > 4 M cells still exceed 100 e- (classical 10.5%). Only 4% of false cells hold even half the predicted charge [V] |
| side effect: hits-truth incompleteness flatters faint-region linearity | linearity at 1-4 k drops 0.08 (M6dc) / 0.06 (M6d) against complete truth | the effect is the same sign for all methods and the ranking is unchanged [V] |

**Verdict F.** The hypothesis holds for about a quarter of the dense false cells, but it is not the main cause. The haze is mostly genuine over-prediction of a few hundred e- next to bright charge. Classical tuned behaves the same, with a slightly smaller truth-artefact share.

## B. Faint false charge in sparse regions

**The failing.** Gated M6d has 0.0116 of empty cells > 100 e- in ≤ 10 k neighbourhoods against classical 0.0020, and 0.0243 vs 0.0057 at 10-100 k. In isolated surroundings (no charge within 8 cells) it has 0.0002 against classical's 0.0014.

**Explanations considered.**
1. Lone noise coefficients that survive the corpus threshold (an information limit).
2. A context prior.
3. Truth incompleteness.

**Decisive test.** Every sparse false cell (Th = 0, M6d gated > 100 e-) goes into the first category it meets, in this order:
1. **Truth incomplete:** complete truth Tf ≥ 25 e-.
2. **Spill next to true charge:** complete-truth charge within 2 cells. 68% of these have a kept coefficient within 3 × 3; that is usually the deposit's own signal.
3. **Lone noise coefficient:** no complete-truth charge within 2 cells but a kept coefficient within 3 × 3. Response plus band footprint is at most 1 cell, so this coefficient cannot be the deposit's own signal: it is noise.
4. **Context:** no coefficient, but charge within 8 cells.
5. **Far:** no coefficient, no charge.

Share of false cells (share of false charge):

| category | M6d, ≤ 10 k | M6d, 10-100 k | M6d, iso | classical, ≤ 10 k | classical, 10-100 k |
|---|---|---|---|---|---|
| truth incomplete (iii) | 0.035 (0.029) | 0.073 (0.059) | 0 | 0.019 | 0.072 |
| spill next to true charge (ii, local) | **0.523** (0.453) | **0.686** (0.662) | 0 | 0.300 | 0.766 |
| lone noise coefficient (i) | **0.296** (0.364) | 0.167 (0.201) | **0.80** | 0.043 | 0.012 |
| context, no coefficient, 3-8 cells from charge (ii) | 0.146 (0.155) | 0.074 (0.077) | 0.007 | 0.638 | 0.150 |
| no coefficient, far | 0 | 0 | 0.19 | 0 | 0 |

M6dc is identical to within 0.01.

**Excess over classical in the ≤ 10 k stratum** (false fraction × category share): 0.0116 − 0.0020 = 0.0096, made up of
- spill: +0.0055 (**57%**);
- lone noise coefficients: +0.0033 (**34%**);
- context, no coefficient: +0.0004 (4%);
- truth incompleteness: +0.0004 (4%).

**Is the model's response to a lone coefficient a calibrated likelihood?** The test uses truly empty cells (Th = 0, Tf < 25, no complete-truth charge within 2 cells) whose own cell holds a kept coefficient of amplitude a. Their prediction rate is compared with that of faint charged cells (0 < Th < 1 Q0) with the same amplitude. ≤ 10 k stratum:

| amplitude (σ) | lone cells (weighted) | M6d fires on lone (mean presence) | fires on charged at same amplitude (mean presence) | P(charged \| amplitude) in the data | classical fires on lone |
|---|---|---|---|---|---|
| 3-4 | 273 k | 0.005 (0.035) | 0.45 (0.48) | 0.04 | 0.000 |
| 4-5 | 106 k | 0.007 (0.038) | 0.70 (0.71) | 0.08 | 0.000 |
| 5-6 | 5.2 k | 0.022 (0.062) | 0.91 (0.92) | 0.42 | 0.002 |
| 6-8 | 1.1 k | 0.011 (0.037) | 0.94 (0.96) | 0.72 | 0.014 |

At 10-100 k the lone firing rate is 0.012 / 0.015 / 0.044 for 3-4 / 4-5 / 5-6σ, and the posterior is 0.19 / 0.35 / 0.81.

**Context prior without any coefficient.** These are truly empty cells with no kept coefficient in 3 × 3, at distance d from charge. At d = 1 / 2 / 3-4 / 5-8 cells, M6d fires 0.026 / 0.014 / 0.0048 / 0.0017 of the time (mean presence 0.13 / 0.13 / 0.09 / 0.05). Classical fires 0.0023 / 0.0018 / 0.0018 / 0.0019, flat in distance [V].

| explanation | share of the sparse excess over classical (≤ 10 k) | evidence |
|---|---|---|
| (ii) context / spill: charge predicted because true charge is within 2 cells, or a prior without any coefficient | **about 60%** (57% spill + 4% coefficient-free context); at 10-100 k spill alone is 69% of the false cells | the false rate on coefficient-free empty cells decays with distance to charge, 0.026 at d = 1 to 0.0017 at d = 5-8, while classical is flat at about 0.002. 75% of sparse false cells are within 4 cells of true charge [V] |
| (i) lone noise coefficients, an information limit of the sparsified input | **about 34%** in ≤ 10 k (30% of false cells, 36% of false charge); **80% in iso** surroundings, where the model is still 7× cleaner than classical | the model fires on only 0.5-2% of lone 3-6σ coefficients and gives them mean presence 0.04-0.06, below the data's posterior of 0.04-0.42. Equally strong coefficients on faint charged cells get presence 0.48-0.92. So the response is conservative and discriminative, not over-trusting. The residual false hits are the price of detecting 45-70% of faint charged cells at the same amplitude, whose coefficients are statistically indistinguishable from noise in amplitude alone. Classical reads the full waveform, where these coefficients do not exist as isolated objects [V]. Whether a calibrated likelihood could do better is [A] |
| (iii) truth incompleteness | **about 4%** (3.5-7% of false cells) | false cells holding Tf ≥ 25 e-: 0.035 at ≤ 10 k, 0.073 at 10-100 k; classical 0.02 / 0.07 [V] |

**Verdict B.** The sparse excess is mostly local spill and a context prior around faint deposits, roughly 60%. A third comes from lone noise coefficients, to which the model already responds conservatively. Truth incompleteness is negligible. Far from activity (iso) the model is 7× cleaner than classical, and what it does predict there sits on lone noise coefficients.

## C. Gating zeroes true charge

**The failing.** Gating zeroes 0.74% of all true charge: 33% of the cell charge in iso neighbourhoods, 25% at ≤ 10 k, 9.4% at 10-100 k. For 1-2 k e- cells it zeroes 39.5% / 46.5% / 14.5% on U / V / Y.

**Competing explanations.**
1. No input signal survives (sensor zero-suppression or the corpus threshold): unrecoverable.
2. Presence mis-calibration.
3. The 0.5 threshold choice.

**Decisive test.** For every zeroed true cell (Th > 0, p ≤ 0.5):
- Is it token-covered?
- Does a kept coefficient lie in the cell or the 3 × 3?
- What is the noise-free signal coefficient amplitude Smax3, in σ? This is the per-coefficient SNR of the corpus input before thresholding.
- What is its presence value?

Plus reliability diagrams and the zeroed-charge-vs-threshold curve.

Share of the zeroed charge (M6d; M6dc within 0.01):

| | all | iso | ≤ 10 k | 10-100 k | 100 k-1 M | 1-4 M | > 4 M |
|---|---|---|---|---|---|---|---|
| zeroed / bin charge | 0.0074 | 0.33 | 0.25 | 0.094 | 0.016 | 0.005 | 0.002 |
| **input absent** (no token, or no kept coefficient in 3 × 3) | **0.35** | **0.70** | **0.67** | **0.58** | 0.37 | 0.14 | 0.045 |
|   of which noise-free signal < 1σ (the sensor holds essentially nothing) | 0.05 | 0.21 | 0.13 | 0.09 | 0.05 | 0.01 | 0.00 |
|   of which signal 1-3.35σ (erased by the corpus threshold) | 0.27 | 0.45 | 0.49 | 0.44 | 0.29 | 0.11 | 0.03 |
|   of which signal ≥ 3.35σ (lost to the noise draw) | 0.03 | 0.04 | 0.05 | 0.05 | 0.04 | 0.02 | 0.01 |
| kept coefficient only in an adjacent cell | 0.33 | 0.17 | 0.21 | 0.29 | 0.40 | 0.37 | 0.24 |
| kept coefficient in the cell, p ≤ 0.3 | 0.18 | 0.10 | 0.10 | 0.10 | 0.16 | 0.28 | 0.27 |
| kept coefficient in the cell, p 0.3-0.5 | 0.13 | 0.03 | 0.02 | 0.04 | 0.08 | 0.21 | 0.44 |
| (kept in cell with Kmax ≥ 5σ) | 0.19 | 0.00 | 0.01 | 0.02 | 0.11 | 0.36 | 0.61 |

The median zeroed cell holds 280 e-.

**Reliability** (presence bin centre → observed fraction of cells with hits charge):

| | 0.05 | 0.25 | 0.45 | 0.55 | 0.75 | 0.95 |
|---|---|---|---|---|---|---|
| ≤ 10 k | 0.01 | 0.01 | 0.03 | 0.05 | 0.16 | 0.83 |
| 10-100 k | 0.01 | 0.03 | 0.07 | 0.11 | 0.28 | 0.93 |
| 100 k-1 M | 0.01 | 0.07 | 0.16 | 0.25 | 0.48 | 0.98 |
| > 4 M | 0.04 | 0.19 | 0.38 | 0.48 | 0.70 | 1.00 |

**Threshold curve** (charge zeroed / empty cells > 100 e-, all densities):

| threshold | 0.1 | 0.2 | 0.3 | 0.4 | **0.5** | 0.6 | 0.7 |
|---|---|---|---|---|---|---|---|
| charge zeroed | 0.11% | 0.25% | 0.44% | 0.62% | **0.74%** | 0.83% | 0.90% |
| empty cells > 100 e- | 4.50% | 4.39% | 3.07% | 1.25% | **0.51%** | 0.26% | 0.15% |

M6dc's admitted false cells are the same at ≥ 0.3 and higher below (13% at 0.1).

| explanation | share of the zeroed charge | evidence |
|---|---|---|
| (i) the input holds no surviving signal: unrecoverable from the corpus | **35% overall; 67-70% in sparse (iso / ≤ 10 k), where the failing is**; 4.5% at > 4 M | **Split by cause:** ~5 points have < 1σ of noise-free signal in any coefficient (sensor zero-suppression / sub-ADC charge); ~27 points have 1-3.35σ signal that the corpus threshold erases (an information limit of the sparsified input, the same one `DENOISE.md`'s stage replay finds); ~3 points had ≥ 3.35σ signal that the noise realisation pushed below threshold. **Plane split:** for 1-2 k e- cells, 30 / 35 / 45% (U / V / Y) of the zeroed ones have no surviving coefficient [V] |
| (ii) presence mis-calibration, i.e. p too low despite visible signal | **not supported in sparse regions** (the 10-13% "kept in cell, p ≤ 0.3" there is ambiguous evidence, not mis-calibration). In dense regions, up to the 18% overall / 27% at > 4 M "kept in cell, p ≤ 0.3", but that evidence is not cell-specific | **Sparse:** the presence head is over-confident, not under-confident. At p = 0.45 only 3-7% of cells hold charge, and at p = 0.75 only 16-28%, so a calibrated threshold would zero more charge, not less. Sparse zeroed cells with a kept coefficient in the cell have Kmax < 5σ in 96% (≤ 10 k) / 86% (10-100 k) of their charge, the amplitude range where noise coefficients dominate (B: posterior 0.04-0.35). **Dense (> 1 M):** presence is nearly calibrated (p 0.45 → 0.30-0.38). The zeroed cells sit next to bright charge whose coefficients (Kmax ≥ 5σ for 61% at > 4 M) are the neighbour's, so the cell's own 280 e- is not separately visible [V; cell-specificity A] |
| (iii) the 0.5 threshold choice | **at most about 40%** of the zeroed charge (0.30 of 0.74 points) is recoverable by lowering to 0.3, but empty cells > 100 e- rise 6× (0.51% → 3.07%). 0.4 recovers 16% at 2.4× the false cells | given the reliability above, 0.5 is near the knee of the trade-off: below 0.4 the admitted false cells grow much faster than the recovered charge [V] |

**Verdict C.** Where gating hurts most, the sparse strata (iso / ≤ 10 k zero 25-33% of cell charge), about 70% of the zeroed charge has no surviving coefficient in the model's input. Most of that is signal the corpus threshold erased; a smaller part is signal the sensor never held. That is an input limit, not a model error. The presence head is over-confident there, so mis-calibration in the direction of under-detection is ruled out. Threshold tuning can recover at most about 40% of the zeroed charge, at a 6× false-cell cost.

In dense regions the small zeroed fraction (0.2-0.5% of bin charge) sits next to bright tracks, where coefficient evidence cannot separate a 280 e- cell from its neighbour.

## Infrastructure

- **Reads.** /pscratch reads were probed on the host with `lfs getstripe -i`. The only OST-61 files were `dense/cls_fit/ev275_g5` and `ev301_g2`, read from `dense2/cls_fix` on CFS. The corpus, model maps, step, hits and sensor files were all read from CFS.
- **Outputs.** All are under `/global/cfs/cdirs/m5238/users/oalter/helix_runs/causes/`. Compute ran on job 59625957.

```bash
H=baselines/uboone_sp/causes; C=/global/cfs/cdirs/m5238/users/oalter/helix_runs/causes
JAX_PLATFORMS=cpu scripts/helix_run.sh python $H/build.py --events 260-387 --workers 30 --ost61 $C/ost61.txt --out $C/cells
scripts/helix_run.sh python $H/analyze.py --cells $C/cells --out $H/results
```
