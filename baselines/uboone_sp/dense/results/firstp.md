# First principles: are the denoiser's "failings" posterior quantities under the training distribution?

2026-10-11. Post hoc only: no training, no new losses or heads. Labels:
- **[V]** = measured here;
- **[A]** = assumed or inferred.

Files:
- data: `firstp.json`;
- figures: `../figs/firstp_reliability.png`, `../figs/firstp_linear.png`;
- code: `firstp.py`, `figs_firstp.py`, `run_firstp_dump.sh`;
- intermediates: `/global/cfs/cdirs/m5238/users/oalter/helix_runs/firstp/`.

## Set-up

**Models.**
- M4n, M6d, M6dc: the earlier dumps.
- The four fast-protocol label arms, `lb_dec_{hits,cond_hits,ct,cond_ct}/best.pt`. The *_cond arms are hurdles; the *_ct arms were trained on the complete truth.
- Every token-covered cell of the 128 test events was dumped with `model_dump2.py` on the stored corpus. The "deployment distribution" = all covered cells, about 680 k per event.

**Truth.**
- Hits truth: `fine_cells(hits)`.
- Complete truth: the other agent's `denoise_truth_complete`, run_0027575715 shards.

**A first-principles fact about the complete label** [V, its files]. 23.5% of complete-truth cells hold < 1 e-, and 37.6% hold < 25 e-; values go down to 1e-37 e-. The *_ct arms' presence target is q > 0 of that truth, i.e. "any diffusion tail". It is not the same event as hits q > 0.

**Training distribution.** I applied `DenoiseEvents.rows` with the checkpoints' recipe to each event's own truth (the hits or complete truth, per model): sig_cap 20000, neg 1, near 1, cov 2, win 48, near_any 1, win_near 24. Rows were restricted to covered cells, as training does.

**Ratios.** Fitted on 63 calibration events: probe-train ev000-063, minus ev041 (hung OST). These events are disjoint from the test events and from denoiser training. The ratios were then applied to test.
- Global log(r_train/r_deploy), by plane U / V / Y: **2.20 / 2.35 / 2.15** for the hits label (odds ratio 9.0 / 10.5 / 8.6) and 2.07 / 2.18 / 2.06 for the complete label.
- The other agent's train-event measurement (`causes/results/firstp_sampling.md`, 200 train events) gives 9.1 / 10.6 / 8.6 per plane and the same density profile. The two agree, so I used mine (out-of-sample) [V].

**Stratification of the prior correction.**
- **glob:** one ratio per plane. Deployable.
- **tdens:** per stratum of the true charge within ±8 cells. This uses the truth, so it is a diagnostic only.
- **pdens:** per stratum of the model's own predicted charge within ±8 cells. Deployable.
- The ratio falls from about 15 (sparse) to about 1.4 (shower cores); in log terms, U for example goes 2.72 / 2.72 / 2.12 / 1.17 / 0.58 / 0.31 from iso to > 4 M e-. So a single global ratio cannot calibrate everywhere.

**Jensen correction.** q_J = Q0·(exp(ŷ + σ²/2) − 1), with σ² the variance of the log-space residual (y_true − ŷ). σ² is estimated on the calibration events per (ŷ bin, predicted-density bin) and applied to test. Both stratifiers use predictions only. For hurdles, σ² is the conditional residual on charged cells.

## Test 1: presence calibration on the deployment distribution (own training label; ECE on 19 probability bins)

| model | ECE raw | ECE glob | ECE pdens | ECE tdens | Brier raw → pdens | log-loss raw → pdens |
|---|---|---|---|---|---|---|
| M4n | 0.062 | 0.009 | **0.0011** | 0.0011 | 0.0302 → 0.0178 | 0.118 → 0.069 |
| M6d | 0.063 | 0.009 | **0.0013** | 0.0013 | 0.0310 → 0.0177 | 0.120 → 0.069 |
| M6dc | 0.064 | 0.009 | 0.0025 | 0.0013 | 0.0313 → 0.0179 | 0.121 → 0.070 |
| lb_dec_hits | 0.053 | 0.011 | 0.0083 | 0.0082 | 0.0360 → 0.0221 | 0.131 → 0.090 |
| lb_dec_cond_hits | 0.054 | 0.011 | 0.0069 | 0.0051 | 0.0346 → 0.0214 | 0.127 → 0.086 |
| lb_dec_ct (label: complete q > 0) | 0.064 | 0.022 | 0.0145 | 0.0140 | 0.0554 → 0.0376 | 0.186 → 0.146 |
| lb_dec_cond_ct (label: complete q > 0) | 0.060 | 0.023 | 0.0155 | 0.0144 | 0.0537 → 0.0377 | 0.184 → 0.150 |

All intervals are ±0.0002-0.001.

**Per plane and per density stratum** [V]:
- Raw ECE is 0.055-0.083 on every plane.
- After the pdens correction it is ≤ 0.003 for M4n, M6d and M6dc on every plane, and ≤ 0.010 in every density stratum.
- The raw miscalibration is worst in the sparse-to-medium strata: 0.13-0.18 at ≤ 10 k and 10-100 k e-. That is where the odds ratio is about 9-15.
- A single global correction leaves dense strata over-corrected (ECE 0.07-0.12 at > 1 M e-), because their odds ratio is only 1.4-2.

**Against other labels** (complete q > 25 / > 102 e-) the hits models after pdens are 0.006 / 0.002 (M6d). The ct arms are worse against these labels, 0.033-0.045, than against their own: they predict a different event.

## Test 2: linear-space charge

**Binning by truth always shows regression.** Cells binned by TRUTH always show faint cells over-predicted and moderate cells under-predicted, for any posterior-mean estimator (E[q̂|q] ≠ q even when E[q|q̂] = q̂). Unbiasedness is therefore tested in PREDICTION bins: truth / prediction should be 1. The truth-binned tables are in the JSON (`lin_*`).

Truth / prediction in prediction bins (complete truth, all planes); 1 = unbiased, < 1 = over-prediction. The first column is the share of all true charge carried by cells predicted exactly 0.

| model | estimator | predicted 0 (share of truth) | 0-25 e- | 25-102 | 102-500 | 0.5-1k | 1-2k | 2-4k | 4-8k | 16-64k | > 64k | plane total pred/truth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M6d | raw | 0.001 | **0.21** | **0.28** | **0.47** | 1.02 | 1.09 | 1.07 | 1.04 | 1.01 | 1.00 | 1.010 |
| M6d | raw + Jensen | 0.001 | 0.20 | 0.27 | 0.43 | 0.91 | **1.02** | **1.04** | 1.03 | 1.01 | 1.00 | 1.014 |
| M6d | raw × p_corr/p (prior shift) | 0.001 | **1.33** | **1.38** | **1.43** | 1.26 | 1.14 | 1.07 | 1.04 | 1.01 | 1.00 | 0.987 |
| M6d | prior shift + Jensen | 0.001 | 1.28 | 1.28 | 1.31 | 1.17 | 1.08 | 1.04 | 1.03 | 1.01 | 1.00 | 0.991 |
| M6d | gated (p > 0.5) | **0.008** | 5.7 | 1.64 | 0.78 | 1.03 | 1.09 | 1.07 | 1.04 | 1.01 | 1.00 | 0.984 |
| M4n | raw / prior shift + Jensen | 0.001 | 0.20 / 1.29 | 0.28 / 1.28 | 0.48 / 1.31 | 1.04 / 1.17 | 1.10 / 1.09 | 1.07 / 1.05 | 1.05 / 1.04 | 1.02 / 1.02 | 1.02 / 1.02 | 0.995 / 0.977 |
| M6dc | hurdle p_corr × cond | 0.001 | 1.46 | 1.49 | 1.42 | 1.26 | 1.14 | 1.07 | 1.05 | 1.02 | 1.01 | 0.981 |
| M6dc | hurdle + Jensen | 0.001 | 1.36 | 1.38 | 1.33 | 1.19 | 1.09 | 1.05 | 1.04 | 1.02 | 1.01 | 0.983 |
| M6dc | gated | 0.009 | 6.4 | 1.64 | 0.67 | 0.65 | 0.93 | 1.05 | 1.05 | 1.02 | 1.01 | 0.982 |
| lb_dec_hits | raw / prior shift + Jensen | 0.001 | 0.57 / 2.03 | 0.46 / 1.42 | 0.45 / 1.20 | 1.08 / 1.06 | 1.19 / **1.06** | 1.16 / **1.04** | 1.11 / 1.03 | 1.03 / 1.00 | 1.01 / 0.99 | 0.996 / **1.003** |
| lb_dec_ct | raw / prior shift + Jensen | 0.001 | 0.41 / 1.59 | 0.48 / 1.16 | 0.90 / 1.07 | 1.26 / 1.08 | 1.18 / **1.04** | 1.13 / **1.01** | 1.10 / 1.00 | 1.03 / 1.00 | 1.01 / 0.99 | 0.985 / **1.003** |
| lb_dec_cond_ct | hurdle + Jensen | 0.001 | 2.54 | 1.16 | 1.07 | 1.09 | 1.03 | 1.00 | 0.99 | 0.99 | 0.98 | **1.010** |

Regions (16 × 8 cells, prediction bins of the region sum) show the same pattern:
- **Raw maps.** Regions predicted < 64 k e- hold only 12-74% of what they predict. That is the integrated haze.
- **Prior shift.** It brings them to 1.06-1.33.
- **Gated maps.** Regions predicted 1-4 k hold 2.0-2.3× their prediction, the faint-region under-estimate.

**Per plane, full-plane totals** (M6d, complete truth):

| estimator | U | V | Y |
|---|---|---|---|
| raw | 1.012 | 1.012 | 1.005 |
| gated | 0.984 | 0.980 | 0.990 |
| prior shift + Jensen | 0.991 | 0.987 | 0.994 |

The other models follow the same pattern (`plane_ct_*_U/V/Y`).

What these numbers say [V]:
- **The raw maps' near-unbiased plane totals (0.985-1.010) are a cancellation.** A factor 2-5 over-prediction in cells predicted < 500 e- (the haze) offsets a 4-19% under-prediction at 1-8 k e-.
- **Prior shift.** It removes the faint over-prediction and slightly overshoots: 0.2-0.5 → 1.3-1.4 for M4n and M6d. In log terms it removes the bias and over-corrects by about 30%. That is expected, because a log-space mean is not p × conditional mean.
- **Jensen.** It removes 60-90% of the 1-8 k e- under-prediction: M6d 1.09 → 1.02, lb arms 1.16-1.19 → 1.01-1.06.
- **Most nearly unbiased estimator above 1 k e-** (within 1-6%, every plane): prior shift + Jensen for the non-hurdle models, and hurdle + Jensen for the hurdles. No estimator is unbiased below about 500 e- predicted; there they are 30-60% low after correction.

## Test 3: proper scores, and a re-ranking

| model | presence log-loss (pdens-corrected; own label) | presence Brier (pdens) | charge CRPS in y (own label) |
|---|---|---|---|
| M6dc | 0.070 | 0.0179 | **0.0059** (point mass + Gaussian) |
| lb_dec_cond_hits | 0.086 | 0.0214 | 0.0095 (mixture) |
| lb_dec_cond_ct | 0.150 (complete label) | 0.0377 | 0.0109 (mixture; complete label) |
| M6d | **0.069** | **0.0177** | 0.0114 (Gaussian) |
| M4n | 0.069 | 0.0178 | 0.0114 (Gaussian) |
| lb_dec_hits | 0.090 | 0.0221 | 0.0167 (Gaussian) |
| lb_dec_ct | 0.146 (complete label) | 0.0376 | 0.0173 (Gaussian; complete label) |

**Presence (hits label).** M6d = M4n ≈ M6dc. The fast arms are worse, which is the shorter training. The ct arms predict a different and harder event (any tail charge), so their log-loss is not comparable.

**Charge.** The hurdle's point mass at zero makes it the best predictive distribution by a factor of about 2 in CRPS. M6dc beats M6d (0.0059 vs 0.0114), the reverse of the thresholded detection ranking.

On CRPS, hurdle > non-hurdle at equal training (M6dc > M6d; cond_hits > hits). M4n and M6d tie. Thresholded counts ranked M6d first.

## Test 4: the label arms

| arm pair | presence ECE raw → pdens (own label) | empty-cell charge (share of complete truth): raw | gated | p_corr hurdle |
|---|---|---|---|---|
| lb_dec_hits vs lb_dec_ct | 0.053 → 0.008 vs 0.064 → 0.015 | 0.023 vs **0.016** | 0.004 vs 0.006 | — |
| lb_dec_cond_hits vs lb_dec_cond_ct | 0.054 → 0.007 vs 0.060 → 0.016 | (conditional) | 0.005 vs 0.008 | **0.005 vs 0.005** |

What these numbers say [V]:
- **The complete-truth arms' extra haze is a gated-map phenomenon.** It is +0.002 to +0.003 of the truth in empty cells. In the raw map the ct arm puts LESS charge in empty cells than the hits arm.
- **After the prior correction (p_corr hurdle) the extra haze disappears exactly:** 0.005 vs 0.005.
- **The extra gated haze has two causes.** The ct presence target includes infinitesimal tails (more cells cross 0.5), and the uncorrected prior.
- **So the complete label is fine for charge.** The ct arms' linear calibration is the best of the seven (prior shift + Jensen: 1.01-1.04 from 2 k e- up, plane total 1.003).
- **The presence target is the problem.** For a presence head, the complete label should be thresholded, e.g. q > 25 e-. As is, the ct arms are mis-calibrated against q > 25 / > 102 even after correction (ECE 0.033-0.045) [V].

## Verdict per hypothesis

| hypothesis | verdict | share of the failing explained | evidence |
|---|---|---|---|
| **H1** outputs are training-distribution posteriors; presence is over-confident by the case-control prior; raw haze and the faint over-estimate are mostly prior shift | **Confirmed [V]** | **Presence:** 98% of ECE (0.063 → 0.0013, M6d) and 42% of log-loss, with the deployable pdens correction; the global ratio alone removes 86%. **Charge:** the sign and essentially all of the factor 2-5 faint over-prediction; it over-corrects by about 30% because the charge head is a log mean | Training odds of "charged" are 9-10× deployment on average (15× sparse, 1.4× cores); the ratio is a recipe property, the same on train events (other agent) and held-out events (here). After correction presence is calibrated on every plane and density stratum |
| **H2** exponentiating a log-space mean underestimates E[q] by ~σ²/2 | **Confirmed for moderate charge [V]; not the faint failing** | 60-90% of the 4-19% under-prediction at 1-8 k e-; about 0 at > 16 k (σ_log 0.05-0.08 there); it makes the faint over-prediction worse, so it is not its cause | σ_log (M6d) 0.21-0.24 at ŷ 0.2-0.75 (400-2,000 e-), 0.05-0.11 above 16 k; lb arms 0.34-0.37 |
| **H3** gating at p > 0.5 is a decision rule whose bias follows from the posterior | **Confirmed [V]** | All of the gated maps' zeroed charge (0.8-1.1% of the truth vs 0.1% raw) and the faint-region under-estimate (16 × 8 regions predicted 1-4 k hold 2.0-2.3× their prediction) | Under the deployment prior, raw p > 0.5 means corrected p > 1/(1 + OR): about 0.1 in sparse regions, about 0.4 in shower cores. The gate is a density-dependent threshold, not a 50% rule [A, arithmetic from the measured OR] |
| **H4** per-cell spread includes irreducible posterior width | **Partly supported; size not separable [A]** | Not quantified | σ_log 0.21-0.24 (M6d/M4n) vs 0.34-0.37 (fast arms) at the same predicted charge: much of the arms' width is training, not irreducible. The irreducible part needs an information bound per cell (causes_ade: the corpus keeps < 2σ of the evidence for half of faint deposits) [A] |

**Consequences** (post hoc, no retraining):
1. **Presence.** Report presence after the deployable pdens prior correction. It is calibrated. The thresholded detection metrics are rank-based and unchanged by the correction.
2. **Charge.** The linearly unbiased charge estimators are prior shift + Jensen for non-hurdle models and p_corr × cond + Jensen for hurdles. Gating at 0.5 should be viewed as a decision rule with a density-dependent threshold.
3. **Model ranking.** Rank on proper scores: hurdle models lead on charge CRPS.
4. **Complete-truth presence label.** Use q > ~25 e-, not q > 0.

Caveats:
- The calibration set is 63 events, all from one run (as are the test events).
- Prediction-density strata for the hurdles use the conditional charge map.
- The Gaussian CRPS for hurdle models is not meaningful and is omitted.
- The prior correction assumes the sampling depends on the label only within a stratum. Rows actually depend on distance to charge [A], which is why tdens/pdens beat glob.
