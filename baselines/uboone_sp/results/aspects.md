# Where each method wins, and why: identical cells and windows

All results below use the 128 test events (ev260-387). Each method is scored on the same
fine cells (2 wires × 16 ticks) and the same eval windows. Sources:
- `v2/results/aspects.json` (cell level, ROC, near-activity distance);
- `v2/results/benchmark.json` (detection with bootstrap);
- `v2/results/pairs.json` (pair separation).

Figures are `figs/aspects_*.png` and `figs/validity_summary.png`. Threats and validation are
in `validity.md`. Labels: [V] verified by running; [A] assumed.

## Methods

**Classical chain.** Input is the gated waveform; noise is seed 1000 + ev. Selection used
the tuning split ev000-095 only.
- `cls_default`: the MicroBooNE configuration of the faithful Wire-Cell port. It outputs ROI
  Gaus_wide charge.
- `cls_tuned_q`: the tuned Wire-Cell ROI charge. U is thresholds ×1.25 with the fake cut
  ×0.25; V is ×1.5 with no fake cut; Y is ×0.67 with no fake cut. It is a charge output, so
  it is used for every charge aspect.
- `cls_tuned_any`: the tuned best detection statistic. On U and V it is an LF-filtered Wiener
  significance with no ROI; on Y it equals `cls_tuned_q`. It is a detection score, not a
  charge.
- `cls_noroi`: Gaus_wide cell sums without ROIs (the R2nrp-like reference).
- `cls_oracle_roi`: Gaus_wide inside truth ROIs. It is a reference only and not a detector.

**Model.** Input is the sparsified corpus.
- `M4n_q`, `M4n_p`: the M4n_ft_full_near charge head and presence probability.
- `N64_q`, `N64_p`: the same for le3_ft_N64 (64 labelled events).
- The model predicts 0 on cells no token covers. Token coverage is 97.6-98.1% of truth cells
  and 35-49% of empty cells [V].

Q0 = 2,043.26 e- (~0.1 MeV).

## Summary: who wins what

| aspect | winner | size |
|---|---|---|
| detection far from activity (all planes) | **model** | 0.85 vs 0.50 (0.1-0.2 MeV, per-plane 1% threshold) |
| detection near activity, collection (Y) | **classical** (tuned) | 0.92 vs 0.90 (64-label model 0.82) |
| detection near activity, induction (U, V) | **model** | U 0.50 vs 0.37, V 0.38 vs 0.30 |
| detection per deposit SNR, near activity | **tie** | 5-7: 0.34 vs 0.30; 7-10: 0.82 vs 0.80; 10-15: 0.98 vs 0.98 |
| near-activity haze (charge-free windows above the far threshold) | **classical** | 0.01 vs 0.15 |
| faint false cells (> 0.05 Q0 on empty cells) | **classical** | U+V+Y 0.91M vs 6.7M |
| bright false cells (> 1 Q0 on empty cells) | **model** | 87k vs 0.9k |
| per-cell charge fidelity (true 2-16k e-) | **model** | 68% half-width 0.09-0.51 vs 0.33-1.0 |
| charge in cells with true 1-2k e- | **model** | classical zeroes 67% (U), 77% (V), 12% (Y) of them; model 2-3% |
| total charge, induction | **model** | 1.01 vs 0.85 / 0.84 (U / V) |
| total charge, collection | tie | 1.005 vs 1.000 |
| separation of two found deposits | **tie** (charge outputs) | w ≤ 4 wires: 0.55 vs 0.48; ≥ 8 wires: equal |
| finding both deposits of a pair | **model** | 0.34 vs 0.19 (w ≤ 8) |
| localisation (eval loc, 0.2-0.5 MeV) | **classical charge** | 0.17 wire / 1.5 tick vs M4n charge 0.82 / 3.4 |

## 1. Noise suppression: predicted charge on truth-empty cells

Cells are binned by chessboard distance, in cells, to the nearest truth charge. The full
table is in `aspects.json` → `noise`; the figure is `aspects_noise.png`.

Fraction of empty cells above 0.25 Q0 / above 1 Q0, by distance:

| plane | method | 1 cell | 2 | 3-4 | 5-8 | 9-16 | > 16 |
|---|---|---|---|---|---|---|---|
| U | classical tuned ROI charge | 0.032 / 0.0081 | 0.0006 / 0.0002 | 0.0003 / 0.0001 | 0.0002 / 0.00004 | 0.0001 / 0.000004 | 0.00005 / 0 |
| U | model M4n charge | 0.008 / 0.00006 | 0.003 / 0.00001 | 0.002 / 0.00001 | 0.001 / 0.00001 | 0.0004 / 0.000003 | 0.00002 / 0 |
| V | classical tuned | 0.027 / 0.0074 | 0.0001 / 0.00005 | 0.00006 / 0.00002 | 0.00003 / 0.00001 | 0.00003 / 0.000003 | 0.00002 / 0.000002 |
| V | model M4n | 0.009 / 0.00009 | 0.003 / 0.00004 | 0.0016 / 0.00002 | 0.0007 / 0.00001 | 0.0002 / 0.000003 | 0.000006 / 0 |
| Y | classical tuned | 0.042 / 0.0114 | 0.004 / 0.0004 | 0.001 / 0.0001 | 0.001 / < 0.00005 | 0.001 / < 0.00005 | 0.001 / < 0.00005 |
| Y | model M4n | 0.003 / 0.000005 | 0.0006 / 0 | 0.0004 / 0 | 0.0002 / 0 | 0.00005 / 0 | 0.000003 / 0 |

Counts over all 128 events (U / V / Y):

| method | empty cells > 0.05 Q0 | empty cells > 1 Q0 |
|---|---|---|
| classical, default | 696k / 157k / 188k | 30k / 25k / 28k |
| classical, tuned ROI charge | 204k / 158k / 546k | 26k / 23k / 39k |
| classical, no ROI | 27.1M / 29.5M / 17.3M | 5.6M / 14.7M / 38k |
| classical, oracle ROI | 349k / 357k / 424k | 24k / 23k / 27k |
| model M4n, charge | 2.65M / 3.42M / 0.59M | 330 / 536 / 17 |
| model 64-label, charge | 605k / 629k / 429k | 8.1k / 7.3k / 3.8k |

The empty-cell totals are 66.7M / 66.7M / 48.3M.

Per-cell noise level, as 90th / 99th / 99.9th percentiles of empty cells:
- **Classical tuned ROI charge:** 0 / 0-170 / 750-1,900 e-. The ROI gate zeroes ≥ 99% of cells.
- **M4n:** 47-60 / 120-211 / 380-530 e-.

What these numbers say:

- **The classical chain is clean wherever it opens no ROI.** Its false charge is
  concentrated one cell from real charge: 3-4% of those cells exceed 0.25 Q0. That is the
  deconvolution filters smearing real charge (Gaus_wide σ_t = 2.9 ticks; wire filter), not
  noise.
- **The oracle-ROI rung has the same 1-cell excess.** So it is the filters, not ROI finding.
- **The model leaves a faint haze that decays with distance:** 0.1-0.3% of cells above
  0.25 Q0 at 2-8 cells. It gives ~7× more faint false cells than the tuned classical chain,
  but ~100× fewer bright ones.
- **The 64-label model is cleaner in the faint haze** (0.6M vs 2.6-3.4M on U/V) but brighter
  where it fails (8k vs 0.3k above 1 Q0).
- **Without ROIs the classical induction output is unusable as a charge map:** 30-40% of
  empty cells exceed 0.25 Q0, because of the near-DC wander of the bipolar
  deconvolution [V].

## 2. Deconvolution fidelity, per cell (truth > 0 cells)

The table gives median predicted / true, with the 68% half-width in brackets. The figure is
`aspects_fidelity.png`; `aspects.json` → `fidelity` also gives the fraction of cells
predicted 0.

| plane | method | true 1-2k e- | 2-4k | 4-8k | 8-16k | 16-64k | > 64k |
|---|---|---|---|---|---|---|---|
| U | classical tuned | 0.00 (0.79), 67% zero | 0.57 (1.00) | 0.91 (0.72) | 1.06 (0.41) | 0.91 (0.14) | 0.87 (0.16) |
| U | classical, oracle ROI | 1.06 (0.95) | 1.05 (0.84) | 1.11 (0.66) | 1.18 (0.41) | 0.96 (0.14) | 0.93 (0.07) |
| U | M4n charge | 0.36 (0.52) | 0.84 (0.44) | 0.97 (0.18) | 0.99 (0.11) | 0.99 (0.06) | 0.99 (0.05) |
| U | 64-label charge | 0.06 (0.48) | 0.34 (0.64) | 0.80 (0.63) | 0.94 (0.54) | 1.00 (0.33) | 0.95 (0.28) |
| V | classical tuned | 0.00 (0.51), 77% zero | 0.00 (0.95), 57% zero | 0.79 (0.94) | 1.04 (0.45) | 0.91 (0.15) | 0.86 (0.17) |
| V | M4n charge | 0.26 (0.50) | 0.81 (0.51) | 0.96 (0.19) | 0.99 (0.12) | 1.00 (0.06) | 0.99 (0.05) |
| Y | classical tuned | 0.84 (0.65), 12% zero | 0.92 (0.69) | 1.00 (0.57) | 1.11 (0.32) | 0.96 (0.10) | 0.95 (0.05) |
| Y | classical, oracle ROI | 1.00 (0.52) | 0.99 (0.64) | 1.02 (0.55) | 1.11 (0.32) | 0.96 (0.10) | 0.95 (0.05) |
| Y | M4n charge | 0.83 (0.49) | 0.96 (0.21) | 0.99 (0.12) | 1.00 (0.09) | 1.00 (0.05) | 0.99 (0.04) |

Total charge recovered (sum of predictions over all cells / truth, U / V / Y):

| method | U | V | Y |
|---|---|---|---|
| classical tuned | 0.855 | 0.840 | 1.000 |
| classical default | 0.868 | 0.839 | 0.973 |
| classical, oracle ROI | 1.004 | 1.005 | 1.004 |
| M4n | 1.012 | 1.015 | 1.005 |
| 64-label model | 0.979 | 0.984 | 0.992 |

What these numbers say:

- **The model is much better per cell at every charge level.** Its 68% half-width is
  0.05-0.2 above 4k e-. The classical chain's is 0.3-0.7 at 4-16k e-, even with oracle ROIs.
- **The classical per-cell limit is the filter smearing, not noise.** A Gaus_wide σ_t of 2.9
  ticks and the wire filter move charge across 16-tick / 2-wire cell edges. With oracle ROIs
  the same width remains (U 4-8k: 0.66).
- **On a per-wire or ROI-integral basis the classical chain would look better.** The paper's
  own metric integrates the ROI on one wire [A, not computed here].
- **Induction loses 15-16% of the total charge in the found-ROI classical chain.** Oracle
  ROIs recover 1.00, so the loss is ROI boundaries and the linear baseline. Faint cells are
  zeroed.
- **Localisation (eval loc, median |centroid offset|, 0.2-0.5 MeV).**
  - Classical tuned ROI charge: 0.17 wire / 1.5 ticks.
  - M4n charge head: 0.82 / 3.4.
  - 64-label charge head: 0.41 / 2.0.
  - The classical chain's ROI-masked output is compact; the model's haze broadens its
    centroid.
  - Signed offsets show no bias for either method: |Δw| ≤ 0.06, |Δt| ≤ 0.4 (validity.md §4).
- **Induction vs collection.** Every gap is larger on induction. On collection the classical
  chain is within 0.02 in total charge and its per-cell bias is ≤ 10% above 2k e-.

## 3. Detection

**ROC curves** at 0.1-0.2 MeV, per plane, at the far and near thresholds:
`aspects_roc_{far,near}.png`. Numbers are in `aspects.json` → `roc`.

**Efficiency vs deposit SNR** (clean isolated deposits): `aspects_snr.png`. Numbers are in
`benchmark.json` (`{far,near}_snr_<plane>_<bin>`).

At the 1% FPR threshold, 0.1-0.2 MeV, per plane, near / far (16-84% in brackets):

| method | U | V | Y |
|---|---|---|---|
| classical default | 0.21 / 0.21 | 0.02 / 0.02 | 0.01 / 0.01 |
| classical tuned (best statistic) | 0.37 [0.34-0.39] / 0.36 | 0.30 [0.28-0.32] / 0.25 | 0.93 [0.92-0.94] / 0.93 |
| 64-label model, presence | 0.28 [0.25-0.30] / 0.53 | 0.23 [0.21-0.25] / 0.48 | 0.82 [0.80-0.84] / 0.91 |
| M4n presence | 0.50 [0.48-0.53] / 0.83 | 0.38 [0.35-0.41] / 0.78 | 0.90 [0.88-0.91] / 0.95 |

By deposit SNR at the near threshold, pooled with per-plane thresholds, for tuned classical /
M4n presence:

| SNR bin | tuned classical | M4n presence |
|---|---|---|
| no signal | 0.01 | 0.01 |
| < 3 | 0.00 | 0.01 |
| 3-5 | 0.05 | 0.06 |
| 5-7 | 0.30 | 0.34 |
| 7-10 | 0.80 | 0.82 |
| 10-15 | 0.98 | 0.98 |

At the far threshold the classical numbers are unchanged, while the model's become 0.31 /
0.35 / 0.46 / 0.66 / 0.92 / 1.00 (same bins).

What these numbers say:

- **Near activity, on clean deposits ranked by their optimal SNR, the tuned classical chain
  and M4n are equal.** Both sit well below the known-location matched-filter bound on the
  gated waveform (0.91 at SNR 3-5, 0.99 at 5-7, from `docs/DENOISE.md`) [A].
- **The model's per-plane advantage on U/V near activity is on deposits with foreign charge
  nearby,** which the SNR bins exclude (clean = foreign ≤ 25%). Neighbouring charge is what
  breaks the classical significance statistics (the matched-filter ladder in DENOISE.md).
- **Far from activity the model's efficiency is much higher, but part of it is not
  sensitivity.** It scores 0.31 on deposits with no clean signal at all. The far threshold
  sits below its near-activity haze. Its far noise windows are also selected for high
  noise, which goes against the model (validity.md #10).
- **The two biases point in opposite directions, and their sizes are not separated.** The
  near threshold is the fair comparison.
- **Collection is where the classical chain is strongest.** Collection is a unipolar
  response with no low-frequency amplification, and a 5σ-scale Wiener threshold at 0.67-0.83
  of the default already reaches ~0.93.

## 4. Two-deposit separation (the eval's pair windows)

The eval's dip metric is confounded by detection: dip = 1 whenever the weaker deposit's
peak is ≤ 0.
- The classical ROI outputs score median dip 1.0 everywhere, because the weaker deposit is
  usually not found.
- The no-ROI Gaus_wide output scores 0.0, because clipped noise valleys read as "resolved".

`v2/pairs.py` separates the two questions:
- **both found:** both deposits' 3×3-cell peaks exceed the method's own far 1% threshold;
- **resolved:** given both are found, valley < 0.5 × the weaker peak.

| separation | n | classical tuned: both / resolved\|both | M4n charge | M4n presence | 64-label charge | truth: resolved |
|---|---|---|---|---|---|---|
| ≤ 4 wires | 203 | 0.30 / 0.48 | 0.46 / 0.55 | 0.57 / 0.19 | 0.36 / 0.51 | 0.80 |
| ≤ 8 wires | 575 | 0.19 / 0.84 | 0.34 / 0.82 | 0.40 / 0.50 | 0.27 / 0.86 | 0.96 |
| ≤ 16 wires | 1,106 | 0.20 / 0.98 | 0.32 / 0.97 | 0.37 / 0.83 | 0.27 / 0.99 | 1.00 |
| ≤ 32 ticks | 293 | 0.17 / 0.86 | 0.31 / 0.90 | 0.37 / 0.61 | 0.26 / 0.95 | 0.98 |
| ≤ 64 ticks | 958 | 0.19 / 0.99 | 0.32 / 0.99 | 0.36 / 0.89 | 0.26 / 0.98 | 1.00 |

What these numbers say:

- **Once both deposits are found, the classical ROI charge and the model's charge head
  separate them equally well.** The model is slightly better at ≤ 4 wires: 0.55 vs 0.48
  (truth itself gives 0.80 on the 2×16 grid).
- **The model's advantage is finding both** (0.34 vs 0.19).
- **The presence head is a poor separator.** It blurs: 0.19 at ≤ 4 wires. Use the charge
  head for topology.
- Figure: right panel of `aspects_near_pairs.png`.

## 5. Robustness near activity

**Near-activity charge-free windows above each method's own FAR 1% threshold**, by the
window's distance to the nearest charge (left panel of `aspects_near_pairs.png`):

| method | 1-2 cells | 3-4 | 5-8 |
|---|---|---|---|
| classical tuned (any) | 0.009-0.021 | 0.005-0.013 | 0.004-0.017 |
| classical default | 0.000-0.014 | 0.000-0.016 | 0.000-0.006 |
| M4n presence | 0.09 (Y) - 0.31 (V) | 0.07-0.22 | 0.04-0.13 |
| 64-label presence | 0.14-0.22 | 0.07-0.16 | 0.03-0.09 |

Ranges are over U / V / Y.

What these numbers say:

- **The classical detector's false-positive rate does not depend on the distance to
  activity** (~1%). Its noise is the waveform noise, and its thresholds are per-wire RMS.
- **The model's false-positive rate rises 4-30× next to charge** (0.04-0.31, against its
  1% far threshold). That is the context prior, or haze.
- **Consequence for efficiency.** Moving the threshold from far to near activity costs the
  classical chain nothing: 0.50 → 0.52 pooled at 0.1-0.2 MeV, because its two thresholds
  coincide. The model goes 0.85 → 0.58.
- **Both methods still lose efficiency to neighbouring charge itself.** That is why the
  per-plane near numbers on U/V sit below the clean-deposit SNR curves.

## 6. Shared upstream limits

**Sensor zero-suppression** affects both methods: integer ADC with |v| < 2 dropped before
noise [V, earlier].
- Share of isolated deposits whose clean sensor holds no signal at all (test events):

  | plane | < 0.1 MeV | 0.1-0.2 MeV | ≥ 0.2 MeV |
  |---|---|---|---|
  | U | 0.43 | 0.008 | ~0.01-0.02 |
  | V | 0.47 | 0.007 | ~0.01-0.02 |
  | Y | 0.18 | 0.013 | ~0.01-0.02 |
  | pooled | 0.36 | 0.009 | ~0.01-0.02 |

- Both methods score ~0.01 (chance) on these at the near threshold.
- This caps every method below 0.1 MeV: near-threshold efficiency at < 0.1 MeV is 0.09
  (classical tuned) vs 0.07 (M4n), largely on Y (0.22 vs 0.15).
- It is irrelevant at 0.1-0.2 MeV.

**Corpus sparsification** affects the model only.
- The per-band hard threshold (3.35-3.9σ, D1 dropped) erases most of the signal of
  SNR 3-7 deposits. A known-location matched filter drops from 0.91 / 0.99 to 0.19 / 0.48 at
  SNR 3-5 / 5-7 (`docs/DENOISE.md`) [A, not re-run].
- That M4n still matches the tuned classical chain at SNR 5-7 near activity (0.34 vs 0.30)
  means it recovers from the sparsified input about what the classical chain gets from the
  full waveform.
- The classical chain re-run on a corpus-like input is in `validity.md` §7.
