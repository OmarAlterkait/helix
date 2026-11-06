# Dense / full-spectrum comparison, v2: presence-gated outputs and the decoder head (M5d, M5dc)

2026-10-09. This page repeats `dense.md` (same events, cells, strata, metrics, bootstrap and classical maps) for the new
models. Labels: [V] = verified by running or by reading the code; [A] = assumed.

- **Models.**
  - M4n, raw and gated.
  - M5d (`M5d_ft_full_dec/final.pt`, FM encoder + CellDecoder head), raw and gated.
  - M5dc (`M5dc_ft_full_dec_cond`, hurdle loss), gated only.
  - **Gated** is the charge map kept where sigmoid(presence) > 0.5 and set to 0 elsewhere: eval_denoise's `_gated` map.
- **Classical references.** MicroBooNE default, tuned (ROI charge `tuned_q`; best detection statistic `tuned_any`) and
  oracle ROI. All use the fitted kernel; `dense.md` showed the exact kernel changes nothing by more than 0.02.

Files:
- tables: `tables_v2.md` (full tables, every entry with event-bootstrap 16-84%);
- JSON: `dense_v2.json`;
- figures: `../figs/*_v2.png`.

## Answers to the three questions

**1. Does gating fix the faint-region over-estimate? Yes. It now under-estimates mildly, except M5dc, which is close to
linear.** [V]

Sum predicted / sum true for 16 x 8-cell regions (32 wires x 128 ticks), all planes:

| region true charge | M4n raw | M4n gated | M5d raw | M5d gated | M5dc gated | classical tuned | oracle ROI |
|---|---|---|---|---|---|---|---|
| < 1 k | 10.0 | 0.85 | 11.1 | 0.96 | 1.65 | 0.75 | 2.22 |
| 1-4 k | 3.04 | 0.64 | 3.28 | 0.68 | **0.94** | 0.36 | 1.24 |
| 4-16 k | 1.84 | 0.85 | 1.92 | 0.87 | **1.00** | 0.56 | 1.12 |
| 16-64 k | 1.19 | 0.93 | 1.20 | 0.92 | 0.97 | 0.81 | 1.07 |
| 64 k-1 M | 1.00-1.03 | 0.97-0.99 | 0.99-1.02 | 0.96-0.98 | 0.99 | 0.93-0.96 | 1.01-1.03 |

Median ratio (68% half-width) in the 1-4 k bin:

| M4n raw | M4n gated | M5d gated | M5dc gated |
|---|---|---|---|
| 2.73 (2.24) | 0.36 (0.62) | 0.43 (0.65) | 0.68 (0.87) |

Empty 16 x 8 regions, fraction above 1 Q0 / mean predicted charge:

| M4n raw | M4n gated | M5d raw | M5d gated | M5dc gated | classical tuned |
|---|---|---|---|---|---|
| 6.3% / 461 e- | 0.29% / 23 e- | 7.7% / 597 e- | 0.33% / 27 e- | 0.81% / 49 e- | 0.16% / 91 e- |

What these numbers say:
- The 1.8-7.8x over-estimate is gone. Gated M4n and M5d now lose 15-36% of the charge of regions below 16 k e-. That
  is half the classical deficit (44-64%).
- M5dc gated is within 6% of truth from 1 k e- upward, the best of any detector here. The oracle-ROI reference
  over-shoots by 7-24% at the low end.

**2. Does it fix the dense-region haze? Mostly yes.** Gating removes 80-90% of the faint false cells. Above 100 k e- of
neighbourhood charge the gated model now has fewer faint false cells than the classical chain. [V]

Fraction of truth-empty cells predicted above 0.05 Q0 (100 e-), all planes, by neighbourhood density:

| density | classical tuned | M4n raw | M4n gated | M5d raw | M5d gated | M5dc gated |
|---|---|---|---|---|---|---|
| ≤ 10 k | 0.0019 | 0.092 | 0.010 | 0.104 | 0.012 | 0.013 |
| 10-100 k | 0.0057 | 0.231 | 0.022 | 0.256 | 0.025 | 0.027 |
| 100 k-1 M | 0.025 | 0.184 | 0.018 | 0.202 | 0.020 | 0.020 |
| 1-4 M | 0.068 | 0.202 | 0.031 | 0.214 | 0.031 | 0.032 |
| > 4 M | 0.124 | 0.428 | 0.109 | 0.443 | 0.109 | 0.107 |

The same, per plane, in > 4 M neighbourhoods:

| plane | M5d gated | classical tuned |
|---|---|---|
| U | 0.113 | 0.111 |
| V | 0.134 | 0.096 |
| Y | 0.075 | 0.172 |

Mean predicted charge per empty cell (e-):

| density | classical tuned | M5d gated |
|---|---|---|
| ≤ 10 k | 1 | 4 |
| 10-100 k | 3 | 9 |
| 100 k-1 M | 25 | 7 |
| 1-4 M | 82 | 12 |
| > 4 M | 172 | 45 |

What these numbers say:
- **The haze no longer sets the comparison.** The "20-60% of empty cells above 100 e-" becomes 2-3%, and 11% in shower
  cores. The classical chain is cleaner only in sparse surroundings below 100 k e-: 4-6x fewer faint false cells.
- **Gating does not touch bright false cells.** They need presence > 0.5 and > 1 Q0 together, so the > 1 Q0 fractions are
  unchanged by construction: 0.00001-0.0004 for M4n and M5d gated against 0.0034-0.022 for classical tuned.
  - M5dc gated has 2-3x more bright false cells than M5d (up to 0.0007 in > 4 M). It is still 30x below classical.

**3. How much faint true charge does gating zero, and where?** In total, 0.74-0.75% of all true charge, for every gated
model. The raw model zeroes 0.06% (uncovered cells only), classical tuned 1.47% and the oracle ROI 0.04%. [V]

Share of true charge in cells set to 0, all planes and all densities:

| cell true charge | classical tuned | M4n raw | M4n gated | M5d gated | M5dc gated |
|---|---|---|---|---|---|
| < 500 e- | 0.76 | 0.05 | 0.59 | 0.59 | 0.59 |
| 0.5-1 k | 0.66 | 0.04 | 0.49 | 0.49 | 0.49 |
| 1-2 k | 0.51 | 0.02 | 0.32 | 0.32 | 0.32 |
| 2-4 k | 0.29 | 0.00 | 0.08 | 0.07 | 0.08 |
| 4-8 k | 0.08 | 0 | 0.00 | 0.00 | 0.00 |
| > 8 k | ≤ 0.01 | 0 | 0 | 0 | 0 |

The gated share of true charge set to 0, by neighbourhood density:

| iso | ≤ 10 k | 10-100 k | 100 k-1 M | 1-4 M | > 4 M |
|---|---|---|---|---|---|
| 0.33 | 0.25 | 0.09 | 0.016 | 0.005 | 0.002 |

In 1-2 k cells, by plane:

| | U | V | Y |
|---|---|---|---|
| gated | 0.38 | 0.45 | 0.13 |
| classical tuned | 0.66 | 0.76 | 0.11 |

The density x charge map is in `gating_zeroed_v2.png`.

What these numbers say:
- **Gating zeroes the faint (< 2 k e-) cells of sparse surroundings.** In dense activity it zeroes almost nothing.
- **On induction it zeroes about 60% of what the classical ROI gate zeroes. On collection it is level with the classical
  chain.**
- **Per cell, the gated maps inherit the classical failure mode below 2 k e-, at a lower rate.** The median gated ratio
  of < 1 k e- cells in sparse surroundings is 0 (the raw model gave 0.1-0.4).
- **Above 4 k e- gated and raw are identical.** In 2-4 k cells gating zeroes 4-17% of the charge, depending on density.
- **The three presence heads agree almost cell for cell:** same > 0.5 decision on 99.8% of covered cells, M4n vs M5d vs
  M5dc. Hence the identical zeroed fractions. These are decisions about the data, not about the heads.

**The dense-activity wins survive gating.** Every dense-region margin of `dense.md` is unchanged within 0.01:

| dense-region win | classical tuned | gated model |
|---|---|---|
| per-cell error, 4-16 k cells, > 1 M density | 0.27-0.57 | 0.07-0.15 |
| bright false cells | — | 60-150x fewer than classical |
| induction charge in bright regions (16 x 8, 1-4 M and > 4 M; U/V) | 0.75-0.86 | 0.96-0.99 |
| resolved double peaks along wire / time | 0.68 / 0.75 | 0.89 / 0.95 (M5d gated) |
| per-particle charge, > 1 MeV, 68% half-width | 0.06-0.14 | 0.01-0.06 |

Cell detection is a property of the presence score, so it is unchanged (§3).

## 1. Charge fidelity on charged cells (all planes; `margin_v2.png`, `margin_M4ngated_v2.png`)

Median |pred − true| / true:

| density | method | 1-2 k | 2-4 k | 4-8 k | 8-16 k | 16-64 k | > 64 k |
|---|---|---|---|---|---|---|---|
| ≤ 10 k | classical tuned | 1.00 | 0.50 | 0.27 | 0.17 | 0.13 | 0.12 |
| | M4n raw / gated | 0.86 / 1.00 | 0.20 / 0.20 | 0.07 | 0.04 | 0.03 | 0.03 |
| | M5d raw / gated | 0.84 / 1.00 | 0.19 / 0.19 | 0.07 | 0.04 | 0.03 | 0.02 |
| | M5dc gated | 1.00 | 0.18 | 0.06 | 0.03 | 0.02 | 0.02 |
| 1-4 M | classical tuned | 1.00 | 0.99 | 0.49 | 0.27 | 0.10 | 0.09 |
| | M4n raw / gated | 0.49 / 0.49 | 0.21 | 0.11 | 0.07 | 0.04 | 0.03 |
| | M5d raw / gated | 0.47 / 0.47 | 0.20 | 0.10 | 0.07 | 0.04 | 0.03 |
| | M5dc gated | **0.40** | **0.18** | **0.09** | **0.06** | **0.03** | **0.02** |
| > 4 M | classical tuned | 1.00 | 1.00 | 0.57 | 0.32 | 0.14 | 0.12 |
| | M4n raw / gated | 0.53 / 0.53 | 0.29 | 0.15 | 0.10 | 0.05 | 0.04 |
| | M5d raw / gated | 0.53 / 0.53 | 0.28 | 0.15 | 0.10 | 0.05 | 0.04 |
| | M5dc gated | **0.42** | **0.25** | **0.13** | **0.08** | **0.04** | **0.03** |

Median pred/true of 1-2 k cells in > 4 M neighbourhoods:

| M4n | M5d | M5dc gated |
|---|---|---|
| 0.60 | 0.60 | 0.72 |

What these numbers say:
- **M5dc gated has the best per-cell charge at every density above 1 k e-.** The hurdle loss removes the pull of faint
  cells toward 0.
- **M5d's decoder improves on M4n only marginally at the cell level**, by up to 0.02 in error.
- **Finding: the decoder head saturates in the brightest cells** [V]. Its output layer is LayerNorm → Linear, which
  bounds the log-charge logit:

  | head | logit cap | charge cap per cell |
  |---|---|---|
  | M5d | 5.941 | 775 k e- |
  | M5dc | 6.037 | 853 k e- |

  The caps were computed from the weights, and the observed maxima sit at the caps: 772 k / 875 k. M4n's MLP head is
  uncapped, with a maximum of 2.6 M.

  Median pred/true by cell true charge:

  | cell true charge | M5d raw | M5dc gated | M4n |
  |---|---|---|---|
  | 0.5-1 M e- | 0.94 | 0.98 | — |
  | > 1 M e- (247 cells) | 0.62 | 0.70 | 0.95 |

  - M5d raw is also about 2% low across all bright cells: 0.97-0.98 above 64 k.
  - In 4 x 4 regions above 4 M e- (159 regions) the U/V sum ratio is 0.84 for M5d and 0.89 for M5dc, against 0.98 for
    M4n.
  - Few cells are affected. It is a real architectural ceiling, and a scale on the output (or removing the final
    LayerNorm) would remove it [A].

## 2. Empty cells: see question 2. Quantiles of predicted charge on empty cells (e-; 90th / 99th / 99.9th)

| density | classical tuned | M4n raw | M4n gated | M5d gated | M5dc gated |
|---|---|---|---|---|---|
| ≤ 10 k | 0 / 0 / 462 | 98 / 241 / 570 | 0 / 115 / 552 | 0 / 167 / 596 | 0 / 304 / 1,036 |
| 1-4 M | 0 / 2,163 / 7,568 | 154 / 424 / 1,030 | 0 / 400 / 1,024 | 0 / 405 / 1,030 | 0 / 669 / 1,529 |
| > 4 M | 206 / 3,563 / 12,713 | 269 / 750 / 1,585 | 154 / 750 / 1,585 | 153 / 739 / 1,585 | 188 / 1,103 / 1,911 |

The false charge that survives gating is the model's confident errors. M5dc's are about 1.5-2x brighter, because its
charge is never pulled toward 0.

## 3. Context-matched cell detection (1% FPR on empties of the same plane and density; planes pooled)

Efficiency, classical best statistic vs M5d presence (M4n presence within ±0.01 of M5d everywhere):

| density | 0.5-1 k | 1-2 k | 2-4 k | all charged cells |
|---|---|---|---|---|
| ≤ 10 k | 0.20 / 0.27 | **0.48** / 0.45 | **0.87** / 0.81 | 0.28 / 0.35 |
| 10-100 k | 0.26 / 0.38 | 0.53 / 0.55 | **0.88** / 0.85 | 0.37 / 0.48 |
| 100 k-1 M | 0.19 / 0.44 | 0.43 / 0.61 | 0.84 / 0.89 | 0.49 / 0.63 |
| 1-4 M | 0.14 / 0.48 | 0.24 / 0.64 | 0.58 / 0.90 | 0.55 / 0.73 |
| > 4 M | 0.10 / 0.41 | 0.16 / 0.58 | 0.37 / 0.85 | 0.59 / 0.76 |

What these numbers say:
- **The decoder's eval gain does not show at the cell level.** On the near-activity windows M5d presence scores 0.592
  vs 0.558. At the cell level it is level with M4n (AUC equal to 0.001). The window gain is a window-level effect, a
  better max over 16 x 8 cells; it is not better per-cell ranking [A on the mechanism].
- **The sparse-faint regime is unchanged:** classical ahead by 0.03-0.06. Per plane at ≤ 10 k:

  | plane, cell charge | classical | M5d presence |
  |---|---|---|
  | U, 2-4 k | 0.84 | 0.77 |
  | V, 2-4 k | 0.76 | 0.67 |
  | Y, 1-2 k | 0.84 | 0.75 |

- **Use the presence score for detection, not the gated charge.** Used as a detection score, the gated charge is worse
  for the faintest cells. For < 500 e- cells in > 4 M neighbourhoods:

  | M5d gated | M5dc gated | presence |
  |---|---|---|
  | 0.09 | 0.05 | 0.24 |

## 4. Region linearity, bright end (`regions_v2.png`)

Sum ratio for 16 x 8 regions, U / V / Y:

| region true charge | classical tuned | M4n gated | M5d gated | M5dc gated |
|---|---|---|---|---|
| 1-4 M | 0.86 / 0.86 / 1.00 | 0.99 / 0.99 / 0.99 | 0.97 / 0.97 / 0.98 | 0.99-1.00 |
| > 4 M | 0.76 / 0.75 / 1.00 | 0.98 / 0.98 / 0.99 | 0.96 / 0.96 / 0.97 | 0.99-1.00 |

The faint end is in question 1.

Log-log fit on regions > 4 k e- (slope / r):

| M4n raw | gated models | M5dc gated | classical tuned |
|---|---|---|---|
| 0.89 / 0.97 | 1.07-1.11 / 0.92 | 1.07 / 0.92 | 1.47 / 0.81 |

The gated slope exceeds 1 because faint regions are now low rather than high.

## 5. Per particle (884 k particle-plane entries)

Predicted / true on each particle's majority cells, median (68% half-width):

| | E 0.1-0.3 MeV | 1-3 MeV | 10-30 MeV | > 100 MeV | overlap > 0.6 |
|---|---|---|---|---|---|
| classical tuned | 0.47 (0.48) | 0.96 (0.14) | 0.93 (0.14) | 0.97 (0.06) | 0.92 (0.46) |
| M4n raw / gated | 0.88 (0.41) / 0.88 (0.46) | 0.99 (0.06) | 0.99 (0.03) | 0.99 (0.01) | 0.98 (0.15) / 0.98 (0.16) |
| M5d raw / gated | 0.89 (0.40) / 0.88 (0.44) | 0.98 (0.05) | 0.98 (0.03) | 0.98 (0.01) | 0.97 (0.15) |
| M5dc gated | **0.91 (0.39)** | **1.00 (0.04)** | **1.00 (0.02)** | **1.00 (0.01)** | **0.99 (0.14)** |

Gating costs particles almost nothing, because particles live in charged cells. M5d's 2% low bias is visible here;
M5dc is unbiased.

## 6. Close double peaks (truth peaks 2-4 cells apart; resolved fraction)

| | along wire | wire, 4 wires apart | along time | time, 32 ticks apart |
|---|---|---|---|---|
| classical tuned | 0.68 | 0.55 | 0.75 | 0.64 |
| oracle ROI | 0.85 | 0.69 | 0.86 | 0.70 |
| M4n raw / gated | 0.89 / 0.88 | 0.81 | 0.94 / 0.94 | 0.93 |
| M5d raw / gated | 0.89 / 0.89 | 0.82 | 0.95 / 0.95 | 0.93 |
| M5dc gated | 0.90 | 0.83 | 0.95 | 0.94 |

Gating does not change separation. The decoder adds at most 0.01-0.02.

## 7. Event totals (predicted / true per plane)

| | U | V | Y | per plane-event, 16-84% |
|---|---|---|---|---|
| classical tuned | 0.855 | 0.840 | 1.000 | [0.83, 1.00] |
| oracle ROI | 1.004 | 1.005 | 1.004 | [0.999, 1.010] |
| M4n raw | 1.012 | 1.015 | 1.005 | [1.004, 1.025] |
| M4n gated | 0.984 | 0.983 | 0.990 | [0.981, 0.991] |
| M5d raw | 1.000 | 1.003 | 0.993 | [0.992, 1.016] |
| M5d gated | 0.969 | 0.967 | 0.976 | [0.966, 0.977] |
| **M5dc gated** | **0.994** | **0.991** | **0.997** | **[0.990, 0.999]** |

What these numbers say:
- **Gating trades a +1-1.5% haze excess for a −1 to −3% deficit.**
  - 0.75% of the true charge is zeroed. The rest of the deficit is the slight low bias of the charge heads in charged
    cells: M5d about 2%, plus its cap.
- **M5dc gated is the best calorimeter here:** within 1% on every plane, with the narrowest event-to-event spread of any
  method except the oracle ROI.
- **Log-map correlation over the activity footprint** is 0.990-0.997 for every model output, against 0.964-0.980 for
  classical tuned.

## Displays (`displays_v2.png`)

These are the same three dense windows as `dense.md`: ev311 U, ev296 V, ev293 Y.
- **Panels:** truth, classical tuned, M4n gated and M5d gated, then the residuals of classical and of M5d gated.
- **The gated maps lose the uniform faint carpet of the raw maps.** Some confidently-predicted false patches between
  tracks remain, for example the V window near ticks 1250-1450 x wires 565-620.

Median |residual| on charged cells and window total:

| window | classical tuned | M5d gated | M4n raw (dense.md) |
|---|---|---|---|
| ev311 U | 3,464 e-; 48.6 M | 1,292 e-; 61.4 M | 1,320 e- |
| ev296 V | 2,936 e-; 45.1 M | 1,094 e-; 57.4 M | 1,037 e- |
| ev293 Y | 901 e-; 24.4 M | 529 e-; 24.0 M | 505 e- |

Window truth totals are 63.5 M, 61.0 M and 24.6 M.

## Synthesis

| regime | best | margin vs classical tuned |
|---|---|---|
| sparse (≤ 100 k), faint 1-4 k cells, detection | classical | 0.03-0.06 efficiency, unchanged by gating or decoder |
| sparse, faint false charge | classical | 4-6x fewer cells above 100 e- (was 20-50x before gating) |
| 100 k-4 M density, faint false charge | **gated model** | 0.46-0.8x the classical rate |
| shower cores, faint false charge | tie | 0.11 vs 0.12 |
| bright false cells, any density | **model** (gated or raw) | 30-150x fewer |
| faint regions (< 16 k e- per 32 x 128) | **M5dc gated** | 0.94-1.00 vs 0.36-0.56; M4n/M5d gated 0.64-0.87; raw 1.8-3.3 |
| dense per-cell charge | **M5dc gated** | error 0.09-0.25 vs 0.49-1.00 at 2-8 k (> 1 M density) |
| induction calorimetry, bright regions / particles / events | **M5dc gated ≈ M4n** | 0.99 vs 0.75-0.86 |
| close double peaks | **model** (all variants) | 0.89-0.95 vs 0.68-0.75 |
| faint true charge kept | **raw model** (0.06% zeroed); gated 0.75%; classical 1.47% | — |
| brightest cells (> 0.5 M e-) | **M4n** | M5d/M5dc capped at 0.78 / 0.85 M e- per cell |

**Recommended map.** M5dc gated (hurdle loss + gating) fixes both faint-end failures of the raw maps: region
over-estimate and haze. It keeps every dense-activity win and is the most linear and least biased output here. It needs
two caveats:
- the decoder's output cap above about 0.8 M e- per cell;
- 2-3x more bright false cells than M5d, still 30x below classical.

The cost of gating is localized: about a third to three-quarters of the charge in < 2 k e- cells in sparse
surroundings. That is about 0.75% of all charge, half of what the classical ROI gate drops. Raw presence remains the
right detection score.

## Infrastructure and validity notes

- **Hung OST 61.** I checked every /pscratch input with `lfs getstripe` (metadata only, no data read). Four of my
  earlier intermediates sit on OST 61; three of them are relevant here:
  - `dense/cls_fit/ev275_g5` and `ev301_g2`: recomputed with `cls_maps.py --reuse v2/cls_full` (their cls_full sources
    are on other OSTs) into CFS `dense2/cls_fix`;
  - `dense/model_m2n/ev382`: M2n is not part of this rerun.

  None of the truth_v2 / truth_v2_near test files, the ptruth files, the 38 test-event corpus shards or the M4n
  checkpoint is on OST 61. The models read the CFS corpus copy. Every new output is on CFS:
  `/global/cfs/cdirs/m5238/users/oalter/helix_runs/dense2/` (`model`, `rec`, `cls_fix`, logs).
- **Model reproduction** [V]. `dense/model_dump2.py` reproduces every window score of eval_denoise's
  `dn_full_{M4n_ft_full_near, M5d_ft_full_dec, M5dc_ft_full_dec_cond}{,_pres,_gated}_trained_windows.json` exactly on
  all 128 test events (26,674 windows each, max rel. diff 0).
- **Noise.** The classical chain uses seed 1000 + ev and the models use the corpus realisation, as in `dense.md`.

## Reproduce

```bash
H=baselines/uboone_sp/dense; C=/global/cfs/cdirs/m5238/users/oalter/helix_runs/dense2
$H/run_dump2.sh                                                     # M4n/M5d/M5dc maps, 4 GPUs -> $C/model
scripts/helix_run.sh python $H/cls_maps.py --events 275-275 --planes 5 --reuse /pscratch/.../v2/cls_full --out $C/cls_fix   # + 301 g2
scripts/helix_run.sh python $H/collect2.py --cls $C/cls_fix,/pscratch/sd/o/oalter/helix_work/uboone_sp/dense/cls_fit --out $C/rec
scripts/helix_run.sh python $H/analyze.py --rec $C/rec --out $H/results/dense_v2.json
scripts/helix_run.sh python $H/tables2.py --tag v2 > $H/results/tables_v2.md
scripts/helix_run.sh python $H/figs2.py --tag v2 --cls $C/cls_fix,/pscratch/.../dense/cls_fit --model $C/model
```
