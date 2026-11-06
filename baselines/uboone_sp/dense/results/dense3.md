# Dense / full-spectrum comparison, v3: M6d vs M6dc (decoder charge cap fixed)

2026-10-09. This page repeats `dense2.md` for the retrained decoder models:
- **M6d**: `M6d_ft_full_dec/final.pt`, scored raw, presence and gated.
- **M6dc**: `M6dc_ft_full_dec_cond/final.pt`, hurdle loss, scored presence and gated.

Both use CellDecoder with a linear readout of the residual stream (commit 87b4caa).

M4n gated and classical tuned are scored alongside on the same cells. The old M5d/M5dc numbers are quoted from `dense_v2.json`; their checkpoints no longer load.

Everything else is unchanged from dense2: the same 128 test events, 2 x 16 fine cells, strata, metrics and 200-sample event bootstrap, and the same classical maps (fitted kernel; dense2 cls_fix plus dense cls_fit).

Event-bootstrap 16-84% intervals are ±0.001-0.01 on every number quoted here. Full tables are in `tables_v3.md`, the JSON in `dense_v3.json`, figures in `../figs/*_v3.png`. Labels: [V] = verified by running; [A] = assumed.

**Reproduction [V].** `model_dump2.py` reproduces every window score of `dn_full_{M6d_ft_full_dec, M6dc_ft_full_dec_cond}{,_pres,_gated}_trained_windows.json`. That is 26,674 windows per tag over all 128 events, with max relative difference 0.

## Verdict per use

| use | pick | why (numbers below) |
|---|---|---|
| detection / triggering on faint deposits | **M6d presence** (M6dc presence is equivalent) | cell-level presence is identical: AUC equal to 0.001, efficiency ±0.01 in every density x charge bin; the eval's window-level 0.591 vs 0.579 is the only difference |
| a single map used both as a detector and as charge (gated map as the score) | **M6d gated** | at 1% FPR, all charged cells: 0.42 vs 0.38 (10-100 k), 0.69 vs 0.65 (1-4 M); faint cells in showers 0.09 vs 0.05 |
| clean empty space / fewest confident false hits | **M6d gated** | empty cells above 1 Q0: 0.00037 vs 0.00077 in > 4 M (2x); 99.9th percentile 596 vs 1,030 e- (≤ 10 k); empty 32 x 128 regions above 1 Q0: 0.35% vs 0.80% |
| calorimetry of faint / small clusters (< 16 k e- per region) | **M6dc gated** | 32 x 128 regions: 0.91 / 0.98 vs 0.67 / 0.87 (1-4 k / 4-16 k); 8 x 64 regions: 0.69 / 0.92 vs 0.59 / 0.89 |
| per-cell charge resolution, dense activity | **M6dc gated** | > 4 M neighbourhoods, median relative error 2-4 k / 4-8 k / 8-16 k: 0.25 / 0.13 / 0.08 vs 0.28 / 0.15 / 0.10; 68% half-width 0.38 / 0.21 / 0.13 vs 0.44 / 0.23 / 0.15 |
| per-particle charge | **M6dc gated** (narrowly) | > 1 MeV: median 0.997-1.001, half-width 0.010-0.045, vs M6d 1.005-1.008 and 0.013-0.055 |
| plane / event totals | **tie** | M6d gated 1.000 / 0.995 / 1.005 (U / V / Y); M6dc gated 0.999 / 0.994 / 0.999. M6d's total comes from cancellation (below) |
| bright cells and bright regions | **tie** (cap fix works for both) | cells > 1 M e-: 0.96 / 0.98 (was 0.62 / 0.70); 8 x 64 regions > 4 M on U/V: 1.00 / 0.99 (was 0.84 / 0.89) |
| close double peaks | **M6dc** (marginal) | resolved along wire 0.898 vs 0.886; 4 wires apart 0.833 vs 0.817 |

**Bottom line.** The calorimetry gap from dense2 holds for M6d vs M6dc.

- **Faint-region linearity:** M6dc still wins by 0.11-0.24 in the 1-16 k region bins.
- **Dense per-cell resolution:** M6dc still wins by 10-15%.
- **Plane totals:** the gap closes, but for the wrong reason. M6d's charge head now over-predicts by about 0.4% in charged cells. That cancels the 0.74% gating loss plus its leftover haze.
- **Detection and false hits:** M6d wins on the gated-map detection score and has half as many confident false hits.
- **Recommended set:** M6d presence for detection, M6dc gated for calorimetry. If one model must be chosen:
  - M6d for a detection/reconstruction product;
  - M6dc for an energy-measurement product.

  On dense-activity physics the difference between the two is small. Both beat classical tuned by the same large margins as in dense2.

## 1. Did the cap fix change anything? Yes: bright cells and bright regions are now unbiased [V]

Median pred/true by cell true charge, all planes:

| cell true charge | n | classical tuned | oracle ROI | M4n | M5d raw (capped) | M5dc gated (capped) | **M6d raw / gated** | **M6dc gated** |
|---|---|---|---|---|---|---|---|---|
| 64-128 k | 333 k | 0.909 | 0.943 | 0.991 | 0.977 | 0.997 | 1.006 | 0.998 |
| 128-256 k | 122 k | 0.897 | 0.940 | 0.988 | 0.972 | 0.996 | 1.005 | 0.999 |
| 0.5-1 M | 2.5 k | 0.843 | 0.932 | 0.973 | 0.937 | 0.983 | 1.002 | 1.000 |
| 1-4 M | 247 | 0.769 | 0.911 | 0.951 | **0.619** | **0.698** | 0.958 | 0.975 |

The maximum predicted cell charge is now 2.17 M e- (M6d) and 2.46 M e- (M6dc), against 0.77 M and 0.88 M before. M4n reaches 2.62 M.

Sum ratio of regions with true charge above 4 M e-, U / V:

| region | M5d gated | M5dc gated | M6d gated | M6dc gated | M4n gated | classical tuned |
|---|---|---|---|---|---|---|
| 8 x 64 | 0.84 / 0.83 | 0.89 / 0.89 | **1.00 / 0.99** | **0.99 / 0.99** | 0.98 / 0.98 | 0.60 / 0.62 |
| 32 x 128 | 0.96 / 0.96 | 0.99 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 | 0.98 / 0.98 | 0.76 / 0.75 |

What these numbers say:
- **The fix also removed M5d's general ~2% low bias.** M5d was low across all bright cells (0.97-0.98 above 64 k e-); M6d is +0.2-0.6%.
- **Per particle the fix moves both models by about +2.5%.** M6d is 1.005-1.008 against M5d's 0.979-0.986.
- **Nothing else moved beyond 0.01** between M5x and M6x: detection, false charge, separation and zeroed charge.

## 2. M6d vs M6dc on the dense2 tables

### 2a. Faint-region linearity

Sum ratio pred/true, all planes:

| region true charge | M6d raw | M6d gated | M6dc gated | M4n gated | classical tuned | oracle ROI |
|---|---|---|---|---|---|---|
| 32 x 128, 1-4 k | 3.10 | 0.67 [0.668-0.679] | **0.91** [0.903-0.917] | 0.64 | 0.36 | 1.24 |
| 32 x 128, 4-16 k | 1.86 | 0.87 | **0.98** | 0.85 | 0.56 | 1.12 |
| 32 x 128, 16-64 k | 1.20 | 0.94 | 0.96 | 0.93 | 0.81 | 1.07 |
| 8 x 64, 1-4 k | 1.15 | 0.59 | **0.69** | 0.58 | 0.44 | 1.27 |
| 8 x 64, 4-16 k | 1.04 | 0.89 | **0.92** | 0.89 | 0.73 | 1.16 |

Empty 32 x 128 regions:

| | above 1 Q0 | mean predicted |
|---|---|---|
| M6d gated | 0.35% | 27 e- |
| M6dc gated | 0.80% | 48 e- |
| M6d raw | 6.5% | 475 e- |
| classical tuned | 0.16% | 91 e- |

What these numbers say:
- **The gated M6d map under-reads small clusters**, much like M4n gated.
- **M6dc is near-linear from 4 k e-.** Its cost is about 2x more charge in empty regions.

### 2b. Per-cell charge

All planes, median |pred-true|/true, with the 68% half-width in brackets:

| density | method | 1-2 k | 2-4 k | 4-8 k | 8-16 k | 16-64 k | > 64 k |
|---|---|---|---|---|---|---|---|
| 100 k-1 M | M6d gated | 0.54 | 0.18 (0.36) | 0.08 | 0.05 | 0.03 | 0.02 |
| | M6dc gated | **0.46** | **0.16 (0.31)** | **0.07** | **0.04** | 0.02 | 0.02 |
| 1-4 M | M6d gated | 0.48 | 0.20 (0.37) | 0.10 | 0.07 | 0.03 | 0.03 |
| | M6dc gated | **0.40** | **0.18 (0.32)** | **0.09** | **0.06** | 0.03 | 0.02 |
| > 4 M | M6d gated | 0.53 | 0.28 (0.44) | 0.15 | 0.10 | 0.05 | 0.04 |
| | M6dc gated | **0.43** | **0.25 (0.38)** | **0.13** | **0.08** | 0.04 | 0.03 |
| > 4 M | classical tuned | 1.00 | 1.00 (1.22) | 0.57 | 0.32 | 0.14 | 0.12 |

What these numbers say:
- **Median ratio of 1-2 k cells in > 4 M neighbourhoods:** M6dc 0.71, M6d 0.60. The hurdle loss stops faint charged cells being pulled toward 0.
- **In sparse surroundings (≤ 10 k) the two are within 0.01.**

### 2c. Empty cells

Fraction above 100 e- / above 1 Q0:

| density | M6d gated | M6dc gated | M4n gated | M6d raw | classical tuned |
|---|---|---|---|---|---|
| ≤ 10 k | 0.012 / 0.00001 | 0.012 / 0.00007 | 0.010 / 0.00001 | 0.097 / 0.00001 | 0.0019 / 0.00001 |
| 100 k-1 M | 0.019 / 0.00004 | 0.020 / 0.00016 | 0.018 / 0.00003 | 0.180 / 0.00004 | 0.025 / 0.0034 |
| > 4 M | 0.107 / 0.00037 | 0.106 / **0.00077** | 0.109 / 0.00037 | 0.417 / 0.00037 | 0.124 / 0.0218 |

99th / 99.9th percentile of predicted charge on empty cells (e-):

| density | M6d gated | M6dc gated |
|---|---|---|
| ≤ 10 k | 166 / 596 | 283 / 1,030 |
| > 4 M | 728 / 1,571 | 1,097 / 1,926 |

What these numbers say:
- **The faint haze after gating is the same for both models.**
- **The confident false hits that survive gating are about 2x more frequent and about 1.5x brighter for M6dc.** Its charge is never regressed toward 0, so a false presence decision carries full charge.
- **Both are 30-60x below classical tuned** for empty cells above 1 Q0 in dense activity.

### 2d. Detection

The threshold is set at 1% FPR on empty cells of the same plane and density stratum, and planes are pooled. Each entry is efficiency on 1-2 k cells / 2-4 k cells / all charged cells:

| density | classical best statistic | M6d presence | M6dc presence | M6d gated charge | M6dc gated charge |
|---|---|---|---|---|---|
| ≤ 10 k | 0.48 / 0.87 / 0.28 | 0.45 / 0.81 / 0.35 | 0.44 / 0.80 / 0.35 | 0.45 / 0.81 / 0.34 | 0.44 / 0.80 / 0.31 |
| 100 k-1 M | 0.43 / 0.84 / 0.49 | 0.61 / 0.89 / 0.63 | 0.61 / 0.88 / 0.63 | 0.62 / 0.90 / 0.59 | 0.60 / 0.89 / 0.56 |
| > 4 M | 0.16 / 0.37 / 0.59 | 0.58 / 0.84 / 0.76 | 0.58 / 0.85 / 0.76 | 0.55 / 0.86 / 0.72 | 0.47 / 0.83 / 0.69 |

What these numbers say:
- **Presence scores are interchangeable.** AUC per plane agrees to 0.001.
- **As a detection score, the M6d gated map beats the M6dc gated map** in every density bin: 0.02-0.05 on all charged cells, and up to 0.08 on 1-2 k cells in showers.
- **This matches the eval's gated-map detection gap** (0.566 vs 0.416), though it is smaller at the cell level.
- **Sparse faint cells still favour classical** by 0.03-0.06 (dense.md, dense2.md).

### 2e. Zeroed true charge

The models are identical here. Both gated maps zero 0.74% of all true charge (M4n gated 0.75%, classical tuned 1.47%), split by cell charge as:

| < 500 e- | 0.5-1 k | 1-2 k | 2-4 k |
|---|---|---|---|
| 0.59 | 0.49 | 0.32 | 0.08 |

The three presence heads make the same > 0.5 decisions.

### 2f. Plane totals, decomposed [V]

| | total | charged cells | empty cells | ratio on charged cells kept |
|---|---|---|---|---|
| classical tuned | 0.898 | 0.889 | +0.009 | 0.902 |
| M4n gated | 0.985 | 0.982 | +0.003 | 0.990 |
| M6d raw | 1.026 | 0.998 | +0.027 | 0.999 |
| **M6d gated** | **1.000** | 0.997 | +0.003 | **1.004** |
| **M6dc gated** | **0.997** | 0.992 | +0.006 | **0.999** |

Per-plane totals:

| | U | V | Y | 16-84% per plane-event |
|---|---|---|---|---|
| M6d gated | 1.000 | 0.995 | 1.005 | [0.995, 1.007] |
| M6dc gated | 0.999 | 0.994 | 0.999 | [0.994, 1.001] |

What these numbers say:
- **M6d's exact total is a cancellation.** It combines +0.4% over-prediction on kept charged cells, −0.74% lost to gating and +0.3% haze. The M6d raw charge head over-predicts by 2.6% in total; most of that excess (2.7%) is haze in empty cells.
- **M6dc is unbiased where it predicts** (0.999). Its −0.3% total is the gating loss partly refilled by its false charge.
- **For a calibrated energy scale, M6dc's components are the cleaner ones** [A on how that matters downstream]. On totals alone the two tie.

### 2g. Per particle (884 k particle-plane entries)

Predicted / true on each particle's majority cells, median (68% half-width):

| | 0.1-0.3 MeV | 1-3 MeV | 10-30 MeV | > 100 MeV | overlap > 0.6 |
|---|---|---|---|---|---|
| M6d gated | 0.89 (0.45) | 1.006 (0.055) | 1.007 (0.027) | 1.008 (0.013) | 0.99 (0.16) |
| M6dc gated | **0.91 (0.40)** | **0.997 (0.045)** | **1.000 (0.022)** | **1.001 (0.010)** | 0.99 (0.14) |
| M4n gated | 0.88 (0.46) | 0.995 (0.058) | 0.992 (0.028) | 0.994 (0.013) | 0.98 (0.16) |
| classical tuned | 0.47 (0.48) | 0.963 (0.142) | 0.932 (0.136) | 0.969 (0.060) | 0.92 (0.46) |

The same pattern holds on every plane: for 10-30 MeV, U / V / Y, M6d gives 1.007 / 1.003 / 1.009 and M6dc 1.001 / 0.998 / 1.001.

### 2h. Close double peaks

Resolved fraction:

| | along wire | wire, > 4 M | wire, 4 wires apart | along time | time, 32 ticks apart |
|---|---|---|---|---|---|
| M6d gated | 0.886 | 0.899 | 0.817 | 0.945 | 0.927 |
| M6dc gated | **0.898** | **0.913** | **0.833** | **0.950** | **0.933** |
| M4n gated | 0.884 | 0.898 | 0.813 | 0.943 | 0.925 |
| classical tuned | 0.677 | 0.687 | 0.552 | 0.754 | 0.636 |

## Displays (`displays_v3.png`)

These are the same three dense windows as before: ev311 U, ev296 V, ev293 Y. The panels show truth, classical tuned, M6dc gated, M6d gated and both model residuals.

| window | truth total | M6dc gated | M6d gated |
|---|---|---|---|
| ev311 U | 63.5 M | 63.9 M | 63.8 M |
| ev296 V | 61.0 M | 60.7 M | 59.9 M |
| ev293 Y | 24.6 M | 24.7 M | 24.8 M |

Median |residual| on charged cells is 1,171 / 952 / 472 e- for M6dc and 1,297 / 1,097 / 559 e- for M6d. For classical tuned it is 3,464 / 2,936 / 901 e-.

The visible difference is that M6d over-shoots along the brightest tracks (red ridges) where M6dc does not. Both keep the same confident false patches between tracks, for example V near ticks 1250-1450 x wires 565-620.

## Infrastructure

- **/pscratch reads.** The only /pscratch inputs are the truth_v2 / truth_v2_near test files, the dense cls_fit maps and the ptruth files. All of them were checked off OST 61 in the dense2 run; the two cls_fit planes on OST 61 come from CFS `dense2/cls_fix`.
- **Models.** They read the CFS corpus. M4n maps come from `dense2/model`.
- **Outputs.** All new intermediates are in `/global/cfs/cdirs/m5238/users/oalter/helix_runs/dense3/` (`model`, `rec`, `logs`, `scratch`). Nothing was written to /pscratch.

## Reproduce

```bash
H=baselines/uboone_sp/dense; C=/global/cfs/cdirs/m5238/users/oalter/helix_runs
$H/run_dump3.sh                                    # M6d / M6dc maps, 4 GPUs -> $C/dense3/model
scripts/helix_run.sh python $H/collect3.py --cls $C/dense2/cls_fix,/pscratch/sd/o/oalter/helix_work/uboone_sp/dense/cls_fit --out $C/dense3/rec
scripts/helix_run.sh python $H/analyze.py --rec $C/dense3/rec --out $H/results/dense_v3.json
scripts/helix_run.sh python $H/tables3.py --tag v3 > $H/results/tables_v3.md
scripts/helix_run.sh python $H/figs3.py --tag v3 --cls $C/dense2/cls_fix,/pscratch/sd/o/oalter/helix_work/uboone_sp/dense/cls_fit --model $C/dense3/model
```
