# Learned denoiser vs the classical Wire-Cell chain on everything: dense activity and the full charge spectrum

2026-10-08. This extends the validated benchmark (`../../results/validity.md`, `../../results/aspects.md`).
That benchmark's headline numbers are about isolated deposits: the 0.1-0.2 MeV near-activity detection
windows. This study scores every fine cell (2 wires x 16 ticks) of every plane of the 128 test events
(ev260-387). The cells are split by local activity density and by true charge, from 0 to 3.5 M e- per
cell. It also scores regions, particles, close double peaks and whole events.

Labels:
- **[V]** verified: I ran it, or I read the cited code.
- **[A]** assumed: not checked.

The full tables, with event-bootstrap 16-84% intervals on every entry, are in `tables_fit.md`. The JSON
is `dense_fit.json`. Figures are in `../figs/`. All numbers below use the fitted kernel, which v2 uses by
default. The run was repeated with the simulator's exact kernel; see "Exact kernel" near the end.

## Bottom line

Most of the charge is in dense activity, and that is where the model wins by the largest margins:
- 56% of all charged cells have more than 1 M e- of true charge within +-8 cells, roughly 50 MeV in a
  box of 34 wires x 272 ticks. Only 0.05% are truly isolated.

At the cell level, the model (M4n) beats the tuned classical chain in every density x charge bin above
2 k e-, and its margin grows with density. The classical chain is better in four places:
1. **Faint cells in sparse surroundings, for detection.** Below 10 k e- of neighbourhood charge, the
   classical detection statistic matches or beats the model at a context-matched 1% FPR:
   - 2-4 k e- cells: 0.87 vs 0.80;
   - collection 1-2 k e- cells: 0.84 vs 0.75.

   This is the isolated-deposit regime the published benchmark measured, and it agrees with it.
2. **Faint false charge.** Classical outputs are exactly 0 wherever no ROI opens. The model leaves a haze
   of about 50-270 e- per empty cell near activity (90th percentile).
3. **Region sums at the faint end.** That haze integrates. A 32-wire x 128-tick region with < 16 k e- of
   true charge is predicted 1.8-7.8x too high (median). For 6.3% of the empty regions the model predicts
   more than 1 Q0; the tuned classical chain does so for 0.16%.
4. **Collection-plane totals,** where both are unbiased: Y totals 1.000 vs 1.005.

Everything else goes to the model, and dense activity is where it pulls furthest ahead:
- **Per-cell charge.** In regions above 1 M e-, the 4-16 k e- cells:
  - median |error|/true: 0.07-0.15 (M4n) vs 0.27-0.57 (classical tuned);
  - classical median bias: +0 to +17%; with oracle ROIs +25-34%.
- **Bright false cells.** Empty cells above 1 Q0 in regions above 1 M e-: 0.007-0.04% vs 1.1-2.2%, 60-150x fewer.
- **Detecting faint cells inside showers.** Cells of 1-4 k e- in neighbourhoods above 4 M e-:
  - 2-4 k e-: 0.84 vs 0.37;
  - 1-2 k e-: 0.57 vs 0.16.
- **Induction-plane calorimetry in bright regions.** In regions above 1 M e-:
  - classical: 0.75-0.86 of the charge (U/V);
  - model: 0.98-0.99.
- **Close double peaks.** Resolved peak pairs 4 wires apart: 0.81 vs 0.55.
- **Per-particle charge.** For particles above 1 MeV, the 68% half-width is 0.01-0.06 vs 0.06-0.14.

The oracle-ROI classical chain (truth ROIs) shows the same dense-region smearing as the found-ROI
chain. So the classical per-cell limit in dense activity is the deconvolution filters, not ROI finding or
noise.

Mechanisms:
- **Classical, per cell.** Filters (Gaus_wide, σ_t ≈ 2.9 ticks, plus the wire filter) push bright charge
  into neighbouring cells [V, the oracle-ROI rung].
- **Classical, induction charge.** The found-ROI logic loses 14-25% of the charge on U/V in bright
  regions. The oracle-ROI rung recovers 1.00 [V], and fewer than 2% of the charge sits in zeroed cells
  [V]. The loss is therefore in the amplitude inside ROIs; the linear ROI baseline is the likely cause [A].
- **Model.** A learned 2D deconvolution with context gives sharp, unbiased charge. Its cost is a
  context-driven haze and faint cells regressed toward that haze.

## Setup (same cells, same events)

| method | what is scored | source |
|---|---|---|
| classical, MicroBooNE default | ROI Gaus_wide charge, U q_s1_f0, V/Y q_s1_f1 (faithful port) | `v2/classical.py` maps (`v2/cls_full`), [V] identical code |
| classical, tuned | charge: `tuned_q` (U q_s1.25_f0.25, V q_s1.5_f0, Y q_s0.67_f0); detection: `tuned_any` (U zlf06, V zor06a2, Y = tuned_q). Picks frozen from ev000-095 | `dense/cls_maps.py`; the tuned_any maps reproduce v2's window lookups exactly (max \|diff\| = 0) [V] |
| classical, oracle ROI | Gaus_wide inside truth ROIs (truth > 0, pad 10 ticks). A reference, not a detector | v2 `qor` |
| M4n | charge head and presence head, `M4n_ft_full_near/final.pt` | `v2/model` (v2/model_dump.py, reproduces eval_denoise exactly) |
| 64-label model | `le3_ft_N64/best.pt` | `v2/model` |
| M2n (from scratch) | `M2n_scratch_full_near/final.pt` | `dense/model_m2n`, from v2/model_dump.py run unchanged |

- **Model cells.** The model predicts every token-covered cell; uncovered cells are 0. Coverage is
  99.2-99.997% of charged cells for neighbourhood density above 100 k e-, and 63-94% below it
  (`tables_fit.md`, last table).
- **Truth.** Fine-cell truth is `fine_cells(hits)`, identical to `build_denoise_truth.event_cells` within
  the readout. Particle identity is hits group -> track, as in `dump_resolution_truth.py`, with energies
  from step `de` (`dense/particle_truth.py`). Per-cell totals reproduce the truth maps to float precision
  [V].
- **Noise.** The classical chain uses seed 1000 + ev, as v2 does. The model reads the corpus, which has its
  own noise draw from the same generator. In dense regions the signal dominates, so this cannot matter
  there [A]. For the faint strata the statistics are the same, but the realisations are not.

**Strata.** Density is the true charge within +-8 cells, excluding the cell itself. The bins were chosen
from the data: the median charged cell sees 1.3 M e-, the 25th percentile 250 k, the 10th 27 k.

Number of charged cells per density bin (all planes):

| iso (0) | <=10 k | 10-100 k | 100 k-1 M | 1-4 M | > 4 M |
|---|---|---|---|---|---|
| 2,519 | 253,819 | 669,464 | 1,295,004 | 1,884,734 | 903,972 |

A single MIP crossing the box deposits about 0.4-1 M e-, so the "> 4 M" bin is shower cores and
overlapping tracks.

Empty cells per density bin:

| iso (0) | <=10 k | 10-100 k | 100 k-1 M | 1-4 M | > 4 M |
|---|---|---|---|---|---|
| 136.8 M | 17.0 M | 12.7 M | 10.0 M | 4.6 M | 0.56 M |

Charge bins are those of the brief. The density x charge counts are the first table of `tables_fit.md`;
all 48 bins above "iso" hold at least 68 cells, most of them more than 10 k.

**Statistics.** Each interval is the 16-84% range over 200 event-bootstrap resamples. With about 5 M
charged cells the intervals are narrow, typically ±0.002-0.01 on efficiencies and ratios. One stratum was
checked against an explicit resample-and-recompute: 0.381 [0.376-0.385], identical to the histogram
method [V]. Differences quoted below are far outside these intervals unless stated otherwise.

**Topology.** The step and hits files carry no PDG code and no parent link: only group -> Geant4 track and
per-deposit `de`. A track/shower label could only be guessed, so I did not make one. The density strata
and the particle-overlap strata stand in for it: shower cores are the "> 4 M" and "overlap > 0.6" bins.
"Particles" here are Geant4 tracks, so a shower is hundreds of them.

## 1. Charge fidelity on charged cells (figure `margin_fit.png`, top row)

All planes. Each entry is the median pred/true, with the 68% half-width in brackets.

| density | method | 1-2 k | 2-4 k | 4-8 k | 8-16 k | 16-64 k | > 64 k |
|---|---|---|---|---|---|---|---|
| <=10 k | classical tuned | 0.00 (0.40), 68% zero | 0.51 (0.44) | 0.73 (0.25) | 0.83 (0.13) | 0.87 (0.09) | 0.88 (0.06) |
| <=10 k | M4n | 0.15 (0.49) | 0.85 (0.50) | 0.98 (0.11) | 1.00 (0.07) | 1.00 (0.04) | 1.00 (0.04) |
| 100 k-1 M | classical tuned | 0.00 (0.70), 53% zero | 0.75 (0.93) | 0.94 (0.63) | 1.03 (0.31) | 0.93 (0.09) | 0.91 (0.07) |
| 100 k-1 M | classical oracle ROI | 1.00 (0.69) | 0.98 (0.64) | 1.01 (0.53) | 1.07 (0.29) | 0.93 (0.09) | 0.92 (0.06) |
| 100 k-1 M | M4n | 0.54 (0.52) | 0.91 (0.37) | 0.98 (0.13) | 1.00 (0.08) | 1.00 (0.04) | 0.99 (0.04) |
| > 4 M | classical tuned | 0.10 (1.23), 46% zero | 0.74 (1.22) | 1.00 (0.97) | 1.13 (0.50) | 0.93 (0.20) | 0.88 (0.19) |
| > 4 M | classical oracle ROI | 1.33 (1.52) | 1.29 (1.16) | 1.34 (0.82) | 1.30 (0.50) | 1.00 (0.18) | 0.96 (0.07) |
| > 4 M | M4n | 0.60 (0.54) | 0.84 (0.44) | 0.95 (0.24) | 0.98 (0.15) | 0.99 (0.08) | 0.99 (0.06) |

Median |pred − true|/true, classical tuned vs M4n, all planes:

| density | 2-4 k | 4-8 k | 8-16 k | > 64 k |
|---|---|---|---|---|
| <=10 k | 0.50 vs 0.20 | 0.27 vs 0.07 | 0.17 vs 0.04 | 0.12 vs 0.03 |
| 1-4 M | 0.99 vs 0.21 | 0.49 vs 0.11 | 0.27 vs 0.07 | 0.09 vs 0.03 |
| > 4 M | 1.00 vs 0.29 | 0.57 vs 0.15 | 0.32 vs 0.10 | 0.12 vs 0.04 |

[V] What these numbers say:

- **The model's margin grows with density.** At 4-8 k e- the classical error doubles from sparse to
  dense (0.27 -> 0.57); the model's goes from 0.07 to 0.15. The oracle-ROI rung, which has perfect ROIs,
  degrades the same way and is biased **+29-34%** at 1-16 k e- in > 4 M neighbourhoods. Moderate cells
  next to bright ones receive filter-smeared charge (Gaus_wide σ_t ≈ 2.9 ticks; on induction the wire
  filter as well). That is a property of the deconvolution filters, not of ROI finding or noise.
- **The brightest cells.** The classical chain loses charge in its brightest cells:
  - > 64 k e-: 0.88-0.91 overall, and 0.81 on U/V in > 4 M neighbourhoods. The oracle gets 0.95.
  - The model gives 0.99 with a half-width of 0.04-0.06.
- **Faint cells (0.5-2 k e-).** Both methods are poor here, in opposite directions:
  - The classical chain zeroes 56-99% of them on U/V (11-58% on Y), falling with density. The share of
    charge in zeroed cells is in the table "Share of true charge in cells the output sets to 0".
  - The model rarely zeroes them, but regresses them toward its haze floor: median 0.1-0.6 on induction.
  - Its < 500 e- cells sit at median 0.3-1.4x truth with a half-width near or above 1. Cells that small are
    dominated by the haze.
- **Collection plane in sparse surroundings: the one per-cell region the classical chain wins.** At Y
  1-2 k e- the median |error|/true is 0.39 vs 0.76 isolated and 0.28 vs 0.32 at <=10 k. Above 2 k e- the
  model wins on Y too.
- **The other models.** M2n equals M4n to ±0.02 everywhere. The 64-label model sits between M4n and the
  classical chain: half-width 0.13-0.84, and it degrades faster with density (> 4 M, 4-8 k: 0.69 (0.84)).

## 2. Empty cells inside and near activity (figure `false_charge_fit.png`)

Fraction of truth-empty cells whose prediction exceeds 0.05 Q0 (100 e-) / 1 Q0 (2,043 e-), all planes:

| method | <=10 k | 10-100 k | 100 k-1 M | 1-4 M | > 4 M |
|---|---|---|---|---|---|
| classical default | 0.0021 / 0.00002 | 0.0047 / 0.00016 | 0.022 / 0.0030 | 0.062 / 0.0106 | 0.115 / 0.0223 |
| classical tuned | 0.0019 / 0.00001 | 0.0057 / 0.00016 | 0.025 / 0.0034 | 0.068 / 0.0109 | 0.124 / 0.0218 |
| classical oracle ROI | 0.0038 / 0.0 | 0.015 / 0.00014 | 0.038 / 0.0025 | 0.094 / 0.0095 | 0.194 / 0.0220 |
| **M4n** | 0.092 / 0.00001 | 0.231 / 0.00003 | 0.184 / 0.00003 | 0.202 / 0.00007 | 0.428 / 0.00037 |
| 64-label | 0.008 / 0.00003 | 0.032 / 0.0001 | 0.051 / 0.0003 | 0.118 / 0.0023 | 0.290 / 0.0132 |
| M2n | 0.106 / 0.00001 | 0.260 / 0.00003 | 0.207 / 0.00004 | 0.222 / 0.00008 | 0.453 / 0.00043 |

Mean predicted charge per empty cell (e-), classical tuned vs M4n:

| <=10 k | 10-100 k | 100 k-1 M | 1-4 M | > 4 M |
|---|---|---|---|---|
| 1 vs 33 | 3 vs 68 | 25 vs 59 | 82 vs 67 | 172 vs 125 |

99.9th percentile (e-), classical tuned vs M4n:

| <=10 k | 10-100 k | 100 k-1 M | 1-4 M | > 4 M |
|---|---|---|---|---|
| 462 vs 570 | 714 vs 782 | 3,812 vs 821 | 7,568 vs 1,030 | 12,713 vs 1,585 |

[V] What these numbers say:

- **Which is worse in dense regions: haze or smearing? It depends on the threshold.**
  - **Bright false charge (> 1 Q0).** The classical chain is far worse above 100 k e- (U/V/Y alike), and
    its oracle-ROI rung is just as bad. In numbers: 60-150x worse. This is filter smearing of bright neighbours into empty cells. The
    model's tails stop at about 1.6 k e- (99.9th percentile).
  - **Faint false charge (> 100 e-).** The model is worse at every density: 2-50x in sparse regions,
    3.5x in shower cores. On V, 59% of empty cells in shower cores carry more than 100 e-.
  - **Total false charge.** It crosses over at about 1 M e- of neighbourhood charge. Below that the
    model's haze adds more charge than the classical smearing; above it the classical smearing adds more.
- **The 64-label model's haze is 4-11x fainter than M4n's** below 1 M e-, but in shower cores its bright
  false cells are almost as frequent as the classical chain's (1.3%).
- **Truth-definition caveat.** The hits truth drops each group's samples below 25 e-. That is 1.7% of the
  charge, mostly in diffusion tails (`results/exact_kernel.md` [V there]; per-sample floor 24.98-25.0 e-
  confirmed here [V]).
  - Some of the "faint false charge" next to tracks is therefore real charge absent from the truth.
  - The classical chain sees it in the waveform. The model was trained to suppress it.
  - It cannot exceed 25 e- per group-sample, so it cannot explain the > 1 Q0 numbers. It may explain part
    of the > 100 e- numbers for both methods [A on the size].

## 3. Cell-level presence: context-matched detection (figure `margin_fit.png`, bottom row)

The threshold for each method is set at 1% FPR on the empty cells of the same plane and the same density
stratum; planes are then pooled. The classical score is the tuned best statistic (on U/V a significance
map, not a charge). The model score is the M4n presence head. Entries are classical / M4n:

| density | 0.5-1 k | 1-2 k | 2-4 k | 4-8 k | all charged cells |
|---|---|---|---|---|---|
| <=10 k | 0.20 / 0.27 | **0.48** / 0.44 | **0.87** / 0.80 | 1.00 / 1.00 | 0.28 / 0.35 |
| 10-100 k | 0.26 / 0.38 | 0.53 / 0.54 | **0.88** / 0.85 | 1.00 / 1.00 | 0.37 / 0.48 |
| 100 k-1 M | 0.19 / 0.44 | 0.43 / 0.61 | 0.84 / 0.89 | 0.98 / 1.00 | 0.49 / 0.63 |
| 1-4 M | 0.14 / 0.47 | 0.24 / 0.64 | 0.58 / 0.90 | 0.90 / 1.00 | 0.55 / 0.73 |
| > 4 M | 0.10 / 0.41 | 0.16 / 0.57 | 0.37 / 0.84 | 0.73 / 0.99 | 0.59 / 0.76 |

Intervals are ±0.01 or tighter. Per plane, the classical chain is ahead at <=10 k on:
- U, 2-4 k: 0.84 vs 0.76;
- V, 2-4 k: 0.76 vs 0.67;
- Y, 1-2 k: 0.84 vs 0.75.

The AUC per plane (charged vs empty, same stratum, all charges) favours the model at every density on
U/V, rising with density: U 0.70-0.94 vs 0.65-0.82. On Y it ties isolated (0.675 / 0.675) and favours the
model above that, up to 0.958 vs 0.908.

[V] What these numbers say:

- **The density dependence is the result.**
  - In sparse surroundings the full-waveform classical significance is as good as the model, or better,
    for 1-4 k e- cells. This is the isolated regime, consistent with the published benchmark: Y classical
    ahead, SNR bins tied.
  - In dense activity the classical statistic collapses. Its empty-cell scores there are raised by the
    residual of neighbouring bright charge: bipolar tails and filter leakage, against a per-wire RMS
    normalisation.
  - The model keeps 0.84-0.90 for 2-4 k e- cells inside showers.
- **N64 achieved-FPR caveat.** The 64-label presence head saturates in shower cores. Its achieved FPR
  there is 3-5%, not 1% (bf16 sigmoid ties), so its > 4 M row is optimistic.
- **The oracle-ROI rung is not a detector.** It scores 0 outside truth ROIs. Even so, its efficiency falls
  to 0.50 for 2-4 k e- in > 4 M: smearing again.

## 4. Region-level linearity (figure `regions_fit.png`)

Regions are tiles of 16 x 8 cells (32 wires x 128 ticks) and 4 x 4 cells (8 wires x 64 ticks), each with
its predicted and true summed charge. Entries are median pred/true (68% half-width) / sum ratio, all
planes, 16 x 8 tiles:

| true region charge | classical tuned | classical oracle ROI | M4n | 64-label |
|---|---|---|---|---|
| 1-4 k | 0.00 (0.40) / 0.36 | 1.16 / 1.24 | 2.73 (2.24) / 3.04 | 0.66 / 0.85 |
| 4-16 k | 0.53 (0.46) / 0.56 | 1.08 / 1.12 | 1.80 (0.68) / 1.84 | 0.92 / 0.90 |
| 16-64 k | 0.85 (0.24) / 0.81 | 1.04 / 1.07 | 1.19 (0.18) / 1.19 | 0.95 / 0.95 |
| 64-256 k | 0.95 (0.09) / 0.93 | 1.02 / 1.03 | 1.03 (0.05) / 1.03 | 0.98 / 0.98 |
| 256 k-1 M | 0.98 (0.06) / 0.96 | 1.01 / 1.01 | 1.00 (0.02) / 1.00 | 0.99 / 0.99 |
| 1-4 M | 0.96 (0.09) / 0.91 | 1.00 / 1.00 | 0.99 (0.01) / 0.99 | 0.99 / 0.99 |
| > 4 M | 0.82 (0.15) / 0.82 | 1.00 / 1.00 | 0.99 (0.01) / 0.99 | 0.96 / 0.95 |

The same, broken down by plane:

| true region charge | classical tuned, U / V / Y | M4n, U / V / Y |
|---|---|---|
| 1-4 M | 0.86 / 0.86 / 1.00 | 0.99 for every plane |
| > 4 M | 0.76 / 0.75 / 1.00 | 0.99 / 0.98 / 0.99 |

On 4 x 4 tiles the classical U/V sum ratio above 4 M is 0.60 / 0.62.

Empty 16 x 8 regions (1.12 M of them):

| method | > 1 Q0 | > 4 Q0 | mean predicted |
|---|---|---|---|
| classical tuned | 0.16% | 0.018% | 91 e- |
| M4n | 6.3% | 1.0% | 461 e- |
| 64-label | 0.66% | 0.002% | 264 e- |

[V] What these numbers say:

- **Bright regions.** The model is linear to 1% from 64 k e- up to the brightest regions, on every plane.
  - The tuned classical chain is linear on Y (0.99-1.00).
  - On induction it loses 14% (1-4 M) to 25% (> 4 M) of the charge of the brightest regions, and 38-40%
    on 8-wire x 64-tick tiles (sum ratio 0.60 / 0.62).
  - The oracle-ROI rung gets 1.00, and fewer than 2% of the true charge in dense regions sits in cells
    the classical chain zeroes. So the loss is in the amplitude inside the found ROIs [V].
  - Induction `apply_roi` subtracts a straight line between the ROI ends. In dense activity those ends
    sit on neighbouring signal, which is the likely cause [A]. The MicroBooNE default shows the same loss,
    so it is not an artefact of tuning.
- **Faint regions.** The model is badly non-linear at the low end on 32 x 128 regions. Its haze, about
  50-150 e- per cell over 128 cells, adds 5-20 k e- to any region near activity: 1-4 k e- regions come out
  at 2.7x. On 8 x 64 tiles the model is linear from 4 k e- (1.06, half-width 0.20); the classical chain
  needs > 16 k e-.
  - For calorimetry of small clusters near activity, the model's haze is a real liability.
  - The 64-label model and the oracle-ROI rung are the closest to linear at the low end.
- **Log-log fit, regions > 4 k e-** (slope / r):

  | method | 16 x 8 | 4 x 4 |
  |---|---|---|
  | M4n | 0.89 / 0.970 | 1.00 / 0.988 |
  | classical tuned | 1.47 / 0.81 | 1.45 / 0.78 |
  | classical tuned, Y only | 1.04 / 0.99 | - |

  The classical slope is pulled up by zeroed faint regions.

## 5. Per-particle charge in dense and overlapping regions (figure `particles_separation_fit.png`, left)

There are 884 k particle-plane entries, counting each Geant4 track with at least 1 Q0 of its own charge
on a plane. The score is the predicted charge summed over the particle's majority cells, divided by the
true charge in those cells. Entries are median (68% half-width):

| | classical tuned | classical oracle ROI | M4n | 64-label |
|---|---|---|---|---|
| E 0.1-0.3 MeV (437 k) | 0.47 (0.48) | 0.98 (0.36) | 0.88 (0.41) | 0.36 (0.53) |
| E 1-3 MeV | 0.96 (0.14) | 0.99 (0.16) | 0.99 (0.06) | 1.00 (0.32) |
| E 10-30 MeV | 0.93 (0.14) | 0.99 (0.04) | 0.99 (0.03) | 0.99 (0.18) |
| E > 100 MeV (5.2 k) | 0.97 (0.06) | 0.99 (0.02) | 0.99 (0.01) | 0.99 (0.08) |
| overlap < 0.1 (424 k) | 0.71 (0.49) | 0.97 (0.19) | 0.97 (0.29) | 0.75 (0.56) |
| overlap > 0.6 (327 k) | 0.92 (0.46) | 1.01 (0.47) | 0.98 (0.15) | 0.88 (0.53) |
| E > 100 MeV and overlap > 0.6 | 0.89 (0.12) | 0.98 (0.03) | 0.99 (0.02) | 1.01 (0.12) |

Overlap is the fraction of the particle's own charge that lies in cells where other particles carry at
least 10%.

[V] What these numbers say:

- **Overlapping particles.** The model's per-particle charge is unbiased (0.98-0.99) and its spread
  shrinks with overlap: 0.15 at overlap > 0.6, against 0.46 for the classical chain.
- **Induction.** The classical per-particle charge carries the same induction loss as above: 0.86-0.87
  on U/V for 10-100 MeV particles, against 0.99 on Y.
- **The faintest particles.** For 0.1-0.3 MeV particles the oracle-ROI rung is best (0.98). The model is
  biased low (0.88), and the found-ROI chain loses half of them (0.47).
- **Low-overlap particles are mostly faint fragments.** This explains the low classical median in the
  "overlap < 0.1" row.

## 6. Close-deposit separation in dense regions (figure `particles_separation_fit.png`, right)

A truth double peak is two 1D local maxima of at least 1 Q0, 2-4 cells apart along wire or time, with a
truth valley at most 0.5x the weaker peak. There are 100 k such pairs along wire and 234 k along time,
almost all in > 100 k neighbourhoods. A pair is **resolved** when both predicted peaks exceed 0.25 Q0 and
the predicted valley is below 0.5x the weaker predicted peak.

Resolved fraction, all planes; "given found" in brackets:

| | classical tuned | classical oracle ROI | M4n | 64-label |
|---|---|---|---|---|
| along wire, all | 0.68 (0.87) | 0.85 (0.86) | **0.89 (0.95)** | 0.60 (0.80) |
| along wire, > 4 M | 0.69 (0.85) | 0.83 (0.84) | **0.90 (0.94)** | 0.60 (0.75) |
| along wire, 2 cells (4 wires) apart | 0.55 | 0.69 | **0.81** | - |
| along time, all | 0.75 (0.90) | 0.86 (0.86) | **0.94 (0.98)** | 0.77 (0.91) |
| along time, 2 cells (32 ticks) apart | 0.64 | 0.70 | **0.93** | - |

[V] What these numbers say:

- **The model wins in every density bin above 10 k e-, on every plane and on both axes,** including the
  "given found" rows. The classical chain leads only in the near-empty <=10 k bin (92 pairs along wire,
  pooled: given found 1.00 vs 0.92; on Y resolved 1.00 vs 0.97).
- **On Y along wire** the classical chain reaches 0.89 resolved; the model reaches 0.96.
- **The oracle rung shows the cause.** Its "given found" rate (0.84-0.86) is below the model's, so even
  perfect ROIs leave valleys filled by the filters.
- **Caveat.** A 2 x 16 grid, and a learned prior that knows what deposits look like, both favour the
  model on this metric. The truth itself is the arbiter here, and the model is closer to it.

## 7. Event level

Total predicted / true charge, with the 16-84% interval for M4n:

| | U | V | Y |
|---|---|---|---|
| classical tuned | 0.855 | 0.840 | 1.000 |
| classical default | 0.868 | 0.839 | 0.973 |
| classical oracle ROI | 1.004 | 1.005 | 1.004 |
| **M4n** | 1.012 [1.011-1.013] | 1.015 | 1.005 |
| 64-label | 0.979 | 0.984 | 0.992 |
| M2n | 1.025 | 1.030 | 1.018 |

Spread of the per-plane-event ratio (16-84%), all planes:
- classical tuned [0.83, 1.00];
- M4n [1.004, 1.025].

Median per-plane correlation of log1p(q/Q0) over the activity footprint (cells within ±8 of charge):

| method | U / V / Y |
|---|---|
| classical tuned | 0.968 / 0.964 / 0.980 |
| M4n | 0.992 / 0.991 / 0.996 |

[V] What these numbers say:
- **Induction totals.** The model's induction totals are right to 1.5%; the classical chain's are low by
  15%.
- **Collection totals.** On collection both are right to 0.5%, and the classical chain is slightly closer
  (1.000 vs 1.005).
- **M2n** over-predicts by 2-3%. That is the most visible cost of training from scratch at the cell level.

## Exact kernel (`dense_exact.json`, `tables_exact.md`, `../figs/*_exact.png`)

The whole classical side was re-run from the gated waveform with `--kernel exact`: the simulator-derived
drift-averaged kernel `response/exact_<P>.npz` from `v2/exact_kernel.py`. That covers the decon, the ROI
chain and the statistics, with the fitted-kernel picks frozen. My exact-kernel maps reproduce the other
agent's exact-kernel window lookups (`v2/cls_exact`) exactly: max |diff| = 0 for tuned_q, tuned_any,
the oracle and the default on ev260 [V].

**Nothing in this study moves by more than 0.02** [V]:

| quantity | fitted kernel | exact kernel |
|---|---|---|
| median rel. error, 4-8 k cells, > 4 M (classical tuned) | 0.57 | 0.59 |
| classical tuned, 1-2 k cells, efficiency in > 4 M | 0.16 | 0.17 |
| classical tuned, 2-4 k cells, efficiency in > 4 M | 0.37 | 0.37 |
| classical tuned, empty cells > 1 Q0 in > 4 M | 2.18% | 2.26% |
| classical tuned, bright-region sum ratio (> 4 M), U / V / Y | 0.76 / 0.75 / 1.00 | 0.77 / 0.76 / 1.01 |
| classical tuned, resolved double peaks, wire / time | 0.68 / 0.75 | 0.67 / 0.75 |
| classical tuned, particles 10-30 MeV | 0.93 (0.14) | 0.94 (0.14) |

Totals rise by 0.3-1.2%. Examples: oracle ROI 1.004 -> 1.009-1.015, tuned Y 1.000 -> 1.011.
- This is the expected sign. The exact kernel does not absorb the 1.7% of charge the hits truth drops
  below 25 e-, while the least-squares fit partly did.
- On the exact kernel the classical Y total is therefore 1.011 vs M4n's 1.005, and the collection-total
  "tie" holds either way.

**Every conclusion above is kernel-independent.**

## Synthesis: where each method wins, by how much, and why

| region of density x charge | winner | margin | mechanism |
|---|---|---|---|
| sparse (<= 100 k e- around), faint 1-4 k cells, **detection** | classical (tuned statistic), especially Y | 0.03-0.09 in efficiency at 1% FPR | full-waveform matched significance at the noise limit; the model reads a sparsified input (corpus threshold 3.35-3.9σ per band) and pays for it at threshold |
| sparse, empty cells, **faint false charge** | classical | 2-50x fewer cells above 100 e- | the ROI gate outputs exact zeros; the model's context prior leaves 50-150 e- of haze |
| faint **regions** (< 16 k e- per 32 x 128) | neither; classical under, model over | model 1.8-7.8x high, classical 0-0.56x | haze integrates over 128 cells; classical zeroes sub-threshold charge |
| collection totals | tie | 1.000 vs 1.005 | unipolar response; both unbiased |
| dense (> 1 M e- around), **per-cell charge** at 2-64 k | **model** | median rel. error 2-5x smaller; classical biased up to +17% at 4-16 k (oracle ROI +25-34%) | filter smearing (Gaus_wide, wire filter) of bright neighbours; present with oracle ROIs |
| dense, **bright false cells** (> 1 Q0) | **model** | 60-150x fewer | same smearing |
| dense, **faint-cell detection** (0.5-4 k) | **model** | +0.3 to +0.5 efficiency at 1% FPR | neighbouring bright charge's residual raises the classical significance of empty cells; the model uses 2D context across planes and scales |
| bright induction **regions / particles / events** | **model** | classical loses 14-25% (U/V); model ±1-2% | found-ROI amplitude loss on induction [A: linear ROI baseline] |
| brightest cells (> 64 k) | **model** | 0.99 vs 0.81-0.91 | smearing out of the brightest cells |
| close double peaks (4-8 wires, 32-64 ticks) | **model** | 0.81-0.93 vs 0.55-0.75 resolved | filters fill the valley (oracle ROIs also 0.69-0.70 at 2 cells) |
| faint particles (0.1-0.3 MeV) | **model** over found-ROI classical (0.88 vs 0.47); oracle ROI best | - | ROI threshold vs regression bias |

**The published claims, re-read against this study.**
- **"Model per-cell resolution 2-5x tighter above 4 k e-": holds, and widens in dense activity** (up to
  4x in shower cores at 4-8 k).
- **"Classical zeroes most 1-2 k e- induction cells": holds.** That share falls with density (U 1-2 k:
  89% at <=10 k -> 56% in > 4 M) because in dense regions ROIs are open anyway. There the faint cells are smeared instead.
- **"Model has more faint haze, far fewer bright false cells": holds at every density.** The bright-cell
  advantage reaches 60-150x in dense activity.
- **"Classical localises better": an isolated-deposit statement that does not extend to dense activity.**
  There the classical outputs fill valleys, and the model resolves double peaks better (0.89 vs 0.68
  along wire).
- **"Model 0.579 vs 0.516 near-activity detection": the cell-level analogue is density-dependent.**
  Classical >= model below 10 k e- of neighbourhood charge; the model is far ahead in dense activity.

**What the model should fix.**
- The haze: region sums near activity and 20-60% of empty cells above 100 e- in dense regions.
- The regression of 0.5-2 k e- cells toward the haze floor.

**What the classical chain cannot fix by tuning.**
- Filter smearing in dense regions; it is present with oracle ROIs.
- The induction ROI amplitude loss. It is present in the MicroBooNE default and the tuned chain alike;
  only oracle ROIs avoid it.

## Figures (`../figs/`; `_fit` = fitted kernel, `_exact` = exact kernel)

- **`margin_fit.png`.** Heatmaps over density x cell charge, per plane and pooled.
  - Top row: median |pred − true|/true, classical tuned (upper number) vs M4n (lower number). The colour is
    the difference; blue means the model is better.
  - Bottom row: efficiency at 1% FPR set on empties of the same stratum, classical best statistic vs M4n
    presence.
- **`false_charge_fit.png`.** Fraction of empty cells above 0.05 Q0 and above 1 Q0, against density, for
  every method. Bands are 16-84%. Points drawn at 1e-7 are zero.
- **`regions_fit.png`.** Predicted / true region charge (median, 68% band; dotted = sum ratio) against true
  region charge, for 32 x 128 and 8 x 64 regions, per plane.
- **`particles_separation_fit.png`.**
  - Per-particle pred/true against energy and against overlap.
  - Resolved double-peak fraction along wire and along time, against density at the valley. Dotted =
    given both found.
- **`displays_fit.png`.** The densest 40 x 40-cell windows (80 wires x 640 ticks) of the test set, one per
  plane type:
  - ev311 U, ev296 V, ev293 Y (`displays_fit.json`);
  - panels: truth, classical tuned, M4n, and the residuals (shared LogNorm in e-, symlog residuals).

  Median |residual| on the charged cells, classical vs M4n:

  | | U | V | Y |
  |---|---|---|---|
  | classical tuned | 3,464 e- | 2,936 e- | 901 e- |
  | M4n | 1,320 e- | 1,037 e- | 505 e- |

  What the displays show:
  - Window totals: classical 48.6 / 45.1 / 24.4 M e- against truth 63.5 / 61.0 / 24.6 M e-. M4n gives
    63.1 / 59.1 / 24.3 M e-.
  - The classical induction residual is dominated by deficits along the bright tracks and by unrecovered
    moderate charge between them.
  - The model's residual is a faint positive haze between tracks, with small deficits at the very
    brightest cells.

## Threats and limits

- **Different noise realisations** (classical: seed 1000 + ev; model: the corpus). Irrelevant for the
  dense and bright strata [A]; same statistics for the faint ones.
- **Grid.** Everything is scored on the 2 x 16 cell grid. That penalises a filter-based method relative
  to a per-wire ROI-integral metric (the paper's own). The 4 x 4 and 16 x 8 regions are the
  grid-insensitive check, and the bright-region conclusions survive them. The faint-end conclusions
  reverse at the region scale, against the model.
- **Truth truncation** (25 e- per group-sample, 1.7% of the charge). Affects faint false charge for both
  methods; cannot affect > 1 Q0 numbers [V by arithmetic].
- **No PDG or parent information**, so no track/shower label and no shower-level grouping of particles
  (Geant4 tracks only).
- **Model coverage.** Uncovered cells are 0, which is part of the method. Coverage is 63-94% of charged
  cells below 100 k e- of neighbourhood charge, and ≥ 99.2% above.
- **Classical detection statistic on Y** is the tuned ROI charge (the frozen pick). On U/V it is a
  significance map, which is not a charge and is not used for any charge metric.
- **ROI baseline mechanism** for the induction charge loss is inferred, not isolated [A]. The measured
  facts are: oracle ROIs give 1.00, and fewer than 2% of the charge is in zeroed cells.
- **Wire-Cell itself was not run** (validity.md §1). This is the validated port.

## Reproduce (compute node, container; ~25 min on one GPU node)

```bash
H=baselines/uboone_sp/dense; S=/pscratch/sd/o/oalter/helix_work/uboone_sp/dense; V=/pscratch/sd/o/oalter/helix_work/uboone_sp/v2
$H/run_m2n_dump.sh                                       # M2n maps with v2/model_dump.py (4 GPUs); M4n/N64 maps are v2's
scripts/helix_run.sh python $H/cls_maps.py --events 260-387 --kernel fit --reuse $V/cls_full --out $S/cls_fit
scripts/helix_run.sh python $H/particle_truth.py --events 260-387 --out $S/ptruth
scripts/helix_run.sh python $H/collect.py --events 260-387 --cls $S/cls_fit --out $S/rec_fit
scripts/helix_run.sh python $H/analyze.py --rec $S/rec_fit --out $H/results/dense_fit.json
scripts/helix_run.sh python $H/tables.py --tag fit > $H/results/tables_fit.md
scripts/helix_run.sh python $H/figs.py --tag fit --cls $S/cls_fit
# exact kernel (or any other): the same with --kernel exact (no --reuse), out dirs *_exact, --tag exact
```
