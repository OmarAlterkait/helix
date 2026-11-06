# Classical baseline: MicroBooNE (Wire-Cell) 2D deconvolution + ROI signal processing

> **2026-10-08 review: read `results/validity.md` first.**
> The headline below (and in `roi.md`) uses the MicroBooNE defaults and one threshold pooled
> over all planes. Tuned on non-test events (ev000-095) and scored with per-plane
> thresholds, the classical chain reaches **0.516** near-activity 1%-FPR efficiency at
> 0.1-0.2 MeV. That compares with 0.116 at the default, 0.579 for M4n and 0.429 for the
> 64-label model.
>
> - Code: `v2/`.
> - Numbers: `v2/results/*.json`.
> - Aspect-by-aspect comparison: `results/aspects.md`.
> - The simulation's exact response (JAXTPC 1ea2282) and its effect: `results/exact_kernel.md`. Nothing changes; the fitted kernel equals it to 1-4%.

A faithful re-run of the user's MicroBooNE signal-processing example
(`reference/uboone_deconv/`, unzipped verbatim from
`uboone_deconv_code_only.zip`) on OUR simulation, scored on the SAME 128 events,
windows and map rows, with the SAME code, as our learned denoiser
(`scripts/eval_denoise.py`). It is evaluated as a ladder that starts from
"perfect everything" and removes one piece of perfection per rung, so the loss
of each stage of a classical chain can be read off and set against the model.

## Results (128 events, ev260-ev387; full tables in `results/ladder.md`, raw in `results/ladder.json`)

Headline, eval_denoise keys, the model's numbers from `dn.jsonl` (tag `ab_ft1024_win`):

| rung | map r | eff@1% FPR <0.1 / 0.1-0.2 / 0.2-0.5 MeV | floor AUC <0.1 | cell resolution, truth 4-16k / 16-64k e- (68% half-width) |
|---|---|---|---|---|
| R0syn ideal sensor, oracle ROI | 0.973 | (0.998 / 0.997 / 0.998)* | 0.999* | 0.67 / 0.20 |
| R0zs  + integer ADC, \|v\|<2 dropped | 0.969 | (0.852 / 0.938 / 0.990)* | 0.953* | 0.68 / 0.20 |
| R0    stored clean sensor | 0.965 | (0.867 / 0.940 / 0.984)* | 0.958* | 0.70 / 0.21 |
| R1    + incoherent noise | 0.933 | (0.826 / 0.895 / 0.951)* | 0.946* | 0.98 / 0.26 |
| R2    + coherent noise, gate removal | 0.932 | (0.824 / 0.903 / 0.955)* | 0.946* | 0.98 / 0.26 |
| **R3  + MicroBooNE ROI finder (full chain)** | **0.908** | **0.006 / 0.004 / 0.007** | **0.634** | **1.16 / 0.28** |
| **our model** | **0.988** | **0.413 / 0.784 / 0.982** | **0.903** | **0.26 / 0.12** (CPU replay) |

\* oracle-ROI rungs: background windows score exactly 0 (verified, 100% of the
7,680), so the 1%-FPR threshold is 0 and "efficiency" is "the oracle ROI kept at
least one positive cell" -- charge survival, not detection. Only R3 and the model
are detectors.

Total charge recovered (sum of the ROI charge / sum of hits), U, V, Y:
R0syn 0.88, 0.88, 1.00 · R0 0.88, 0.87, 1.00 · R1 0.88, 0.87, 0.92 · R2 0.88, 0.87, 0.91 · R3 0.81, 1.01, 0.92.

### What limits each rung

- **R0syn (deconvolution + the example's software filters, nothing else).**
  Collection is exact in total (0.996) and per wire (median 0.000, 68% 0.054).
  Induction keeps 0.88: the induction wire filter (a = 0.11 cycles/wire, sigma
  ~1.4 wires) spreads charge onto wires the oracle ROI (truth > 25 e-) does not
  cover -- with no ROI (R0synnr) all three planes return exactly 1.000. Per cell
  the filters smear bright cells by ~20% and faint ones by ~90%.
- **R0zs (storage).** Integer rounding + dropping |v| < 2 ADC empties 30% of
  faint cells (<1k e-; 7% at R0syn) and 14% of 1-4k e- cells (3%), moves faint
  deposits by 1.4 ticks instead of 0.2, and takes the Y-plane F0 from 1.00 to 0.93.
  This is the zero-suppression effect at R0; induction is hit hardest (on U/V,
  37-42% of <0.1 MeV deposit windows have no stored clean signal at all, measured on the first 16 events).
- **R0 (response knowledge).** The stored sensor vs our best translation-invariant
  kernel (2-4% residual, from sub-wire position dependence the per-wire hits cannot
  carry): small inside oracle ROIs (map r 0.969 -> 0.965). Outside ROIs it is not
  small: the noise-free R0 deconvolution with no ROI (R0nr) has background cells
  up to 13-16k e- (q99 of the window max; 2-4k for R0zsnr, 0.1-3k for R0syn) --
  mostly near-DC wander on induction, where the bipolar response has R(0) ~ 0.
  The example's kernel (R0ex) costs F0 (0.87 -> 0.79 on Y) but little per cell.
- **R1 (intrinsic noise)** is the largest per-cell loss of the oracle ladder:
  map r 0.965 -> 0.933, 4-16k e- cells 0.70 -> 0.98, half of faint cells emptied.
  Y total charge drops 0.995 -> 0.92 because in noisy mode the example zeroes DC
  for collection too and restores a baseline only for induction ROIs (per wire
  this costs the ROI's share of the readout times the wire's charge; long ROIs of
  drift-parallel tracks lose most) -- measured, not a noise effect (clean input
  with noiseless=False loses the same).
- **R2 (coherent removal) costs nothing measurable**: our gate leaves 0.5-0.7 ADC
  rms of the 2.5 ADC coherent noise, and R2 = R1 within errors on every key. The
  image-space multipass removal (R2mp) is worse (V F0 0.57 -> 0.31, map r 0.912),
  it subtracts signal.
- **R3 (ROI finding) is where the classical chain fails at the floor.**
  Collection: the 5 sigma threshold is 1,260 e-/tick on the Wiener output
  (rms 250 e-/tick), so the ROI finder catches 1% of <0.2 MeV deposits, 23% at
  0.2-0.5, 94% at 0.5-1 MeV; Y fakes are rare (4% of charge-free windows) and the
  large ones all sit at the readout start (t0 < 160 ticks). Induction: the
  example's tight/loose LF chain, never run with noise by its author, opens ROIs in
  44% (U) / 14% (V) of charge-free windows and the Gaussian output inside them,
  after a two-point baseline, carries up to 25-33k e- per cell (q99). The pooled
  1%-FPR threshold is therefore 28.8k e- per cell; only >1 MeV deposits clear it.
  The model's background q99 is 327 e-. R3inc (no coherent noise) is the same, so
  this is the ROI logic on intrinsic noise, not coherent noise.

### Per-plane floor (diagnostic: each plane's own 1%-FPR threshold)

R3: U 0.00/0.00/0.00, V 0.02/0.01/0.01, Y 0.00/0.00/0.00 (<0.1 / 0.1-0.2 / 0.2-0.5 MeV).
Model: U 0.33/0.63/0.98, V 0.52/0.76/0.98, Y 0.40/0.95/0.99.
With the readout-start windows (t0 < 160) excluded, the Y background q99 of R3 is
0 and its Y efficiency becomes the ROI efficiency: 0.01 / 0.01 / 0.23 / 0.94 / 1.00
for <0.1 / 0.1-0.2 / 0.2-0.5 / 0.5-1 / >1 MeV -- the best case for this chain.

### What was verified, and what was assumed

Verified by execution:
- our deconvolution + the example's ROI code reproduce the example's own
  `run_pipeline` exactly (identical ROIs, C_charge to 3e-5 e-), U and Y, both ROI
  modes, both DC modes (`dev/check_equivalence.py`);
- Q0 recomputed from ev000-ev259 equals the model's (2043.258 e-); the 18,994
  windows are the model's, in the same (ev, g, w0, t0, kind) order; the model's
  `dn_ab_ft1024_win_trained_windows.json` re-scored by our scorer reproduces every
  floor key of dn.jsonl to 4 decimals; map rows are restricted to the same
  token-covered subset eval_denoise scores (701,744 rows);
- the CPU fp32 replay of the model (`model_cells.py`) gives map r 0.9881 (dn.jsonl
  0.9881) and floor 0.385 / 0.767 / 0.983 vs 0.413 / 0.784 / 0.982 (bf16 GPU) --
  the 1%-FPR threshold is precision-sensitive, so the table quotes dn.jsonl for the
  floor and uses the replay only for per-cell resolution and figures;
- the response (below); gain and lag; the sensor's zero suppression (min |v| = 2).
Assumed / not checked:
- the example's filter parameters and thresholds (taken as given; not compared
  with the paper's tables);
- (the readout-start background excess is now traced: the cyclic-FFT edge effect, see Addition 1);
- one noise realisation per event (seed 1000 + ev); bootstrap errors are over
  events only.

### Figures (`figs/`)

- `summary.png` -- the headline keys per rung next to the model (oracle rungs hatched);
- `images_ev300_g2_*` (Y) and `images_ev300_g1_*` (V) -- truth, R0, R1, R2, R3 and
  the model on the 2-wire x 16-tick grid, one shared LogNorm in electrons floored
  at 0.05 Q0: full plane, a 0.15 / 0.19 MeV isolated deposit, a charge-free window;
- `floor_scores.png` -- window score vs deposit energy for R1, R3 and the model,
  with each one's 1%-FPR threshold;
- `response.png` -- example vs fitted kernel.

## Addition 1: detection WITHOUT the MicroBooNE ROI logic (no-ROI rungs), and the readout-edge effect

The oracle rungs cannot measure detection (every background window scores 0). The
no-ROI rungs take the full Gaussian-filtered deconvolved image, sum it per
2-wire x 16-tick cell, clip at 0 -- the classical detector's floor without ROIs:

| rung | map r | eff@1% <0.1 / 0.1-0.2 / 0.2-0.5 MeV | AUC <0.1 | bg q99 U / V / Y (e-) | per-plane eff@1% (own threshold), Y: <0.1 / 0.1-0.2 / 0.2-0.5 |
|---|---|---|---|---|---|
| R1nr  incoherent noise, no ROI | 0.950 | 0.004 / 0.005 / 0.010 | 0.474 | 8,842 / 11,891 / 11,386 | 0.00 / 0.00 / 0.00 |
| R2nr  + coherent, gate, no ROI | 0.939 | 0.007 / 0.009 / 0.015 | 0.498 | 9,236 / 11,876 / 11,112 | 0.00 / 0.00 / 0.00 |
| R1nrp R1nr, readout zero-padded | 0.950 | 0.015 / 0.018 / 0.082 | 0.471 | 3,933 / 8,297 / 3,784 | 0.01 / 0.04 / 0.57 |
| R2nrp R2nr, readout zero-padded | 0.939 | 0.029 / 0.031 / 0.084 | 0.497 | 3,982 / 8,457 / 3,620 | 0.01 / 0.04 / 0.60 |
| R0nr / R0nrp  noise-free, no ROI (artefacts only) | 0.955 / 0.957 | 0.006 / 0.003 / 0.006 -> 0.010 / 0.009 / 0.021 | 0.547 / 0.573 | 16,241 / 13,009 / 12,971 -> 10,296 / 9,809 / 1,446 | 0.00 / 0.00 / 0.00 -> 0.05 / 0.71 / 0.98 |
| R3  full chain (for reference) | 0.908 | 0.006 / 0.004 / 0.007 | 0.634 | 32,960 / 26,382 / 12,926 | 0.00 / 0.00 / 0.00 |
| our model | 0.988 | 0.413 / 0.784 / 0.982 | 0.903 | 484 / 240 / 274 (all planes 327) | 0.40 / 0.95 / 0.99 |

Without ROIs the classical chain is not a detector below 0.5 MeV either: the
Gaussian output's per-cell noise (std ~1,000-1,300 e- per 2x16 cell, induction
near-DC wander on top) puts the charge-free windows' 99th-percentile max at
4-12k e-, above a 0.1-0.5 MeV deposit's brightest cell (2-10k e-).

**The readout-start excess is an edge effect of the deconvolution.** The example
deconvolves with a cyclic FFT over the 4321-tick readout. Charge near the end
of the readout has a response that runs past the last tick, and charge near the
start has a precursor (up to ~150 ticks on induction) that would start before
tick 0. Both responses are cut off, and the cyclic division misplaces what is
missing at the opposite edge. Verified (`dev/edge_check.py`, `dev/edge_check2.py`):
- **Not out-of-window charge:** no hits lie outside [0, 4321).
- **Charge-free wires show no excess:** their early, middle and late cells are alike.
- **Readout-start windows (t0 < 160) on the noise-free R0 deconvolution:**

  | | median score | q90 |
  |---|---|---|
  | cyclic (as the example does) | 1,580 | 6,618 (middle of the readout: 3,830) |
  | readout zero-padded 512 ticks each side before the FFT | 1,094 | 2,561 |
  | windows whose wires carry charge in the last 200 ticks (cyclic) | 2,162 | |
  | windows whose wires carry no such charge (cyclic) | 986 | |

  Padding brings the windows with late charge down to 1,160.
- **Over all 128 events (the *nrp rungs):** padding cuts the Y background q99 from
  ~11k to ~3.6k e- with noise (from 13k to 1.4k without). That raises the
  Y-plane no-ROI efficiency at 0.2-0.5 MeV from 0.00 to 0.60.

On induction the tail is near-DC wander, not the edge, and padding only moves the
q99 from 9-12k to 4-8k e-. So the edge effect is real and dominates the
collection-plane tail. It is not what keeps the classical chain off the floor:
pooled over planes, padding takes R2's no-ROI efficiency only from 0.007 / 0.009 /
0.015 to 0.029 / 0.031 / 0.084.

## Addition 2: near-activity noise windows, SNR bins, false-positive cells

Scored exactly as `scripts/eval_denoise.py --near --annot`:
- **Windows:** the floor windows plus the 7,680 `bgn` windows of `truth_v2_near`.
  These are charge-free windows with charge within 16 wires / 128 ticks.
- **Annotations:** each iso window gets `snr` / `clean` from `noise_vs_hits.npz`.
- **Metrics:** `helix.probe.resolution.near_scalars`, with the 200-resample event bootstrap.
- **Model:** its numbers come from `dn_near_debug_ab_ft1024_win_trained_windows.json`.
  Its windows are ours, in the same order (verified), and its floor keys
  re-scored here equal dn.jsonl.

Full tables: `results/near.md`, `results/near.json`. Oracle-ROI rungs (R0-R2)
score near windows 0 as well (q99 = 0), so their near numbers are again
charge survival, not detection.

| rung | near_fpr_at_far1pct | near_eff1pct <0.1 / 0.1-0.2 / 0.2-0.5 / 0.5-1 | far / near bg q99 (e-) |
|---|---|---|---|
| R0syn / R0 / R1 / R2 (oracle) | 0.000 | (0.998/0.997/0.998/1.000) / (0.867/0.940/0.984/1.000) / (0.826/0.895/0.951/0.978) / (0.824/0.903/0.955/0.984) | 0 / 0 |
| R1nr | 0.004 | 0.016 / 0.019 / 0.083 / 0.587 | 10,501 / 6,686 |
| R2nr | 0.008 | 0.009 / 0.009 / 0.019 / 0.232 | 10,529 / 9,481 |
| R2nrp | 0.023 | 0.013 / 0.012 / 0.025 / 0.322 | 6,765 / 8,876 |
| R3 | 0.004 | 0.014 / 0.013 / 0.023 / 0.052 | 28,844 / 23,615 |
| our model | **0.174** | 0.003 / 0.151 / 0.913 / 0.989 | 327 / 2,708 |

Efficiency by deposit SNR (clean isolated deposits), at the far / near 1%-FPR threshold:

| rung | no signal | 0-3 | 3-5 | 5-7 | 7-10 | 10-15 | >15 |
|---|---|---|---|---|---|---|---|
| R2nr | 0.01 / 0.01 | 0.00 / 0.01 | 0.01 / 0.01 | 0.01 / 0.01 | 0.01 / 0.01 | 0.01 / 0.01 | 0.52 / 0.54 |
| R2nrp | 0.03 / 0.01 | 0.02 / 0.01 | 0.04 / 0.01 | 0.02 / 0.01 | 0.02 / 0.01 | 0.01 / 0.01 | 0.65 / 0.57 |
| R3 | 0.00 / 0.02 | 0.01 / 0.03 | 0.01 / 0.02 | 0.00 / 0.01 | 0.01 / 0.02 | 0.00 / 0.01 | 0.27 / 0.35 |
| our model | 0.30 / 0.00 | 0.29 / 0.00 | 0.37 / 0.00 | 0.54 / 0.05 | 0.87 / 0.29 | 1.00 / 0.56 | 1.00 / 0.98 |

Reading:
- **The classical detectors are blind on both sides.** Their thresholds are set by
  noise and artefacts far from activity as much as near it. near_fpr_at_far1pct
  is about 0.01 for R2nr and R3, and below SNR 15 they detect nothing at either
  threshold.
- **The model is excellent far from activity but hazes near it.** 17.4% of
  near-activity noise windows exceed its far threshold, and its near-activity
  1%-FPR threshold is 8x the far one (2.7k vs 327 e-). At the near threshold it
  still keeps 0.91 of 0.2-0.5 MeV deposits, but only 0.15 at 0.1-0.2 and 0.00
  below 0.1.
- **The model scores 0.30 at SNR 0 ("no signal": deposits whose clean sensor
  holds none of their charge) at the far threshold.** At the near threshold this
  drops to 0.00. So those far-threshold "detections" are the same haze, or
  context, not signal.

Cells above f x Q0 where the truth is 0, summed over all 128 test events. U, V, Y:

| rung | > 0.05 Q0 (102 e-) | > 0.25 Q0 (511 e-) | > 1 Q0 (2,043 e-) |
|---|---|---|---|
| R0syn | 126k, 126k, 66k | 64k, 64k, 16k | 23k, 23k, 2.2k |
| R0 | 384k, 360k, 294k | 195k, 181k, 46k | 43k, 41k, 3.8k |
| R1 | 438k, 431k, 347k | 370k, 363k, 190k | 194k, 191k, 13k |
| R2 | 438k, 430k, 350k | 370k, 360k, 194k | 194k, 185k, 17k |
| R3 | 1.52M, 925k, 65k | 1.40M, 874k, 46k | 1.02M, 719k, 12k |
| R2nr | 24.1M, 26.2M, 15.4M | 15.6M, 19.9M, 10.2M | 1.60M, 5.01M, 0.94M |
| R2nrp | 24.4M, 27.3M, 15.7M | 14.9M, 21.0M, 9.9M | 1.29M, 5.75M, 0.61M |
| our model | 6.23M, 7.32M, 5.47M | 144k, 123k, 51k | 5.7k, 4.5k, 1.2k |

How the cells were counted:
- **Classical rungs:** every cell of every plane, 68.3M / 68.3M / 50.1M cells.
- **Model:** token-covered cells only, 34.0M / 34.1M / 18.5M. It predicts 0
  elsewhere (`model_cells.py --fp-only`, CPU replay).

Reading:
- **The model's false positives are almost all faint haze.** It has more empty
  cells above 102 e- than any ROI rung. Above 2,043 e- it has
  fewer than any classical rung, the oracle-ROI rungs included.
- **R3's induction false positives are bright.** About 1.7M cells above 2,043 e-
  come from the fake ROIs.

## Addition 3: the ROI side, investigated (full write-up: `results/roi.md`; audit: `results/roi_audit.md`)

1. **Audit.** I compared the example's ROI code against Wire-Cell Toolkit master (@ b7bd2a1a, MicroBooNE configuration, sources in `reference/wirecell/`) and against the paper, with file:line citations [V].
   - **Filters.** The example's filter numbers are wrong by large factors. Its Wiener is ~2× too wide, its induction wire filter 3.6× too narrow, and its LF filters 1.75-4× off.
   - **Missing steps.** It lacks Wire-Cell's loose-ROI algorithm (rebin ×6, prominence rules), CheckROIs, Break/Shrink as written, and the absolute fake-signal removal (a bin > 1000 e or a mean > 500 e).
   - **Thresholds.** It thresholds |x| at 3.5σ, where Wire-Cell uses a signed 3σ on induction.
   - **Invented ROIs.** It copies tight ROIs onto neighbouring wires.
   - **Wire-Cell itself, as executed.** At master the U-plane fake removal never removes anything. Wire-Cell zero-pads the FFT, which is the readout-edge fix of Addition 1.
2. **Diagnosis.** The example's fake induction ROIs come from its first step. Raw tight ROIs already sit in 54% (U) / 17% (V) of charge-free windows, and nothing downstream removes them.
   - **Not the RMS.** Its noise RMS matches the true noise to 1-5%, and the crossing rates are Gaussian. The fakes come from a ~3.5σ per-sample cut over 4,096 samples per window.
   - **Collection misses.** A 0.1-0.2 MeV deposit's per-tick Wiener peak is 0.26× the 5σ threshold (1,280 e).
3. **Faithful port** (`sp/wirecell_sp.py`), with a threshold sweep (`roi_sweep.py`, `roi_score.py`).
   - **As configured** (U fake removal a no-op): eff@1% 0.010 / 0.096 / 0.390 (far) and 0.016 / 0.116 / 0.397 (near), for <0.1 / 0.1-0.2 / 0.2-0.5 MeV.
   - **What kills V and Y below 0.2 MeV** is the absolute fake-signal cut. Without it, Y alone reaches 0.85 at 0.1-0.2 MeV, against the model's 0.95 on Y.
   - **Pooled best (post hoc).** No classical setting exceeds 0.14 at 0.1-0.2 MeV pooled, or 0.28 for a post-hoc per-plane hybrid. The model reaches 0.827 (far) and 0.558 (near).
   - **Near activity** is where the classical chain is clean (near_fpr_at_far1pct ≤ 0.01, model 0.141). Even so, the model's near-threshold efficiency exceeds every classical setting above 0.1 MeV.
4. **Decomposition.**
   - **ROI efficiency** per plane and SNR. Without the fake cut, U finds 0.93 and Y 0.55 at SNR 7-10. The model's thresholded cells find 0.90 (far) / 0.80 (near).
   - **Boundary loss.** A found faint deposit has 72% of its charge inside the Wire-Cell ROI.
   - **Charge.** Found ROIs give 0.83 ± 0.37, oracle ROIs 0.92 ± 0.26, at 0.1-0.2 MeV.
   - **Purity.**

     | ROIs | U | V | Y |
     |---|---|---|---|
     | Wire-Cell port | 0.29 | 0.15 | 0.11 |
     | example | 0.65 | 0.53 | 0.05 |
     | model cells, far threshold | 0.37 | 0.43 | 0.23 |
     | model cells, near threshold | 0.01 | 0.01 | 0.01 |

5. **Figures:** `figs/roi_roc_{far,near}.png` and `figs/roi_display_ev{300,321,350}.png`.

## What was recreated from the example, and how faithfully

The example's own modules are IMPORTED, not re-typed: `sp/pipeline.py` puts
`reference/uboone_deconv` on `sys.path` and calls, unchanged,

| example module | used for | paper section |
|---|---|---|
| `filters.py` (`PLANE_PARAMS`, Wiener-inspired / Gaussian time filters, wire filter, LF filters) | every rung | 3.2.2, Eq. 3.10/3.12/3.13/3.14 |
| `roi.py` (`form_rois_induction`: tight/loose LF ROIs, containment, connectivity, clean, break, shrink; `form_rois_collection`; `apply_rois` incl. induction linear baseline subtraction; `form_rois_true`) | R3 / oracle ROIs / charge | 3.2.3-3.2.4 |
| `pipeline.filter_truth` | the example's F0 metric vs filtered truth | 4.2.5 |
| `metrics.pixel_metrics` | F0 / correlation | 4.3 |

Only the deconvolution entry point is re-implemented (`sp/pipeline.deconvolve_2d_dual`),
line for line, with the kernel passed in instead of read from
`avg_response_<plane>.npz`, and real FFTs instead of complex ones (identical
result for real input and Hermitian filters, half the cost).
`dev/check_equivalence.py` runs both on a real plane of our sim with the
example's kernel: deconvolution outputs agree to float32 rounding
(max |diff| 3e-5 e- on values of 2.5e4), the LF-filtered outputs likewise, and
the full `run_pipeline` gives IDENTICAL ROI dictionaries and C_charge for U and
Y, oracle and algorithmic ROIs, both DC modes.

Per-plane parameters are the example's (`PLANE_PARAMS`, read by the example's
author off the paper's Figs. 15-17, NOT refit to our noise): Wiener-inspired
`a` = 0.28/0.28/0.35 MHz, `b` = 4; Gaussian `a` = 0.24/0.24/0.30 MHz; wire
filter `a` = 0.11/0.11/0.30 cycles/wire; LF loose/tight 0.010/0.035 MHz
(induction only); ROI threshold 3.5/3.5/5.0 x robust RMS (noisy mode), pad
10/10/5 ticks; oracle ROIs: truth > 25 e- per (wire, tick), padded 10 ticks.

DC handling is the example's (its README §5.3): R0-type rungs (noise-free input)
run with `noiseless=True` (DC kept), every noisy rung with `noiseless=False`
(F(0) = 0; the induction ROI baseline subtraction restores it).

Charge per fine cell = the sum of the example's `C_charge` (ROI-masked,
baseline-subtracted Gaussian-filter output -- its "naive" extraction) over each
2-wire x 16-tick cell, clipped at 0. Map rows are scored as log1p(q/Q0), windows
as charge, exactly as the model's prediction is.

### Deviations (all deliberate, none changes the algorithm)

1. **Response kernel.** The example's `avg_response_*` (JAXTPC kernel,
   trapezoid-averaged over the pitch) is close to, but not exactly, what our
   simulation used. "Perfect knowledge" rungs use a kernel FITTED to our sim:
   exact least squares, 21 wires x 211 ticks, from (hits, clean sensor) pairs of
   8 events of the probe split's TRAINING part (never a scored event), both
   volumes pooled per plane type, with two passes imputing the zero-suppressed
   samples (`fit_response.py`). R0ex keeps the example's kernel to show what a
   slightly wrong response costs. See "Response" below.
2. **Noise filter.** The example has none (it never ran with noise, HANDOVER §5).
   R2 uses OUR coherent removal two ways: the corpus's coefficient-space gate
   (DWT -> `helix.tpc.coherent_gate` -> inverse DWT, no thresholding; the main
   R2) and image-space multipass (`helix.tpc.coherent.remove_coherent`, R2mp),
   the closer analogue of MicroBooNE's group-median filter (JINST 12 P08003).
   MicroBooNE's other noise-filter stages (harmonic/RC/misconfigured-channel
   corrections) have no counterpart in our noise model and are not needed.
3. **Charge extraction** is per fine cell, not per wire / per hit, because the
   scoring is per cell. The example's Gaussian hit finder (`hit_finder.py`)
   returns per-wire charge and is not used; its README notes it changed results
   by 1-2 points in the regimes it was run.
4. **Noisy-mode thresholds** (`roi_th_factor x RMS`) are the example's, never
   tuned with noise by its author (HANDOVER §4.2/§5 step 4); left as is.

### Example vs the published papers (JINST 13 P07006 / P07007)

Where the example simplifies the paper we follow the example, as instructed;
the differences we know of:
- filter parameters were read off the paper's frequency-domain figures, not
  taken from its tables or re-fitted (example HANDOVER §4.1);
- the position-averaged response uses 11 trapezoid sub-positions over 19 wires
  (paper: 6 positions, Eq. 3.8; ±10 wires); we replace it with a direct fit;
- `clean_rois` is per wire (+-1 neighbour), not per connected loose-ROI cluster
  as the paper describes ("Simplified" in the example's docstring), and the
  valley-splitting / shrink tolerances in `break_rois`/`shrink_rois` were filled
  in by the example's author (HANDOVER §4.6);
- the paper extracts charge with Gaussian hit fits on the ROI output; the
  example's default and ours is the naive integral;
- in noisy mode the example zeroes DC for collection too (F(0) = 0) but only
  masks collection ROIs (no baseline restore), so a collection wire loses its
  time-mean inside its ROIs -- measured 0.995 -> 0.92 of the Y charge at R1 (see
  R1 below); whether Wire-Cell does the same was not checked.

## Response

The example's `avg_response_*` was built from the JAXTPC kernel of the same
`cubic_wireplane` geometry, and it is CLOSE to our sim: its integral matches
(Y +0.0540 vs the clean/hits ratio +0.0544 ADC per electron; U -0.0090 vs -0.0102;
V -0.0022 vs -0.0030, the U/V ratios biased by zero suppression of small lobes),
and the Y peak lags the hit tick by +6 ticks (3 us) in both the kernel and a
direct hits-sensor cross-correlation. But forward-folding hits with it does not
reproduce the stored clean sensor well: on held-out events (ev260-262, after
emulating the writer's rounding and zero suppression)
||clean - store(hits (*) R)|| / ||clean|| = 0.113 (U), 0.164 (V), 0.073 (Y).

The least-squares kernel fitted to our sim (`response/fit_{U,V,Y}.npz`, 21 wires
x 211 ticks, dt in [-150, +60]) brings that to **0.041 (U), 0.039 (V), 0.021 (Y)**
(correlation 0.9991 / 0.9993 / 0.9998). It differs from the example's by 22% / 24%
/ 10% in norm, mostly a broader, earlier induction lobe; 1% of its norm lies
outside the example's support. Facts from it: central-wire peak 2.6 / 2.8 / 8.3
ADC per 1000 e- (a deposit spread over wires and ticks peaks lower -- the 2.0 /
1.65 / 4.5 quoted for real deposits); U bipolar zero crossing at -1 tick, V at +4,
Y peak at +6. The residual 2-4% is what a translation-invariant, per-wire kernel
cannot express (the sim responds to the sub-wire position, which the per-wire
hits do not carry) plus rounding. All "perfect knowledge" rungs use the fit;
R0ex uses the example's kernel.

## Ladder definitions

| rung | input | ROIs | removes which perfection |
|---|---|---|---|
| R0syn | hits (*) fitted kernel, float (no rounding, no zero suppression) | oracle | -- the deconvolution + software filters alone |
| R0zs | R0syn rounded to integer ADC, \|v\| < 2 dropped (what the sensor writer does) | oracle | the storage |
| R0 | the stored clean sensor | oracle | response-model mismatch (sub-wire position etc.) |
| R0ex | the stored clean sensor, example's kernel | oracle | a slightly wrong kernel |
| R1 | digitize(clean + incoherent noise) -- coherent noise removed perfectly | oracle | intrinsic noise |
| R2 | digitize(clean + incoherent + coherent) -> our gate | oracle | coherent removal |
| R2mp | as R2 with image-space multipass removal | oracle | (variant) |
| R3 | R2's input | MicroBooNE algorithm | ROI finding = the full classical chain |
| R3inc | R1's input | MicroBooNE algorithm | (diagnostic: ROI finding without coherent noise) |
| R1nr / R2nr | R1 / R2 input, Gaussian output with NO ROI mask | none | detection without the ROI logic |
| R1nrp / R2nrp / R0nrp | as *nr, waveform zero-padded 512 ticks at both ends before the FFT | none | removes the cyclic readout-edge artefact |
| R0synnr / R0zsnr / R0nr | noise-free inputs, no ROI | none | (diagnostic: deconvolution artefacts alone) |

Noise is one forward-model realisation per event (seed 1000 + ev, as
`scripts/snr_stages.py`), shared by every rung, generated exactly as the corpus
does (`helix.tpc.noise.generate_noise` with the measured "microboone" spectrum,
per-wire ENC from the wire length, coherent 2.5 ADC per 64-wire group,
`digitize` with the plane's pedestal).

**Oracle ROIs make the floor metric degenerate.** Background windows contain no
truth within +-16 wires / +-128 ticks, so an oracle ROI never reaches them: their
score is exactly 0 for every oracle rung (verified: 100% of 7,680), the 1%-FPR
threshold is 0, and "efficiency" becomes "the oracle ROI holds at least one
positive cell". R0-R2 floor numbers therefore measure charge survival, not
detection; only R3 (and the no-ROI diagnostics) is a detector. Map r and per-cell
resolution are meaningful on every rung.

## Files

| file | what |
|---|---|
| `reference/uboone_deconv/` | the example, verbatim |
| `sp/pipeline.py` | deconvolution with injected kernel + the example's chain (imported) + fine-cell binning |
| `sp/response.py` | kernel I/O, forward model, sensor-storage emulation, least-squares kernel fit |
| `sp/data.py` | event loading (truth file -> shard -> source file), hits decoding, forward noise |
| `sp/noise_filter.py` | coherent removal for R2 (gate / multipass) |
| `fit_response.py` | fits `response/fit_{U,V,Y}.npz`, writes `response/residuals.json` |
| `run_ladder.py` | runs the rungs per event, stores per-cell predictions (scratch) |
| `model_cells.py` | CPU replay of the model's per-cell predictions (eval_denoise's predict) |
| `sp/wirecell_sp.py` | Python port of Wire-Cell OmnibusSigProc's ROI path (MicroBooNE configuration), file:line cited |
| `roi_diag.py`, `summarize_roi_diag.py` | stage-by-stage fake-ROI accounting, noise RMS vs truth, deposit peak vs threshold -> `results/roi_diag.md` |
| `roi_sweep.py`, `roi_score.py`, `make_roi_md.py` | threshold sweep of both ROI chains, ROC, decomposition -> `results/roi_sweep.{json,md}` |
| `model_presence.py` | CPU replay of a presence-headed model's per-cell outputs for the ROI comparison |
| `plot_roi.py` | ROC curves and ROI event displays |
| `score_near.py` | near-activity windows (near_scalars, SNR bins) and false-positive cell counts -> `results/near.{json,md}` |
| `score_ladder.py` | eval_denoise's scoring, applied to every rung and to the model replay -> `results/ladder.json`, `results/windows_<rung>.json` |
| `make_table.py` | `results/ladder.json` -> `results/ladder.md` |
| `plot_ladder.py` | `figs/` |
| `dev/` | equivalence check, response/zero-suppression/noise diagnostics |

Reproduce (inside the container, on a compute allocation):

```bash
H=baselines/uboone_sp; S=/pscratch/sd/o/oalter/helix_work/uboone_sp
scripts/helix_run.sh python $H/fit_response.py --fit-events 8 --test-events 3 --impute 2
scripts/helix_run.sh python $H/run_ladder.py --events 260-387 --workers 15 --out $S/preds --dump 300
scripts/helix_run.sh python $H/run_ladder.py --events 260-387 --workers 12 --rungs R0syn,R0zs --out $S/preds_syn --dump 300
scripts/helix_run.sh python $H/model_cells.py --events 260-387 --plane-event 300 --out $S/model
scripts/helix_run.sh python $H/run_ladder.py --events 260-387 --workers 14 --rungs R0nr,R0synnr --out $S/preds_nr
scripts/helix_run.sh python $H/run_ladder.py --events 260-387 --workers 14 --rungs R0zsnr --out $S/preds_zsnr
N=/pscratch/sd/o/oalter/helix_work/resolution/truth_v2_near
scripts/helix_run.sh python $H/run_ladder.py --events 260-387 --workers 15 --near $N --out $S/preds_near \
    --rungs R0,R0ex,R1,R2,R2mp,R3,R1nr,R2nr,R3inc,R0syn,R0zs,R0nr,R0synnr,R0zsnr
scripts/helix_run.sh python $H/run_ladder.py --events 260-387 --workers 15 --near $N --out $S/preds_nrp --rungs R0nrp,R1nrp,R2nrp
scripts/helix_run.sh python $H/model_cells.py --fp-only --events 260-387 --out $S/model_fp
scripts/helix_run.sh python $H/score_ladder.py --preds $S/preds,$S/preds_syn,$S/preds_nr,$S/preds_zsnr,$S/preds_nrp --out $H/results
scripts/helix_run.sh python $H/score_near.py --preds $S/preds_near,$S/preds_nrp --out $H/results
scripts/helix_run.sh python $H/make_table.py && scripts/helix_run.sh python $H/plot_ladder.py
# ROI investigation
scripts/helix_run.sh python $H/roi_diag.py --events 260-291 --workers 16 --out $S/roi_diag
scripts/helix_run.sh python $H/summarize_roi_diag.py --diag $S/roi_diag --out $H/results/roi_diag.md
scripts/helix_run.sh python $H/roi_sweep.py --events 260-387 --workers 80 --out $S/roi_sweep \
    --skip ex_s0.5,wc_s0.5,wc_fake0.5,wc_s0.67_nofake
scripts/helix_run.sh python $H/model_presence.py --events 260-387 --out $S/model_m4n \
    --checkpoint /pscratch/sd/o/oalter/helix_work/denoise/M4n_ft_full_near/final.pt
scripts/helix_run.sh python $H/roi_score.py --sweep $S/roi_sweep --out $H/results
scripts/helix_run.sh python $H/make_roi_md.py && scripts/helix_run.sh python $H/plot_roi.py
```
