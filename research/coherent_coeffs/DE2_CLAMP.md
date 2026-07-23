# de2_clamp — coherent-noise remover (reference)

Sample-space, detect-then-estimate coherent removal for TPC wire planes, anchored by the
coefficient-space `smart` estimate. Operates on one pedestal-subtracted plane
`(n_wires × n_ticks)`; wires are grouped into blocks of `GS=64` (the hardware coherent group).
Code: `research/coherent_coeffs/induction.py` (`iterate(...,seed='amp')` = de2; clamp applied on top).

`de2_clamp(noisy)`:
  smc  = smart_baseline(noisy)                       # coeff-space coherent estimate (anchor)
  de2  = iterate(noisy, n_iter=4, klo=0.7, khi=3.5, dilate=15, seed='amp', reducer='mean', minc=4)
  coh  = clip(de2, smc-4, smc+4)                      # clamp to the smart anchor +/- 4 ADC
  cleaned = noisy - coh                               # then -> standard sparsify for compression

## What it does, step by step

0. **smart baseline** `smc` = `sm.smart_removal(noisy, kgate=4.0)` coherent estimate (per-block,
   broadcast). smart = per-wire coif3-L4 DWT -> per-(block,coeff) k-sigma masked-mean common-mode
   -> level-aware GATE (keep |m|<kgate*sigma_coh,band as coherent, drop large=signal) -> IDWT.
   Used twice: as the initial detection baseline AND as the clamp anchor.

1. **per-wire noise scale** `sigw[w]` = MAD over time of `noisy[w]-median(noisy[w])` / 0.6745 ~ sigma_intrinsic.

2. **iterate n_iter times** (refine the coherent estimate `coh`, init = smc):
   a. DETECT signal (hysteresis on residual `z=|noisy-coh|/sigw`):
        seeds = z>khi ;  low = z>klo ;  2D (wire x tick, 8-conn) connected components of `low`;
        keep components containing a seed; dilate the mask `dilate` ticks in time.
   b. ESTIMATE coherent: per (block,tick) MEAN over the un-flagged (clean) wires; where <minc clean
        wires, linearly INTERPOLATE coh from reliable ticks (block median if <2 reliable).
   The refined `coh` becomes the detection baseline for the next iteration.

3. **clamp**: `coh = clip(de2, smc - clamp, smc + clamp)` — bounds how far the sample-space estimate
   may deviate from the robust smart anchor (prevents dense-region over-removal, mainly on U).

Why it works: off-signal, the masked-mean over ~all 64 wires recovers coherent to the intrinsic
floor. On-signal/dense regions, hysteresis finds the contiguous track (so clean wires are excluded),
the clamp stops the estimate running away where few clean wires remain.

## Knobs (role / default / sensitivity / tuning)

DETECTION (hysteresis):
- `klo`  grow threshold (sigma_int), default **0.7**. Lower -> grow into weak track shoulders
  (more recall) but risk PERCOLATION (connecting noise; <=0.45 blows up). MOST IMPORTANT detection
  knob. Tune in [0.5, 0.9].
- `khi`  seed threshold (sigma_int), default **3.5**. Strong-signal cores. Robust; [3, 4]. Low sensitivity.
- `dilate` temporal dilation (ticks), default **15**. Catches pulse shoulders. PLANE-DEPENDENT:
  U likes ~15-31, V ~11-15, Y indifferent. Moderate sensitivity. Too large over-flags.
- `n_iter` baseline-refinement passes, default **4**. Diminishing returns after ~2. Low sensitivity.

ESTIMATION:
- `reducer` per-tick center over clean wires, default **'mean'** (MVUE; best). Alternatives 'median'
  (robust but higher variance -> worse), 'trim'. Keep 'mean'.
- `minc` min clean wires to trust a tick (else interpolate), default **4**. Low sensitivity.

CLAMP:
- `clamp` max deviation from smc (ADC), default **4**. KEY noise-vs-fidelity / per-plane knob.
  Tighter (2) -> hug smart (keep more coherent, safest); looser (6-8) -> remove more but risk
  over-removal on bad-detection events. 4 = sweet spot (U safety + recovery). Moderate, plane-dependent.

ANCHOR:
- smart `kgate` (sigma_coh per band), default **4.0** — smart's gate; sets the anchor/baseline quality.

FIXED (structural, not tuned): block size GS=64 (hardware); smart transform coif3/L4/periodization;
smart per-band sigma + threshold-approx.

## Choices (alternatives at each stage; what we measured)

- DETECTOR: **hysteresis** (`seed='amp'`, used) — needed for INDUCTION (U/V; bipolar). PLAIN amplitude
  (`seed='ampthr'`, mask_amp) — simpler, fine/best for COLLECTION (Y), but HURTS induction (injects
  on-signal noise). Matched-filter (mask_mf, baseline-free, low recall) and energy (mask_energy) — no win.
- BASELINE: **smart** (used, robust anchor) vs interp-median (fails on U long dense runs) vs per-tick median.
- ESTIMATOR: **mean** (used) vs median/trimmed (worse) vs IRLS/mode/consensus (fail under majority signal).
- ADD-ONS tested and NOT worth it: joint per-wire wavelet signal-removal (~0 on all metrics -> dropped;
  de2_clamp == de3c), run-consensus, GLS/Wiener with coherent spectrum (HURTS: short corr + high SNR).
- PLANE-ADAPTIVE option: Y -> plain-amp detect, no clamp (simplest, ~optimal); U/V -> hysteresis+dilation
  +clamp. OR one UNIVERSAL config (hysteresis+dilation+clamp, this doc) — near-best on all (Y over-engineered
  by ~-0.0003, negligible).

## What you actually have to tune
Mainly **klo** (~0.7), **dilate** (per plane-type), and **clamp** (~4). khi/n_iter/minc/reducer keep
defaults. For a single universal setting use the defaults above; for best-per-plane, switch the
DETECTOR (plain for collection, hysteresis for induction) and drop the clamp on collection.

## Measured behaviour (12 ev; F0 / kept / nz_in / nz_out)
  Y: 0.9635 / 40k / 1.93 / 1.60   (best; clamp/joint neutral-to-slightly-negative here)
  U: 0.8926 / 45k / 2.83 / 1.66   (ties helix on F0 at fewer coeffs; helix nz_in 2.72 @ 55k)
  V: 0.8975 / 37k / 2.15 / 1.66   (dominates smart & helix on all four)
nz_out hits the intrinsic floor (~1.6) everywhere; the remaining work is nz_in (on-signal,
dense-region coherent). See RESULTS.md sec 6e-6l, figs 15-25.
