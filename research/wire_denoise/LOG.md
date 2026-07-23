# Running log

## 2026-05-30 05:46 PDT — P0 setup
- Folder `research/wire_denoise/` + `PLAN.md`. Budget 7h → stop ~12:45 PDT.
- `common.py`: pimm_data clean loader (cached), JAXTPC noise (intrinsic FFT-shaped+white,
  optional coherent via tools.coherent_noise), digitize, F0/noise metric, train/test split.
- Smoke test (Y, intrinsic-only): occupancy 1.46%, noise floor std **1.61 ADC** (matches
  σ_Y≈1.58 model), raw-noisy self-F0 **0.974**, noise_rms 1.61.
- **Insight:** signal pulses ≫ noise, so F0-on-signal is high even with no denoising. The
  discriminating axes are **compression ratio** and **off-pixel noise_rms** at matched F0.
  → report all methods as rate–distortion (compression vs F0 and vs noise_rms), per plane.

## Stage A = intrinsic noise only (no coherent). Stage B adds coherent later.

## 06:00 P1 baselines done — per-wire DWT VisuShrink, sweep wavelet/level/κ.
- Y best db8/L4, U coif3/L4, V db8/L4; level 4 > level 8 (boundary effects); family ~irrelevant.
- DWT frontier strong: Y 9×/0.974 … 101×/0.950; U 13×/0.945; V 9×/0.933. This is the bar.

## 06:13 P3 dict (1D non-overlap patches, OMP) — UNDERperforms (Y 10.7×/0.915, U 10.7×/0.872).
   V didn't save (background task cut during idle gap). Caveat: non-overlap + fixed-nnz is a weak setup.

## 10:33 P2 KLT + P4 learned-wavelet done (after idle gap; clock 10:3x, deadline 12:45).
- KLT (per-plane PCA): matches DWT ~10–20×, edges it only at ≥100× on Y. Reaches very high compression.
- learned-wavelet (lifting, 6 lvl, len-4 filters, trained per plane): matches DWT at ~10–20×,
  falls behind by ~50×; capped ~50× by always-kept level-6 approx. PR exact (3.6e-7).
- **Stage A: DWT wins/ties everywhere. No learned transform beats it.** (Bug found+fixed in
  lifting _fir circular pad; OOM-killer when sklearn n_jobs=-1 dict ran concurrently w/ lwave → run sequentially.)

## 10:44 P5 Stage B (coherent) done.
- Per-wire transforms CANNOT remove coherent noise (raw nrms ≈2.5 all methods). helix coherent
  removal first → nrms ≈1.2, F0 recovers. DWT best after removal (Y .964/U .903/V .899 @10×).
- **Coherent removal (cross-wire) is the lever, not the per-wire transform.**

## 10:50 P6 interpretability + plots.
- Learned lifting filters = localized oscillatory ~zero-mean (wavelet-like, coif3-ish); learned
  per-level shrinkage ≈ uniform 0.4× → the NN rediscovered wavelet+VisuShrink, didn't beat it.
- Figures: figures/rd_{Y,U,V}.png, stage_b_frontiers.png, learned_wavelets.png. See RESULTS.md.

## 10:50 P7 2D (wire×time) separable wavelet on raw coherent — does NOT remove coherent noise
   (nrms ~2.8, worse than per-wire). Coherent waveform = constant across 64-wire group → lands in
   low-wire-freq bands overlapping signal; generic dyadic threshold can't separate. helix removal
   wins because it's group-structured. → real lever = learn the (group-aware) removal, not the basis.

## 10:56 P8 group-aware LEARNED coherent removal — THE WIN.
- 19k-param CNN estimates per-64-wire-group coherent waveform from median/mean/std + temporal convs,
  supervised on synthetic coherent. Subtract → DWT. Trains ~100s/plane.
- Beats helix removal on ALL planes @10×: Y 0.970 vs 0.964, U 0.915 vs 0.903, V 0.914 vs 0.899.
  Residual nrms (1.61/1.68/1.66) = intrinsic floor → near-perfect coherent removal (cleaner than helix).
- Caveat: trained on this specific coherent model (helix is model-agnostic); modest (~0.01 F0) but consistent.

## ~13:45 P9 R-D push (F0 + #coeffs, unbiased): tried CSC (learned pulse templates + MP),
   top-k / energy selection, oracle ceiling. CSC fails (biased -0.3..-0.6, F0 0.77-0.89 — amplitude
   underfit). top-k beats visu ONLY at shallow L4 (where visu is approx-capped); at fair L8 VisuShrink
   wins and top-k gets badly biased at high compression (-3.28 @100x). energy bad. VisuShrink-hard at
   deep level is best practical (100x @ F0 .91-.94, low bias). Oracle ceiling ~0.02-0.04 F0 above visu
   @100x -> the only headroom is LEARNED COEFFICIENT DETECTION (keep/drop classifier), not the transform.

## SUMMARY: (1) per-wire transform is saturated — fixed DWT VisuShrink is near-optimal; learned
   transforms (KLT, neural wavelet) match but don't beat; dict learning worse; 2D wavelet doesn't help
   coherent. (2) The lever is the cross-wire coherent removal, and a small group-aware CNN that LEARNS
   that removal beats the hand-crafted helix step on all planes. Actionable: keep DWT compression,
   swap in a learned group-aware removal front-end. Full details + figures in RESULTS.md.
