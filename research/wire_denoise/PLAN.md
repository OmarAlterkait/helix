# Wire-plane denoising/compression — learned-transform study

**Goal.** Beat (or characterize) the production per-wire DWT + VisuShrink-hard
pipeline for TPC wire-plane denoise→compress, using data-driven transforms:
PCA/KLT, dictionary learning, and a learnable (neural) wavelet. The optimum is
expected to differ per plane type (U, V induction vs Y collection).

**Started** 2026-05-30 05:46 PDT · **Budget** 7h → hard stop ~12:45 PDT.

## Problem definition
- Data: doraemon `run_0026628546` clean wire truth (noise-free, threshold 2 ADC),
  schema via `pimm_data` reader. Plane images `(n_wires, n_ticks=4321)`.
- Noise (synthetic, added by us): JAXTPC model.
  - **Stage A (simpler, do first): intrinsic only** = FFT-shaped colored series
    (noise_spectrum.npz) + flat white (NOISE_X=0.90). σ≈1.3–2.5 ADC/wire.
  - **Stage B: + coherent** = group-correlated waveform (rms 2.5, β coupling),
    broadcast to 64-wire groups. Harder (cross-wire structure).
- Pipeline shape: noisy → (optional coherent removal) → transform → threshold/sparse-code → inverse.
- **Metric (matches production `compute_metrics_jax`):**
  - `F0 = 1 - Σ|recon-clean| / Σ|clean|` over signal pixels (|clean|>0)  ← primary fidelity
  - `noise_rms` = RMS(recon-clean) over non-signal pixels
  - **Rate**: kept nonzero coeffs / n_pixels (→ compression = n_pixels/n_kept for
    orthonormal transforms). Dictionary is overcomplete → also report index overhead.
- **Comparison = rate–distortion**: sweep each method's sparsity knob → (compression, F0)
  curve PER PLANE TYPE. A method "wins" if its R–D frontier dominates.

## Phases (time-boxed; adjust from LOG)
- **P0 setup** (done): folder, plan, `common.py` (loader/noise/metric/data split).
- **P1 baselines** (~45m): production DWT (coif3 L4) at production κ; sweep
  wavelet family/level/κ per plane; time-domain hard-threshold floor. → R–D bar.
- **P2 KLT/PCA** (~30m): per-plane signal KLT (1D along time), threshold in KLT basis.
  Strong linear baseline; tests whether a learned *orthonormal* basis beats wavelet.
- **P3 dictionary learning** (~1.5h): per-plane overcomplete dict (sklearn
  MiniBatchDictionaryLearning) on 1D time patches (and 2D wire×time if time allows);
  sparse-code noisy (OMP/threshold), reconstruct. R–D per plane.
- **P4 learned wavelet (neural)** (~2h): torch lifting-scheme wavelet (learnable
  predict/update filters → guaranteed perfect reconstruction & invertibility) +
  learnable soft-threshold, trained end-to-end (noisy→clean, L1-on-signal loss),
  PER PLANE. Inspect learned filters vs coif3. Optional: small 1D conv denoiser.
- **P5 coherent noise** (~1h): take the best 1–2 methods from P1–P4, redo Stage B
  (with/without coherent removal front-end). Does the learned transform absorb
  coherent structure, or is explicit removal still needed?
- **P6 synthesis** (~30m): RESULTS.md frontier plots per plane, conclusions,
  recommendation, honest caveats.

## Ground rules
- Fixed train/test event split; never evaluate on training events.
- Same noise seeds across methods for paired comparison at Stage A.
- Document every run in LOG.md with a timestamp + the numbers.
- Check `date` between phases; if behind, cut P3/P4 scope (fewer planes/atoms/epochs),
  never skip documentation or the honest summary.
- Code stays self-contained in this folder; heavy caches in `temp/doraemon_run/explore_cache/`.
