# scripts/optical — optical compression analysis toolkit

The production optical pipeline lives in `helix.optical` (default method:
VisuShrink-hard DWT + quantization). These scripts are the **research/analysis
tools** used to validate it and explore the rate-distortion space. All read goop
`light_output.h5` and shard across available GPUs (`torch.cuda.device_count()`).

## Sweep engine
- `sweep_big.py <stage> [n_events]` — the campaign engine. Also defines the
  canonical helpers `threshold()` (visu/leveldep/sure/bayes/minimax/hybrid/fdr/
  block), `quant()`, `raw_sigma()` that the other scripts import.
  Stages: `depth`, `complete_wav` (all 105 wavelets), `complete_lev` (levels 1–13),
  `complete_meth` (all threshold methods).
- `analyze_big.py <stage> [peak|rms]` — aggregate + Pareto + write `final_configs.json`.

## Rate / distortion / fidelity
- `final_frontier.py` — Pareto configs with entropy-coded + fixed-bit event-total rate.
- `abs_rms_gpu.py` — absolute-RMS-vs-noise frontier data → `abs_rms.json`.
- `active_vs_kept.py` / `kept_per_event.py` — per-chunk / per-event coefficient stats.
- `coeffs_per_level.py` → `plot_coeffs_per_level.py` — **per-DWT-band** kept-coefficient
  distributions for the default method (where in the multi-scale decomposition the
  surviving coefficients live). → `big/coeffs_per_level.json`.

## Method/transform comparisons
- `transform_compare.py` — DWT vs DCT vs FFT.  `wpt_compare.py` — wavelet-packet best-basis.
  `zerotree_estimate.py` — EZW/SPIHT position-coding gain.

## Plots (read the saved *.json — instant restyle)
- `plot_rms_noise.py`, `plot_pareto_clean.py`, `plot_final.py`, `plot_active_vs_kept.py`,
  `plot_coeffs_per_level.py`

Results + manifest: `temp/figures/big/` (`RESULTS.md`, `*.jsonl`, `*.json`).
