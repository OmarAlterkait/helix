# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

HELIX is a signal-processing toolkit for liquid-argon TPC detector data, organized as three packages:

- **`helix.core`** — detector-agnostic wavelet sparsification + a lazy multi-backend dispatcher.
- **`helix.tpc`** — wire-plane pipeline: coherent noise removal → wavelet sparsification. Each plane is a dense `(n_wires, n_ticks)` float32 image (pedestal-subtracted ADC). Quality metric **F0** = `1 - sum|recon-clean|/sum|clean|`.
- **`helix.optical`** — PMT optical-waveform pipeline for goop "light" files. Operates on goop's stored chunks (gap-compressed "stitches") directly; wavelet-sparsifies them for compression.

Top-level modules (`helix.config`, `helix.coherent`, `helix.wavelet`, `helix.pipeline`, `helix.io`, `helix._backend`, `helix._numpy_ops`, `helix._jax_ops`, `helix._dwt_matrix`) are **back-compat shims** re-exporting from the new packages — single source of truth lives in `core`/`tpc`. Don't add logic to the shims.

## Commands

```bash
pip install -e .                  # base: numpy + pywt + scipy + h5py
pip install -e ".[jax-cpu]"       # + jax CPU      (or .[jax-gpu] for cuda12)
pip install -e ".[torch-cpu]"     # + torch CPU    (torch is the optical/GPU wavelet backend)
pip install -e ".[dev]"           # + pytest

pytest                            # full suite (~10s; torch/jax tests importorskip if absent)
pytest tests/test_optical.py -q   # one file
python tests/bench_numpy.py       # legacy numpy-ops microbenchmarks

helix-tpc --input sensor.h5 --output out.h5 --backend jax   # TPC CLI (alias: helix); --coh-only, --events 0-19
python scripts/optical/sweep_big.py depth 100              # optical rate-distortion campaign, sharded across all GPUs (stages: depth|complete_wav|complete_lev|complete_meth)
```

## Architecture

### Lazy multi-backend dispatch (`helix.core.backend`) — the core structural pattern

Heavy frameworks (`jax`, `torch`) are imported **only when their backend is selected and used**, never at `import helix` time (numpy is the default; the test suite dropped from 15s→0.6s because of this). Mechanism:

- Each op *family* ships one module per backend: `<family>_<backend>.py`. Currently `core/wavelet_ops_{numpy,jax,torch}.py` and `tpc/coherent_ops_{numpy,jax}.py`.
- `backend.ops("helix.core.wavelet_ops")` imports and returns **only the active backend's** module (`importlib`, cached). The framework import lives inside that file.
- Selection precedence: `set_backend(name)` > `$HELIX_BACKEND` > `"numpy"`. Valid: `numpy`, `jax`, `torch`.

To add a backend to a family, drop in `<family>_<newbackend>.py` implementing the same functions — no dispatcher change needed. `coherent` has no torch backend yet (raises a clear error); the optical/torch path is the wavelet workhorse.

### Wavelet core (`helix.core.wavelet`)

`sparsify(image, *, wavelet, level, mode, threshold, sigma=None)` → `SparseResult`; `reconstruct(result, n_time)`. `ThresholdSpec` selects the strategy:

- **universal = VisuShrink** (`func='hard'|'soft'|'garrote'`): `t = scale·σ·sqrt(2 ln N)` with a **per-signal** noise σ — caller-supplied via `sigma=`, else MAD of the *finest* detail band per row. (Do NOT use per-band σ — it's signal-contaminated in coarse bands and over-thresholds; that was a bug.)
- **topk**: keep top `keep` fraction of detail coeffs per signal.
- **energy**: keep smallest set holding `energy` fraction of detail energy per signal.

Approximation band kept untouched when `include_approx`. **Validated finding (100-event campaign): for this denoise-then-compress task, noise-relative VisuShrink-hard is the right method — energy/topk are content-relative and either leak on noise-only regions (energy) or are noise-blind (topk); SURE/Bayes barely threshold high-SNR signal. Wavelet family/level barely matter (coif/sym, level≈8–12). `scale` (κ) is the rate knob.**

Backend differences that callers must respect:
- **numpy** = pywt (`per_wire DWT`, any length, exact). `coeffs` = list `[cA, cD_L, …, cD_1]`.
- **jax** = matmul DWT via precomputed pywt-exact matrices (`core/dwt_matrix.py`). `coeffs` = flat `(n_sig, n_coeffs)` array. Only for SHORT signals (a 36k² matrix is ~5 GB — never use jax wavelet on optical chunks).
- **torch** = FFT-based periodic DWT (`wavelet_ops_torch.py`), batched on GPU, scales to long signals, machine-precision perfect reconstruction. `coeffs` = list of torch tensors. It is *self-consistent* (PR exact) but **not coefficient-identical to pywt** (different DWT phase) — compare sparsity within-backend, not across.

### TPC pipeline (`helix.tpc`)

`process_plane` → `remove_coherent` (multi-pass mask-accumulation: group median → residual → kσ mask → temporal dilation → masked group mean → subtract `α·estimate`, `α=n_unflagged/group_size`; passes 2..n detect on the *cleaned* output but re-estimate/subtract from the *original*) → `core.sparsify` (universal hard, built from `DetectorConfig.threshold_spec()`). `DetectorConfig` is a frozen dataclass; geometry comes from `tpc.io.config_from_file`. Sensor HDF5: `event_<N>/<plane>` with sparse COO (`wire`,`time`,`values`) + pedestal/n_wires attrs.

### Optical pipeline (`helix.optical`)

goop light files are **two-sided** (`event_NNN/{east,west}` + `pe_counts_{side}`), each side a CSR-style `SlicedWaveform` (`adc`/`offsets`/`t0_ns`/`pmt_id`). **The upstream goop loader expects `label_N` groups and will NOT read these** — use `optical.io` (east/west schema).

**Default method** (`OpticalConfig`): per-chunk DWT (coif3, level 10, periodization) → **VisuShrink-hard** (`ThresholdSpec("universal","hard",scale=1.2)`) → quantize survivors to `quant_bits` (12). The default **κ=1.2 is the 1× noise-RMS operating point** (recon RMS over signal ≈ noise floor ~2.57 ADC, ~33×); κ≈1 is pure denoising (~25×), κ≈1.75/2.5 → 2×/3× noise (~49×/85×). `process_event` reads **stored chunks directly** (no deslicing), pedestal-subtracts, pad-batches to a multiple of `2^level`, and runs one batched `core.sparsify`. **σ is computed by `io.chunk_noise_sigma` from the UNPADDED chunk (db1 MAD) and passed to `sparsify` — never estimate σ from the padded batch (the zero-padding collapses the MAD → threshold collapses → keeps ~12× too many; this was a real bug).** Note `core.sparsify`'s `universal` thresholds each detail band with its **own** length in `√(2 ln N)` (level-dependent universal), so production κ is not bit-identical to the campaign's total-N "visu" helper — within a few % on total count (coarse bands keep marginally more). `viz.plot_decomposition(signal, style='bars'|'map')` renders the multi-level decomposition with correct dyadic time–scale placement. Metrics in `metrics.py` score peak/area over SIGNAL chunks only (|x|max>50 ADC). `deslice_side` is viz-only.

**Per-level coefficient budget** (κ=1.2, 100 events, `scripts/optical/coeffs_per_level.py` → `plot_coeffs_per_level.py`): ~994 kept coeffs / signal chunk, concentrated at **mid-scales** — D4+D3 hold ~41% of the budget, D1 (finest) keeps ~0.04% of its band (noise-dominated, thresholded away), survival fraction declines monotonically coarse→fine (A10 100% → D10 28% → D1 0.04%). Noise-only chunks keep ~50 coeffs (~920× each).

Data characteristics (goop light_output.h5): 162 ch (81/side), 1 ns ticks, pedestal ~29490, 15-bit, SERKernel (10 µs), noise 2.57 ADC (white, verified). Pulses are large negative-going; single PE (~0.36 ADC) is sub-noise. Stored chunk ~36k samples ⊃ active >3σ ~6.8k. Honest compression: ~3× would be wrong (a padding-σ bug) — the real faithful frontier is **~30× at the noise floor**, since the bright prompt is high-information and only the noise tail compresses for free.

### Gotchas

- `config.beta`/`xblock_kernel` (TPC) exist but are **not wired into** `remove_coherent` (defined-but-unused spatial high-pass).
- jax/torch wavelet `coeffs` differ in type (flat array vs list); `tpc.io.write_processed` handles both.
- `temp/` holds a throwaway goop clone + scratch analysis (gitignored intent; pytest `norecursedirs` skips it).

## scripts/ layout

- `scripts/optical/` — optical compression analysis toolkit (sweeps, frontier, transform/WPT/zerotree comparisons, plots). `sweep_big.py` is the campaign engine + canonical method helpers; see `scripts/optical/README.md`. All shard across available GPUs. Campaign results + manifest live in `temp/figures/big/` (`RESULTS.md`, `*.jsonl`, `*.json`).
- Other `scripts/*.py` (TPC metrics/plots) hardcode another machine's paths and import external `pimm_data.jaxtpc` / `tools.coherent_noise` — reference only, not runnable here.
