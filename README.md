# HELIX

**Hierarchical Encoding for Learned Inference on eXperimental data**

Signal processing pipeline for liquid argon TPC wire data: coherent noise removal followed by wavelet sparsification.

## Installation

```bash
pip install -e .            # CPU only (numpy + pywt)
pip install -e ".[gpu]"     # with JAX GPU acceleration
```

## Usage

### Python API

```python
from helix import DetectorConfig, process_plane, process_event

config = DetectorConfig()
result = process_plane(image, config)

result.cleaned          # coherent noise removed
result.reconstructed    # wavelet-denoised reconstruction
result.sparse.sparsity  # fraction of zero coefficients
```

### CLI

```bash
helix --input sensor.h5 --output processed.h5
helix --input sensor.h5 --output processed.h5 --events 0-19
helix --input sensor.h5 --output processed.h5 --coh-only    # coherent removal only
helix --input sensor.h5 --output processed.h5 --backend jax  # force JAX/GPU
```

## Pipeline

Two sequential stages, each derived from first principles:

### 1. Coherent noise removal

Multi-pass mask-accumulation algorithm. Each pass detects additional
sub-threshold signal on the cleaned output, augments the mask, then
re-estimates coherent noise from the **original** image:

| Pass | Operation |
|------|-----------|
| 1 | group_median -> residual -> mask(3 sigma) -> dilate(11) -> masked_mean -> alpha x subtract |
| 2-3 | detect on cleaned -> augment mask -> re-estimate from original -> alpha x subtract |

The linear alpha = n_unflagged / group_size scales subtraction by estimation confidence.

### 2. Wavelet sparsification

Per-wire coif3 DWT (level 4) with Donoho-Johnstone universal hard threshold (kappa=1.0).
GPU path uses matmul-based DWT/IDWT for full pipeline acceleration.

## Configuration

```python
DetectorConfig(
    group_size=64,              # wires per coherent group
    mask_threshold_nsigma=3.0,  # signal detection threshold
    temporal_dilation_ticks=11, # mask dilation along time axis
    n_passes=3,                 # coherent removal passes
    wavelet="coif3",
    dwt_level=4,
    threshold_kappa=1.0,
)
```

## Performance

Measured on 200 edepsim events (SBND geometry, 1969/1443 wires x 4321 ticks,
5 noise seeds). Noise model matches JAXTPC: FFT-shaped series noise (empirical
MicroBooNE spectrum) + flat white noise + coherent group noise.

### Full pipeline (coherent removal + wavelet)

| Plane | F0 (median) | Bias (ADC/pixel) | RMS (ADC) | Coefficients |
|-------|-------------|------------------|-----------|--------------|
| U (induction) | 0.9015 | -0.146 | 2.480 | 56.8k |
| V (induction) | 0.8884 | -0.190 | 2.496 | 46.3k |
| Y (collection) | 0.9567 | -0.282 | 2.566 | 50.7k |

### Mode comparison

| Mode | F0 (U) | F0 (V) | F0 (Y) |
|------|--------|--------|--------|
| Coherent removal only | 0.899 | 0.898 | 0.963 |
| Wavelet only | 0.845 | 0.832 | 0.933 |
| Full pipeline | 0.902 | 0.888 | 0.957 |

### Timing

| Backend | Per plane | Per event (6 planes) |
|---------|-----------|---------------------|
| NumPy (CPU) | 176 ms | 1.1 s |
| JAX (GPU) | 1.5 ms | 9 ms |

## Scripts

- `scripts/run_metrics.py` -- Run 200-event metrics (3 modes x 5 seeds)
- `scripts/plot_metrics.py` -- Key metrics plots from run_metrics output
- `scripts/plot_steps.py` -- Step-by-step pipeline visualization
- `scripts/plot_gallery.py` -- Multi-event before/after gallery
- `scripts/plot_wavelet_levels.py` -- Per-level DWT band visualization
- `scripts/plot_wavelet_coeffs.py` -- 2D wavelet coefficient maps
- `scripts/passes_sweep.py` -- Sweep n_passes for different sigma/dilation
