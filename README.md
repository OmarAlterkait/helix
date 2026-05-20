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
from helix import DetectorConfig, process_plane

config = DetectorConfig.sbnd()
result = process_plane(image, config)

result.cleaned          # coherent noise removed
result.reconstructed    # wavelet-denoised reconstruction
result.sparse.sparsity  # fraction of zero coefficients
```

### CLI

```bash
helix --input sensor.h5 --output processed.h5
helix --input sensor.h5 --output processed.h5 --config configs/microboone.toml --events 0-19
```

## Pipeline

Two sequential stages, each derived from first principles:

### 1. Coherent noise removal

Multi-pass mask-accumulation algorithm. Each pass detects additional
sub-threshold signal on the cleaned output, augments the mask, then
re-estimates coherent noise from the **original** image:

| Pass | Operation |
|------|-----------|
| 1 | group_median → residual → mask(3σ) → dilate(±5) → masked_mean → α×subtract |
| 2–3 | detect on cleaned → augment mask → re-estimate from original → α×subtract |

The adaptive α = n_unflagged / group_size scales subtraction by estimation confidence.

### 2. Wavelet sparsification

Per-wire coif3 DWT (level 4) with Donoho-Johnstone universal hard threshold (κ=1.0).
GPU path uses matmul-based DWT/IDWT for full pipeline acceleration.

## Configuration

```python
DetectorConfig(
    group_size=64,              # wires per coherent group
    mask_threshold_nsigma=3.0,  # signal detection threshold
    temporal_dilation_ticks=11, # mask dilation (±5 ticks)
    n_passes=3,                 # coherent removal passes
    wavelet="coif3",
    dwt_level=4,
    threshold_kappa=1.0,
)
```

## Performance

Measured on 200 edepsim events (SBND geometry, 1969/1443 wires × 4321 ticks, 5 noise seeds):

| Plane | F0 (median) | Bias (ADC/pixel) | RMS (ADC) | Sparsity |
|-------|-------------|------------------|-----------|----------|
| U (induction) | 0.9145 | −0.069 | 2.218 | 99.22% |
| V (induction) | 0.9059 | −0.113 | 2.175 | 99.38% |
| Y (collection) | 0.9634 | −0.164 | 2.230 | 99.05% |

| Backend | Per plane | Per event (6 planes) |
|---------|-----------|---------------------|
| NumPy (CPU) | 176 ms | 1.1 s |
| JAX (GPU) | 1.5 ms | 9 ms |
