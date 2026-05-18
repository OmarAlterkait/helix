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

Single-pass algorithm with four operations matched to the noise correlation structure:

| Step | Operation | Why |
|------|-----------|-----|
| Mask | \|dig − group_median\| > 3σ | Identify signal pixels |
| Temporal dilation | ±4 ticks | Coh and leakage both correlated in time → exclude |
| Spatial kernel | (−β, 1, −β) across groups | Coh anti-correlated, leakage correlated → filter |
| α subtraction | n_unflag / group_size | Scale by estimation confidence |

### 2. Wavelet sparsification

Per-wire coif3 DWT (level 4) with Donoho-Johnstone universal hard threshold (κ=1.0).

## Configuration

```python
DetectorConfig(
    group_size=64,           # wires per coherent group
    beta=0.15,               # inter-group anti-correlation
    temporal_dilation_ticks=9,
    wavelet="coif3",
    dwt_level=4,
    threshold_kappa=1.0,
)
```

Presets: `DetectorConfig.sbnd()`, `.microboone()`, `.icarus()`

## Performance

| Backend | Per plane | Per event (6 planes) |
|---------|-----------|---------------------|
| NumPy (CPU) | 176 ms | 1.1 s |
| JAX (GPU) | 1.5 ms | 9 ms |
