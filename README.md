# helix

**hierarchical encoding for learned inference on experimental data**

A foundation model for liquid-argon TPC wire data, and the signal processing that
produces what it trains on. helix covers the whole path: raw wire ADC -> coherent
noise removal and wavelet sparsification -> a coefficient corpus -> a masked
autoencoder over those coefficients -> a 3D probe that asks whether the learned
representation knows where charge is.

## Start here

| you want to | read |
|---|---|
| take this over from someone | **[HANDOVER.md](HANDOVER.md)** |
| run something | **[docs/RUNBOOK.md](docs/RUNBOOK.md)** |
| know why it is shaped this way | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| know what was measured | [docs/SCIENCE.md](docs/SCIENCE.md) |
| run the tests | [TESTING.md](TESTING.md) |
| know what is not in git | [docs/INVENTORY.md](docs/INVENTORY.md) |

First command in a new environment:

    python -m helix.paths

It prints every external path, whether it came from the environment or a
default, and whether it exists. The defaults are where things live on the machine
helix was developed on — they are defaults, not truths.

## Install

    pip install -e .                 # DSP only: numpy, h5py, PyWavelets, scipy
    pip install -e ".[pimm]"         # + the data layer, for the corpus and training
    pip install -e ".[gpu]"          # + jax and torch
    pip install -e ".[probe]"        # + hdf5plugin and torch, to read shards and fit

Or skip all of it and use the image, which has everything:

    /sdf/data/neutrino/omara/images/helix-train.sif

torch and jax are **optional**. `helix.core` and `helix.tpc` import with neither,
and a test enforces it — that is what lets the DSP half run where the training
stack does not exist.

## Scope: the wire path

This repository contains two pipelines. Everything above — corpus, foundation
model, probe — is the **wire** path, and it is what the documentation, the
tests and the runbook describe.

`helix/optical` and `scripts/optical` (18 files) are a **separate PMT-light
pipeline**: wavelet compression of goop waveform chunks, with its own config and
metrics. It shares `helix.core` and nothing else — no corpus, no tokenizer, no
model. It is live code, not dead, but it is not part of the FM pipeline and is
not exercised by the handover validation.

The intent is for optical to grow the same way the wire path did — its own
wavelet compression, its own corpus, feeding the same model. See
`docs/ARCHITECTURE.md` §8 for what that would require and why it has not been
designed yet.

## The two packages

helix works with **pimm-data**, a generic data layer serving several detector
families. The split: helix owns LArTPC physics; pimm-data owns everything
detector-agnostic.

The forward model (intrinsic noise, coherent noise, digitization) is helix's.
Going-to-dense is pimm-data's. They are **lockstep** — the transform registry
raises on duplicate registration, so the two must move together and the image
must be rebuilt when pimm-data changes. `docs/ARCHITECTURE.md` §3 has the detail.

## Pipeline

### 1. Coherent noise removal

Multi-pass mask accumulation. Each pass detects additional sub-threshold signal
on the cleaned output, augments the mask, then re-estimates coherent noise from
the **original** image:

| Pass | Operation |
|------|-----------|
| 1 | group_median -> residual -> mask(3 sigma) -> dilate(11) -> masked_mean -> alpha x subtract |
| 2-3 | detect on cleaned -> augment mask -> re-estimate from original -> alpha x subtract |

`alpha = n_unflagged / group_size` scales subtraction by estimation confidence.

The production gate also applies an occupancy tolerance, `tau = 0.05` — 3 wires
of 64. Measured over 1,200 (event, plane) pairs it gives **5.32x lower stripe
residual** and **2.61x fewer off-signal pixels above 5 ADC**. It is the only
difference between the two corpus generations, and mixing them is the failure
`helix/data/identity.py` exists to prevent.

### 2. Wavelet sparsification

Per-wire coif3 DWT (level 4) with Donoho-Johnstone universal hard threshold
(kappa = 1.0). GPU paths use matmul-based DWT/IDWT.

### 3. The model

A masked autoencoder over the surviving coefficients. Masking whole planes rather
than random cells buys **+0.160** on the cross-plane task and takes the 3D probe
from **0.53 to 0.85**, at a cost on random masking within noise of zero. See
`docs/SCIENCE.md` §2, including a hypothesis that measurement refuted.

## Python API

```python
from helix import DetectorConfig, process_plane, process_event

config = DetectorConfig()
result = process_plane(image, config)

result.cleaned          # coherent noise removed
result.reconstructed    # wavelet-denoised reconstruction
result.sparse.sparsity  # fraction of zero coefficients
```

## CLI

```bash
helix --input sensor.h5 --output processed.h5
helix --input sensor.h5 --output processed.h5 --events 0-19
helix --input sensor.h5 --output processed.h5 --removal gate   # gate|multipass|none
helix --input sensor.h5 --output processed.h5 --to-coeffs      # write a coefficient shard
helix --input sensor.h5 --output processed.h5 --backend jax
```

## DSP performance

200 edepsim events (SBND geometry, 1969/1443 wires x 4321 ticks, 5 noise seeds).
Noise model matches JAXTPC: FFT-shaped series noise (empirical MicroBooNE
spectrum) + flat white noise + coherent group noise.

| Plane | F0 (median) | Bias (ADC/pixel) | RMS (ADC) | Coefficients |
|-------|-------------|------------------|-----------|--------------|
| U (induction) | 0.9015 | -0.146 | 2.480 | 56.8k |
| V (induction) | 0.8884 | -0.190 | 2.496 | 46.3k |
| Y (collection) | 0.9567 | -0.282 | 2.566 | 50.7k |

| Mode | F0 (U) | F0 (V) | F0 (Y) |
|------|--------|--------|--------|
| Coherent removal only | 0.899 | 0.898 | 0.963 |
| Wavelet only | 0.845 | 0.832 | 0.933 |
| Full pipeline | 0.902 | 0.888 | 0.957 |

| Backend | Per plane | Per event (6 planes) |
|---------|-----------|---------------------|
| NumPy (CPU) | 176 ms | 1.1 s |
| JAX (GPU) | 1.5 ms | 9 ms |

The corpus builder defaults to `--backend torch` (10.6 ms/plane), which is what
built the current corpus.

## Scripts

**Corpus** — `build_coeff_corpus.py` (shards), `submit_coeff_corpus.sh` (the
Slurm array), `calibrate_norm_sigma.sh` (the frozen norm_sigma table),
`derive_coeff_bins.py` (the categorical bin grid), `viz_2x2_corpus.py`.

**Training** — `scripts/chain_coeff_fm_train.sh` (a chain of links across the
wall-clock limit and preemption), `scripts/submit_coeff_fm_train.sh` (one link),
`smoke_train_fm.py`, `plot_train_progress.py`.

**Evaluation** — `eval_checkpoint.py` (score a frozen checkpoint),
`dump_probe_truth.py` + `run_probe.py` (the 3D probe, two stages),
`feats_rank.py`, `probe_xattn.py`, `viz_mask_recon.py`.

**Tools** — `scripts/export_artifact.py` promotes a `pimm export` into a helix
eval artifact: weights plus the operating point, the corpus `basis_digest` and
the helix commit. An export cannot say which weight set it holds, which corpus
it trained on, or which code defined its tokenizer; an artifact can, and that is
what makes a probe number attributable.
