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
    pip install -e ".[gpu]"          # + torch
    pip install -e ".[probe]"        # + hdf5plugin and torch, to read shards and fit

Or skip all of it and use the image, which has everything:

    /sdf/data/neutrino/omara/images/helix-train.sif

torch is **optional**. `helix.core` and `helix.tpc` import without it,
and a test enforces it — that is what lets the DSP half run where the training
stack does not exist.

## Scope: the wire path

This repository contains two pipelines. Everything above — corpus, foundation
model, probe — is the **wire** path, and it is what the documentation, the
tests and the runbook describe.

`helix/optical` holds the **optical path**: one codec (`helix.optical.codec`) for
every light system - SBND PMT and X-ARAPUCA data (`helix.tpc.sbnd_run`) and goop's
simulated PMTs (`python -m helix.optical.goop_run --input light_output.h5 --output
goop_shard.h5 [--device cuda]`): deconvolution with the system's measured
single-photon response, integer-Haar differences kept where the light (or the
departure from it) reaches threshold, coarse sums always; written as optical shards
(`helix.optical.shard`, one modality per file next to the TPC coefficient shards) with 128-slot tokens (`helix.optical.tokens`),
so simulation and data share one representation. The optical tokens are not yet fed to the model.

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

Both the simulation corpus and SBND data run through one chain,
`helix.tpc.chain.TPCChain` (simulation: `sim_config(layout.num_wires)`; SBND:
`SBND_DEFAULT`): pedestal and channel status, DWT, coherent removal, threshold.

### 1. Coherent noise removal

A coefficient-space gate (no image round trip: the DWT is linear). Per band and
per pass, a robust common mode is estimated per block of wires (64-wire blocks
in simulation, `helix.tpc.coherent_gate`; front-end-board groups across
planes on SBND, `helix.tpc.coherent_gate_labels`), subtracted where it looks like
coherent noise and refused where it looks like signal. Two passes: the second
excludes signal detected on the first pass's output.

The production gate also applies an occupancy tolerance, `tau = 0.05` — 3 wires
of 64. Measured over 1,200 (event, plane) pairs it gives **5.32x lower stripe
residual** and **2.61x fewer off-signal pixels above 5 ADC**. It is the only
difference between the two corpus generations, and mixing them is the failure
`helix/data/identity.py` exists to prevent.

### 2. Wavelet sparsification

Per-wire coif3 DWT (level 4, periodization) with a hysteresis hard threshold
(`helix.tpc.threshold_rows`): seeds at the universal threshold (kappa = 1.0),
grown through 2 sigma; the finest detail band D1 is dropped. The DSP is torch (GPU,
or CPU) with a DWT that is coefficient-identical to pywt.

### 3. The model

A masked autoencoder over the surviving coefficients. Masking whole planes rather
than random cells buys **+0.160** on the cross-plane task and takes the 3D probe
from **0.53 to 0.85**, at a cost on random masking within noise of zero. See
`docs/SCIENCE.md` §2, including a hypothesis that measurement refuted.

## Python API

```python
from helix.tpc.chain import TPCChain, sim_config
from helix.tpc.corpus import process_planes
from helix.tpc.io import sensor_layout
from helix.tpc import chain_coeff

layout = sensor_layout("sensor.h5")
chain = TPCChain(sim_config(layout.num_wires))      # device="cuda" for torch
results = process_planes(chain, noisy_planes)       # {gid: (wires, ticks) image}
ce = chain_coeff.coeff_event(chain, results)        # -> CoeffEvent
```

## Command line

```bash
python scripts/build_coeff_corpus.py --out <corpus dir> ...        # simulation corpus
python -m helix.tpc.sbnd_run --input RUN_DIR/*.h5 --output OUT/PREFIX --device cuda   # SBND
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

The corpus builder runs the chain on the GPU: 10.6 ms/plane on a 2080 Ti.

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
