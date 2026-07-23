# Scoping: a learned coefficient-space denoiser for HELIX TPC planes

What the data is, what the current pipeline produces, and what would be needed to
build a model that takes noisy planes → does removal + our transform → operates on
the wavelet coefficients per plane per band level. **Scoping only — no model built.**

## 1. The data and how it is extracted

The supervised pair is generated, not stored: clean truth + a *forward* noise model.

**Clean truth** (the target): doraemon LArTPC simulation, noise-free,
pedestal-subtracted, 2-ADC threshold. Loaded via `pimm_data.JAXTPCDataset` as a
SPARSE COO list `(wire, time, value)` and densified to `(n_wires, n_ticks)`
(`research/_tpc_common.py:35`, `research/wire_denoise/common.py:load_clean`).

**Forward noise** (`pimm_data.noise.generate_noise` / JAXTPC, applied per event with a seed):
- **coherent**: one waveform per 64-wire block, broadcast identically to all wires
  in the block (rank-1 common-mode), neighbor blocks anti-correlated by β=0.15,
  spectrum `1/(1+f/20kHz)^0.75` (low-frequency), ~2.5 ADC RMS.
- **intrinsic**: per-wire independent, FFT-shaped colored series + flat white;
  per-wire σ = √(x² + (y + z·L)²) from wire length L (`config.py:wire_sigma_intrinsic`).
- then `digitize(clean + noise, pedestal)`.

So `noisy = digitize(clean + coherent + intrinsic)`. In simulation **all three
components are individually available** — we can supervise on the clean image, on the
coherent component alone, or on the coefficients of any of them.

**Planes** (3 per event, geometry from the registry):

| plane | type | n_wires | signal character | difficulty |
|---|---|---|---|---|
| U | induction | 1969 | bipolar | hardest (dense parallel tracks) |
| V | induction | 1969 | bipolar | medium |
| Y | collection | 1443 | unipolar, large | easiest |

`n_ticks` ≈ 2701 (older config default) or 4321 (newer doraemon run). **Drift to
reconcile:** two data roots / tick counts are referenced — `_tpc_common.py` uses
`/sdf/home/o/omara/data/omara/doraemon` (N≈2701) and `wire_denoise/common.py` uses
`/sdf/home/o/omara/neutrino_data/omara/doraemon` (N=4321). Pick one canonically.

> Note: these data roots and the loader deps (`pimm_data`, `JAXTPC`,
> `tools.coherent_noise`) live OUTSIDE the helix repo (home dirs + sibling repos).
> A model effort needs a stable, in-repo dataset boundary (see §4.1).

## 2. What the pipeline produces — the model's input/output tensor

Current pipeline (`tpc/pipeline.py`): `remove_coherent` → `sparsify` → `reconstruct`.
The DWT is **per-wire, 1-D, linear, fixed** (coif3, level 4, periodization). The
coefficients are a list of per-band 2-D arrays, **one row per wire**:

```
SparseResult.coeffs = [cA4, cD4, cD3, cD2, cD1]   # numpy/torch backend: list of (n_wires, len_band)
```

Per-band lengths (periodization halves each level):

| band | scale | len @ N=2701 | len @ N=4321 | content (from research) |
|---|---|---|---|---|
| cA4 | coarsest approx | 169 | 271 | signal (Y) + coherent (Gaussian) |
| cD4 | coarse detail | 169 | 271 | coherent-dominated; signal common-mode |
| cD3 | | 338 | 541 | mixed |
| cD2 | | 676 | 1081 | intrinsic-leaning |
| cD1 | finest | 1351 | 2161 | intrinsic/white-dominated (thresholded away) |

This **is** the (wire × level × time) lattice the model idea targets: each level is a
2-D field `(n_wires, len_level)`, with a **different time resolution per level**
(dyadic: coarse bands are short/low-rate, fine bands long/high-rate). A coefficient at
band j, position p corresponds to a time support of ~2^j ticks centred near p·2^j.

After production thresholding (`config.threshold_spec()`: per-band MAD σ, hard, also
thresholds approx) the bands are **sparse** (~50–60k nonzeros/plane out of ~2700·n_wires)
— the "kept coefficients." Pre-threshold they are dense.

## 3. Physics priors that should shape the architecture

From `research/coherent_coeffs/RESULTS.md` (extensive prior analysis):
1. **Coherent is exactly rank-1 within each 64-wire block** (bit-identical coeffs
   across the 64 wires, every level). → strong block structure; a model should be
   ~equivariant over the 64-wire group, and the block axis is the natural place to
   read the common mode. Block boundaries are sharp → not translation-invariant across
   blocks in the wire direction.
2. **Components separate by scale**: coherent → coarse (cA4/cD4, Gaussian, dense,
   high σ); signal → coarse but large+sparse (Y) or bipolar-split (U/V); intrinsic →
   mid-band (cD3) flatter; finest band cD1 ≈ pure white noise. → per-level processing
   with **per-level normalization** is essential (~10× amplitude differences between
   levels; one global threshold fails — established).
3. **The separable part is essentially solved** by classical methods: smart level-aware
   gating already reaches the no-coherent oracle on *coefficient count* (compression),
   and de2_clamp gets within ~0.004 (Y) / ~0.012 (V) / ~0.024 (U) F0 of the reachable
   ceiling. **The only open lever is DETECTION of signal-occupied (block,tick) cells in
   dense regions at SNR≈1** — and the research explicitly names *"a learned detector is
   the only untried lever for the detection gap."* That is the model's real target,
   especially on U/V. (Y is already near-optimal classically.)
4. **U has an information-theoretic floor**: tracks parallel to wires for >50 ticks make
   signal the majority of a block with no temporal anchor — coherent there is
   unrecoverable by any estimator that lacks the true mask. Set expectations: the model
   can chase the detection gap, not the floor.

## 4. What would be needed to build it

### 4.1 Data / dataloader (the biggest missing piece)
- A **torch `Dataset`** yielding paired tensors: noisy planes (input) + clean planes
  and/or clean-coefficient targets + (optionally) the isolated coherent component for
  auxiliary supervision. On-the-fly noise generation (seeded) gives unlimited pairs;
  cache clean truth (the `wire_denoise` npy cache pattern already does this).
- **Decouple from external repos**: today loading needs `pimm_data` + `JAXTPC` +
  `tools.coherent_noise` with hardcoded home paths. Either vendor a minimal forward
  model into helix or wrap it behind one stable interface. Reconcile the 2701/4321 +
  two-data-root drift first.
- **Batching across ragged planes**: U/V (1969) and Y (1443) differ in wire count, and
  every band differs in length. Either train per-plane-type (3 models / 3 heads) or pad
  + mask. Per-plane heads is the cleaner default given induction vs collection differ
  physically.

### 4.2 Backend gaps in helix (must close for a torch training loop)
- **No torch `remove_coherent`** — `coherent.py:26` raises for torch (numpy/jax only).
  If the model *replaces* removal this is moot; if it consumes pre-cleaned input, you
  need a torch port or to precompute cleaned planes (numpy/jax) into the cache.
- **torch `sparsify` ignores `per_band_sigma` and `threshold_approx`**
  (`wavelet_ops_torch.py:132-136` always keeps approx untouched, single per-signal σ),
  but TPC production uses both (`config.py:threshold_spec`). For training targets to
  match production coefficients, either reconcile the torch path or generate targets via
  numpy. The **DWT/IDWT themselves are coefficient-identical across backends** (verified
  ~1e-14), so the *transform* is safe to run in torch and is differentiable (FFT-based,
  `wavelet_ops_torch.py`) — good: the model can backprop through IDWT to an image-space
  (F0) loss.

### 4.3 Representation decisions (the core of the architecture)
The model operates on per-level fields `(n_wires, len_level)`. Map the stated ideas:
- **Patchify per level (wire × time), different per level**: because coarse bands are
  short and fine bands long, patch sizes should be level-specific (e.g. align time
  patches to a fixed *tick* footprint → larger position-patches at fine levels). The
  64-wire block is the natural wire-patch unit (matches the coherent rank-1 structure).
- **Sparse convolution per level, then mix**: choose dense vs sparse input. Sparse conv
  (spconv, profiled and available) fits the *post-threshold* kept set (~2–4% occupancy)
  and the block structure; but removal/detection likely needs the *pre-threshold* dense
  field (the signal you must detect is sub-threshold). Likely: dense per-level conv for
  detection/removal, sparse representation only for the kept output / compression head.
  "Mix" = cross-level fusion (coarse↔fine), which must respect the dyadic time
  alignment (a coarse cell spans 2^j fine cells).
- **Cross-plane**: U/V/Y see the same event from 3 angles; a late-fusion option exists,
  but planes have different wire counts and the wire↔wire correspondence is non-trivial.
  Start per-plane; treat cross-plane as a later lever.

### 4.4 Supervision / targets / metrics
- **Targets available**: clean image (for F0/L1 image-space loss via differentiable
  IDWT), clean coefficients (oracle = `sparsify(signal+intrinsic)` or `sparsify(signal)`),
  the isolated coherent component (direct regression of what to subtract), and the
  true signal-occupancy mask (the detection-gap target — this is the lever §3.3).
- **Metrics already defined** (`_tpc_common.py`, research): **F0** (charge fidelity on
  signal pixels), **nz_in / nz_out** (residual RMS on/off signal — nz_out is the solved
  floor ~1.6 ADC, **nz_in is the real discriminator**), **kept-count** (compression).
  The classical baselines (helix, smart, de2_clamp, oracle, no-coherent ceiling) are all
  quantified per plane — use them as the bar to beat, plane by plane.
- **Honest framing** (from research §6n): gaps beyond smart are ~1% charge, below LArTPC
  calorimetric resolution and inside per-event scatter. A learned model is worth it only
  if it closes the *U/V detection gap* (≈+0.03 F0 on U reachable mask) that no classical
  method reached — that should be the explicit success criterion, not marginal Y gains.

## 5. Key decisions to settle before coding
1. **Does the model replace `remove_coherent`, or post-process its output?** (Drives
   whether a torch coherent backend is needed and what the input is.)
2. **Operate on pre-threshold (dense) or post-threshold (sparse) coefficients?** (Dense
   for detection/removal; sparse for the compression head. Probably both, staged.)
3. **One model with per-plane heads, or 3 separate models?** (Induction vs collection
   differ physically; per-plane heads recommended.)
4. **Primary loss: image-space F0 (backprop through IDWT) or coefficient-space?**
   (IDWT is differentiable in torch — image-space loss is available and matches the
   real metric.)
5. **Canonical dataset**: which doraemon root / tick count, and where the forward-model
   boundary lives inside helix.

## 5b. The pimm_data dataloader — what exists, what must change

Data: **5.9 TB** sparse sensor HDF5 at `/sdf/data/neutrino/omara/JAXTPC_Wire/test_00_00_02/`
(`sensor/`, `hits/`, `step/` per `run_*`), loaded by `pimm_data.JAXTPCDataset`
(submodule: `particle-imaging-models/libs/pimm-data`). Foundation-model scale.

**What already exists (and is exactly right for the input side):** a post-collate,
**on-device (torch, GPU), born-on-GPU** dense pipeline — only the sparse hits cross
PCIe; dense grids + noise are created on the device:
- `JAXTPCDataset.get_data` → sparse COO `{wire,time,value,plane_gid,...}` + `plane_geometry()`
  registry (`{gid: n_wires, n_ticks, pedestal, wire_lengths}`).
- `batch_transforms.build_sensor_gpu_stages` →
  `BatchDensify` (sparse→`{gid:(B,W,T)}`) → `BatchAddIntrinsicNoise`
  (coherent numpy-oracle bit-exact to JAXTPC + incoherent torch-FFT) → `BatchDigitize`.
- Output `batch['sensor_dense'] = {gid: (B, W_p, T)}` = **the noisy planes the model
  takes in.** Per-event content-addressed seeds (`content_seed`): fold `epoch` in →
  fresh noise per epoch (augmentation); fix it → static. Noise defaults
  (`noise.py`: ENC 0.90/0.79/0.22, coh_rms 2.5, β 0.15, group_size 64) are **identical
  to the helix research physics** — same forward model, no drift.

**What must change / be added for the coefficient foundation model:**
1. **Preserve the clean target.** `dense_ops.add_intrinsic_noise` mutates grids
   **in place** (`grids[gid][b] += …`), destroying the clean image. The dense path was
   built to train detectors on noisy data (no clean label needed). For a denoiser/coeff
   model the clean grid IS the label → add a stage/flag that clones `sensor_dense` →
   `sensor_dense_clean` before noise, and optionally returns the **isolated coherent /
   incoherent components** (auxiliary supervision + the true target the research wants).
2. **Add a wavelet-coefficient stage (does not exist — confirmed: zero dwt/coeff refs
   in pimm_data).** New `BatchWavelet` stage: `{gid:(B,W,T)} → {gid:{band:(B,W,len_band)}}`
   via helix `core.wavelet_ops_torch._wavedec` (differentiable, GPU, backend-exact).
   Produce **pre-threshold** coeffs — thresholding/removal is the *model's* job
   (end-to-end), not the loader's.
   - Requires T padded to a multiple of `2^level` (torch periodization); add a pad
     (the optical pipeline already does this pattern). 
   - Creates a **pimm_data → helix dependency** (direction matches the cross-repo
     contract: helix owns algorithms). `wavelet_ops_torch` imports `pywt` at load for
     the filter bank — **`pywt` must be in the training env** (it was absent in the pimm
     container) or precompute/vendor the filter coeffs.
3. **Geometry config**: set `wire_lengths_per_plane` (U/V ≈ (0.42, 4.63) m, Y = 2.33 m
   from `wire_denoise/common.py`) so the incoherent stage has L; confirm the sensor
   reader surfaces `n_wires_per_plane`/`num_time_steps` for this run (lazy — read one
   event first).

**Architecture note for "foundation model":** pimm's backbones (PTv3/Sonata) are
**point-cloud** networks (coord/feat/offset). Our model operates on the **per-level 2-D
coefficient lattice** (`(W, len_band)` images, dyadic), not a point cloud — so the
*backbone is new* (sparse-conv-per-level + cross-level mixing, the stated plan), but
pimm's **training infra is reusable**: config system, DDP/launch, hooks, and especially
the **Sonata-style self-supervised pretraining** machinery (the natural way to "train a
large-scale foundation model" on 5.9 TB of mostly-unlabeled events — the denoise/coeff
task is a strong pretext). Reuse infra, write a new coefficient backbone.

## 5c. Prototype — per-level coefficient tensors (`research/coeff_prototype.py`)

Verified end-to-end on the real 5.9 TB data (run_0027575766, 19999 events). Flow
respects the split: **pimm_data extracts sparse clean sensor + geometry; helix does
densify → forward noise → torch DWT**. Output = the model's input contract,
`{gid: {band: (B, W, len_band)}}`, pre-threshold, for clean (target) and noisy (input).

Confirmed facts from the data/config: stored sensor is **clean truth**
(`include_coherent_noise=False, include_intrinsic_noise=False, include_digitize=True`,
threshold 2 ADC, 12-bit, electrons_per_adc=182); **n_ticks=4321**; geometry from
`pimm_data/.../cubic_wireplane_geometry.json` — U/V **1969** wires (ped 1843), Y **1443**
(ped 410); coherent {group 64, β 0.15, rms 2.5, corner 20 kHz, slope 1.5}. Per-band
lengths (T padded 4321→4336 for 2^4): cA4=cD4=271, cD3=542, cD2=1084, cD1=2168.

**The learning problem, visible in coeff space** — occupancy (|coeff|>2 ADC):

| band | clean | noisy |
|---|---|---|
| cA4 | 1.3–1.7% | ~80% |
| cD4 | 0.9–1.3% | ~64% |
| cD3 | 0.5–0.8% | ~51% |
| cD2 | 0.1–0.3% | ~39% |
| cD1 | ~0.005% | ~31% |

Clean signal is sparse + coarse-concentrated (cD1 ≈ empty); noise fills the lattice
densely, worst in coarse bands (coherent's home). Model maps dense noisy → sparse clean.

**Sizes / budget:** one batch B=2 × 3 planes (one volume), pre-threshold float32 =
**187 MB**. So per-event per-volume ≈ 31 MB of dense coeffs; a real batch needs either a
sparse representation, per-level streaming, or bf16 — relevant to the sparse-conv-per-level
design and to GPU-memory budgeting (cf. the A100 profiling in `../particle-imaging-models/tools/`).

**Env note (read-only container):** `pywt` and `hdf5plugin` (blosc/zstd codecs, required
to read the compressed sensor HDF5) are not in the pimm container and its site is
read-only → installed to `helix/.pylibs` and added to `sys.path` in the prototype.
`hdf5plugin` must be imported before any h5py read. A real training env should add both
as proper deps.

## 6. What already exists to build on
- Differentiable, backend-exact torch DWT/IDWT (`core/wavelet_ops_torch.py`).
- Full classical baseline suite + metrics + per-plane numbers
  (`research/coherent_coeffs/`, `research/wire_denoise/`).
- The forward noise model and clean-truth loaders (`research/_tpc_common.py`,
  `research/wire_denoise/common.py`) — to be wrapped behind a stable Dataset.
- spconv / flash-attn / PTv3 availability + GPU profiling already done in
  `../particle-imaging-models/tools/` (sparse-conv-per-level is feasible on the A100).
