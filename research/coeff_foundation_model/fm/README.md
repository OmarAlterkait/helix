# fm/ — building the wavelet-coefficient foundation model

First implementation of the architecture in `../FULL_ARCHITECTURE.md`. Goal:
overfit sanity → scale up. **The framework (data loading / scaling) is
deliberately thin and THROWAWAY** — it reuses the existing on-the-fly TPC
pipeline and will be replaced by proper pimm-data / pimm integration later. The
*model* is the real deliverable.

## Files
- `model.py` — the architecture, modular (maps 1:1 to FULL_ARCHITECTURE):
  linear embed[values,occ] → **ResponseFiLM(band, plane, wire)** → learned
  band/plane identity → **axial RoPE(physical_time, wire)** in attention →
  K encoder ViT blocks → asymmetric decoder blocks → two heads (occupancy logit
  + value; optional Gaussian-NLL value head via `--nll`). Plain SDPA full
  attention over one event's tokens (cross-plane); batch=1.
- `data.py` — THROWAWAY shim: reuses `star_tpc`/`vit_tpc`/`baseline_tpc` to make
  per-band 2D patch token batches and re-keys fields for the model. To be
  replaced by pimm-data.
- `train.py` — `--overfit` (memorize a tiny set, fixed mask → loss→0 sanity) and
  scale mode (more events, bigger model, fresh masks).

## Status
- **Overfit sanity PASSED** (2 events, d=128, 4 enc + 2 dec blocks, 1.28M params,
  ~100 ms/step): value loss 12 → 0.0125 over 4k steps; train MSE 0.022 = **45×
  below the classical baseline** → the full architecture trains end-to-end and
  has capacity. (Occupancy BCE plateaus ~0.078 — masked-support prediction from
  context is the harder sub-task; value recon is the fidelity signal.)

## Run
```
python train.py --overfit --events 2 --steps 4000 --lr 1e-3      # sanity (passed)
python train.py --events 64 --steps 5000 --d 256 --blocks 8      # scale up
python train.py --overfit --film band,plane --nll               # ablate FiLM / NLL head
```

## Implemented from FULL_ARCHITECTURE (SETTLED + DEFAULT)
per-band patchify (reused) · linear embed · **response-FiLM (band+plane+wire)** ·
axial RoPE(physical-time, wire) + learned scale/plane embeddings · plain
full-attention ViT trunk (cross-plane, single sequence, no windowing/tree-op) ·
MAE masking · asymmetric two-head decoder · masked cross-plane clean-coeff recon ·
L2 head (NLL = `--nll` A/B).

## Not yet / deferred (intentionally)
- Eval backbone (label-efficiency curve + frozen probe) — add before scaling
  conclusions (the one adopted process change; currently we only report MSE).
- MAE encoder-drop (encode visible only) — efficiency; currently mask-in-place.
- flash-varlen multi-event packing — currently batch=1.
- optical modality + cross-modal fusion (shared coord, sim pairs).
- extract-once cache; proper pimm-data loader (replaces `data.py`).
- contingent A/Bs: plane-FiLM-first, NLL head, wavelet, geometry-bias toy.

## Next
1. scale run (64→ events) + held-out generalization (needs train/test split).
2. add the eval backbone (probe on tpc_de/pe_counts) so we judge representation,
   not MSE.
3. the cheap A/Bs (plane-FiLM, masking ratio, NLL) under the large-effect filter.
