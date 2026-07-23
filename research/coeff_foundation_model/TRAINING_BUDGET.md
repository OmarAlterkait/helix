# Training budget — parallel loading, batching, iterations, wall-clock

Measured on 1 A100 (this container; cluster = 5 GPU projection). 28 CPUs.
Batch = 1 event (all 6 planes -> one cross-plane token set, ~30k tokens cap).

## Per-event cost (CORRECTED, measured free GPU, bf16; batch = 1 event = ~30k tok)
| stage | ms | hideable |
|---|--:|---|
| CPU h5 extract (build_batch, 1 ev/6 planes) | 63 | workers |
| numpy assemble | 141 | workers |
| GPU: noise stages (densify+coh+incoh+digit) | 78 | — |
| GPU: DWT (x2 noisy+clean) | ~30 | cache clean -> x1 |
| GPU: smart_gate_bands | 49 | — |
| GPU: clean stages | ~50 | cache clean -> 0 |
| **model fwd+bwd (30k tok, 4 blk d192) — bf16** | **156** | DDP / MAE-drop |
| serial total (bf16) | **~520** (measured 522) | |

CORRECTIONS to the first estimate (the user caught these):
- **"model 652 ms" was fp32 with no autocast.** SDPA fp32 falls back to the
  math kernel (one 30k attn: 41 ms fp32 vs 6 ms bf16). **bf16 model = 156 ms
  (6.4x).** Now fixed in baseline_tpc (autocast added).
- **"GPU pipeline ~400 ms" was a mis-attribution.** Real GPU pipeline ~207 ms;
  the rest was the numpy assemble (141, CPU) + h5 read done TWICE (the noisy
  and clean passes each call build_batch -> 126 ms reading the same event;
  dedupe saves ~63). True split: CPU ~204 (hideable) / GPU ~207 / model 156.
- batch size = **1 event = all 6 planes = ~30k tokens**, one cross-plane set.

## Parallelization levers (quantified)
1. **DataLoader workers (24 CPU)** hide CPU extract+assemble (~200 ms)
   -> step ~1050 ms. (The explicit ask; the onfly_optical pattern.)
2. **Cache clean DWT** (clean is deterministic across epochs; only noise
   resamples) -> halve GPU prep (~-200 ms) -> ~850 ms.
3. **MAE encoder-drop** (encode only the ~50% VISIBLE tokens; lightweight
   decoder predicts masked) -> attention 30k->15k, ~4x model -> model ~170 ms
   -> step ~370 ms. BIGGEST single lever; it is also standard MAE. (Current
   code keeps masked tokens in-place = simpler but pays full attention.)
4. **Multi-GPU DDP (cluster, 5 GPU)** -> throughput x5 (events independent;
   near-linear, grads all-reduced once/step). GPU-hours unchanged; wall /5.

## Iterations & wall-clock (1 event/step; corpus = 580k unique TPC events)
"1 epoch" = 580k steps. GPU-hours = wall x #GPU (independent of #GPU).

| config (bf16) | ms/step | GPU-h/epoch | wall/epoch 1GPU | wall/epoch 5GPU |
|---|--:|--:|--:|--:|
| serial (no workers) | 520 | 84 | 3.5 d | 0.7 d |
| + workers (hide ~204 CPU) | ~360 | 58 | 2.4 d | 0.48 d |
| + clean-cache (skip clean pipeline) | ~290 | 47 | 1.9 d | 0.39 d |
| + MAE encoder-drop (~50% tokens) | ~190 | 31 | 1.3 d | 0.26 d |

10 effective epochs over 580k unique events (likely enough at this corpus
size), config "+workers+clean-cache" (~290 ms): **~470 GPU-h, wall ~19 d
(1 GPU) / ~3.9 d (5 GPU)**. With MAE-drop: ~310 GPU-h, ~13 d / 2.6 d.
GPU-hours sit well inside 10-50k; wall-clock is the real constraint and scales
1/#GPU.

## Recommended for the SCALED run (when validated)
- workers for read (lever 1) + clean-cache (2) + MAE encoder-drop (3): the
  three together take 1261 -> ~370 ms/step, no quality cost (MAE-drop is the
  standard recipe). Then DDP across whatever GPUs the cluster gives.
- pack 2-4 events/GPU with a block-diagonal (per-event) attention mask to lift
  GPU utilization if single-event batches underfill (cross-plane stays intra-
  event via the mask).

## Near-term PILOT (this container, validate the objective learns)
- ~400 train events, 10k steps, frac 0.5, current model: ~1 s/step -> ~3 h.
- Purpose: does masked CROSS-PLANE coeff AE learn (loss decrease, beats a
  masked-neighbor baseline) and does cross-plane info flow (whole-plane mask).
  Optimize (levers 1-3) only AFTER the objective is shown to learn.
