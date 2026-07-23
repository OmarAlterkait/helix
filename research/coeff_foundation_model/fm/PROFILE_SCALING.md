# MAE training-pipeline profiling + scaling (2026-06-16, 2× A100-40GB)

Measured on the REAL pipeline. Tools: torch 2.5, flash_attn 2.7.3, flex_attention, DDP.
Scripts: bench_batch.py (batching), profile_mae.py (MAE step + kernels), deconv_ddp.py (multi-GPU).

## 1. Tokenization
~32k tokens/event (p50 32039, range 26482–38411, p90 35439). Tight → ~17% pad-waste-to-max.

## 2. Batching — SDPA per-event + grad-accum is optimal; varlen packing is WORSE
Fair bf16, L=10 blocks, fwd+bwd (tok/s, peak GB):
| d | SDPA per-event | flash-varlen K=1 | varlen scaling |
|---|--:|--:|---|
| 512 | 75,198 tok/s (7.1GB) | 43,753 (9.8GB) | flat tok/s, OOM@K4 (29.7GB@K3) |
| 768 | 36,678 tok/s (12.6GB) | 19,220 (17.5GB) | flat tok/s, OOM@K3 |
- torch SDPA flash backend BEATS flash_attn varlen ~1.7× here. **Don't use varlen packing.**
- Packing gives NO throughput gain (block-diagonal = same attention FLOPs) + more activation mem → OOM.
- **Grad-accumulation** (per-event backward+free) = flat throughput, memory bounded to 1 event → the
  correct way to grow effective batch. Wall-clock lever = multi-GPU, not packing.

## 3. MAE step breakdown (mask 0.75 encoder-drop, dec_blocks=4, single A100)
| phase | d=512 (44M) | d=768 (114M) |
|---|--:|--:|
| data (CPU assemble) | 183 ms (40%) | 200 ms (30%) |
| fwd | 74 ms (16%) | 125 ms (19%) |
| bwd | 188 ms (41%) | 323 ms (48%) |
| h2d+mask+loss+opt | 13 ms | 19 ms |
| **TOTAL serial** | **459 ms** | **666 ms** |
| GPU-bound (excl data) | 276 ms | 466 ms |
| **with 8 workers (measured)** | **320 ms** | ~510 ms |
peak mem: 6.9 GB (d512) — encoder-drop → huge headroom.

KERNEL breakdown (d=768, % CUDA time): flash-attn **bwd 50% + fwd 16% = 65% attention**, matmul 6%,
elementwise/LN ~28%. The **decoder dominates**: encoder sees 25% of tokens but the decoder runs all
32k (32k²×4blk ≈ 5× the encoder's 8k²×12blk). => **decoder depth is the efficiency knob.**

Data assembly is 30–40% of the serial step and is HIDDEN by the worker DataLoader (459→320 ms, 1.43×,
near the 276 ms GPU floor). Workers are essential.

## 4. Multi-GPU (DDP, 2 GPU)
Works (needs find_unused_parameters — deconv/occ-head unused). ~2× throughput (linear scaling),
both GPUs ~60% util. Needed: shard events per rank, grads all-reduced.

## 5. Throughput + scaling (workers + DDP, MEASURED rates)
| model | ev/s/GPU | 2-GPU ev/s | GPU-h per 1M event-passes |
|---|--:|--:|--:|
| d=512 / 44M | 3.12 | 6.25 | 89 |
| d=768 / 114M | 1.96 | 3.92 | 142 |

Campaigns (GPU-h | days@2GPU | days@24GPU):
- 1M ×3ep, d512: 267 GPU-h | 5.6d | 0.5d
- 1M ×3ep, d768: 425 GPU-h | 8.9d | 0.7d
- 10M ×3ep, d512: 2667 GPU-h | 55.6d | 4.6d
- 10M ×3ep, d768: 4250 GPU-h | 88.5d | 7.4d

## Takeaways
1. Current per-event SDPA path is already throughput-optimal; varlen/packing is a dead end here.
2. **Workers are mandatory** (data is 30–40% of the serial step; hidden → 1.43× faster).
3. **Decoder full-N attention is the compute bottleneck** (65% of GPU time) — shrink dec_blocks or
   make the decoder also drop tokens to speed the MAE step.
4. Multi-GPU scales ~linearly (DDP). At 1M–10M events you need a real cluster (10s of GPUs): 1M×3ep
   ≈ 0.5 GPU-days (24 GPU), 10M×3ep ≈ 4.6 GPU-days (24 GPU) for the 44M model.
5. Memory is NOT the constraint at mask 0.75 (6.9 GB) — room for much bigger models on 40 GB.
6. NOTE: TRAINING (~320 ms/event) ≈ 10× the decomposition PREPROCESSING (~37 ms/event GPU). At
   10M events the wire cache is ~29 TB → use on-the-fly decompose-in-dataloader, not extract-once.
