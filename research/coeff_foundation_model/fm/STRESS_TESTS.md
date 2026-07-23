# Stress tests — "push slightly to see where they break" (2026-06-14, 10h autonomous)

Goal: find the EDGES of the current MAE design, not converge. Breakage (divergence,
overfit, inference collapse) shows up early → use SHORT runs (~2–15k steps), many of them.
Base config = current best: d=512 enc10 dec4, mask 0.75, vis_w=1.0, workers=8, 6k ev,
warmup+cosine, eval curve. All resilient (checkpoint every eval).

Order chosen so cheap/foundational edges come first (LR informs all later runs).

## E3 — LR edge (cheapest; find divergence) [~45 min]
We're optimization-bound, so higher LR may help until it breaks. Short 2000-step probes.
- lr ∈ {8e-4, 1.5e-3, 3e-3}, warmup 300, steps 2000, eval_every 1000.
- Break signal: loss NaN/explodes, or var_expl worse than 4e-4 baseline.
- Outcome: pick the highest STABLE lr for E1/E2/E4.

## E1 — mask-ratio edge (maps difficulty + doubles as ceiling) [~2.5 h]
- mask ∈ {0.5, 0.9, 0.95} (0.75 already have), steps 10k, eval_every 2500.
- Break signal: at high mask, masked var_expl collapses (too little context).
  At low mask, R² → the interpolation ceiling (the recon-ceiling measurement, mask→0 limit).
- Outcome: the R²-vs-visible-fraction curve; where inference stops being learnable.

## E2 — capacity edge (find the data/capacity wall) [~2.5 h]
- d=768 enc12 dec4 (~90M), mask 0.75, steps 15k, eval_every 2500.
- Break signal: gap blows up (overfits 6k events = data wall) OR no gain over 44M (capacity
  saturated). Width gave +2.9 (44M); does ~2x more params still pay?
- Watch GPU mem (d=768 ~2x; should fit 40GB at batch=1).

## E4 — data-starve edge (overfit boundary) [~1.5 h]
- events ∈ {1000, 2000}, d=512, mask 0.75, steps 10k, eval_every 2500.
- Break signal: train/test gap grows (currently ~0.03 at 6k/5ep). Where does it overfit?
- Outcome: minimum data for this capacity.

## (stretch) E5 — vis_w edge
- vis_w ∈ {0.25, 4.0}: does heavy denoising weight drown masked inference (hurt) / does tiny
  weight lose the dual-target benefit? Judge by masked var_expl vs vis_w=1.0.

## Logging
All write to fm_results.jsonl + fm_curve.jsonl (tagged). Summary table appended here as runs land.

---
## RESULTS (filled in as runs complete)
(pending)

### E3 LR edge (2k steps, warmup 300, d512) — DONE
| LR | masked var_expl @2k | verdict |
|---|--:|---|
| 8e-4 | 17.6% | stable |
| 1.5e-3 | 2.3% | BROKEN (masked stuck ~2%, denoise still 98%) |
| 3e-3 | 2.1% | BROKEN |
Edge between 8e-4 and 1.5e-3. 4e-4 has ~2x headroom, no more. Denoising is LR-robust;
only the hard masked-inference task collapses → model is optimization-LIMITED (push via
steps, not LR).

### E1 mask edge (10k steps, d512, lr4e-4) — DONE
| mask | masked var_expl @10k | denoise |
|---|--:|--:|
| 0.5 | 64.7% (climbing) | 98.7% |
| 0.75 | 49.5% | 98.7% |
| 0.9 | 30.3% | 98.7% |
| 0.95 | 23.0% | 98.8% |
No catastrophic break — smooth difficulty gradient. mask 0.5 @10k already > mask0.75
converged (57.6%) => low-mask recon ceiling ~70%+ (matches 65-80% first-principles).
0.75 sat at 58% because it's HARDER, not weaker. Denoise mask-independent (~98.7%).

### E4 data-starve edge (10k steps, d512, mask0.75) — DONE
| events | test var_expl @10k | gap_mse |
|---|--:|--:|
| 1000 | 45.8% | 0.053 |
| 2000 | 47.4% | 0.017 |
| 6000 (ref) | 49.5% | ~0.03 |
NO overfit edge found — gap stays ~0.02-0.05 even at 1k events/10 epochs. Masked-MAE is
strongly self-regularizing. Data is NOT the bottleneck (marginal var_expl gain, no breakage).

### E2 capacity edge (d=768/114M, 15k steps, mask0.75) — DONE
@15k: 56.6% vs d512/44M @15k 54.0% => +2.6 for 2.5x params (gap 0.028, no overfit).
Same slope as 384->512 (+2.9). No capacity wall, no overfit; diminishing returns ~+2.6/2x params.

## SYNTHESIS — only LR breaks the design
- LR: hard break ~1e-3 (masked inference collapses, denoising survives) -> optimization-LIMITED.
- Mask: smooth knob, no break; recon ceiling is mask-dependent (~70% @0.5, ~60% @0.75).
- Data: self-regularizing to 1k events, no overfit -> not the lever.
- Capacity: no wall, +2-3 pts per 2x params, no overfit at 114M.
Binding constraints = task difficulty (mask) + optimization (steps/params, slow). NOT data/overfit/capacity.
Denoising (visible R²) ~99% everywhere, LR/mask/size-robust.
