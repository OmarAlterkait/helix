# Sparse-attention connectivity, hops, and gap-robustness (design note)

Analysis of the proposed **geometry-gated sparse attention** for the coeff-FM: can information reach
where the physics needs it, in how many layer-hops, and what happens at **gaps** (dead wires, thresholded
segments, empty cross-plane time-slabs). Grounded in occupancy stats measured on real cached events.
**kNN windows are explicitly NOT used** (design decision) — the register hierarchy below is the connectivity
mechanism, so the local tier stays a simple fixed physical window.

## Token grid (measured, `artifacts/fm_cache_tpc`)

- **N ≈ 31k tokens/event** (27k–35k). Current encoder = full `O(N²)` ≈ 1e9 QK pairs.
- **24 groups** = 6 planes × 4 wavelet bands (A4,D4,D3,D2). Each band has its OWN time grid; the only
  coordinate shared across bands AND planes is physical `(wire_pos, t_phys)`. RoPE already uses this pair.
- Per-plane token counts ≈ [6127, 6074, 3065, 6267, 6291, 3552] (planes 2 & 5 are the short ones).
- Per (plane,band) grid: up to ~119 wire-blocks × 33–135 time-blocks, **occupancy only 0.05–0.32**
  (finest band D2 in short planes = 0.05 → mostly holes). Physical: wire 0–1968, t_phys −38…4334
  (~1.4 tokens/tick/plane).

## The two-tier proposal (recap)

- **Tier A — within-plane**: local fixed physical window over each plane's sparse (wire,t_phys) grid.
- **Tier B — cross-plane**: attention gated to the shared **drift-time slab** (t_phys only) — the degenerate
  epipolar locus (U/V/Y sense the same charge at the same drift time).

## What works and what fails (measured)

**Tier B (cross-plane drift-time) is excellent.** Empty-slab fraction (≥1 token in a target plane within ±dt):
| dt (phys ticks) | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| empty fraction (mean/p90) | 0.001/0.004 | 0.000 | 0.000 | 0.000 |
The global t-axis is dense (~1.4 tok/tick/plane), so an empty cross-plane slab essentially never happens.
Cross-plane triangulation reaches the corresponding plane in **1 hop** and is gap-robust — *provided Tier B
gates by `t_phys` only* (all wires in the slab; let attention sort the epipolar wire correspondence). Do NOT
pre-gate by a predicted wire band or emptiness returns.

**Tier A (within-plane local) FAILS on its own — three failure modes:**
- **F1 — diameter > layer budget.** Hops to cross a p90 extent for window `w` blocks: w=4 → 28–34 hops;
  w=8 → 14–17; w=16 → 7–9. Only w=16 fits the 12-layer budget, but a 16-block window (256 wires × 128 ticks)
  holds hundreds of tokens in dense cores → drifts back toward quadratic. A full through-going track (~1900
  wires) exceeds the budget for any cheap window.
- **F2 — gaps create PERMANENT islands.** Connected components of a (plane,band) grid under a local window:
  | window | mean #components | largest-comp frac |
  |---|---|---|
  | ±1×±1 | 212 | 0.51 |
  | ±2×±2 | 64 | 0.81 |
  | ±3×±3 | 29 | 0.90 |
  At 0.05–0.32 occupancy a small window shatters each plane-band into dozens–hundreds of islands; stranded
  fragments are **never** connected by Tier A at any depth. A dead-wire band or a thresholded gap along a
  track puts the two halves in different components — local attention cannot bridge them.
- **F3 — cross-band siblings severed.** The 4 bands have different time-index grids; windows defined on the
  per-band index grid never connect a coefficient to its A4↔D4↔D3↔D2 siblings at the same physical
  `(wire,t_phys)` — yet denoising a masked coeff wants exactly those. Tier A windows must be defined in
  **physical `(wire, t_phys)`** (pooling all bands), not per-band indices. (The column tokenizer — see the
  band-patch note — fixes this at the tokenizer level by bundling bands into one column.)

## The fix: a 2-level register hierarchy (BigBird global tokens, specialized to the geometry)

No kNN. Instead, guarantee bounded-hop connectivity with learned hub tokens:

1. **Per-plane registers** — R ≈ 8–16 learned tokens per plane, full attention to/from ALL tokens of that
   plane (across all 4 bands, addressed in physical `(wire,t_phys)`). Any two tokens in a plane connect in
   **exactly 2 hops** (token→register→token), independent of extent and of gaps → kills F1 and F2, and
   reconnects the bands (F3).
2. **Global register tier** — G ≈ 4–8 tokens attending all 6·R plane registers (and back) → **any-to-any in
   ≤4 hops** across volumes, and guarantees a masked/partial-view token still receives the surviving planes'
   information (dead-wire partial views).
3. **Tier A** stays a simple fixed physical window (bands pooled in `(wire,t_phys)`) — it now only supplies
   fine local texture; the registers carry global reach, so the window need not be large.
4. **Tier B** gates by `t_phys` only; cap/sample to a bounded #keys in dense cores. Optional adaptive slab
   widening if ever empty (measured ~0, so a cheap safety belt, not load-bearing).

## Hop budget (before → after)
| path | pure 2-tier | + register hierarchy |
|---|---|---|
| full-length within-plane track | 14–34 (**fails >12**) | **2** |
| stranded fragment across a gap | ∞ (**fails**) | **2** |
| cross-plane triangulation (same t) | 1 | 1 |
| cross-band siblings | ∞ if per-band grid | **2** |
| cross-volume any-to-any | multi-hop | **≤4** |

Every physically-corresponding pair connects in ≤4 hops → ~8 of 12 layers left for refinement, not transport.

## Cost
Per-plane registers O(R·N); global O(G·R·6); Tier B ~225 keys/token; Tier A fixed-window k_local keys.
Total ≈ O((k_local + 225 + R)·N) ≈ a few ×1e7 QK pairs vs full-attention ~1e9 → **~30–100× cheaper**,
firmly sub-quadratic, with connectivity that full attention had for free restored by the register hubs.

## Verdict
Adopt **Tier B as-is**; replace Tier A's local-only reach with a **per-plane + global register hierarchy**
(fixed physical windows for local texture, no kNN). This is the minimal change that guarantees bounded hops
(≤4), gap-robustness (no islands; partial-view still routed), and sub-quadratic cost. Composes with the
column tokenizer (which bundles bands → fixes F3 at the token level and halves N).
