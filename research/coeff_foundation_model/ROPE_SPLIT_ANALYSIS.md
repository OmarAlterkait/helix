# Why the cross-plane RoPE split destroyed 3D — analysis + fix

**TL;DR:** The "drop wire-RoPE on cross-plane layers" idea was a fix for a non-problem, and it
broke a mechanism the model actually uses. Cross-plane triangulation *requires* wire information
on exactly the cross-plane layers. Keep axial (t, wire) RoPE on every layer, encoder and decoder.
If we want to *improve* cross-plane 3D, add an epipolar (matched-drift-time) attention prior /
tighter cross-plane connectivity — never remove wire.

## What the split did
Encoder RoPE, two regimes:
- within-plane layers (plane-time, plane-wire-time orders): axial RoPE on **(t_phys, wire_pos)**.
- cross-plane layers (drift-time orders o_t / o_ts): **t_phys only** (wire dropped).
- decoder: axial (an earlier bug dropped wire here too → reproduced the `wire_rope=0` recon
  freeze: var_expl 10.6% + gradient-norm collapse; fixed to always-axial).

**Motivation (the premise):** `wire_pos` is a per-plane projection axis. For two tokens in
different planes, the RoPE relative phase depends on (w_U − w_V), which has no metric meaning
across planes. So cross-plane relative wire-RoPE looked like injected noise → drop it.

## Result (fair 3D probe, fisher_r on u = cross-plane along-wire coord; geo floor 0.190)
| variant | var_expl | 3D trained_r |
|---|---|---|
| base, full attn, axial everywhere, 300k | 63.7% | **0.415** |
| serial, global axial RoPE, 25k warm | 58.3% | 0.298 |
| serial, cross-plane time-only (the split) | 57.6% | **0.136** (BELOW geo floor) |

## Why it failed
Triangulation physics: the along-wire 3D coordinate is `u = f(w_U, w_V, w_Y)` — a fixed geometric
function of *which wire fired in each plane* at a matched drift time. A cross-plane layer computing
u for a U-token must attend to the V/Y tokens at the same drift time **and know their wire index**.

RoPE is the only per-token channel that carries wire index into the attention q·k interaction
(FiLM / embeddings shift the token vector, but the *positional* dependence of the attention score
comes from RoPE). Dropping wire-RoPE on the cross-plane layers makes cross-plane attention
**wire-blind**: a U-token sees "a V hit exists at this drift time" but not *which* V-wire → u is
uncomputable. Those are precisely the layers that do cross-plane mixing, so u never forms. 3D
collapses *below* the geo-only floor because the learned features become an actively-worse
(wire-washed) representation than raw geometry.

## The conceptual error
The premise conflated **"not a clean metric"** with **"not useful."** Cross-plane relative wire
phase is not a Euclidean distance — true. But it is a deterministic, learnable signal: given the
two plane identities (plane_emb), the pair (w_U, w_V) maps to u by fixed detector geometry, and the
model learns that map from the RoPE phases. It *did* learn it — that's the 0.415. The split removed
the input to a mechanism the model had already built. "Geometrically inconsistent" ≠ "uninformative."
The original problem (cross-plane wire-RoPE inconsistency) was never shown to *hurt* — it was a
hypothesized defect, and the experiment falsified the hypothesis.

## What to actually do
1. **Keep axial RoPE on all layers, encoder + decoder.** (Evidence: base = 0.415.)
2. To *raise* cross-plane 3D, strengthen the correspondence mechanism, don't remove wire:
   - **Epipolar attention prior**: score bias favoring cross-plane pairs at matched drift time
     (|Δt| small) — a positive prior on the triangulation partners. This is exactly what coarse
     grouped attention loses.
   - **Drift-tight cross-plane groups** (not generic equal-count) so cross-plane attention resolves
     the ~±5-tick epipolar band. Addresses both the grouped-attention 3D deficit and triangulation.
   - **Richer wirefeat** (per-plane geometric angle/offset) so absolute cross-plane wire is available
     additively, not only through RoPE — could make a future time-only-cross-plane variant viable
     (absolute wire via features, relative time via RoPE). Speculative, secondary.

## Caveat / open door
The split run was warm-started from a wire-RoPE-everywhere model, so its cross-plane layers had to
unlearn/relearn — 25k steps may be too few, and a from-scratch split *might* beat 0.136. But the
mechanism argument holds regardless of training length, the split solves a non-problem, and global
RoPE already works — so a from-scratch split run is low value. Not recommended.

Separate question (the reason for the 500k `serial_long` run): grouped attention with *global*
RoPE also lost 3D (0.415→0.298) at the 25k warm budget. Under-convergence or a fundamental cost of
coarse cross-plane grouping? **ANSWERED by `serial_long`** (from scratch, global RoPE, lr 4e-4
matched to base, 500k steps):

| step | var_expl | 3D trained_r |
|---|---|---|
| 100k | ~51% | 0.225 |
| 200k | ~57% | 0.312 |
| 300k | ~61% | 0.343 |
| 400k | 62.2% | 0.357 |
| base full-attn @300k | 63.7% | **0.415** |

### PLANE-MASKING regime (the 3D lever): grouped saturates, full keeps climbing
Trained grouped-serial with plane_frac 0.1/0.25 (400k, lr 4e-4) + probed vs existing full-attn
plane_frac ckpts (150k, lr 1.6e-3). 3D fisher_r on u:

| plane_frac | grouped-serial | full-attention | grouped PLANE var_expl |
|---|---|---|---|
| 0 (random) | 0.352 | 0.387 | ~0% |
| 0.1 | ~0.59 | 0.543 | 13% |
| 0.25 | ~0.55 | **0.662** | 25% |

- Plane masking is the 3D lever for grouped too: 0.35→0.59 at pf0.1 (above ALL random-mask models,
  incl. full 0.415). The "grouped costs 3D" worry was a random-mask-regime artifact.
- BUT full-attn 3D scales monotonically with plane_frac (0.39→0.54→0.66) while grouped SATURATES
  (~0.59 pf0.1, ~0.55 pf0.25 — no gain). Grouped's coarse equal-count drift groups hit a cross-plane
  resolution CEILING (~0.59); full attention's precise connectivity keeps extracting more triangulation
  as the pretext pushes harder. This is the predicted cross-plane deficit, amplified where triangulation
  is stressed most. Caveat: LR/steps differ (grouped 4e-4/400k vs full 1.6e-3/150k) — qualitative trend
  robust, but a clean version needs grouped reruns at lr 1.6e-3. Fix for the ceiling = epipolar-aware /
  drift-tight cross-plane groups.

### RANDOM-MASKING regime (earlier serial_long run)
3D increments halve every 100k (+0.087 → +0.031 → +0.013) → **plateauing ~0.36, below full-attn 0.415**.
So: (a) reconstruction under grouping ESSENTIALLY MATCHES full attention (62.3% vs 63.7%); (b) 3D has
a real but MODEST fundamental deficit (~0.36 vs 0.415, ~13% relative) that persists at convergence and
extra steps (400k vs base's 300k) don't close. The 25k-warm 0.298 understated it — converged grouped
gets ~0.36, not 0.30. **Interpretation:** coarse equal-count drift groups blur the tight cross-plane
epipolar alignment triangulation needs; reconstruction (local/mid-scale) is unaffected, 3D is. **Fix if
we want grouped attention at scale without the 3D tax = epipolar-aware / drift-tight cross-plane groups,
NOT more training and NOT a RoPE split.**
