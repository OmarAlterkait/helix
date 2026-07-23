# Blocks & Assumptions Audit — the full decision ledger

Companion to `DESIGN_HANDOFF.md` and `cross_level_operator_analysis.md`.
Purpose: step back and enumerate every big block, the decisions inside it, the
assumptions those decisions rest on, and how each block uses the **scale axis**
— the defining feature of this architecture. Provenance tags as in the handoff.

Settled since the handoff: optical uses **dense** within-level convs (measured
62.5% coarse-cell occupancy — sparsity lives at the chunk level), TPC uses
submanifold sparse convs; all other machinery is shared.

---

## The scale story (read this first)

Scale (the wavelet level axis) is used differently — and deliberately — in
every block. This is the architecture's organizing principle:

| block | how scale is used |
|---|---|
| 0 interface | scale-aware threshold (per-band σ·√(2 ln N_b)); the input support is scale-structured by construction |
| 1 embedding | per-band σ normalization = scale conditioning at entry; **physical-time PE shared across levels** = the device that makes scales alignable |
| 2 within-level | constant kernel per level ⇒ physical RF ∝ 2^ℓ = **scale-equivariance** (exploits signal self-similarity); FiLM(ℓ) absorbs where self-similarity *breaks* (detector response scale) |
| 3 cross-level | scale as a **recurrence axis**: level-sequential V-cycle sweeps, shared edge weights, deeper tree = more iterations of the same operator |
| 4 tokenizer | scale **folded into channels** at a chosen anchor level; the anchor level is a per-modality measured decision, not a constant |
| 5 trunk | scale-agnostic by assumption (scale already folded) — flagged below as an assumption, not a fact |
| 6 SSL | scale as the **predictive axis**: coarse→fine prediction is a native SSL task for wavelet data (VAR lesson); the measured tree conditionals are its difficulty curve |
| 8 compute | per-level cost is count-balanced for TPC (flat bands [MEASURED]); mid-heavy for optical (D4/D3 peak [MEASURED]) |

---

## Block 0 — The interface (what the model sees)

**Decisions [settled]:**
- Input = post-threshold surviving coefficients. TPC: smart removal (kgate=4)
  → per-band-σ VisuShrink hard, κ=1. Optical: per-chunk σ, κ=1.2, A10 kept.
- Drop D1: TPC (measured noise floor); optical (budget — 0.9% of survivors,
  but it IS signal-bearing, sig:noise 41× [MEASURED]).

**Assumptions:**
- [SETTLED by user] κ operating points are fixed — they were designed to
  remove the noise while maximally keeping the signal; not a swept variable.
- [ASSUMED] Classical smart removal (kgate=4) is good enough preprocessing
  that the model needn't see pre-removal data. (The original "end-to-end
  replaces remove_coherent" ambition is deferred — the model sits after the
  classical remover, not instead of it.)
- [ASSUMED, minor] Optical 12-bit survivor quantization is irrelevant to the
  representation (production quantizes; measurement scripts don't).

**Falsifiers:** κ-sensitivity in the AE sweep; L3 physics observables vs κ.

## Block 1 — Embedding & normalization

**Decisions:** `asinh(c/σ_b)` signed, per-band; linear → d_s (~32–64); level
embedding + plane/sensor embedding; physical-time PE (delay-aware) shared
across levels.

**Assumptions:**
- [MEASURED, supports] value ranges: TPC 125× p1→p99; optical **1350×** at D10
  — compressive signed norm is mandatory, more so for optical.
- [ASSUMED] σ_b is stable/available at inference (comes free from the
  threshold step).
- [ASSUMED] physical time (not band index) is the right shared coordinate —
  supported by the alignment shift sweep (naive τ>>1 near-optimal [MEASURED]).

**Falsifiers:** norm ablation (asinh vs log-mag+sign vs learned affine) —
cheap, in AE phase.

## Block 2 — Within-level operator

**Decisions:** depthwise conv + pointwise MLP (ConvNeXt split). TPC:
submanifold sparse, anisotropic (~7 wire × 3 time). Optical: **dense** per
chunk [settled by MEASURED occupancy]. Constant kernel size across levels.
Weights shared across levels with FiLM(ℓ) conditioning (per-level as ablation
arm).

**Assumptions:**
- [ASSUMED — the deep one] **Signal self-similarity across scale.** Constant
  kernels = scale-equivariant RF assumes features look alike at every dyadic
  scale. This is true of the underlying physics (track dE/dx, photon arrival
  statistics) but **broken by detector response, which imprints a fixed
  physical scale** (TPC field response ~ few ticks → D1/D2 territory; optical
  SER 10 µs spans ~10 coarse cells). The measured per-band statistics
  gradients (survival 36%→0.05%, lift 2×→57×, p99 14k→46 across optical
  depth) are the visible signature of broken self-similarity. FiLM is the
  hedge: shared structure, per-scale modulation. If FiLM can't absorb it,
  unification weakens to per-level weights and "one model" dies.
- [ASSUMED] within-band locality suffices pre-trunk (global is the trunk's
  job).

**Falsifiers:** shared vs FiLM vs per-level (TPC first, then transfer to
optical depth) — the single most informative cheap ablation.

## Block 3 — Cross-level operator (tree V-cycle)

**Decisions:** residual bidirectional adjacent-level message passing —
up: `parent += MLP(LN([child_even, child_odd, masks]))` (ordered 2-slot,
exact); down: `child += MLP(LN(parent))`. Interleaved with within-level convs
(stitches dyadic cousins). A-band = root partner of the deepest detail.
Naive τ>>1 alignment [MEASURED near-optimal; coarse levels ~1–2% better at −1,
absorbable]. 1–2 rounds.

**Assumptions:**
- [MEASURED, supports] tree edges carry the dominant dependency (TPC 31×/29×;
  optical 2×→57× adjacent-dominant at every depth).
- [MEASURED, complicates] **observed activity is NOT first-order Markov**
  (grandparent residual lift 3–21× given parent inactive). The V-cycle
  *topology* still suffices — multi-hop info composes through node states —
  but now the **node state capacity (d_s) is load-bearing**: it must carry
  subtree information across hops, not just one-hop context.
- [ASSUMED] one shared edge-MLP works at all depths. The conditional
  P(child|parent) varies 0.65→0.03 across optical depth → the edge-MLP likely
  needs FiLM(ℓ) too, same hedge as Block 2.
- [ASSUMED, UNVERIFIED] tree-op cost ~ sparse-conv cost (handoff open #1 —
  microbench pending; now implementable as pure index gathers, both dumped
  events available).

**Falsifiers:** V-cycle vs direct column ops at L=4 (the unification gate);
rounds 0/1/2 (0 = the control that proves cross-level communication matters at
all); d_s sweep; edge-FiLM on/off; the microbench.

## Block 4 — Tokenizer (aggregation)

**Decisions:** anchor tokens at a coarse level, fold scale into channels via
the up-sweep, learned pooling (strided conv vs attention pooling — never
mean), core-aware evaluation.

**Assumptions:**
- [ASSUMED — what the AE phase exists to measure] the d-dimensional token
  bottleneck retains per-coefficient fluctuations, especially on saturated
  cores (TPC: max load = cell capacity at every patch size [MEASURED]).
- [ASSUMED, newly visible] **the anchor level is the right one.** TPC: 8×4 on
  the A4/D4 grid → fan-out mean ~5–7, p95 ~35–46 [MEASURED]. Optical anchored
  at the level-10 cell → fan-out mean 43, p95 206, max 398 [MEASURED] — ~5×
  heavier per token than TPC. Anchoring optical at level 9 (~512 ticks) or 8
  (~256 ticks) trades token count (5.8k → ~11.5k → ~23k/event) against
  per-token load. This is a measured knob, not a constant; the optical anchor
  sweep is a cheap npz recompute (patch_sweep-style).
- [MEASURED] token count event-independent for TPC (±10%); optical cell count
  set by chunk lengths (~5.8k/event), survivors vary ~3×.

**Falsifiers:** AE N×d sweep stratified by core saturation; optical
anchor-level sweep on `typical_event_coeffs_optical.npz`.

## Block 5 — Trunk

**Decisions:** TPC: hierarchical (within-plane → within-volume → light
same-tick cross-volume), ~3:1, no serialization, MAE-style visible-subset
encoding. Optical: ~5.8k tokens/event → **flat global attention suffices**
(measured 1.9 ms/layer at 5k tokens) — the hierarchy machinery is a TPC-scale
necessity, a config not an architecture difference.

**Assumptions:**
- [ASSUMED] scale can be fully folded into tokens before the trunk (no scale
  axis in trunk attention). Alternative (axial attention over scale) rejected
  on token-economics, not measurement.
- [ASSUMED] detector-mirroring hierarchy is the right factorization (vs latent
  bottleneck — handoff open #12, decided by busy-core fidelity).
- [SCOPE, important] **unification ≠ fusion.** Training one shared-weight
  model on *unpaired* TPC + optical corpora needs no paired data. Joint
  event-level charge+light fusion (physically motivated: optical gives
  t0/position priors) **requires paired simulation that emits both** — which
  does not exist on disk today (TPC: JAXTPC_Wire; optical: goop light, 100
  events, different sims). Keep the two ambitions separate.

**Falsifiers:** small-model trunk sweeps (handoff §13.9–13).

## Block 6 — SSL objective

**Decisions:** masked/predictive in latent space (MAE/JEPA primary);
cross-view (plane-holdout) prediction; optional DINO event token; pure DINO
rejected (washes out fluctuations).

**Assumptions:**
- [ASSUMED] 30% visible → ~3× cheaper/step.
- [ASSUMED] masked-latent prediction preserves per-coefficient fluctuations.

**Scale-native addition [DESIGN]:** the scale axis gives a *free* predictive
task images don't have: **coarse→fine band prediction** (mask entire fine
bands / subtrees, predict from coarse — wavelet super-resolution; VAR's
next-scale lesson). The measured conditionals are its difficulty curve:
P(child|parent) ≈ 0.65 at coarse optical levels (learnable), 0.03 at D1
(hard); the non-Markov residual says the task is non-trivial (parent alone
underdetermines the child). Candidate masking menu: random-token /
subtree-structured / whole-band (scale-predictive) — compare at small scale.

**Falsifiers:** objective comparison on ViT-S with linear probes.

## Block 7 — Decoder / heads (AE phase + dense tasks)

**Decisions:** symmetric decoder, two-head (occupancy + values), skips OFF
through the bottleneck (honesty constraint), U-Net skips only for production
dense tasks later.

**Assumptions:**
- [ASSUMED] support prediction tractable. Note asymmetry: optical occupancy is
  nearly class-balanced at the coarse grid (62.5% [MEASURED]) — easier than
  TPC's sparse support.
- L1-lossless ≠ L2 ≠ L3 (metric ladder; relationships to be reported, not
  assumed).

## Block 8 — Systems & data scale

**Decisions:** A100 bf16, spconv (TPC) / dense (optical), fixed-shape
batching, sequence packing for the ~5×/3× coefficient-count spread, ViT-L
target with ViT-H as config reach.

**Assumptions:**
- [ASSUMED, UNVERIFIED] §11 cost model: tree-op cost + batched-stem
  amortization (the two pending microbenches).
- [ASSUMED — the biggest practical one] **the corpora exist.** 10M TPC events
  is ~500× what is on disk (19,999 events / 5.9 TB sparse). Optical: **update
  June 2026** — 20,000 noise-free events with truth found at
  `/sdf/data/neutrino/doraemon/optical_test_00_00_02/` (210 GB, label_N
  schema, needs a small loader; see `mechanism_tests.md` §4) — prototype-scale
  optical is covered; FM-scale (10M-class) still needs a generation plan for
  both modalities.
- [ASSUMED] synthetic forward noise ≈ real detector noise (sim-to-real:
  coherent per-64 group structure, intrinsic ENC values are sim parameters).

---

## Ranked: the assumptions most likely to hurt

(updated June 2026: κ settled by user — removed; FiLM/V-cycle point-choices
replaced by the variant spaces + test battery in `mechanism_tests.md`)

1. **Corpus existence at FM scale** (Block 8) — 10M-class generation plan
   unowned for both modalities (prototype scale now covered: TPC 20k, optical
   20k doraemon events).
2. **Token bottleneck loses saturated cores** (Block 4) — the AE phase exists
   for exactly this; N, d, anchor level are the knobs.
3. **Weight structure across levels** (Blocks 2–3) — sharing is a hypothesis,
   not a principle (the per-level task morphs regression→detection); decided
   by the W0–W7 ladder + transfer tests (T-A2..A5).
4. **Cross-scale information travel at optical depth** (Block 3) — V-cycle
   connectivity ≠ fidelity (attenuation/oversquashing over ~10 hops); decided
   by the C0–C4 topology shoot-out + ancestor-ablation curves (T-B1..B7).
5. **Cost model** (Blocks 3, 8) — microbenches ready to run on the dumped
   events (extended to all C0–C3 topologies, T-B7).
6. **Paired charge+light data** (Block 5) — only gates event-level fusion,
   not shared-weight unification; the doraemon optical set is the same sim
   family as the TPC set (test_00_00_02) so pairing is plausible — verify
   before asking for new sim.
7. **Optical token anchor at level 10** (Block 4) — 5× heavier tokens than
   TPC; anchor sweep is a cheap recompute (T-C1).
8. **Sim-to-real noise** (Block 8) — monitor; real-data finetune is the
   standard answer.
9. **Trunk scale-agnosticism** (Block 5) — revisit only if AE shows
   scale-folded tokens plateau.
