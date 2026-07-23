# Cross-Level Operator Analysis — unified TPC/optical architecture over sparse wavelet coefficients

Companion to `DESIGN_HANDOFF.md`. Scope: the four design questions for the
pre-attention stage — per-level operations, aggregation, skips/MLPs, and (the
central one) full cross-level communication — resolved into a single operator
that works for both TPC (2D, 4 levels) and optical (1D per-sensor, 8–11 levels).
Provenance convention follows the handoff ([MEASURED]/[SOURCED]/[DESIGN]/[ASSUMED]).
Everything here is [DESIGN] unless tagged otherwise; the §8 ladder is what tests it.

> **Superseded point-choices (June 2026):** the "shared + FiLM" weight
> recommendation and the "V-cycle as the one operator" recommendation were
> premature hard choices. Both are now variant *spaces* with discriminating
> experiments — see `mechanism_tests.md` (weight-structure ladder W0–W7,
> topology menu C0–C4, test battery T-A/B/C/D). The analysis below stands as
> the reasoning record; treat its "recommendations" as baseline variants.

---

## 1. The unified data model

Both modalities reduce to one object: a sparse set of coefficients indexed by
`(s, ℓ, τ)` — spatial index `s`, level `ℓ`, within-band time `τ` — living on a
**dyadic tree in time**: parent of `(ℓ, τ)` is `(ℓ+1, τ>>1)` (after group-delay
correction), children `(ℓ−1, 2τ)` / `(ℓ−1, 2τ+1)`. Physical time ≈ `(τ+δ_ℓ)·2^ℓ`
ticks.

- **TPC plane** = W parallel trees (one per wire) laterally coupled by the wire
  axis. L=4 + A-band; D1 dropped [MEASURED noise floor].
- **Optical sensor** = exactly one tree. L=8–11 + A-band; no spatial axis
  (per-sensor independent to first order).

Optical is TPC with the wire axis deleted and the tree made deeper — a
degenerate case, not a sibling design. Consequence: **every operator must be
defined on (spatial axes) × (dyadic tree)**; anything that hard-codes L=4
(channel-concat of levels, per-level weight tables) breaks unification. A
literally-shared model *forces* the weight-sharing-across-levels question
(per-level weights cannot span L=4 and L=11).

## 2. Per-level operations

- **Embed:** `asinh(c/σ_b)` → linear to small `d_s` (~32–64) + level embedding
  + plane/sensor embedding + **physical-time PE shared across levels** (levels
  alignable in one coordinate — required by the tree operator).
- **Mix within-level:** depthwise submanifold conv on the band's *native* grid
  (no resampling). TPC anisotropic (~7 wire × 3 time — track continuity);
  optical 1×k. Pointwise MLP after (ConvNeXt split). Submanifold discipline —
  no support dilation in the encoder.
- **Scale equivariance for free:** constant kernel size at every level ⇒
  physical receptive field scales as `2^ℓ` (3-tap time kernel = 24 µs at A4,
  3 µs at D1). The strongest a-priori argument for sharing conv weights across
  levels (scattering-transform-style scale-equivariant operator).
- **Weight sharing — three options:** per-level / fully shared / **shared +
  level conditioning (FiLM from level embedding)**. Per-band statistics differ
  [MEASURED handoff §4.5: A4 spans 125×, D2 near-uniform] so naked sharing may
  underfit; conditioned sharing is the hedge. A-band gets its own conditioning
  at minimum (lowpass local mean, qualitatively unlike details). Tier-1
  ablation, TPC first.

## 3. Cross-level communication — the central question

Operational definition of "fully": every coefficient can influence, and be
influenced by, every other coefficient in its column-tree through a
learned-projection (never mean) path. One-directional schemes (FPN top-down
only) fail by construction.

### Operator menu

**(a) Column concat at the coarse anchor** (handoff §7 flat coupling). Exact
and cheap for TPC (column = 8 entries, D1 dropped); blows up `2^L` for optical
(`2^10` per coarse cell). NOT unified. Keep as TPC-only baseline.

**(b) Attention along the level axis.** Asymmetric:
- fine-anchored ancestor-chain attention: each coefficient has exactly ONE
  ancestor per coarser level → O(N·L) edges, exact at any depth — but
  broadcast-only (coarse→fine);
- the transpose (descendant aggregation) has `2^Δ` fan-in — blowup again.
Cannot be made symmetric and cheap at depth 11.

**(c) Adjacent-level bidirectional message passing — the tree V-cycle.**
RECOMMENDED unified operator.
- **Up-sweep** (fine→coarse, level-sequential):
  `parent += MLP(LN([child_even, child_odd, presence_masks]))` — children are
  at most 2 and *ordered*, so masked concat is exact; no pooling anywhere.
- **Down-sweep** (coarse→fine): `child += MLP(LN(parent))`.
- **Two sweeps = exact full-tree communication** (upward–downward pass / tree
  prefix-scan / BP on a tree): after up, every node summarizes its subtree;
  during down, each node receives its parent after the parent absorbed the
  sibling subtree + all ancestral context.
- **Cost O(N) total, one parent edge per node, depth-independent per-coefficient
  cost.** Identical at L=4 and L=11.

### Why (c) is the right prior, not just the cheap one

- **Classical anchor [SOURCED]:** structurally the **wavelet hidden Markov
  tree** (Crouse–Nowak–Baraniuk). The established statistical model for this
  data class: the residual dependency the DWT leaves is parent–child on the
  dyadic tree; the up/down sweep is the HMT's exact inference pass. The learned
  version is a strict generalization. The measured lift — P(D3|D4)=0.44 one hop
  vs P(D2|D4 ancestor)=0.11 two hops [MEASURED §4.2] — is the signature of a
  Markov-ish chain: dependence is strongest one hop away. Adjacent coupling is
  where the structure lives, not an approximation of the full column.
- **ML anchor [SOURCED]:** BiFPN/HRNet repeated bidirectional adjacent-resolution
  fusion — the variant dense vision converged to over all-to-all.

### Dyadic cousins — what the tree alone can't fix

Coefficients adjacent in time at a fine level can be distant in the tree
(opposite sides of a dyadic boundary; LCA near root) — the classic
block-boundary artifact of tree-structured wavelet models. Free fix:
**interleave within-level convs with the sweeps** — lateral mixing at each
level stitches cousins at every scale. This is why the block alternates
conv ↔ sweep instead of doing all spatial work first.

### Support handling

Lift is 31× but P(child|parent)=0.44 [MEASURED] → most tree edges have a
missing endpoint.
- Compute only on active sites; edges exist only where the receiving endpoint
  is active (gather semantics; no site creation in the encoder).
- Missing parent → learned null embedding. Missing child → presence mask in the
  2-slot concat. Both exact.
- A-band: root partner of D_L (same resolution) — lateral edge to D_L +
  participates as ancestor; one extra edge type.
- **Alignment:** parent index is `τ>>1` only after per-level group-delay
  correction `δ_ℓ` baked into the index map (handoff §7.2 / open #18).
  Misalignment masquerades as high-frequency loss. See §8 Tier-0 for the
  measurement that settles it.

### Rounds

1 V-cycle = full connectivity; 2 = re-estimation (second BP iteration). The
Markov-like lift decay suggests saturation at 1–2. Ablate 0/1/2; the **0 case
(no cross-level) is the must-have control** quantifying what cross-level
communication buys at all.

## 4. Aggregation to tokens

After fusion, coarse nodes are subtree summaries; tokens anchor on the coarsest
grid (A4/D4), patch 8×4 per the measured budget (~25.4k/event,
event-independent [MEASURED §4.3]).

- **Operator:** learned strided sparse conv (cheapest) vs **attention pooling**
  (PMA-style learned queries per patch). The measured saturated cores (max
  coarse load = full cell capacity at every patch size [MEASURED §4.4]) are why
  attention pooling might earn its cost: 64 co-active individually-meaningful
  slots into d channels is where weighted-sum beats a strided projection.
  Decide by AE reconstruction **stratified by core saturation** — cores fail
  first and are the highest-information regions.
- **Honesty point:** the up-sweep tempts "coarse nodes carry everything →
  pooling lossless." False in general — each node is a d-dimensional
  bottleneck; saturated cores overload it. The §12 autoencoder measures exactly
  this; what's on trial is the up-sweep summarization + pooling jointly, as a
  function of (N, d).
- **Optical:** identical mechanics; coarse nodes (or small patches) per sensor;
  per-sensor encoder is the shared piece, sensor-level aggregation a later
  question.

## 5. Skips / MLPs / norms — assembly rules

- **Pre-norm residual everywhere:** every operator is `x += f(LN(x))` (conv,
  up, down, pool). Residual sweeps = near-lossless coupling (add context, never
  overwrite).
- **Cross-level information moves ONLY through tree edges.** No resampling
  skips across levels; residuals stay grid-aligned. Keeps information routing
  auditable — an AE loss is attributable to the operator that owned the path.
- **MLPs:** pointwise, expansion 2–4×, after each sweep direction — capacity
  lives here (edge-MLPs deliberately small). FiLM(ℓ)-conditioned if shared.
- **Norm:** LayerNorm per coefficient, never BatchNorm (batch stats over
  varying sparse supports are unstable — standard sparse/point practice).
- **Channels:** uniform `d_s` for TPC (band counts measured-flat). Optical may
  taper with level — gated on Tier-0 measurements.
- **U-Net encoder→decoder skips:** dense-decode heads only; OFF during AE
  validation (handoff §12.3, settled).

## 6. The unified block

```
WaveTreeBlock (shared structure, both modalities):
  1. within-level: depthwise SubM conv (spatial axes) + pointwise MLP   [shared across levels, FiLM(ℓ)]
  2. up-sweep:    parent += MLP(LN([child_even, child_odd, masks]))     [one shared edge-MLP]
  3. down-sweep:  child  += MLP(LN(parent))                             [one shared edge-MLP]
  4. pointwise MLP (FiLM(ℓ))
× 2 blocks (ablate 1–3) → learned pooling to tokens → hierarchical attention trunk (handoff §8, unchanged)
```

TPC: spatial axis = wire, L=4+A, D1 dropped. Optical: no spatial axis,
L=8–11+A, droppable fine levels TBD. Structurally identical; literal
cross-modality weight sharing = the handoff §10 scale-recurrence experiment,
reduced to the cheap transfer test in §8.

## 7. Scale / implementation reality

Pure index arithmetic — the "unknown stencil cost" (handoff open #1) becomes
benchable in an afternoon:

- Parent/child indices are `τ>>1`, `2τ`, `2τ+1` (+ `δ_ℓ`) into per-band dense
  index maps (`n_wires × len_b` int32, tiny; scatter once per event
  post-threshold on GPU). No KNN, no serialization, no hashing. Sweeps =
  gather → MLP → scatter_add, level-sequential (≤11 small launches).
- At `d_s=32–64` over ≤743k coefficients, S-stage memory is trivial vs trunk;
  the 15–25%-of-step estimate [ESTIMATED handoff §11] gets a falsifiable form.
- **Batching wrinkle:** token count is event-independent (±10%) but coefficient
  count varies ~5× (64k–743k [MEASURED]) — pad-to-max wastes ~2× in S.
  Standard fix: sequence-packing (multiple events into one fixed packed
  length). Engineering test item, not a design risk.

## 8. Test ladder (ordered, with decision criteria)

**Tier 0 — missing measurements (cheap, no training, first):**

1. **Optical twin of `measure_coeffs`** — the biggest hole. All handoff §4
   numbers are TPC-only. Unification rests on whether the optical tree at
   depth ~10 is also parent-child dominant and whether cross-scale coupling
   stays local (handoff §10's own criterion — unmeasured). Partial facts exist
   (per-level budget: D4+D3 ≈ 41% of kept coeffs, D1 survival 0.04%
   [MEASURED, helix optical campaign]) but no tree lift / value percentiles /
   adjacency at optical depth. Also determines droppable fine levels.
2. **Tree-op microbench** (open #1 concretized): index-map gather + edge-MLP
   V-cycle vs flat column attention vs dense-on-coarse, on
   `artifacts/typical_event_coeffs_smart.npz`.
3. **Batched-stem amortization** (open #2, unchanged).
4. **Alignment by measurement:** tree lift WITH vs WITHOUT `δ_ℓ` correction in
   the parent map. Corrected > naive ⇒ alignment matters and the offset is
   verified simultaneously. Retires open #18 with the existing script + a few
   lines.

**Tier 1 — AE rate–distortion (handoff §12 instrument), factor list:**

- **Cross-level operator:** none (control) / column-concat / column-attention /
  V-cycle×1 / ×2 / lifting (diagnostic). Pivotal read on TPC: **V-cycle ≈
  direct column ops at L=4?** If yes, the unified operator is validated before
  touching optical.
- **Weight sharing:** per-level / shared / shared+FiLM on TPC; then zero-shot
  transfer of the shared operator to optical depth at AE level — cheapest test
  of "one model".
- **Aggregation:** strided conv vs attention pooling × patch {8×4, 16×8} ×
  d {256/512/768}, stratified by core saturation, on the L0/L1/L2 ladder,
  skips off, held-out events.
- Optional: data-gen κ sensitivity (still post-threshold; κ is the input-rate
  knob and interacts with the knee).

**Pre-registered criteria:** V-cycle within few-% of column-attention L2 on
TPC → adopt unified operator. shared+FiLM within tolerance of per-level →
adopt sharing. If the no-cross-level control matches everything → the tree op
isn't earning its place; rethink (unlikely given 31×, but that's the
falsifiable claim).

**Tier 2 — trunk sweeps:** unchanged from handoff §13.9–13; nothing here
touches V.

---

## 9. [MEASURED] Optical tree statistics — Tier-0 item 1, DONE

100 events of `light_output.h5` (the file `helix.optical` loads; 172.6
chunks/event, 161.6 signal = |x|max>50 ADC), production-faithful: coif3 L10
periodization, per-chunk σ from unpadded db1 finest MAD, per-band hard
threshold t_j = 1.2·σ·√(2 ln N_j), A10 kept untouched. Per-chunk padding to
own multiple of 2^10 (exact dyadic tree; threshold N_j differs from
production's per-event common pad by a few % in ln N). Script:
`measure_coeffs_optical.py`; stats `artifacts/optical_tree_stats.json`; dump
`artifacts/typical_event_coeffs_optical.npz` (event_096, 159,512 rows).

### 9.1 Per-band survival & counts (signal chunks)

| band | surv% | act/sig-chunk | act/noise-chunk | sig:noise |
|---|--:|--:|--:|--:|
| A10 | 100 (kept) | 34.9 | 10.6 | 3.3 |
| D10 | 36.3% | 12.7 | 0.6 | 22× |
| D9 | 30.2% | 20.9 | 0.7 | 30× |
| D8 | 26.4% | 36.5 | 1.0 | 38× |
| D7 | 24.0% | 66.2 | 1.0 | 69× |
| D6 | 20.7% | 114.1 | 1.0 | 114× |
| D5 | 15.7% | 172.8 | 1.0 | 172× |
| D4 | 9.8% | 214.8 | 0.7 | 305× |
| D3 | 4.6% | 203.2 | 0.4 | 555× |
| D2 | 1.1% | 97.5 | 0.2 | 464× |
| D1 | 0.05% | 9.2 | 0.2 | 41× |

~983 survivors/signal chunk (reproduces the campaign's ~994; D4+D3 = 44% of
detail budget ≈ campaign's 41%). Per-event totals: min 78k / p5 97k /
p50 159k / p95 208k / max 228k — same order as TPC (p50 321k).
**D1 is NOT a TPC-style constant noise floor** (sig:noise 41×, signal-bearing)
— dropping it costs 0.9% of survivors; justified by budget, not by noise.
(Campaign's "~50 coeffs/noise chunk" counted the full padded A10 band; the
valid-region count here is ~17.)

### 9.2 Tree lift across the full depth

| child | P(act) | cond Δ=1 | lift Δ=1 | lift Δ=2 | lift Δ=3 |
|---|--:|--:|--:|--:|--:|
| D9 | 0.302 | 0.648 | 2.1× | — | — |
| D8 | 0.264 | 0.652 | 2.5× | 2.1× | — |
| D7 | 0.240 | 0.662 | 2.8× | 2.5× | 2.2× |
| D6 | 0.207 | 0.632 | 3.1× | 2.9× | 2.6× |
| D5 | 0.157 | 0.564 | 3.6× | 3.3× | 3.1× |
| D4 | 0.098 | 0.468 | 4.8× | 4.0× | 3.6× |
| D3 | 0.046 | 0.340 | 7.4× | 5.4× | 4.3× |
| D2 | 0.011 | 0.156 | 14.1× | 8.5× | 5.8× |
| D1 | 0.0005 | 0.030 | 57× | 18.3× | 9.6× |

P(child|parent) is remarkably flat (~0.65) over D9–D6 and the lift grows
monotonically fine-ward (2×→57×). A10*(thresholded)→D10 lateral lift 1.9×.

**Strict Markov test** P(child | parent inactive, grandparent active) vs
P(child | parent inactive): residual lift 3.0× (D8) → 20.9× (D1). **Observed
activity is NOT first-order Markov on the tree** — exactly what the *hidden*
Markov tree model predicts (the hidden state chain is Markov; observed
activity is not). Consequence: the V-cycle *topology* is still sufficient
(multi-hop information composes through learned node states), but the node
state must carry it — supports d_s not-too-small and the 2-round ablation.

### 9.3 Alignment shift sweep (open #18, mostly retired for optical)

P(child τ | parent (τ+s)>>1), s=−3..+3: mid/fine levels (D7–D1) peak at
s=0/+1 — **naive τ>>1 is near-optimal**. Coarse levels drift negative (D9/D8
maximal at s≤−3, still rising at the sweep edge) but the effect is ~1–2%
relative (0.657 vs 0.648 at D9) — absorbable by a 3-tap within-level conv or
a fixed −1 offset at coarse levels. No large hidden misalignment.

### 9.4 Values

Active-|c| p50 ~20–55 across details; **p99/p1 ≈ 1350× at D10** (vs TPC's
125×) — the bright prompt makes asinh normalization even more mandatory.
>10× adjacent-pair fraction: 16% (D10) → 0% (D2). Same shape as TPC,
amplified at coarse levels.

### 9.5 Density — the big structural difference from TPC

Coarse-grid (1024-tick cell) **detail-union occupancy on signal chunks =
62.5%** (noise chunks 24.4%); with A10 kept, production occupancy = 100% by
construction. Cone depth: mean 5.0 of 9 detail bands active per active cell,
p95 = 9/9. Fan-out per active cell: mean 43, p95 206, max 398 (of 511 cell
capacity — high but not saturated).

**Optical is dense within chunks; its sparsity lives at the chunk level**
(goop's zero-suppression), not the coefficient level. Submanifold machinery
is optional for optical — dense per-chunk tensors are valid (and likely
faster); the unified operator must therefore be support-agnostic (gathers
degrade gracefully to dense indexing). Quantifies the handoff §10 intuition.

Token analog: ~5,755 valid coarse cells/event, all occupied under production
→ optical adds ~5.8k tokens/event at cell granularity (TPC 8×4: 25.4k);
joint ≈ 31k.

### 9.6 Design consequences (confirm/refute of §§1–8)

- **Adjacent-stacked pyramid (V-cycle) confirmed** for optical: flat all-level
  columns are both infeasible (2^10 fan-out) and unnecessary (Δ-decay is
  gradual, adjacent edge strongest everywhere).
- **Level conditioning strongly indicated**: statistics gradients across depth
  (survival 36%→0.05%, lift 2×→57×, p99 13,915→46) are far steeper than
  TPC's — naked weight sharing across 10 levels is riskier; FiLM is the hedge.
- **Cross-modality comparison point**: at matched sparsity, optical D2
  (P=0.011, lift 14×) vs TPC D3 (P=0.014, lift 31×) — same structure,
  roughly half the strength.
- **Drop-D1**: yes for optical too, but as a budget choice, not a noise-floor
  fact.

## Summary

Treat both modalities as sparse coefficients on (spatial axes) × (dyadic
time-tree); scale-equivariant within-level submanifold convs with
level-conditioned shared weights; cross-level communication via a residual
bidirectional tree sweep (up: masked ordered-2-child concat-MLP; down:
parent-MLP) — exact full-tree communication in O(N), depth-independent, the
learned generalization of wavelet HMTs, the same operator at L=4 and L=11 —
interleaved with within-level convs to stitch dyadic cousins; learned
(attention) pooling on the coarse grid judged on saturated cores; capacity in
FiLM-conditioned pointwise MLPs; pre-norm residuals; cross-level flow only
through tree edges. The two open empirical gates: (i) optical tree statistics
(unmeasured), (ii) V-cycle ≡ column coupling at L=4 (the cheap experiment that
decides whether "one architecture" is real).
