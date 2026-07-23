# LArTPC Wavelet-Coefficient Foundation Model — Design Handoff

**Status:** pre-implementation design, data-grounded. Architecture *shape* is settled; several sizing and ablation choices are explicitly open and flagged.
**Purpose:** enable a new person to take over the project without re-deriving the reasoning.
**Audience:** ML engineer / research scientist with CV + sparse-deep-learning background, working with a LArTPC physicist.

---

## 0. TL;DR — what to build

A self-supervised foundation-model encoder over sparse wavelet coefficients from a 6-plane LArTPC, in two parts:

1. **Pre-attention stage (conv, cheap, fixed cost, value-preserving):** per-band submanifold-conv stems → structured wavelet-tree coupling (channel-lift) → core-aware learned pooling to ~25k tokens/event at an 8×4 patch.
2. **Transformer trunk (attention-dominant, scaled):** hierarchical attention — within-plane → within-volume → light cross-volume — at **ViT-L scale (~300M params, d≈1024, ~24 layers)**, pretrained with **masked-coefficient modeling (MAE family)**.

**Headline position (held with confidence):** scale the model **big** (ViT-L target, ViT-H reachable) because this is a foundation model on a ~2.5×10¹¹-token/epoch corpus where capacity pays; achieve speed not by shrinking the model but by holding the **token count fixed** (the data makes it event-independent) and spending capacity on the cheap **d²/FFN axis** that scales at high MFU. Budget: **~13–17k GPU-hours for ViT-L at 30 epochs**, inside the available 10–50k GPU-hour envelope.

---

## 1. Problem setup

### 1.1 Detector and data

- **Detector:** LArTPC with **2 volumes sharing a central cathode**. Some tracks cross the cathode (genuine cross-volume events exist).
- **Planes:** 6 wire planes total = 2 volumes × {U, V, Y}. **U, V are induction** (bipolar, field-response-smeared signal); **Y is collection** (unipolar).
- **Wire counts:** U = 1969, V = 1969, Y = 1443 (identical across the two volumes).
- **Raw time axis:** 4321 ticks (0.5 µs/tick; 405 µs pre + 405 µs post window). DWT periodization pads to 4336.
- **Transform:** per-wire **coif3, 4-level DWT in TIME ONLY** (not across wires).
- **Bands and grids** (coefficients per wire): A4 = 271, D4 = 271, D3 = 542, D2 = 1084, D1 = 2168. So per plane the coefficient grid is (n_wires × {271,271,542,1084,2168}) for {A4,D4,D3,D2,D1}.
- **Drift-time indexing:** one DAQ clock → tick *t* is the same readout time for all 6 planes. **Cross-volume drift alignment is DIRECT (same tick index), NOT mirrored** — established empirically (see §3, cathode continuity), despite the shared-cathode geometry that would naively suggest mirroring.

### 1.2 Pipeline producing the coefficients (upstream, fixed)

GPU production pipeline: densify → coherent + intrinsic noise → digitize → coif3-L4 torch DWT → **coefficient-space smart removal (kgate=4.0)** → per-band-σ hard threshold (VisuShrink hard, per-band MAD σ, κ=1).

- **Smart removal matters:** working post-smart-removal **roughly doubles** the surviving coefficient/token count vs raw-noisy, because un-removed coherent noise inflates the per-band σ and over-thresholds real signal. All numbers in this document are **post-smart-removal** (the correct basis). Earlier raw-noisy numbers were discarded.
- σ_b (per-band MAD σ) is available for free from the threshold step → reuse it for normalization (§5.1).

### 1.3 Goal

Train a foundation-model encoder via SSL (DINO/JEPA/MAE family) on **~10M events for many epochs**, on academic-scale but real compute (**10k–50k GPU-hours**, A100s). Downstream: classification, regression, segmentation. Fluctuations / per-coefficient detail matter **for the representation** (the encoder must not be invariant to them); a U-Net decoder is needed **only** for dense tasks like segmentation, not for the representation model itself.

---

## 2. Hardware and library facts (grounding for all cost estimates)

### 2.1 A100 ceilings

| Quantity | Value |
|---|---|
| BF16/FP16 tensor core (dense) | 312 TFLOPS (624 with structured sparsity) |
| TF32 | 156 TFLOPS |
| FP32 | 19.5 TFLOPS |
| Memory | 40 GB or 80 GB HBM2e |
| Bandwidth | ~1.55 TB/s (40GB) / ~2.0 TB/s (80GB) |

**Realistic sustained for this workload: ~25–45% MFU (~70–150 TFLOPS).** Sparse gather/scatter and small-window attention are memory-bound; do NOT plan around the 225 TFLOPS attention peak.

### 2.2 Attention kernels

- **FlashAttention-2:** ~225–230 TFLOPS, ~72% MFU on A100. This is what you get on A100.
- **FlashAttention-3:** Hopper-only (SM90; needs WGMMA/TMA/FP8). 740–840 TFLOPS, ~1.2 PFLOPS FP8. **Will NOT run on A100.** Only relevant if you get H100/H200 time. vLLM/SGLang auto-select FA3 on Hopper, FA2 on A100.
- Eager attention: never an option beyond ~8k tokens (O(N²) memory).

### 2.3 Sparse-conv libraries (A100)

Hierarchy, slowest → fastest: **MinkowskiEngine (1×) < SpConv v2 (~3–4×) < TorchSparse++ (~4.6×)**. The reference LArTPC code (`lartpc_mlreco3d`) is on MinkowskiEngine — the slowest. **Use spconv 2.x or TorchSparse++.** (PTv3/Sonata ship on spconv inside Pointcept, so adopting that stack gives this for free.) Profiling in this project used **spconv 2.3.8**.

### 2.4 Measured operator costs (A100, bf16, TF32, spconv 2.3.8, flash_attn 2.7.3; median of 20, 5 warmup)

These are the **anchor numbers** for all estimates. Sizes from the post-smart-removal typical event.

| Operator | Size (active pts / tokens) | fwd ms | fwd MB | fwd+bwd ms | fwd+bwd MB |
|---|---|--:|--:|--:|--:|
| SubMConv3d (5w×3t), 32→64 | A4 typ N=30,437 | 0.52 | 821 | 1.25 | 837 |
| SubMConv3d (5w×3t), 32→64 | A4 busy N=42,232 | 0.52 | 827 | 1.36 | 849 |
| SubMConv3d (5w×3t), 32→64 | D2 typ N=8,639 | 0.46 | 811 | 0.98 | 816 |
| flash attn d=768 h=12 | S1=4,987 (1 plane) | 0.51 | 836 | 1.90 | 905 |
| flash attn d=768 h=12 | S2=29,924 (6-plane global) | 13.3 | 990 | 52.6 | 1,407 |

**Key reads:**
- **SubM conv is flat in active count** (0.88→1.36 ms across an ~8× range) → it is **launch/workspace-bound**, not compute-bound, at these sizes. The ~800 MB is fixed first-call spconv workspace; marginal per-call cost is small. **Implication: batch conv across planes** to amortize the launch floor.
- **Attention is quadratic in tokens** (1.9 ms at 5k → 52.6 ms at 30k). The 6-plane flat global stage (~30k tokens) is the cost driver → **must use hierarchy, never flat global.**
- The FFN-bound↔attention-bound crossover sits around **5–8k tokens**. Within-plane (~5k) is FFN-bound and cheap; per-volume (~12.7k) is mildly attention-bound; 6-plane (~30k) is firmly attention-bound.

---

## 3. Data measurements (post-smart-removal, the decision basis)

All from the GPU production pipeline; 200–300 event scans. Representatives by total active-coefficient percentile.

### 3.1 Per-band active-coefficient counts (typical event, 6-plane sums)

- A4 ≈ 104k (≈26%), D4 ≈ 80k (≈26% by occupancy / co-equal), D3 ≈ 82k (**largest band by absolute count, ~30%**), D2 ≈ 43k (~15%), D1 ≈ noise floor.
- **Bands are FLAT:** A4/D4/D3 are co-equal carriers within ~30% of each other. The old "approximation holds half, details are a thin halo" mental model is WRONG and was discarded. Design must treat A4/D4/D3 as co-equal; only D2 is genuinely sparse.
- Per-event total active spans ~5×: min 64k, p5 152k, p50 321k, p95 518k, max 743k.
- **D1 is a near-constant noise floor** (~2,700 U/V, ~400 Y), uncorrelated with activity → **DROP D1.**

### 3.2 Wavelet-tree parent→child occupancy (Item 2) — THE decisive measurement

Tree connectivity (DWT in time only, same wire): D4 coeff at τ → D3 children {2τ, 2τ+1} → D2 grandchildren {4τ..4τ+3}.

| Quantity | Value |
|---|--:|
| P(D3 active) | 0.0140 |
| P(D3 active \| parent D4 active) | 0.4384 |
| P(D2 active) | 0.0037 |
| P(D2 active \| ancestor D4 active) | 0.1062 |

**Lift = ~31× (D3), ~29× (D2).** (Lower than the raw-noisy 48×/47× because smart removal strips spurious correlated common-mode — this is the *correct* direction.) **Conclusion: the wavelet tree is dominant, real structure → structured tree-coupling is justified and preferred over learned cross-band coupling.** This resolves the single biggest architecture fork (see §6, Fork: structured vs learned coupling).

### 3.3 Token budget vs patch size (sweep over 200 events)

Coarse token grid = A4/D4 resolution (n_wires × 271). Patch = (P_wire × P_ctime); footprint maps to fine bands by downsampling factor. Occupancy defined on **band-union**.

| patch | 6-plane mean tokens | 6-sum p5/p50/p95/max | per-volume mean | per-vol p95 |
|---|--:|--:|--:|--:|
| 8×4 | 25,397 | 23.1k / 25.4k / 27.9k / 29.3k | 12,699 | 14,465 |
| 8×8 | 17,931 | 17.1k / 17.9k / 18.9k / 19.5k | 8,966 | 9,643 |
| 16×4 | 17,881 | 17.0k / 17.9k / 18.9k / 19.4k | 8,941 | 9,633 |
| 16×8 | 11,577 | 11.3k / 11.6k / 11.9k / 12.1k | 5,788 | 6,018 |

### 3.4 Per-token load (pooled over all tokens/events)

| patch | fan-out D3+D2 mean/p95/max | coarse A4+D4 mean/p95/max |
|---|--:|--:|
| 8×4 | 5.20 / 35 / 176 | 7.21 / 46 / 64 |
| 8×8 | 7.37 / 50 / 323 | 10.22 / 68 / 128 |
| 16×4 | 7.39 / 43 / 334 | 10.25 / 73 / 128 |
| 16×8 | 11.41 / 76 / 620 | 15.83 / 114 / 256 |

### 3.5 Coefficient value statistics (typical, pooled over active coeffs)

| band | p1 | p50 | p99 | >10× neighbor (large+small adjacency) |
|---|--:|--:|--:|--:|
| A4 | 6.00 | 23.84 | 750.8 | 38.3% |
| D4 | 5.91 | 23.78 | 461.6 | 49.5% |
| D3 | 6.11 | 19.02 | 236.6 | 25.0% |
| D2 | 6.14 | 9.26 | 64.5 | 1.6% |

- Coarse bands span ~125× dynamic range with **large and small coefficients spatially adjacent** (up to 50% of D4 neighborhoods). → compressive normalization mandatory; **never mean-pool** (would blend adjacent large/small). D2 is nearly uniform scale (the easy band).

### 3.6 Cathode continuity (collection Y, 8×4, 200 events)

direct shared occupancy = **0.253 ± 0.068**; mirrored = 0.228 ± 0.065; chance = 0.219.

- **Cross-cathode coupling is real but weak (~3.4 pts above chance) and DIRECT-aligned (same tick), not mirrored.** ~15% of vol0_Y tokens have a genuine cross-volume partner; ~75% have none. → **light cross-volume coupling only** (summary tokens + localized same-tick cathode-boundary attention), NOT full token-level cross-volume attention.

### 3.7 Critical emergent properties (from the sweep)

1. **Token count is nearly event-independent** (6-plane sum varies only ±10% p5→p95 at 8×4, ±3% at 16×8) even though activity spans ~5×. **Busyness manifests as per-token density, not sequence length.** → fixed-shape batching, near-100% MFU, no ragged sequences, no worst-case OOM. **This is a major operational gift — exploit it.**
2. **Coarsening is conserved repackaging:** tokens × mean-load ≈ 315k coefficients at every patch size. No empty space to absorb; active regions are internally dense. → larger patches trade sequence length for per-token fan-out 1:1; coarsening **relocates** information from the (cheap) sequence axis to the (must-summarize) intra-token axis — it is **not free.**
3. **Wire and coarse-time axes are interchangeable for token count** (8×8 ≈ 16×4). Occupancy is locally isotropic in (wire, coarse-time).
4. **Induction dominates: U/V ≈ 79% of all tokens**, Y ≈ half-weight. The attention cost is an induction-plane problem. → trim induction first if needed; give induction stems more capacity.
5. **Volumes are balanced** (vol0 ≈ vol1 token load) → per-volume attention stage is evenly sized.
6. **Saturated cores exist at every scale:** max coarse-load = exact cell capacity (8×4→64, 16×8→256). Tokens exist where *every* A4/D4 slot is active — fully dense track cores (vertices, dense deposits). Fan-out tail grows with patch area (16×8 max 620). → the highest-information regions are the most at risk from over-compression; tokenizer must be core-aware.

---

## 4. Settled principles (not up for relitigation)

These are forced by the data/profiling and underpin everything:

1. **Conv does the fine-resolution work; attention only ever sees a reduced token set.** Forced by profiling (conv flat & cheap; attention quadratic). Conv is value-preserving (a learned projection, never a mean); the only operation that destroys information is mean-pooling.
2. **The wavelet tree is real structure worth hard-coding** (~30× lift) → structured tree-coupling, not learned-from-scratch cross-band.
3. **Token count is fixed per event** → fixed-shape batching, high MFU.
4. **"Full input access" = every surviving coefficient has an unbroken value-preserving path into its token** (via conv / tree-conv / learned pooling — never an average).
5. **Encoder-only representation; U-Net decoder bolted on only for segmentation.** Conv stems retain full-resolution features on skip connections for that purpose.
6. **Drop D1** (noise floor).
7. **Direct (same-tick) cross-volume alignment, no mirroring** (empirical, §3.6).
8. **At fixed token count, scaling the model is ~pure d²/FFN growth** — the high-MFU, cheap-to-scale axis. The quadratic-in-tokens term and the irregular sparse ops (the costs that don't scale) are already engineered out (fixed tokens + tree-conv + hierarchy). This is why "attention scales" and "we need speed" are NOT in tension here.

---

## 5. The architecture

### 5.1 Pre-attention stage (conv) — full spec and reasoning

Job: **~200k raw coefficients/event → ~25k information-dense tokens, value-preserving, at conv speed.** ~6 sequential conv layers deep on the coarse path. Cost is **flat in model scale** (does not grow B→L→H).

**Stage 0 — Drop D1.** Noise floor (§3.1), largest grid, least information. Free win.

**Stage 1 — Per-band normalization: `asinh(c / σ_b)`.**
- Compressive: coarse bands span ~125× range (§3.5); raw values would let large coeffs dominate gradients and bury the small real signal.
- `asinh` not `log`: coefficients are **signed** (detail coeffs carry sign/phase); asinh handles sign and zero smoothly.
- Per-band σ_b not global: bands live at different amplitude scales; σ_b is free from the threshold step.
- Monotonic → preserves the local large/small contrast that *is* the fluctuation structure.

**Stage 2 — Per-band, plane-specific SubM conv stems (2–3 layers/band).**
- **Submanifold conv** specifically: fires only on active sites, refuses to dilate occupancy → preserves sparsity (bounded memory), value-preserving (learned projection of real neighbors), ~3× cheaper than strided SparseConv, flat in active count (§2.4).
- **Per-band weights:** bands have different statistics (A4 125× range vs D2 ~8×); shared kernels = bad prior.
- **Plane-specific weights, shared across volumes, specialized across plane-type (U/V/Y):** induction (bipolar, smeared) and collection (unipolar) have different detector responses; one filter set can't model both. Share where physics is identical (vol0_U ↔ vol1_U), specialize where it differs (U vs V vs Y).
- **Induction-weighted capacity:** U/V carry ~79% of tokens (§3.7.4) and are the harder, noisier projection → wider stems than Y.
- **Anisotropic kernel (5 wire × 3 time):** wire is the genuine spatial axis (tracks continuous across wires); time-coeff axis is already a frequency-localized DWT projection. Reach further in wire.
- **Shallow (2–3 layers):** SubM receptive field grows only along active sites, not across gaps; depth past ~3 layers wastes parameters (isolated coeffs get context from tree-coupling instead). Keeps stem cheap and scale-invariant.
- **Batched across planes:** SubM is launch-bound (§2.4); 24 separate calls = 24× fixed overhead. Batch band-types into single calls with batch index. **(Unverified amortization assumption — see §7 / microbenchmark.)**

**Stage 3 — Structured tree-coupling / channel-lift (1–2 layers).** See §5.3 for full detail.
- Lifts fine-band features into the coarse grid's **channels** along the wavelet parent→child stencil (same wire; D2 4τ..4τ+3 → D3 2τ,2τ+1 → D4/A4 τ).
- **Why structured conv, not cross-attention:** the ~30× tree lift (§3.2) means the coupling is known, dominant structure; a fixed-stencil sparse conv runs at conv MFU vs an irregular gather's ~20–30% MFU, and is better-justified than learned coupling given the measured prior.
- **Why lift into channels, not pool into space:** coarsening is conserved repackaging (§3.7.2) — spatial pooling relocates real info. Putting detail into the channel dimension lets the spatial grid be pooled to tokens without severing any coefficient's path. Classic "grow channels as you shrink resolution," applied to guarantee full input access.
- **Direction fine→coarse:** the token lives on the coarse grid, so detail flows up the tree to reach it.

**Stage 4 — Core-aware learned pooling to tokens (1 strided conv, 8×4).**
- **8×4 chosen** (not coarser): saturated cores (§3.7.6) make coarser patches over-compress the highest-information regions. 8×4 keeps cores legible (≤64 coarse coeffs/token); control attention cost via hierarchy, not via core-damaging coarsening. "Fine patches + hierarchical attention > coarse patches + flat attention" — pay cost in the cheap place (more tokens through cheap within-plane attention), not the damaging place (over-summarized cores).
- **Learned strided conv, not mean-pool:** large/small adjacency up to 50% (§3.5); mean would blend them and destroy fluctuation. A learned projection can preserve contrast.
- **Core-aware** because stems + tree-coupling build rich features *before* pooling, so dense cores are summarized by conv over structure, not by averaging raw spikes. **Order matters: stems → tree-coupling → pool, never pool-first.**
- **Band-union occupancy:** a token is created wherever ANY band is active, so a feature living mostly in D2/D3 isn't silently dropped.

**Output:** ~25,397 tokens/event (8×4 mean), fixed ±10%, each a d-vector = coarse structure ⊕ tree-lifted fine detail ⊕ value-preserving trace of every contributing coefficient.

### 5.2 Transformer trunk — spec and reasoning

**Scale: ViT-L target (d≈1024, ~24 layers, ~300M params).** Reasoning in §8 (model size). Built so ViT-H is a config change, not a rewrite.

**Hierarchical attention (kills the quadratic without shrinking the model):**
- **Within-plane self-attention** — bulk of the depth (~12+ layers). ~5k tokens/plane, FFN-bound, cheap (~1.9 ms/layer at d=768; more at d=1024). Builds global-within-plane receptive field (conv only reached locally; tracks span the plane).
- **Within-volume cross-plane attention** — the quadratic stage, used sparingly (~3:1 within:across to start). 3 planes ≈ ~12.7k tokens/volume (mildly attention-bound). Shared drift-time positional encoding + plane-type embeddings; learns U/V/Y stereo correspondence rather than hard-coding back-projection.
- **Cross-volume — light only:** summary tokens (per-plane + per-volume + event token) + **localized same-tick cathode-boundary attention** over boundary tokens (the ~15% genuine crossings, §3.6). NOT full token-level (coupling too weak to justify quadratic cost), NOT summary-only (would sever real crossing tracks). Direct alignment, no mirroring.

**Tokens / special tokens:** all coarse tokens (spatially grounded for correspondence + segmentation decoder); per-plane summary tokens; an event token (target for an optional DINO-style global term). Summary/event tokens discarded for dense tasks.

**Shared trunk across planes** (high-level abstractions are common; 3× param saving). Plane-type embeddings + shared drift-time PE carry the asymmetry.

### 5.3 Lifting — detailed

**Basic operation (recommended default).** For each coarse coeff at (wire, τ): gather tree descendants (D3 at (wire, 2τ),(wire,2τ+1); D2 at (wire,4τ..4τ+3), same wire), apply a learned linear map to each child's feature, aggregate (sum or small MLP), concatenate onto the parent's channels. = a sparse conv whose connectivity is the wavelet tree. Inactive children contribute nothing. Output on coarse grid, channels = coarse ⊕ lifted-D3 ⊕ lifted-D2. "Lift" = info raised from fine to coarse grid **without resampling** (carried by tree edges, not interpolated → detail survives in channels; only its spatial address changes).
- **Layers:** 1–2. Recommended **2 thin layers** (D2→D3, then D3→coarse) so D2 passes through D3's learned map rather than jumping straight to coarse (respects tree structure). Add a nonlinearity between hops.
- Value-preserving in practice (nothing averaged) but no *formal* invertibility guarantee.

**Upgrade: true lifting scheme (predict/update, DSP sense).** Replace the one-directional gather with a learned **predict/update cascade** (predict detail from approximation; update approximation using detail), which is **invertible by construction** (like a normalizing flow) → mathematical zero-information-loss guarantee. More constrained, slightly more expensive. Drop-in replacement for the same stage. **Adopt only if** a downstream task needs provable reconstruction or the basic gather is found to bottleneck detail.

### 5.4 Re-injection — detailed

**Default: OFF** (lift once, discard fine bands before trunk → simpler, less memory).

**What it is:** let trunk tokens reach back into the (kept-resident) fine-band features at 1–2 deeper layers and pull detail again, conditioned on accumulated context. Mechanically: cross-attention, trunk tokens = queries, fine-band features = keys/values, windowed to each token's physical footprint. Cheap (fine bands tiny, window local).

**Why:** at layer 0 a token has no global context, so the once-only lift commits to a fixed detail summary blindly. By a middle layer the token "knows" it's e.g. a near-vertex track token and can ask a more informed question of the fine bands. Likely helps exactly on the hardest events (vertices, overlaps).

**Cost:** small compute + must keep fine-band features resident through the trunk (memory).

**Recommendation:** wire as a flag at one middle layer; turn on if linear-probe improves. **0 layers default, 1 as tested upgrade.**

### 5.5 SSL objective

**Primary: masked-coefficient modeling (MAE / Point-BERT family).** Mask active coefficients, reconstruct (or predict latents). Tree-subtree-masking variant (mask a subtree, predict from parent) directly exercises the §3.2 structure.
- **Why primary:** (a) cheapest at scale — encoder sees only visible tokens (~30%), so pretraining is ~3× cheaper *per step*, which is precisely what makes ViT-L affordable for enough epochs; (b) directly forces fluctuation retention (DINO would wash it out); (c) scales cleanly to large ViTs on raw data.
- Mask only active coeffs (sparse) — don't pay to reconstruct empty space.

**Optional add: DINO-style self-distillation term on the event token** for clean linear-probe — only if linear-probe specifically needs it.

**Avoid pure DINO:** rewards invariances, washes out the fine detail the representation must keep.

**JEPA (latent prediction):** ~20–30% cheaper than DINO, no decoder, but representation quality on this data unproven. Fallback efficiency lever.

### 5.6 Layer count summary

| Component | Layers |
|---|---|
| Per-band stems (×bands, batched) | 2–3 each (3 coarse / 2 D2) |
| Tree-coupling lift | 1–2 (recommend 2) |
| Core-aware tokenizer | 1 strided |
| **Pre-attention total (coarse path)** | **~6 sequential, all conv** |
| Transformer trunk (ViT-L) | ~24 (within-plane bulk : within-volume ≈ 3:1 + light cross-volume) |

---

## 6. Cost / scale analysis

### 6.1 Per-event step (fwd+bwd), expected

| Component | Expected | Range / risk |
|---|--:|--:|
| Pre-attention stems (batched) | ~25 ms | 15–40 ms (batching amortization unverified) |
| Tree-coupling lift (1–2 layers) | ~10 ms | 5–15 ms (non-standard stencil cost unknown) |
| Core-aware tokenizer (1 strided) | ~4 ms | 3–5 ms |
| **Pre-attention total** | **~45 ms** | **~35–60 ms** |
| Transformer trunk (ViT-L, MAE 30% visible) | ~200 ms | ~150–250 ms |
| **Full step** | **~245 ms** | — |

**Pre-attention ≈ 15–25% of the step; transformer ≈ 75–85%.** This ratio is intended and correct (cheap fixed compressor << expensive scalable trunk). **Pre-attention cost is flat in model scale**, so the ratio tilts further toward the trunk as you scale B→L→H — the compression overhead amortizes against a bigger model.

**Memory:** pre-attention ~2–4 GB (conv never near the wall; fixed spconv workspace ~1 GB dominates). Trunk ViT-L with flash + checkpointing ~10–20 GB. Comfortable on 40–80 GB.

### 6.2 Full-run budget (10M events)

| Config | ~ms/event | 30 epochs (GPU-hr) | 60 epochs |
|---|--:|--:|--:|
| ViT-B, full tokens (older plan) | ~130 | ~11k | ~22k |
| **ViT-L, MAE 30% visible (recommended)** | ~150–200 | **~13–17k** | **~26–34k** |
| ViT-L, full tokens | ~400+ | ~33k | — |
| Conv-heavy + thin attn cap (null hypothesis) | ~70–90 | ~6–8k | ~12–16k |

**Punchline:** the MAE masking trick + fixed-token hierarchy make **ViT-L cost about what full-token ViT-B would have.** ViT-L is the *right* case, not the stretch case. Budget is **not** the binding constraint — all viable configs fit 30 epochs in 10–50k GPU-hr. Decisions should be made on **efficacy per GPU-hour**, measured on small models.

### 6.3 Token / corpus scale

10M events × ~25k tokens ≈ **2.5×10¹¹ tokens/epoch** — a large corpus comparable to midsize LM pretraining. At this scale undersizing is the expensive mistake (see §8).

---

## 7. Open points (prioritized) — and how to close each

| # | Open question | How to resolve | Priority |
|---|---|---|---|
| 1 | **Tree-coupling lift cost** (non-standard stencil — kernel-map build cost genuinely unknown; could be >spatial conv) | Microbenchmark on dumped event (§9). Fallback: once-only single-layer gather, or dense-on-coarse-grid form (coarse grid is small) | **HIGH** (could move pre-attention total ~2×) |
| 2 | **Batched-stem amortization** (assumption that collapsing 24 calls → few recovers launch floor is unverified) | Microbenchmark: 24 separate vs batched calls | **HIGH** |
| 3 | **Structured tree-conv vs learned coupling** at scale (bitter-lesson concern) | Ablate on ViT-S at fixed token budget; lift is 30× so structured strongly favored, but verify | MED (data favors structured) |
| 4 | **Re-injection on/off** | Flag at one middle layer; measure linear-probe delta | MED |
| 5 | **Within:across attention ratio + within-plane depth** | Sweep on ViT-S (start 3:1, ~12 within-plane) | MED |
| 6 | **ViT-L vs ViT-H ceiling** | Read ViT-L loss/linear-probe scaling curve; depends on unique-event diversity (only a run reveals). Build so H = config change | MED |
| 7 | **Lifting: basic gather vs true invertible scheme** | Adopt invertible only if provable reconstruction needed or gather bottlenecks detail | LOW |
| 8 | **Value semantics: raw vs residual injection** | Look at data: are fluctuations small-amplitude coeffs or small variations on large ones? (asinh + per-band σ handles most of this) | LOW |
| 9 | **H100/H200 access** | If available, FA3 + FP8 is a large multiplier; plan core budget on A100/FA2 regardless | LOW (opportunistic) |

**The two to do first (cheap, on the dumped event, no training): #1 and #2.** They replace the biggest extrapolations in the cost model with real numbers.

---

## 8. Model size — the considered position (important; reverses earlier hedging)

Earlier drafts repeatedly suggested "the data is low-entropy so you don't need a big model." **That was wrong. Discard it.** The settled position:

1. **SSL is the regime where capacity pays most** — the pretext task is unbounded (no label ceiling), so the model keeps finding structure as you add parameters. This is the empirical content of the DINOv2 / JEPA / scaling-law literature. Your pretext signal is raw detector physics, not a small label set.
2. **The data killed the low-entropy premise:** flat co-equal bands (§3.1), saturated dense cores (§3.7.6), 5× busyness as per-token density, strong-but-nontrivial cross-band structure. High information content per event, concentrated and structured → wants width.
3. **The corpus is large:** ~2.5×10¹¹ tokens/epoch. At this scale the binding constraint is "model too small to absorb what's there," not "model too big and overfits." Undersizing a foundation model on a corpus this large is the classic expensive mistake.

**Recommendation:** target **ViT-L (~300M)**, build so **ViT-H** is a config change, decide the ceiling from ViT-L's scaling curve (gated on unique-event diversity, which only a run reveals). Make the big model *fast* by holding tokens fixed and scaling the d²/FFN axis — do **not** make a small model adequate.

### Note on the conv-heavy alternative (considered and de-prioritized)
A deep SpUNet + thin attention cap is cheaper (~6–8k GPU-hr/30ep) and is a legitimate **null hypothesis**. It was de-prioritized as the *recommendation* because conv's locality prior plateaus with scale while attention keeps improving, and you are firmly in the abundant-data regime where the weaker (attention) prior wins. Keep it as the baseline the expensive design must beat, not as the plan.

---

## 9. Recommended next steps (in order)

1. **Microbenchmarks (no training, on `typical_event_coeffs_smart.npz`):** resolve open points #1 (tree-coupling lift cost) and #2 (batched-stem amortization). Replaces the two biggest cost extrapolations. *(Spec below.)*
2. **Build pre-attention stage** (spconv 2.x or TorchSparse++): drop D1 → asinh norm → batched per-band stems → tree-coupling lift → 8×4 core-aware tokenizer. Verify it outputs ~25k fixed-shape tokens at ~45 ms and ~2–4 GB.
3. **Build ViT-S trunk + MAE**, run the small-model sweeps: open points #3 (structured vs learned coupling), #4 (re-injection), #5 (within:across ratio + depth). A few hundred GPU-hours total. Validate the FLOP model against a real multi-layer step.
4. **Commit to ViT-L + MAE**, 30-epoch run (~13–17k GPU-hr). Read the scaling curve → decide ViT-H (#6).
5. **Downstream:** bolt on DPT/U-Net decoder (through conv skips) for segmentation; linear-probe / fine-tune for classification & regression.

### Microbenchmark spec for step 1
On the dumped event, measure (A100, bf16, TF32, median 20 / warmup 5, fwd and fwd+bwd, report ms + peak MB):
- **(A) Tree-coupling lift:** implement the parent→child stencil as (i) a spconv with custom connectivity, (ii) a dense-on-coarse-grid gather. Sizes = typical event per-band counts. Compare both forms.
- **(B) Batched stems:** 24 separate SubM calls (per band × plane) vs band-type-batched calls with batch index, at the §2.4 sizes. Report the amortization factor.
- **(C) Token-budget recompute** (if not already done): occupied tokens + per-token coarse-load p95 at 8×4 and 8×8, to confirm the §3.3/§3.4 numbers and the core-load tail.

---

## 10. Catalogue of all variations mentioned (decision menu)

For the successor — every fork raised in the design process, with the current lean.

**A. Pre-attention / tokenization**
- A1. Patch size: **8×4 (recommended)** / 8×8 / 16×4 / 16×8. Trade token count vs saturated-core fidelity. Wire and time axes interchangeable for token count.
- A2. Tokenizer pooling: **learned strided conv (recommended)** / mean-pool (rejected — destroys large/small adjacency).
- A3. Stem depth: **2–3 layers (recommended)** / deeper (rejected — SubM RF caps usefulness).
- A4. Stem weight sharing: **per-band + per-plane-type, shared across volumes (recommended)** / fully shared (rejected) / fully independent per plane (wasteful).
- A5. Induction capacity: **wider stems for U/V (recommended)** / symmetric.
- A6. Normalization: **asinh(c/σ_b) (recommended)** / log (rejected, signed) / global σ (rejected, scale mismatch) / raw (rejected).
- A7. Sparse library: **spconv 2.x or TorchSparse++ (recommended)** / MinkowskiEngine (rejected — slowest).
- A8. Stem batching: **batched across planes (recommended; unverified — open #2)** / per-band-per-plane.

**B. Cross-band coupling (lifting)**
- B1. Mechanism: **structured tree-conv (recommended, 30× lift)** / irregular cross-attention (low MFU) / learned unconstrained (bitter-lesson hedge, ablate as #3).
- B2. Channel-lift vs spatial pool: **channel-lift (recommended; conserved-repackaging argument)**.
- B3. Lift depth: 1 / **2 thin layers (recommended)**.
- B4. Lifting form: **basic gather (recommended)** / true invertible predict-update scheme (upgrade, open #7).
- B5. Re-injection: **off (recommended default)** / 1 middle layer (tested upgrade, open #4).

**C. Trunk / attention**
- C1. Trunk type: **attention-dominant (recommended)** / conv-heavy + thin cap (null hypothesis, §8).
- C2. Hierarchy: **within-plane → within-volume → light cross-volume (recommended)** / flat 6-plane global (rejected — 53 ms/layer) / extra bottleneck-pool stage then global (Fork A.3, plausible, untested).
- C3. Within:across ratio: **3:1 start (open #5)**.
- C4. Cross-volume: **summary tokens + same-tick cathode-boundary attention (recommended)** / summary-only (severs crossings) / full token-level (wasteful). **Direct alignment, no mirroring** (§3.6).
- C5. Special tokens: per-plane summary + per-volume summary + event token.

**D. Scale**
- D1. Model size: ViT-S (sweeps only) / ViT-B (older plan) / **ViT-L (recommended target)** / ViT-H (config-change reach, open #6).
- D2. Precision: **bf16 (recommended, ≈fp16 in time/mem, better stability)**.
- D3. Hardware: **A100/FA2 (plan basis)** / H100-H200/FA3+FP8 (opportunistic multiplier, open #9).

**E. SSL objective**
- E1. **Masked-coefficient modeling / MAE (recommended primary)**, incl. tree-subtree-masking variant.
- E2. + DINO term on event token (optional, for linear-probe).
- E3. JEPA latent prediction (efficiency fallback).
- E4. Pure DINO (rejected — washes out fluctuations).

**F. Downstream**
- F1. Segmentation: bolt-on DPT/U-Net decoder through conv skips (encoder-only otherwise).
- F2. Classification/regression: linear-probe / fine-tune off the shared trunk + event token.

---

## 11. Reference artifacts

- `typical_event_coeffs_smart.npz` (ev33, ~201,757 active rows) — arrays: `plane_gid` (0–5 = vol0 U,V,Y / vol1 U,V,Y), `band_id` (0=A4,1=D4,2=D3,3=D2,4=D1), `wire`, `time_index`, `value`. Recompute §3.3–3.5 at any patch size without a new run.
- `research/measure_coeffs.py` — measurement script (`--scan N --kappa K --patch-wires ...`).
- `research/coherent_coeffs/smart.py` — smart removal (default `kgate=4.0`).

### External references (architectural inspiration / prior art)
- **PTv3** (serialization + windowed flash attention; 3× speedup, 10–16× memory vs PTv2) — serialization itself NOT used here (regular grid, ~50k coeffs; the Hilbert sort buys nothing) but the conv-PE + flash-attention substrate is relevant.
- **Sonata** (encoder-only PTv3, self-distillation SSL, decoder-free, upcast-and-concatenate, linear-probe with <0.2% params) — SSL recipe reference.
- **VGGT** (alternating frame-wise / global attention for multi-view; learns geometry rather than imposing it) — inspiration for within-plane/cross-plane alternation and "don't hard-code back-projection." NOT copied: bands are not symmetric views (asymmetric coarse-token/fine-injection instead); planes are not weight-shared at the stem (different response).
- **Submanifold sparse conv on LArTPC** (DeepLearnPhysics / Domine & Terao): SSCN gives ~364× memory / ~33× wall-time reduction vs dense CNN in 3D (~93× / ~3.1× in 2D) at no accuracy loss — establishes sparse-native as mandatory.
- **Sparse-ViT foundation model on LArTPC (PILArNet)** (Alonso-Monsalve et al., 2026): MAE + relational objectives, occupancy-masked patch tokens; ~10³ labeled events matches from-scratch with 10× more — validates the SSL premise.

---

*End of handoff. The architecture shape is data-grounded and settled; §7 lists exactly what remains open and §9 the order to close it. Start with the two microbenchmarks (#1, #2) — they convert the largest cost extrapolations into measured numbers before any training commitment.*
