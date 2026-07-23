# SOTA literature synthesis (5 agents, 2023–2026) — verdict on every choice, with the WHY

Five parallel literature searches (MAE/masked-modeling; JEPA/latent-predictive;
efficient/scalable attention; sparse/point-set + positional encoding; scientific
FMs + multimodal fusion). Verbatim agent outputs (with citations) in
`reviews/lit_*.md`. This is the integrated verdict mapped to our design, with
the mechanism/evidence behind each call. "Follow SOTA but understand every choice."

## The single deepest finding (two agents, independently): objective vs the fluctuation requirement
Our hard requirement — preserve per-coefficient fluctuations (the unpredictable
part IS physics signal) — is **mathematically incompatible with an L2 loss and
with the entire JEPA/distillation family**, and **uniquely compatible with
reconstruction-in-coefficient-space + a distributional head**:
- **L2 = conditional mean** (provable; Bregman). It outputs E[c|context] →
  averages away exactly the high-frequency, low-predictability content. DiffMAE:
  "MAE is known to produce blurry reconstructions that lack high-frequency."
- **JEPA / I-JEPA / V-JEPA / DINO / data2vec PROVABLY suppress low-ρ
  (unpredictable, high-variance) directions, exponentially in depth** (Apple,
  NeurIPS 2024, "How JEPA Avoids Noisy Features", arXiv:2407.03475). I-JEPA: "the
  encoder discards precise low-level details." That is the *designed* mechanism —
  it cannot distinguish instrument noise from rare unpredictable signal; both are
  low-ρ and both get suppressed. → **JEPA/DINO are the WRONG primary objective.**
- **MAE-style reconstruction preserves high-variance content** (same proof, other
  side). LHC anomaly-preservation (Phys.Rev.D 2025, arXiv:2502.15926): when rare
  unpredictable excursions are the signal, choose the detail-preserving objective.
- **REVISION (biggest): replace the L2 reconstruction head with a DISTRIBUTIONAL
  head** — per-coefficient Gaussian/MDN NLL (cheap, drops into our decoder; yields
  per-coeff variance) or conditional-diffusion decoder (DiffMAE, strongest). l-DAE
  (ICLR 2025): the *denoising objective* drives representation quality, the
  *stochastic head* drives fidelity — they decouple, so we need the distributional
  head specifically for the fidelity/fluctuation requirement. **Validate
  reconstructed per-band VARIANCE against the known noise floor, not just MSE.**

## Verdict table (our choice → SOTA verdict → why → action)
| our choice | verdict | why (evidence) | action |
|---|---|---|---|
| masked-coeff reconstruction objective | **ENDORSE** | PoLAr-MAE (LArTPC, 1000× label-eff), FASERCal FM (75% MAE), MMR (wavelet-coeff MAE), SIT | keep |
| **L2 reconstruction head** | **REVISE** | L2=conditional mean → kills fluctuations (provable; DiffMAE blur) | → Gaussian/MDN NLL head; diffusion if needed; validate variance |
| reconstruct OWN/raw coeffs (not latent) | **ENDORSE** | latent/feature/codebook targets smooth by design (I-JEPA "discards detail") | keep; reject JEPA/DINO targets for the recon head |
| per-band patchify + linear embed | **ENDORSE + ADD** | Spectral Image Tokenizer (ICCV 2025): per-subband patchify + **separate norm/MLP for approx vs detail** (different stats); our A10→D1 survival 100→0.04% is an even bigger gradient | add band-specialized norms/embeddings (≥ approx vs detail) |
| RoPE on physical time | **ENDORSE** | RoPE injects relative position; handles CONTINUOUS coords natively (angle=coord×freq) | keep |
| RoPE on wire | **ENDORSE** (axial) | N-d/axial RoPE (independent block/axis) is principled (MASA, arXiv:2504.06308); continuous coords fine — our wire hesitation was unwarranted | add as 2nd axial axis; STRING (ICML25) if wire×time coupling matters |
| RoPE on scale/level, plane | **REVISE → don't** | non-metric axes; RoPE injects a spurious distance | learned embeddings (free from per-band patchify) |
| permutation-invariant set backbone | **ENDORSE + FIX** | correct for no-ordering; BUT identical mask tokens → identical outputs (Masked Particle Modeling on Sets) | break symmetry **in the prediction head**; guard coordinate leakage (PCP-MAE) |
| no windowing (no 1D order) | **ENDORSE** | windowing/NSA/Mamba all need a 1D order → OUT | keep full/latent attention |
| 75% masking | **ENDORSE as start** | difficulty∝AvgDist (SimMIM); 75% calibrated to image redundancy | calibrate to our sparse-set redundancy (maybe <75%) |
| mask whole planes | **CONDITIONAL** | helps only if planes mutually redundant (VideoMAE tube); whole-SCALE-BAND masking HURTS −5-6% (MMR) | whole-plane = minority mode; never default whole-band; keep random as baseline |
| two-head occupancy decode | **OK + GUARD** | "occupancy shortcut" (NOMAE): masking only non-empty → model cheats | train support vs plausible-but-empty candidates; restrict to neighborhood of active support |
| dead channels | **ADD** | represent as learned "dead" token at true coord, not absence (MAE mask-token) | + wire-kill augmentation (already have) |
| discrete tokenizer (VQ) | **REJECT** | MPMv2: strong decoder beats VQ codebook (+34 pts); gain is the decoder not the tokenizer | keep continuous coeffs + strong decoder |

## The NEW evaluation backbone (we were missing this — agent 5)
- **NEVER use reconstruction/coeff-MSE as the quality signal** — blind to
  posterior collapse, and non-monotonic with representation quality (MAE: higher
  mask = worse fidelity, better features).
- **Label-free metrics**: RankMe (effective rank, Pearson>0.99 with downstream) →
  **LiDAR** (preferred for masked/generative encoders — RankMe inflated by noise
  directions) → α-ReQ (eigenspectrum slope). Triangulate; use to pick
  hyperparameters and detect collapse with NO neutrino labels.
- **Headline result format**: frozen-backbone **label-efficiency curve vs
  supervised-from-scratch**, swept over label budget — this is what produced the
  1000×/100×/10× claims. Report linear-probe AND fine-tune (expect probe ≪ FT for
  a reconstruction MAE — don't read that as failure; probe intermediate layers,
  MIM-Refiner ICLR 2025).

## Domain precedents / baselines to beat
- **PoLAr-MAE** (arXiv:2502.02558): MAE on LArTPC charge point-cloud, 1000× label
  efficiency, 99.4/97.7 track/shower F. Our differentiator = wavelet-coeff token
  space + fluctuation-preserving distributional head (PoLAr is deterministic).
- **FASERCal FM** (2026, arXiv:2604.07037): MAE 75% + Perceiver-IO fusion, sim→real
  cross-detector transfer WORKS (PILArNet 0.933→0.966). Architectural template.
- **MMR** (arXiv:2601.12215): wavelet-coeff MAE on 1D signals — near-exact method
  precedent; used plain MSE (so does NOT solve our fluctuation requirement = our
  contribution). **WaveToken** (ICLR 2025): wavelet-coeff token space works.

## Fusion (agent 5) — we're in the EASY regime
- **Fuse via the shared PHYSICAL coordinate (xyz,t)** — BEVFusion/PiMAE: coordinate
  bridge beats learned-only alignment; our cleanest differentiator.
- **Sim gives truth-level pairing for FREE** (same event → both detectors), so the
  earlier "unpaired-fusion problem" is solved: train alignment on abundant sim
  pairs, transfer aligned space to unpaired real (ImageBind anchor as fallback).
  Early-fusion ≥ late-fusion at scale (Apple native-MM scaling, ICCV 2025).
- Cross-modal masked prediction (4M/MultiMAE) — predict one modality from the
  other, pseudo-labels-from-forward-model maps onto our sim.

## Scaling reality (agent 5)
Detector/physics data has **weak, early-saturating scaling** (jet-gen β_D≈0.74,
small "learnable window"). **Our 580k corpus is fine** — the FM's value is **label
efficiency, NOT scale gains**. Curate/diversify (dedup, balance topology/noise,
maximize sim-generator diversity; cap ~4 epochs) over collecting more events.
Optical 20k is small → cross-modal binding to the 580k TPC arm earns its keep.

## The two tensions to manage (not resolved by any single choice)
1. **Fidelity vs semantics.** Reconstruction preserves fluctuations but gives WEAK
   frozen features (Dark Secrets of MIM; semantics in intermediate layers).
   Resolution: two heads on one encoder — reconstruction+distributional (fidelity)
   + optional latent/distillation aux (semantics, SALT-style frozen-teacher where
   the teacher is the recon encoder); probe intermediate layers.
2. **Scaling tokens without ordering vs fidelity.** Latent-bottleneck
   (FLARE/Perceiver-IO, BiXT — permutation-equivariant, O(N·M), the no-order SOTA
   scaler) is LOSSY → conflicts with fluctuation preservation. Resolution: use the
   latent bottleneck only for the global/fusion tier (lossy summary acceptable
   there); keep full per-token fidelity in the local encoder + decoder. MoE for
   parameter scaling at fixed FLOPs (order-agnostic, in the FFN). FlashAttention is
   the kernel substrate (FA-3/FP8 if H100).

## ARCHITECTURE-FOCUSED verdict (the trunk/tokenizer/PE/decoder, not the objective)

### Tokenizer / early blocks
- **Per-band patchify + linear embed: ENDORSE.** + **band-specialized blocks**:
  Spectral Image Tokenizer (ICCV 2025) processes approximation vs detail tokens
  with SEPARATE norms/MLPs/attention ("different statistics across scales"). Our
  A10→D1 survival 100%→0.04% is a steeper gradient than images → at minimum
  separate norms for approx vs detail; possibly band-conditioned (FiLM) params.
- Tokenizer is a representation/compression choice, NOT a semantic tokenizer:
  MPMv2 — a strong DECODER beats a VQ codebook (+34 pts); don't over-invest in the
  tokenizer, invest in the decoder. Keep continuous coeffs.

### Positional encoding (architecture of "where")
- **Axial / N-d RoPE — independent 2×2 rotation block per CONTINUOUS axis
  (time, wire)** is the principled SOTA (MASA result, arXiv:2504.06308): it's the
  unique structure preserving relativity+reversibility. Continuous coords are
  native (angle = coord×freq) — no integer indices needed.
- **Do NOT RoPE non-metric axes (scale/level, plane)** → learned embeddings.
- **STRING** (ICML 2025): the upgrade if wire×time coupling carries physics
  (exact translation invariance + cross-axis coupling); else axial is simpler.
- **Inject physics via FEATURES / pairwise interactions** (Particle Transformer),
  not only positions — the HEP-point-cloud standard for geometry without an order.

### The trunk / attention — the central architecture question (no 1D ordering)
Filter: usable as-is only if **permutation-equivariant** OR **content-based**
routing. Windowing / NSA / Mamba/SSM are OUT (all need a 1D order).
- **Latent-bottleneck cross-attention (FLARE / Perceiver-IO / BiXT) = the SOTA
  order-free scaler.** Encode N→M latents (cross-attn), process M latents
  (self-attn, cheap), decode M→N. O(N·M), permutation-equivariant, runs on flash
  kernels (FLARE = 2 SDPA calls), scales to 1M unordered points (PDE meshes).
  **This is the architectural answer to "large unordered set + want to scale +
  want fusion."** BiXT (NeurIPS 2024) = bidirectional variant that fixes the
  bottleneck's weakness on dense/instance tasks.
  - CAVEAT (the tension): the bottleneck is LOSSY → conflicts with fluctuation
    preservation. RESOLUTION: full per-token attention in the LOCAL encoder + the
    DECODER (full fidelity for reconstruction); latent bottleneck only for the
    GLOBAL / cross-plane / cross-modal FUSION tier (lossy summary acceptable there).
    This is also exactly FASERCal's design (sparse-conv tokenizer + **Perceiver-IO
    fusion across detector streams**, sim→real transfer verified).
- **At our current N (~25–30k/event), full FlashAttention is feasible** (measured
  ~5–7 ms/attention) — so the local tier can be plain full attention; the latent
  bottleneck is the lever for going *bigger* (finer patches, more tokens, fusion).
- **Token reduction: ToMe / PiToMe** (content-based bipartite merging, NO order,
  training-free) as a front-end reducer — matches our sparse mid-scale-concentrated
  tokens; merge-then-unmerge for dense decode.
- **MoE FFNs (DeepSeekMoE: fine-grained + shared experts)** = scale PARAMETERS at
  fixed FLOPs, orthogonal to attention and ordering (lives in the FFN), and "helps
  early fusion" (Apple native-MM scaling). The cheap way past 300M params.
- **Factorized / physical-group attention** (within-plane / within-band / region,
  via varlen cu_seqlens) = the structured, order-free alternative to a global
  tier — uses REAL groupings, not arbitrary windows (CaFA precedent).
- **Serialization (PTv3 space-filling curves) is OUT for us** — it's a throughput
  hack for 10⁵–10⁶ points that trades exact geometry for a sorting heuristic; at
  our N, set/coordinate-PE attention is strictly better. Contingency only.

### Decoder
- **Asymmetric**: heavy encoder sees visible tokens only; mask tokens enter the
  DECODER only (MAE/Point-MAE). Decoder depth ~8 blocks if frozen-feature eval
  (keeps encoder abstract), shallow if fine-tuning — but **budget decoder capacity
  for the distributional head** (the NLL/diffusion head lives here; MPMv2: the
  decoder carries the gain).
- **Set/query decoder (OPUS, DETR-style) is trending over two-head** for variable
  support; two-head is a fine baseline but **guard the occupancy shortcut** (NOMAE:
  predict occupancy on a neighborhood incl. plausible-empty, not only survivors).

### Fusion (architecture)
- **Shared-coordinate EARLY fusion transformer** over (xyz,t)-indexed tokens from
  both modalities (PiMAE / 4M / BEVFusion) + **Perceiver-IO cross-stream**
  (FASERCal). Early ≥ late at scale (Apple); MoE modality experts help. Trained on
  sim truth-pairs (free pairing).

### The SOTA-grounded architecture (net)
`asinh → per-band patchify + band-specialized norm/embed → axial-RoPE(time,wire) +
learned(scale,plane) → [local: full FlashAttention or physical-group attn] →
[ToMe reduce] → [global/fusion: latent-bottleneck (FLARE/Perceiver-IO/BiXT)] →
asymmetric decoder with DISTRIBUTIONAL head`, MoE FFNs for parameter scale. Full
fidelity kept in local+decoder; lossy bottleneck only at the global/fusion tier.

## Net revisions to the plan (ranked)
1. **Distributional reconstruction head (Gaussian/MDN NLL)** instead of L2 — the
   fluctuation requirement demands it; validate variance vs noise floor.
2. **Evaluation backbone**: RankMe/LiDAR/α-ReQ + label-efficiency curve; never MSE.
3. **Band-specialized norms/embeddings** (approx vs detail) in the tokenizer.
4. **Masking**: calibrate ratio to sparse-set redundancy; whole-plane = minority
   mode, never whole-band; break symmetry in the prediction head; guard leakage.
5. **RoPE**: axial on (time, wire); learned embeddings for scale/plane (not RoPE).
6. **Decode**: guard the occupancy shortcut; strong decoder, no VQ.
7. **Fusion**: shared-coordinate early fusion, trained on sim truth-pairs.
8. **Scaling**: curate/diversify, ~4 epochs; FM justified by label-efficiency.
