# Full architecture + variation axes (consolidated, post-adjudication)
Integrates the measured decisions (DECISIONS.md), the adjudicated literature
(reviews/insights_adjudicated.md), and the response-conditioning design. Each
stage marks: [SETTLED] (measured) · [DEFAULT] (recommended, untested) · [A/B]
(contingent, gated on a cheap probe) · [VAR] (the variation axis / design space).

## CONDITIONING (the response question) — variation space
A patch = a local time–frequency tile of a wire-group's charge seen *through a
response chain* (field response × electronics × wire geometry). Conditioning =
the model's handle to represent charge rather than response.

Three independent axes:
- **MECHANISM (expressivity ladder, cheapest→richest):**
  additive embedding  ⊂  FiLM (γ,β affine)  ⊂  LoRA/adapter (low-rank per-group
  delta)  ⊂  separate per-group weights [DROP — D-08 measured ~2.6%].
  Orthogonal: **input-feature injection** (append per-wire response features to
  each coeff *before* the linear embed — the only way to get TRUE per-wire, not
  per-block); **explicit normalization** (we already do per-(plane,band) σ via
  asinh/σ_b — a partial response-deconvolution).
- **CONDITIONING VARIABLE:** band/scale · plane-type (U/V/Y = bipolar/unipolar) ·
  plane-id / volume · wire length→capacitance · transverse position · (coherent
  64-group — removed) · per-wire calibration (sim→real only).
- **PLACEMENT:** at embed only · at every block · decoder only.

Recommended: **one response-FiLM at embed** with
`c = [band_emb, plane_emb, MLP(wire_len, transv_pos, volume)]`.
- band FiLM [A/B] — ~2.6% measured, cheap, keep but don't expect much.
- **plane-type FiLM [A/B, test FIRST]** — physically the largest (bipolar vs
  unipolar); unmeasured; most likely to clear the bar band conditioning didn't.
- wire-block FiLM on (length,pos) [DEFAULT, ~free] — the smooth per-wire response.
- per-wire calibration [DEFER] — inject pre-patchify or a learned per-wire table,
  sim→real fine-tuning only (sim has no per-channel constants).

## THE ENTIRE ARCHITECTURE (end to end)

### 0. Data / forward  [SETTLED]
pimm extract → densify → on-the-fly GPU noise (TPC coherent+intrinsic / optical
white σ=2.6) → digitize. Clean truth kept as target. [VAR: on-the-fly vs
extract-once cache — extract-once wins for ≥3 epochs, ~22 GPU-h/580k, ~0.9 TB.]

### 1. Wavelet + threshold  [SETTLED]
coif3 DWT (TPC L4 / optical L10) → bands; TPC smart removal (kgate=4); per-band-σ
threshold; A kept, D1 dropped. [VAR/A-B: coif3 vs db2/Haar for the FM tokenizer —
SIT flagged long-support leakage, but per-band native-grid patchify has no
cross-band patch leakage → expected ~0; low-priority test. Production stays coif3.]

### 2. Normalize  [SETTLED]
asinh(c / σ_band) — signed, compressive, per-(plane,band) σ (partial response norm).

### 3. Patchify  [SETTLED]
Per-band patches in native grid. TPC: 2D 16 wires × {8|16} band-ticks (occupied
only). Optical: hybrid (column cells A10–D4 + per-band P=64 for D3/D2).
[VAR: patch size (token-count ↔ slots/token; bigger needs d ≥ max-active for
lossless), 2D vs 1D, hybrid vs pure per-band. Lossless guaranteed at trunk width.]

### 4. Token embed + conditioning + PE  [DEFAULT + A/B]
- **Linear embed** of [values, occupancy bits] → d_model  [SETTLED — beats deep
  encoder, reaches PCA floor].
- **Response conditioning** (above): band/plane/wire FiLM  [A/B, plane-first].
- **Positional**: axial RoPE on (physical_time=(τ+δ_ℓ)·2^ℓ, wire) [TPC] /
  RoPE(physical_time) [optical]; **learned embeddings for scale & plane** (not
  RoPE — non-metric)  [SETTLED]. [VAR: STRING for wire×time coupling — defer.]
- Dead channels: per-wire dead-bit + wire-kill augmentation  [SETTLED].

### 5. Trunk  [DEFAULT, the main open design]
**Default: plain full-attention ViT blocks** over the whole event's token set
(cross-plane by construction), permutation-equivariant, single shared sequence,
flash attention (30k tokens ≈ 6 ms — cheap). NO windowing (no 1D order), NO
explicit cross-band/tree op (attention + physical-time RoPE do cross-scale).
- depth/width [VAR]: ViT-S/B/L; ViT-L ~232 ms/event at 75% mask.
- [A/B] band-typed FiLM per block (vs embed-only).
- [VAR, scale-only] hierarchical (local full-attn → reduce → global) OR
  latent-bottleneck (FLARE/Perceiver) at the GLOBAL/FUSION tier — only if token
  count balloons; lossy, so keep full fidelity in local+decoder. DROP for baseline.
- [DROP] MoE, ToMe, scale-causal mask, serialization — no measured need.

### 6. Objective / masking  [SETTLED direction + A/B head]
**Masked cross-plane coefficient autoencoder**: all planes (eventually both
modalities) in one token set; mask a fraction; reconstruct masked tokens' CLEAN
coefficients from visible.
- encoder sees VISIBLE tokens only (MAE-drop, ~3× cheaper); mask tokens →
  decoder  [DEFAULT].
- mask ratio on ACTIVE tokens, ~40–50% start; whole-plane = minority curriculum
  mode; never whole-band default  [DEFAULT, calibrate].
- break mask-token symmetry in the prediction head; guard occupancy shortcut
  (train support vs plausible-empty)  [KEEP — the head guard].
- target = clean coeffs (sim); degrades to plain MAE (noisy target) on real data.
- [A/B] distributional (Gaussian/MDN NLL) value head vs plain L2 — gated on the
  variance probe; our data predicts L2 suffices (low-ρ content already
  thresholded away; residual bands are rate-limited, not loss-limited).

### 7. Decoder  [DEFAULT]
Asymmetric, lighter than encoder; two-head (occupancy logit + value) → coefficient
slots; linear/shallow (a linear decode already reaches the floor). [VAR: depth
(~8 if frozen-feature eval); set/query (DETR/OPUS) decoder if variable-support
cardinality hurts two-head; distributional head per #6.]

### 8. Fusion (future tier)  [DEFAULT direction]
Both modalities' tokens in ONE set with modality embedding + shared (xyz,t)
coordinate; cross-modal masking (predict optical from TPC and vice versa). Train
on PAIRED SIM events (sim gives truth-pairing free). Early fusion ≥ late.
[VAR: early token-level fusion vs late contrastive (ImageBind anchor) for the
real-data-transfer fallback; latent-bottleneck cross-stream if token count large;
epipolar/transport geometry bias — DEFER, gated on the line-correspondence toy.]

### 9. Evaluation  [ADOPT — the one process change]
Frozen-backbone **label-efficiency curve vs supervised-from-scratch** (headline)
+ frozen **linear probe on sim truth** (tpc_de, pe_counts) as secondary adoption
gate. NEVER recon-MSE as the representation metric. [A/B: RankMe/LiDAR/α-ReQ
label-free selectors — validate vs probe on existing checkpoints first.]

## The pipeline in one line
`forward(noise→coif3→smart→threshold) → per-band patchify → linear embed +
response-FiLM(band,plane,wire) + axial-RoPE(time,wire)+learned(scale,plane) →
[MAE-drop visible] → plain full-attention ViT (cross-plane) → asymmetric two-head
decoder (clean-coeff recon) ; fusion = both modalities one coord-keyed set on sim
pairs ; eval = label-efficiency curve + probe.`

## The live variation axes (what's genuinely open, ranked)
1. **Response conditioning**: mechanism (emb vs FiLM vs adapter) × variable
   (plane-type first) × placement. [A/B, plane-first]
2. **Trunk depth/width** (ViT-S→L) and whether a hierarchical/latent tier is ever
   needed (only at scale). [VAR]
3. **Masking** ratio/geometry. [DEFAULT, calibrate]
4. **Distributional vs L2 head**. [A/B, predicted L2 suffices]
5. **Patch size / hybrid vs per-band**. [VAR, lossless-bounded]
6. **Wavelet for the FM tokenizer** (coif3 vs shorter). [low-priority A/B]
7. **Fusion**: early vs late, geometry bias. [DEFER to fusion phase]
Everything else is SETTLED or DROPPED.
