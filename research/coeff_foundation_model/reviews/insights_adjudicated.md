# Insights adjudicated against our measurements — the filtered ledger
Skeptic pass over all literature/SIT/expert insights, cross-checked vs DECISIONS.md
(our ground truth). Verdict: SIMPLEST_BASELINE is already the right object; the
literature's NET contribution after filtering = **one process fix + one head guard.**
Everything else is already-in-baseline, a contingent A/B, or a regime-mismatched drop.

## WHAT SURVIVES as actual design changes (only 2)
1. **Evaluation backbone (highest-value, the one real gap).** All our metrics are
   recon-MSE → blind to representation quality (can't tell a good FM from a good
   denoiser). ADOPT: frozen-backbone **label-efficiency curve vs
   supervised-from-scratch** + frozen **linear probe on sim truth (tpc_de,
   pe_counts)** as a secondary adoption gate. (RankMe/LiDAR/α-ReQ = TEST-only:
   validate against the probe on existing D-17 checkpoints — minutes — before
   trusting; RankMe may be inflated by our noise-dominated fine bands.)
2. **Occupancy-shortcut guard + mask-symmetry break in the decoder head.** Real
   failure modes for our two-head decoder; cheap to honor (NOMAE: train support
   against plausible-empty candidates, not only survivors; break mask-token
   symmetry so identical mask tokens give distinct outputs).

## CONTINGENT A/Bs (NOT baseline commitments; each gated on a cheap probe; our
## measurements predict most will NOT pass)
- **NLL/distributional head** → gated on the variance probe. Our data demotes the
  "L2 washes out fluctuations" claim (see Conflict 1).
- **Band-typed norms/FiLM** → gated on a ≤3% AE delta. D-08 says it won't clear
  the bar (see Conflict 2). At most band-embedding + FiLM; never separate QKV/MLP.
- **Epipolar/geometry-bias fusion** → gated on the synthetic line-correspondence toy.
- **C0 tree-op** → gated on the TPC RoPE-subsumption check (the one place it was
  load-bearing, −15% D-12); verify time-RoPE recovers it before dropping for good.

## ALREADY IN BASELINE (literature merely confirms — no change)
masked clean-coeff reconstruction; reject JEPA/DINO/VQ as primary; axial
RoPE(time,wire) + learned(scale,plane); per-band patchify + linear embed
(D-16/17); paired-sim fusion via shared (xyz,t) coordinate; mask-on-active
~40–50% + whole-plane minority mode; curate-don't-collect scaling (580k fine);
full flash attention (no latent bottleneck / no ToMe at our N).

## DROPPED OUTRIGHT (regime-mismatched: generation-vs-SSL, images-vs-sparse-physics,
## internet-scale-vs-sim, or contradicted by our cost measurements)
scale-causal attention (generation affordance; RoPE already auto-aligns
cross-level); Haar-over-coif3 (no cross-band patch leakage in per-band native-grid
patchify; coif3 is the validated production transform); latent bottleneck
Perceiver/FLARE (30k attn = 6 ms; lossy → anti-aligned with our rate-limited fine
bands); ToMe merge (cost-motivated, no measured wall); MoE FFNs (no measured
capacity wall; we're optimization-bound, D-15 shows huge decode headroom);
diffusion decoder (generation machinery); adversarial/perceptual losses (image-
specific); VQ codebooks (continuous wins, MPMv2 +34pts; quantization kills
fluctuations); serialization/space-filling curves (throughput hack for 10^5–10^6
points; our N is fine); asymmetric small-encoder/large-decoder (generation recipe;
MAE wants the opposite); separate per-band QKV/MLP (D-08 ~2.6%).

## The two explicit LITERATURE-vs-MEASUREMENT conflicts (measurements win)
**Conflict 1 — NLL head (lit synthesis's "revision #1") vs our data.** The "L2
washes out high-frequency fluctuations" mechanism is an image-PIXEL result. In our
POST-THRESHOLD coefficient space the low-ρ/noise content is **thresholded away
before tokenization** (D1 dropped, sub-threshold removed); our L2 denoising AE
already goes SUB-classical on the informative bands (A10 0.0806 < classical 0.0852,
D-14); and the bands where L2 leaves residual (D2/D3) are **PCA-rate-limited**
(D-15, D2 floor 4.68/coeff trained near it) — a bottleneck an NLL head **cannot
fix**. → demoted from "revision #1" to a contingent one-line A/B.

**Conflict 2 — SIT "band-specialized QKV/LN/MLP crucial" vs measured D-08 (2.6%)
/ D-10 (1.6%).** SIT is a VQ generation tokenizer with per-band CODEBOOKS (band
stats acutely matter there); we are continuous-MAE with no codebook to band-match,
and we measured the exact quantity an order of magnitude below the large-effect
bar. SIT's own "separate-transformers-with-cross-attention was WORSE" actually
SUPPORTS our single-shared-sequence design. → adopt at most band-embedding+FiLM.

## Bottom line
The accumulated literature/SIT/expert review, after adjudication against our own
numbers, changes the plan in exactly two places (eval backbone + occupancy/mask-head
guard). The fluctuation-preservation alarm and the band-specialization emphasis —
the two loudest literature recommendations — are both demoted by our measurements
to contingent A/Bs that our data predicts will not pass. SIMPLEST_BASELINE stands.
