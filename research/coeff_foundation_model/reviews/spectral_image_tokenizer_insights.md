# Spectral Image Tokenizer (Esteves, Suhail, Makadia — Google, ICCV 2025, arXiv:2412.09607)
Thorough read + insights for our wavelet-coefficient FM. Full text:
/lscratch/omara/tmp/sit.txt (downloaded PDF).

## What SIT actually is (so we transfer the right things)
A **VQ tokenizer + autoregressive GENERATOR** for images — NOT an SSL
representation model. It tokenizes the DWT spectrum coarse-to-fine for
autoregressive image generation/upsampling. So the VQ/codebook/AR-generation
parts do NOT transfer to us (we are continuous + MAE/representation). The
ARCHITECTURE choices do transfer, and several are directly validating/sharpening
our decisions. It is also the closest published "transformer on wavelet
coefficients with per-band design" we have.

## The architecture (exact)
1. **DWT → coarse-to-fine spectral tokens.** Approximation (coarsest scale) +
   detail coefficients (finer scales).
2. **Scale-adaptive patches: LARGER patches at coarser→ no, at HIGHER (finer)
   wavelet scales.** "associating tokens to larger patches at higher wavelet
   scales than at lower scales" — exploits that the power spectrum decays with
   frequency, so high-freq detail can be more coarsely tokenized. Net: token
   count is balanced across scales.
3. **ADTransformer (Approximation-Details Transformer) — the headline design:**
   approximation vs detail coefficients "come from quite distinct distributions,
   so it makes sense to treat them differently. Thus the parameters of the key,
   query, and value embeddings, the layer norms, and the MLP on each transformer
   layer are NOT shared between the approximation and details coefficients" —
   AND separate VQ codebooks per type. Crucially: **still a SINGLE sequence of
   all coefficients** (one self-attention over everything), just band-typed
   params. "We experimented with different transformers per sequence with
   cross-attention for information sharing, but it performed WORSE."
4. **Scale-Causal attention:** "each token attends to its own scale and lower
   [coarser] scales." Enables: train at one resolution → tokenize any # scales
   up to max; partial decode (first tokens → coarse image); upsampling. Costs a
   little at top res (SIT-SC-5 LPIPS 0.161 vs non-causal SIT-5 0.135) but unlocks
   multi-resolution without retrain/resample.
5. VQ, codebook 8192 each (approx, detail); learnable absolute PE; loss = 1.0 L2
   + 0.1 perceptual + 0.1 adversarial + 0.25 commitment. Asymmetric for
   generation: small encoder (8L) + large decoder (32L, d1280).
6. **Results:** beats ViT-VQGAN reconstruction at every resolution; dramatic at
   LOW res (16×16 LPIPS 0.013 vs 0.127 — coarse scales reconstruct cheaply) and
   faster there. Validates the coarse-to-fine spectral representation.

## Ablation findings worth their weight
- **Bigger vocabulary / longer sequence → better reconstruction but WORSE
  generation** ("did not lead to better generative models"). The recon↔downstream
  tradeoff, observed by multiple works. Reinforces: don't optimize the tokenizer
  for reconstruction fidelity alone.
- **Haar (shortest support) > LeGall5/3 > CDF9/7** for the tokenizer — "worse
  reconstruction the larger the [filter] support… due to increased padding and
  leakage of information between neighboring patches." Long wavelets leak across
  patch boundaries and hurt patch-based tokenization.

## INSIGHTS for us (ranked, with action)
1. **Band-typed parameters in ONE shared sequence — and do NOT build explicit
   cross-band operators.** SIT's ADTransformer = separate KQV/LN/MLP per
   approx-vs-detail, single self-attention sequence, and their *separate-
   transformers-with-cross-attention variant was worse*. This (a) validates our
   band-specialized-norms plan and extends it to QKV/MLP, and (b) is independent
   evidence for our measured result that an explicit cross-band/tree operator
   buys little — **put all bands in one sequence with band-typed params and let
   self-attention do the cross-scale mixing.** ACTION: implement per-band-type
   (or per-band) LN+MLP+QKV in the tokenizer/early blocks; drop any explicit
   cross-band machinery beyond attention. (Cheap; high-confidence transfer.)
2. **Scale-Causal attention = a physically-meaningful, order-free-ish structured
   mask we can actually use.** Unlike a 1D spatial window (arbitrary — rejected),
   the SCALE axis IS a meaningful ordering (coarse→fine = the cone of influence).
   A mask where fine tokens attend to their own + coarser scales is legitimate,
   matches the cone, AND gives multi-resolution + partial decoding for free
   (useful for variable detector configs / dropped fine bands). ACTION: consider
   scale-causal masking as a structured-attention option (test vs full attention;
   small top-res cost, multiscale benefit). Note: it's directional (fine←coarse),
   which also matches "coarse conditions fine" — the productive SSL ordering.
3. **Wavelet filter support matters for tokenization: Haar best, long filters
   leak across patches.** WE USE coif3 (long, ~18-tap support) — chosen for
   DENOISING fidelity, not tokenization. SIT says long support hurts patch-based
   tokenization via inter-patch leakage. TENSION: coif3 is our validated
   compression/denoising choice; a shorter wavelet (Haar/db2) might tokenize
   better. ACTION: cheap A/B — tokenizer reconstruction with coif3 vs db2/Haar
   (the production transform can stay coif3; this only questions the FM's input
   wavelet). Worth checking since our per-band patchify is exactly the
   patch-boundary-leakage regime they flag.
4. **Scale-adaptive patches (balanced tokens across scales): validated.** We
   already do per-band patches in native grids (+ the optical hybrid). SIT
   confirms larger footprints at finer scales is the right move; our hybrid
   (column-coarse + per-band-fine) is a reasonable instance.
5. **Recon↔downstream tradeoff: bigger vocab/seq helps recon, hurts the
   generative/downstream model.** For us (continuous, MAE) it reinforces the
   SOTA-review point: don't over-tune the tokenizer for reconstruction; the
   downstream (probe/label-efficiency) is the real metric.
6. **Coarse-to-fine ("literal spectral autoregression") is the productive
   ordering.** Supports our coarse→fine masking SSL idea (mask fine bands,
   predict from coarse) as a principled, validated direction.

## What does NOT transfer
- VQ / discrete codebooks (we keep continuous; MPMv2 + our floors say no VQ).
- Adversarial/perceptual losses (image-specific).
- Asymmetric small-encoder/large-decoder (that's for generation; MAE wants the
  opposite — heavy encoder, light decoder).
- Absolute learnable PE (we use RoPE on physical coords — better for our
  continuous, variable-geometry data).

## Net: 3 concrete things to adopt/test from SIT
A. **Band-typed params, single sequence, no explicit cross-band op** (adopt —
   validates + sharpens band-specialization; independent support for "let
   attention do cross-scale").
B. **Scale-causal attention** as a structured (physical, not arbitrary) mask
   option giving multi-resolution + partial decode (test).
C. **Wavelet support A/B (coif3 vs db2/Haar) for the FM tokenizer** — long
   support may leak across patches (cheap test; production stays coif3).
