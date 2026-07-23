# Simplest baseline — the build target (supersedes OPTIMAL_ARCHITECTURE trunk)

User steer (2026-06-13) + 4-expert convergence: do NOT over-specify the trunk.
Embeddings + RoPE; the model learns what it needs. Plain full attention, no
geometry bias, no pooling hierarchy. Train as a masked cross-plane
coefficient-level autoencoder. Everything fancier must EARN its place against
this baseline.

## Architecture (end to end)
```
asinh(c / sigma_band)
  -> patchify  (per-band/hybrid; built, validated, lossless at trunk width)
  -> LINEAR patch embed  ([values, occ bits] -> d_model)                # 0.2M params, hits PCA floor
  + learned embeddings:  plane/view ,  modality ,  level/band
  + RoPE(physical_time)   on ALL tokens
  + RoPE(wire_index)      on TPC tokens (within-plane)
  -> N x plain ViT block  (pre-LN, full self-attention over the UNIFIED token set, MLP)
                          # no hierarchy, no windowing, no pooling, no attention bias
  -> LINEAR decode -> coefficient slots  (two-head: occupancy + value)
```

## Positional encoding — the one locked choice
RoPE, applied axially (split head dims):
- **physical_time = (tau + delta_ell) * 2^ell**  (delta_ell = measured coif3 group
  delay). NOT band-local index. This is load-bearing:
  - cross-LEVEL attention auto-aligned in time (the wavelet cone, free; no tree op).
  - cross-PLANE attention gets the same-tick correspondence prior for free
    (U/V/Y correspond at same drift time; cathode coupling measured "direct
    same-tick"). The cheap real geometry, via the RoPE variable, NOT a bias.
  - shared across modalities (one DAQ clock) -> universal axis.
- **wire_index**: TPC only, the dominant spatial axis (r=0.685). Naturally
  within-plane (wire index across plane angles is not a shared physical axis;
  a wire is a LINE not a point -> don't fabricate a shared transverse coord).
  Cross-plane signal rides on time-RoPE + plane embedding + content.
- **optical**: time axis only (1D RoPE); sensor identity = learned embedding.
Categorical identity (plane, modality, level) = learned ADD embeddings. That's all.

## Training objective — masked cross-plane AE, coefficient level
- ALL planes (eventually both modalities) in ONE token set.
- Mask a fraction of tokens; reconstruct the masked tokens' COEFFICIENTS from
  the visible ones. Cross-plane prediction is EMERGENT (reconstruct masked U
  from visible V/Y at the same physical time, reachable via time-RoPE) — not
  hard-coded.
- Masking policy = the knob: random tokens (local) + whole-plane curriculum
  (force correspondence). Mask ratio on ACTIVE tokens.
- Target = CLEAN coefficients (free from sim) -> denoiser+MAE in one; degrades
  to plain MAE (noisy target) on real data -> transfers.
- Decode head: plain regression baseline. Gaussian-NLL head = a ONE-LINE A/B
  (the fluctuation-preservation probe decides if needed), NOT a baseline
  commitment.

## Explicitly DEFERRED (must beat this baseline to be added)
pooling / token-merge (ToMe) ; windowed / hierarchical attention ; epipolar &
optical-transport attention bias ; 3D-lift bottleneck ; JEPA ; NLL head ; VQ ;
the C0 tree op (time-RoPE should subsume its -15% TPC effect — verify).

## Fusion
Just put both modalities' tokens in one set with a modality embedding; shared
time-RoPE relates them; cross-modal is emergent. Train on PAIRED SIM events
(extract them — the unpaired state is a storage choice). No special fusion
layer in the baseline.

## First experiments (cheap, decide the deferred items)
1. fluctuation-preservation probe (plain vs NLL).
2. flat baseline vs any hierarchy (matched GPU-h) — does flat suffice at 25k TPC tokens.
3. masking policy: random vs whole-plane vs mix; ratio.
4. synthetic line-correspondence toy: does time-RoPE + learned attention
   recover cross-plane correspondence, or is an epipolar bias needed?
5. frozen linear-probe battery (the eval that distinguishes FM from denoiser).
6. paired-sim extraction spike.
