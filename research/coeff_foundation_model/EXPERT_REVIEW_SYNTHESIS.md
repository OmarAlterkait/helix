# Expert review synthesis — four CV subfields on OPTIMAL_ARCHITECTURE.md

Four independent expert agents reviewed the trunk design (stages 5-10):
hierarchical/efficient-ViT, multi-view/3D geometry, sparse/point-set, and
SSL/multimodal. Verbatim reviews in `reviews/`. They converge strongly toward
**simpler + measure-before-build**, with one expert (geometry) pushing a
specific, well-motivated ADDITION. Below: convergences (high confidence),
conflicts + resolutions, the revised architecture, and the prioritized cheap
experiments.

## CONVERGENT conclusions (multiple experts, act on these)

1. **Build the FLAT / shallow ViT baseline first; the multi-tier pooling
   hierarchy is premature.** [efficient-ViT + sparse, emphatic] The N≈6d
   "hierarchy mandatory" argument is a COST argument, but every measured
   regime (D-02/08/09/15) is OPTIMIZATION-bound, not cost-bound, and D-17
   already showed **2 plain attention blocks** hit 0.164 TPC (15x below deep
   substrate) with no hierarchy/pool/window. Flat ViT-L (FlashAttention, all
   ~25k tokens, MAE) is affordable in-budget. Make the hierarchy EARN its
   place vs flat on quality-per-GPU-h; don't assume it.

2. **Collapse 3 tiers -> 2 (local + one shared global); delete the separate
   "view tier".** [efficient-ViT + sparse] The view tier is sized against the
   cross-plane MI, which is UNMEASURED. Coord relative-bias already makes a
   single global tier attend view-locally. Run the cross-plane MI audit (the
   trunk's M2 analog) BEFORE committing any view/global depth.

3. **If pooling is needed at all, ToMe (training-free, reversible) over
   learned patch-merge.** [efficient-ViT] Reversible un-merge = exact lossless
   skips (kills the "pool-vs-decode fidelity" open question); content-aware
   merging fits the measured 0.001-18% occupancy spread (pool noise hard,
   keep cores split). But default to NO pooling until a measured wall.

4. **The SSL objective is the riskiest under-specified part.** [SSL, emphatic;
   consistent w/ the fluctuation requirement]
   - "Masked-token denoising (clean-from-noisy)" is **supervised, sim-only**,
     and **cannot transfer to real data** (no clean target). Demote to an
     auxiliary; it's also nearly solved by the linear tokenizer already.
   - **L2-to-clean SHRINKS variance = destroys the per-coefficient
     fluctuations the physics requires** (it's a conditional-mean estimator).
     INSIST on a **distributional / NLL head** (predict per-coeff spread), not
     point regression. This is the cheapest, most verifiable way to honor the
     hard requirement.
   - Spine = **masked-reconstruct-clean-from-visible-noisy** (genuine SSL via
     masking AND uses the free clean target; degrades to standard MAE on real
     data). JEPA auxiliary only (its invariance fights fluctuation-keeping).
     DINO rejected (confirmed).
   - **One masked-prediction head, vary the MASKING POLICY**: random / whole-
     band(scale) / whole-view / whole-modality = denoising + cross-view +
     cross-modality pretexts unified. Mask ratio on ACTIVE tokens, lower
     (~40-50%) start. Modality-asymmetric masking axis (TPC scale, optical
     time) per measured anisotropy.

5. **"Config-flip enables fusion later" is wishful — MAKE PAIRED SIM EVENTS.**
   [SSL emphatic, geometry implies] Cross-modality blocks get NO coupling
   gradient from single-modality data; flipping on later = randomly-init
   interface. BUT the sim produces charge AND light from the same truth -
   the 20k/580k unpaired state is a STORAGE/EXTRACTION choice, not physics.
   Extract paired (TPC,optical) sim events -> the entire unpaired-fusion
   problem dissolves. Highest-leverage fusion fix. Add modality-dropout in
   pretraining regardless (avoid single-modality bias in shared global tier).

6. **Keep the tokenizer + strategic frame as-is.** [all four] Patchify +
   linear embed, two-head decode, dead-bits, unified-token interface,
   per-modality towers + shared top. NO PTv3/Perceiver/serialization (grid IS
   the serialization; Perceiver double-bottlenecks the scarce axis). NO VQ/RVQ
   (quantization discards fluctuations first - anti-aligned with the
   requirement). Stay continuous.

## CONFLICTS and resolutions

A. **Geometry vs the simplify-thrust — coordinate design.** [geometry, deep]
   `[x,y,z]` point coord is a CATEGORY ERROR for tomographic data: a wire is a
   LINE INTEGRAL, not a point; a PMT sees DELOCALIZED light (transport
   kernel), not a point. Proposed: TPC token coord = `(n_theta, w, t_drift,
   vol-sign)` line/hyperplane (Plucker-style); continuous angle NOT U/V/Y
   onehot; cross-volume = signed-drift reflection (free, not summary-only);
   cross-plane correspondence via **epipolar/intersection attention BIAS**
   (two wires from different planes intersect at one transverse point - exact,
   differentiable) not naive coord attention; charge-light via **optical
   transport-kernel bias + time(t0) coincidence as PRIMARY key**, not same-xyz.
   RESOLUTION: not in conflict - different layer. Within-modality depth: build
   flat/simple (conv #1,#3). Cross-VIEW/cross-MODALITY tier: use geometry-
   biased attention (#2), not naive coord attention. The biases are additive
   B_ij terms (zero-able, fusion stays config-flippable). Add a **3D-lift /
   intersection bottleneck** (DUSt3R-pointmap analog) between view and fusion
   as the correct shared substrate, and make cross-view PREDICTION the central
   geometry pretext. DECIDED BY the synthetic line-correspondence toy (below).

B. **JEPA**: spine vs auxiliary. RESOLUTION: auxiliary only (SSL expert
   convincing; JEPA collapses the variation we must keep).

C. **TPC hierarchy necessity at 25k tokens**: efficient-ViT says flash makes
   even flat-global affordable; sparse says 25k is the one place global
   genuinely strains -> test windowed-attention-WITHOUT-pool first. RESOLUTION:
   converges - windowed attention (no learned pool) is the first thing to try
   for TPC; flat-global is the baseline; learned pooling only if a wall.

## REVISED architecture (net of the reviews)

- Stages 0-4 (tokenizer): UNCHANGED, validated. Continuous linear embed,
  per-band/hybrid patches, dead-bits, unified-token interface.
- Within-modality trunk: **flat or shallow** (start 2-4 blocks; windowed-
  long-in-wire for TPC, optional). NO learned pooling v1. Most depth here,
  cheap. ToMe if a token-count wall appears.
- Cross-view + fusion: a **geometry-biased global tier** (epipolar bias for
  cross-plane; transport+time bias for charge-light), shared attention KERNEL
  + per-modality value via FiLM/adapter (NOT full weight-tying - D-08's 2.6%
  was within-modality, don't over-extrapolate). Optional explicit 3D-lift
  bottleneck. Trained on PAIRED SIM events.
- Heads: ONE masked-prediction head (policy = masking geometry) with a
  **distributional/NLL** output; U-Net skips for dense decode.
- Keep C0 tree op TPC-only (D-12 -15%); verify windowed attention recovers it.
  Optical D2: residual BYPASS around the bottleneck (D-15 said it's genuinely
  rate-limited; D-14 said more d_tok barely helps).

## PRIORITIZED cheap experiments (the convergent "do these first")

1. **Fluctuation-preservation probe** [SSL] - on the existing trained
   denoiser: predicted-coeff variance vs true clean conditional variance, per
   band. If shrinking, add Gaussian-NLL head and show recovery. Minutes;
   directly tests the hard requirement. **Do first.**
2. **Paired-sim extraction spike** [SSL+geom] - can pimm emit paired
   (charge,light) sim events? ~1 engineer-day; likely dissolves unpaired-
   fusion.
3. **Synthetic line-correspondence toy** [geometry] - K 3D points -> U/V/Y
   line projections (+noise) -> recover 3D. Arm (a) absolute-xyz + plain
   attention vs (b) (theta,w) + epipolar bias. Include >=3 points (ghost
   ambiguity). Falsifies-or-confirms the entire naive-coord fusion bet. Days.
4. **Cross-plane token MI audit** [efficient-ViT+geom] - the trunk's M2 analog
   (only cathode entry measured). Sets view/global depth. Data-only.
5. **Flat-ViT vs hierarchy bake-off** [efficient-ViT+sparse] - 4-block flat vs
   the pooled stack, matched GPU-h, dense-decode primary. Prior predicts flat
   wins/ties.
6. **Frozen linear-probe evaluation battery** [SSL] - small labeled hold-out
   (1-2k ev, track/shower/vertex), eval-only. The field's coin; its absence
   is the biggest PROCESS gap - can't tell a good FM from a good denoiser.
7. **Masking-triviality probe** [SSL] - random vs whole-band masking at the
   D-17 scale; if random ~= NN-copy baseline, structured masking is mandatory.

## One-line meta
All four converged that the doc OVER-built the trunk (pooling hierarchy) on a
cost argument the project's own data shows non-binding, while UNDER-specifying
the two real risks: the SSL objective (fluctuation preservation) and fusion
supervision (paired data + geometry bias). Simplify the trunk; harden the
objective; make paired sim; bias the fusion with geometry.
