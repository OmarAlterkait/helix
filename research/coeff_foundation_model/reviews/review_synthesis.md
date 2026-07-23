# Review synthesis — skeptic + simplifier passes over mechanism_tests.md

Two independent expert reviews of the plan (`skeptic_review.md`,
`simplification_review.md`, both verbatim in this folder). This synthesis
records where they converge, where they conflict (with resolutions), and the
revised minimal plan. The original `mechanism_tests.md` stands as the
reasoning record; **this document supersedes its test battery.**

---

## 1. Where both reviews converge (adopt without debate)

1. **Kill T-A2 (CKA) and T-A3 (gradient conflict) as decision tests.**
   Skeptic: known failure modes (CKA variance-dominance, gradient-cosine noise,
   per-band loss-scale imbalance) can't carry decision weight. Simplifier:
   they are proxies strictly upstream of a sufficient statistic the plan
   already measures (distance-to-ceiling). Resolution: T-A3 becomes free
   telemetry (a logging callback in the star runs); T-A2 is deleted (optional
   no-training substitute: per-band second-order statistics / Wiener argument).

2. **The decision rules were pre-registered in name only.** No ε units, no
   primary metric, no seed/event-fold variance budget, no LR protocol, no
   joint rule for the W×C factorial, ceilings that may not be ceilings.
   Resolution: the star design (below) + per-test registration of
   (primary scalar, ε := k·σ_ceiling over ≥3 seeds × event folds, 3-point LR
   sweep per arm, plateau check).

3. **One ceiling model resolves the ceiling problem structurally.** Full
   attention over a chunk's active coefficients with per-level parameters and
   (physical-time, log-scale) positional encoding is simultaneously the C4
   oracle and the W7 capacity ceiling — and every candidate is this model
   with attention masks + weight ties imposed (the point-set formulation).
   Skeptic's caveat folded in: a connectivity superset is not automatically a
   performance ceiling — define the reference as the **best-of envelope**
   over {full-attn, C2, C3-style} arms; if full-attn < structured, the
   "oracle" framing is void and that ordering is itself reported.

4. **Activity statistics have been doing value-level work.** Lift, the
   non-Markov residual, the shift sweep, occupancy — all thresholded-binary,
   all subject to censoring/common-cause confounds. Resolution: the data-only
   audit (M2) computes value-level twins and converts everything to **bits**.

5. **The doraemon label_N loader is the critical path**, not a footnote.
   M2 (clean-data re-measurement), M3 (clean targets), M4 (transfer) all gate
   on it. First engineering task.

---

## 2. The biggest individual catches (one reviewer each)

**Skeptic:**
- **S2 — the tree-lift common-cause confound.** The optical shift sweep is
  nearly flat (a parent 3 cells away predicts almost as well as the true
  parent: D2 0.152–0.157 over ±3) and the lateral A10*→D10 lift (1.9×) ≈ the
  D10→D9 parent lift (2.1×). Pairwise lift cannot distinguish "tree edge"
  from "shared pulse envelope," and pooled lift is inflated by between-chunk
  heterogeneity (the design-relevant quantity is within-chunk lift). The
  handoff's "tree is real structure, not up for relitigation" is demoted to
  *gated on the M2 controls*: tree-ancestor gain must beat (a) same-band
  neighbor at equal physical distance, (b) shifted non-parent at same level,
  (c) within-chunk stratification.
- **S3 — orphans, the missed implication.** ~27% of active D6 coefficients
  have an *inactive parent but active grandparent* — and the C0 V-cycle
  computes on active sites only, so the relay node doesn't exist; the child
  gets a null embedding. The non-Markov table is really evidence about
  **support holes**, not d_s. New required statistics: orphan fraction per
  band (from existing JSON); new required read-out: recon error on orphaned
  vs parented coefficients per topology (the sharpest C0-vs-C2
  discriminator); new menu variant: instantiate inactive interior tree nodes
  on active cones.
- **S6 — the dense-conv decision was a category error.** 62.5% is *cell-union*
  occupancy; coefficient-level occupancy is ~2.7% (983/36k). Dense may still
  win (small bands, launch-bound gathers) but the choice is un-settled and
  gated on the per-band dense-vs-gather microbench (M1); if dense is kept,
  zero sites must be masked out of losses/normalization.
- **S13 — the "noise chunk" class is contaminated.** |x|max≤50 ADC ≈ 19σ
  admits real light; measured noise-chunk D1 survival is ~300× the Gaussian
  expectation at a 5.3σ threshold. Doraemon (noise-free truth) gives *exact*
  noise chunks — re-derive the class there; re-state the "D1 is
  signal-bearing" conclusion after.
- **S4 — recon alone can't decide; truth labels are unused.** Without the
  production bottleneck pinned, every topology passes coefficients through
  and "within ε of ceiling" is vacuous. AE arms must use the production token
  geometry; and a frozen-feature linear probe on doraemon truth
  (pe_counts / tpc_de) is a pre-registered secondary criterion: a variant may
  not be adopted if it ties on recon but loses >δ on the probe.

**Simplifier:**
- **P5 — convert lifts to bits before arguing.** The D1 non-Markov drama
  (21× residual lift) lives in H(0.0002) ≈ **3 milli-bits/slot**; the
  absolute long-range information is at coarse/mid levels (D8:
  H(0.099) ≈ 0.46 bits/slot) — the opposite end from where the lift table
  draws the eye. Activity-CMI is exact from contingency tables (no training);
  value-CMI via Gaussian-copula / coarse-bin plug-in with conditioning sets
  ≤3. Pre-register edge justification in bits/event.
- **P6 — bit-count the bottlenecks; sweeps become one-point verifications.**
  Global: ~4 Mbit/event source vs ~50 Mbit token capacity — the token stage
  is over-provisioned 10–100×; "fluctuation preservation" is a trainability
  question, not a rate question (retires open #7, invertible lifting,
  permanently). Local: saturated cores p95 ≈ 1.0 kbit → predicted knee
  d ≈ 256–512, residual loss concentrated in the top core-load percentile.
  Up-path: optical anchor-10 p95 fan-out 206 coeffs ≈ 2.5 kbit ≫ d_s=64 —
  **d_s cannot summarize the p95 subtree at anchor 10**; run the anchor
  sweep (T-C1) *first*, size d_s by arithmetic, verify once.
- **P1 — TPC mechanism questions carry ~0 bits at L=4.** All W/C
  discrimination moves to optical chunks (~1k coeffs, dense, clean targets,
  feasible oracle); TPC defaults to per-level weights + adjacent coupling
  with exactly one control (rounds 0 vs 1) and one tokenizer verification.
- **P3/P4 — ladders collapse to endpoints + bisection.** W ladder 8→3
  {W0, W2, W7(=ceiling)}; W4 (hypernet) is transfer-only; W5 cut; W6 is the
  pre-named bisection rung. Topology 5→3 {C0, C2, ceiling}: C1 is dominated
  by C2 at decision scale; C3 inserts the lossy pooling exactly at the
  saturated cells. C1/C3 index maps stay in the cost bench as named
  contingencies.
- **P11 — the point-set formulation.** Everything is a transformer over
  active coefficients in (space, physical-time, log-scale, value); topologies
  are attention masks, weight ladders are tying patterns over the scale
  coordinate, the tokenizer is local pooling. Unification = "one family,
  per-modality masks" — honestly scoped.

---

## 3. Conflicts between the reviews, resolved

| topic | skeptic | simplifier | resolution |
|---|---|---|---|
| Alignment (open #18) | sweep metric insensitive to misalignment by construction; compute coif3 group delay **analytically**; value-level R² vs shift; TPC never swept | close it (within 1–2%, absorbable) | compute analytic δ_ℓ (free, exact) + one value-level R²-vs-shift column in M2 (cheap); then close. TPC shift sweep = few lines in M2. |
| A-band | the 1.9× lateral lift used *thresholded* A10*, not production unthresholded A10; wants an ablation arm | close it (low stakes) | one cheap A-band arm inside M3 (root-partner vs plain-level vs excluded) — near-free, settles it with production semantics |
| W×C interactions | factorial needs a joint decision rule (shared weights ⇒ RNN attenuation ⇒ C2 helps; per-level ⇒ C0 adequate) | star design, no factorial | star + **pre-named contingency**: if any one-factor-off arm lands in the ambiguous band, run the single crossed cell (W2+C2). Adopt cheapest (W,C) cell within ε of best, cost from M1. |
| T-B4/T-B6 (gradient reach / telegraph) | T-B4 underspecified, can't gate anything; T-B6 is the better instrument and gates nothing | both contingent on C0 failing | contingent diagnostics, with the trigger and the joint criterion pre-registered: run iff C0's fine-band or **orphan-stratified** gap > ε |
| T-A5 conditioning | level-index baseline is a straw man; need free per-(modality,level) embedding baseline + held-out-level interpolation AND extrapolation arms | within-modality comparison vacuous (bijection); transfer evals only | merged: transfer-only evaluations of M3 winners, with (a) free-embedding baseline, (b) held-out-level interpolation, (c) held-out-end extrapolation, (d) the SER 5/10 µs pair |

---

## 4. The revised minimal plan (M1–M5, amended)

**M0 (critical path): doraemon label_N loader + forward optical noise model.**
Then re-measure tree statistics on doraemon with pre-registered equivalence
bands vs light_output (per-band survival ±25% rel., occupancy ±10 pts);
pin which dataset is authoritative per decision. Derive exact noise chunks
from truth (zero true PE).

**M1 — cost microbench** (dumped events, no training): all four topology
index maps (benched, not trained), batched stems, **per-band dense vs gather**
(gates the optical dense-conv choice properly).

**M2 — data-only information audit, in bits** (no training): exact
activity-CMI from contingency tables; value-CMI (copula / coarse-bin,
conditioning sets ≤3); **controls:** same-band-neighbor at equal physical
distance, shifted non-parent, within-chunk stratification (the S2 confound);
**orphan fractions per band**; analytic δ_ℓ + value-level R² vs shift (incl.
TPC); optical anchor-level recompute (occupancy/fan-out/saturation at
2^10/2^9/2^8) with the trunk-cost column attached. Event-level bootstrap CIs;
event-split everywhere; average precision (not AUC) for fine bands; censored
vs active-ancestor subpopulations reported separately.

**M3 — one chunk-scale star design on optical** (doraemon, clean targets):
reference = best-of envelope over {full-attn ceiling, C2, C3-style};
arms = {C0+W7, C2+W7, C0+W2, C0+W0} (+ cheap A-band arm; + one quantized-input
column), each at the production bottleneck geometry (anchor from M2's sweep,
d_s from P6 arithmetic ±2×). Registered per arm: primary scalar = per-band
asinh-MSE over a named band set; ε = k·σ_ceiling (≥3 seeds × event folds);
3-point LR sweep; plateau check. **Stratified read-outs:** orphaned vs
parented coefficients; core-saturation percentile. **Secondary criterion:**
frozen linear probe on pe_counts/tpc_de — within-ε-on-recon does not adopt if
probe loss > δ. Contingencies: crossed cell (W2+C2); W-bisection (W6); T-B4 →
T-B6 iff C0 fine-band/orphan gap > ε.

**M4 — transfer evaluations** (no new training): M3 winners evaluated
zero-shot across depth and across the SER pair, under scale- vs
free-embedding conditioning; held-out-level interpolation + held-out-end
extrapolation arms.

**M5 — TPC one-point verification:** pre-attention stack (per-level weights,
C0), rounds 0 vs 1 control, tokenizer at the predicted d, AE stratified by
core saturation, packing/masking equivalence check (packed events must not
attend across event boundaries — correctness, one test).

**Closures adopted (P12, as amended):** invertible lifting (capacity
arithmetic); TPC W/C ladders (P1); drop-D1 (budget; restate the
"signal-bearing" claim after M0's exact noise chunks); cross-volume menu
trimmed; masking-geometry ordering (predicted by measured conditionals —
Group D deferred to trunk phase). Alignment and A-band close *after* their
one-line M2/M3 items, not before.

**Demotions:** handoff §4.2 "tree coupling not up for relitigation" → gated
on M2's controls. blocks_and_assumptions "optical dense conv settled" →
gated on M1. "flat attention suffices for optical" → conditioned on the
anchor outcome ("flat iff anchor stays at level 10").
