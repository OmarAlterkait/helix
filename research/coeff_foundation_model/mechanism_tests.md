# Mechanism Tests — weight structure across levels & cross-scale information travel

Response to two challenged assumptions in `blocks_and_assumptions.md` /
`cross_level_operator_analysis.md`:

1. **"Shared weights + FiLM"** was a premature point-choice. What do levels
   actually *represent*, and what is the space of weight-structure variants?
2. **"Adjacent-only (V-cycle) cross-level coupling"** — connectivity in 2
   sweeps is graph-theory, not fidelity. Information crossing many hops
   attenuates; the measured non-Markov structure says distant levels carry
   unique information. What is the topology design space and how do we measure
   what's needed?

Principle (user-set): **no hard choices — enumerate variants, build the
cheapest experiments that discriminate, pre-register decision rules.**
κ operating points are settled (designed to remove noise while maximally
keeping signal) — not a swept variable.

---

## 1. What the levels actually represent (why sharing is a hypothesis)

Band D_j = the signal's content in frequency octave [fs/2^{j+1}, fs/2^j],
time-localized at 2^j ticks. Walking the depth, the *statistical task* morphs:

| regime | bands (optical / TPC) | content | survival [MEASURED] | implied function |
|---|---|---|--:|---|
| coarse | D10–D7 / A4,D4 | resolved pulse/track envelope, amplitude & shape above the response width | 36–24% | dense regression / shape modeling |
| mid | D6–D4 / D3 | the detector response's internal structure (rise/fall, timing shape); the count budget peaks here | 21–10% | mixed: structure + noise separation |
| fine | D3–D1 / D2,D1 | sharp edges, timing, sub-noise singles | 4.6%→0.05% | rare-event detection against noise |
| A | A10 / A4 | local mean, integrated charge; unipolar; different units | kept / thresholded | baseline & total-energy carrier |

Two forces pull in opposite directions:

**For sharing:** (a) the *underlying physics* is approximately self-similar
(dE/dx fluctuations, photon arrival statistics) — constant kernels give
scale-equivariant physical receptive fields for free; (b) **gradient pooling**
— fine bands have few survivors/event (optical D1: 9/chunk); private weights
starve while shared weights see every level's statistics; (c) cross-modality
transfer requires some sharing.

**Against sharing:** (a) the task morphs (regression → detection) — that's a
*function* change, not a modulation; (b) **the detector response pins a fixed
physical scale** that breaks self-similarity (TPC field response ~ticks → D1/D2;
optical SER 5–10 µs); (c) the measured statistics gradients are steep
(survival 36%→0.05%, lift 2×→57×, p99 14k→46) and asinh normalizes first
moments only.

**The conditioning variable matters as much as the capacity.** Level *index*
does not transfer across modalities; **physical scale** (log2 of coefficient
spacing in seconds) does — and the two modalities tile an almost continuous
physical-scale axis: optical bands span 2 ns–1 µs (D1–D10 at 1 ns ticks),
TPC bands span 1–8 µs (D1–A4 at 0.5 µs ticks), meeting near ~1 µs
(optical D10 ≈ TPC D1 spacing). Under physical-scale conditioning the joint
model sees one scale continuum with the modality/response embedding carrying
what actually differs (response function). Bonus probe: the two optical
datasets have different SER kernels (10 µs in light_output, 5 µs in the
doraemon set) — a controlled response-conditioning test with everything else
fixed.

### The weight-structure ladder (variants to test, capacity-ordered)

| # | variant | params/level | transfers across depth? |
|---|---|---|---|
| W0 | fully shared (pure scale-recurrence) | 0 extra | yes |
| W1 | shared + per-level bias | +d | index-bound |
| W2 | shared + FiLM (channel affine from scale embedding) | +2d | yes if scale-conditioned |
| W3 | shared + per-level LoRA (rank-r delta) | +2dr | partial |
| W4 | hypernetwork: weights = g(physical-scale embedding) | amortized | **yes, continuous in scale** |
| W5 | mixture-of-experts over levels (K small experts, soft routing on scale) | K× | learns regime boundaries |
| W6 | grouped sharing: private weights per measured regime {coarse, mid, fine, A} | 4× | group-bound |
| W7 | fully per-level | L× | no (ceiling) |

W4 and W5 are the interesting middle: W4 makes scale a *continuous* input
(the only variant where a 4-level and an 11-level model are literally the
same function), W5 lets the data discover the regime boundaries that W6
hard-codes from our measurements.

---

## 2. Cross-scale topology (the many-hops problem, taken seriously)

The V-cycle gives **connectivity** in 2 sweeps but each hop is a lossy learned
map. Two distinct failure modes at depth L=10:

- **Attenuation:** with shared edge weights the scale axis is literally a
  depth-10 RNN; a contractive edge-MLP (spectral norm < 1) attenuates
  exponentially in hops. Residuals+LN mitigate, don't eliminate.
- **Oversquashing:** the up-sweep compresses 2^k descendants into d_s dims at
  depth k — the classic tree-GNN bottleneck.

And the measured **non-Markov residual** (grandparent lift 3–21× given parent
inactive) proves distant levels carry information the adjacent hop doesn't
fully relay *in activity space* — whether learned states relay it is exactly
the open question. At TPC depth (L=4) none of this bites; **this is an
optical-depth question.**

**A structural asymmetry worth exploiting:** aggregation (fine→coarse) *must*
summarize — there is physically more content below than fits in one node; a
hierarchical up-path is matched to the token bottleneck anyway. Broadcast
(coarse→fine) has no such excuse — every node has exactly **one ancestor per
level** (≤L of them), so direct attention over the ancestor chain is O(N·L)
and removes hop attenuation entirely in the down direction.

### The topology menu

| # | topology | edges | hops coarse↔fine | cost | notes |
|---|---|---|--:|---|---|
| C0 | adjacent V-cycle (1–2 rounds) | parent only | L | O(N) | baseline |
| C1 | dilated scale edges Δ∈{1,2,4,8} | log-links | log L | O(N log L) | WaveNet-on-scale; matches the measured gradual Δ-decay |
| C2 | **hybrid: hierarchical up + direct ancestor-attention down** | parent up; chain down | 1 (down), L (up) | O(N·L) | kills attenuation where it bites; up stays summarizing |
| C3 | scale-axial attention at anchor cells: pool each band within each coarse cell → L slots → L×L attention → scatter back (residual) | per-cell | 1 both ways | O(cells·L²·d) | price = within-cell pooling of fine bands |
| C4 | full attention over all active coeffs of a chunk/column (**oracle, toy scale only**) | all pairs | 1 | O(N²) | ~1k coeffs/chunk → feasible as the ceiling to judge C0–C3 against |

All are index-arithmetic gathers (no KNN/serialization); all fit the same
packed-tensor implementation; C0–C2 differ only in the index maps.

---

## 3. The test battery

Ordered: data-only (no training) → tiny trained probes (hours, 1 GPU) →
per-chunk/per-plane AEs (the §12 instrument). Every test names its decision
rule. Clean targets exist for BOTH modalities now (TPC: stored sensor = clean;
optical: doraemon set is noise-free, noise added at load).

### Group A — weight structure across levels

- **T-A1. Band-statistics regimes** [DONE — the measured tables]. Defines the
  candidate grouping for W6: {coarse, mid, fine, A}.
- **T-A2. Per-band mini-model filter similarity.** Train tiny independent
  per-band denoisers/AEs (1–2 conv layers, clean targets), compare learned
  filters across bands via CKA / aligned cosine. *Decision:* high cross-band
  similarity (within regime / across regimes) ⇒ sharing viable at that
  granularity. Cost: hours, 1 GPU.
- **T-A3. Gradient-conflict matrix.** One shared tiny model, loss decomposed
  per band; pairwise gradient cosine per layer over training. *Decision:*
  persistent negative blocks ⇒ those level groups need private capacity
  (which W3/W5/W6 grant); diffuse mild conflict ⇒ W2 suffices.
- **T-A4. Conditioning-capacity ladder.** Tiny AE (per-chunk optical, dense;
  per-plane-crop TPC, sparse), sweep W0→W7 parameter-matched where possible.
  Metric: per-band recon + L2 signal-space. *Decision (pre-registered):*
  adopt the smallest capacity within ε of the W7 ceiling, per modality; if
  the winner differs between modalities, W4/W5 (scale-continuous) become the
  unification candidates.
- **T-A5. Physical-scale vs level-index conditioning + transfer.** Train with
  each conditioning variable on one modality; evaluate frozen on the other
  (and across the two optical SER kernels, 5 µs vs 10 µs). *Decision:*
  physical-scale conditioning must beat level-index on transfer to justify
  the unified-axis story; the SER pair isolates response-conditioning.

### Group B — information travel across scale

- **T-B1. Ancestor-ablation predictability curves** (data-only). For each
  level j: fit small predictors of child activity AND asinh-value from
  (i) within-band neighbors only, (ii) + ancestor at Δ=1, (iii) +Δ=2, … full
  chain. Plot incremental gain vs Δ. Extends the measured pairwise lift to
  the actual design quantity: *incremental information given closer context*.
  Run on the dumped events; extend over the 20k-event doraemon set. *Reads
  on:* how far direct edges must reach (C1/C2 justification); whether
  activity non-Markovness survives conditioning on values.
- **T-B2. Descendant-truncation curves** (data-only, up direction). Predict
  coarse-cell summary content from descendants truncated at depth Δ.
  *Reads on:* how much the up-sweep summarization can be localized.
- **T-B3. Topology shoot-out vs oracle.** Per-chunk AE at matched params:
  C0×{1,2 rounds}, C1, C2, C3 vs the C4 full-attention ceiling. Metric:
  per-band recon, worst-band recon, L2. *Decision:* adopt the cheapest
  topology within ε of C4; report the gap per band (fine bands will expose
  attenuation first).
- **T-B4. Gradient-reach (oversquashing) curves.** In each trained T-B3
  model: ‖∂(band-j output)/∂(band-k input)‖ vs |j−k|. *Reads on:* attenuation
  directly — distinguishes "topology can't carry it" from "task doesn't need
  it" when combined with T-B1.
- **T-B5. d_s capacity sweep.** Fixed topology (C0), sweep node width — does
  multi-hop fidelity scale with state capacity as the non-Markov/HMT picture
  predicts? *Decision:* if recon of fine bands saturates only at large d_s,
  prefer C2/C3 over widening.
- **T-B6. Scale-telegraph probe** (synthetic). Micro-task that *requires*
  fine↔coarse communication (e.g., flag coarse cells whose fine descendants
  contain an injected anomaly, and the converse). Sharpest isolated read on
  hop fidelity per topology; immune to "the AE didn't need long range anyway."
- **T-B7. Cost microbench, all topologies.** Extend the queued tree-op bench
  (open #1) to C0/C1/C2/C3 index maps + batched stems (open #2), on both
  dumped events. *Output:* ms and bytes per event per topology — the
  cost axis of the T-B3 decision.

### Group C — tokenizer / bottleneck (sharpened from before)

- **T-C1. Optical anchor-level sweep** (data-only recompute): occupancy,
  fan-out, saturation at cell = 2^10/2^9/2^8 ticks; joint token budget vs
  per-token load curve.
- **T-C2. N×d AE sweep stratified by core saturation** (unchanged §12 plan,
  now with anchor level as a third axis).
- **T-C3. Pooling operator:** strided conv vs attention pooling, judged on
  saturated cores specifically.

### Group D — objective probes (small scale)

- **T-D1. Masking-geometry comparison:** random-token vs subtree vs
  whole-band (coarse→fine scale-prediction) masking on a ViT-S; linear-probe
  + per-band recon read-out. The measured conditionals (P(child|parent)
  0.65→0.03) predict the difficulty ordering — verify.
- **T-D2. Fluctuation retention probe:** does masked-latent training preserve
  per-coefficient variance (regress held-out coefficient values from frozen
  features)?

### Priority & dependencies

1. **T-B1/T-B2** (data-only, this week's class of effort) — they parameterize
   everything: how far edges must reach, how local summarization can be.
2. **T-B7** cost microbench (already queued; extend to topologies).
3. **T-A2/T-A3** (tiny trained probes) → choose ladder rungs worth running.
4. **T-A4 + T-B3/T-B4** (the central tiny-AE factorial: weight-structure ×
   topology; run as one experiment grid, not two).
5. **T-A5** transfer + **T-C1** anchor sweep alongside.
6. Group C2/C3, Group D after the operator is pinned.

---

## 4. Data resources (updated)

| corpus | events | size | noise | schema | truth |
|---|--:|--:|---|---|---|
| TPC `JAXTPC_Wire/test_00_00_02` | 19,999 | 5.9 TB | clean (forward noise at load) | sparse COO sensor | clean waveform |
| optical `light_output.h5` | 100 | — | real noise 2.6 ADC baked in | east/west chunks | pe_counts |
| **optical doraemon `/sdf/data/neutrino/doraemon/optical_test_00_00_02/`** | **20,000** (100 files × 200) | 210 GB | **noise-free** (`baseline_noise_std=0`; add at load) | **label_N** groups (helix.optical.io does NOT read; needs small loader) | per-interaction `tpc_de/pdg/positions`, `pe_counts` |

Doraemon optical details [MEASURED, file 0]: 164 channels, pmt_id 0–81 per
group, SER 5 µs (vs 10 µs in light_output), pedestal/gain/tick identical,
~1.4k–3.4k chunks/event (per-label splitting; same PMT recurs across labels),
chunk lengths 5k–36k (median ~14.8k). Same sim family as the TPC set
(test_00_00_02; different run number) — **event pairing with the TPC corpus is
plausible but unverified.** Consequences: (a) optical clean targets now exist →
supervised denoise/AE targets for optical, symmetric with TPC; (b) the optical
corpus-existence risk is retired at prototype scale; (c) tree statistics
should be re-measured on this set once a label_N loader + forward optical
noise model are in place (the light_output measurements stand for the
noisy-data regime).
