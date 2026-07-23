# Simplification review (verbatim agent output)

**Reviewer stance:** the plan is unusually well-measured, but the test battery has grown to ~17 tests × two ladders (8 weight rungs × 5 topologies) because design questions are being treated as independent empirical unknowns when (a) several are already answered by the numbers in `optical_tree_stats.json` once you convert lifts to *bits*, (b) several variants are nested by expressivity and need only endpoints, and (c) the discriminating regime for every remaining mechanism question is optical depth, where experiments are ~50× cheaper than on TPC planes. The proposals below are ranked by leverage; a minimal test set is given at the end.

---

## P1. Move every open mechanism question to optical-chunk scale; declare the TPC mechanism defaults pinned

**Replaces:** the TPC arms of T-A4 and T-B3; the "V-cycle ≈ column ops at L=4" unification gate (cross_level_operator_analysis §8 Tier-1, called "pivotal"); the implied TPC runs of the W and C ladders.

**Simpler alternative:** TPC ships with per-level weights (W7) and adjacent coupling (C0) *by default, untested*. All W/C discrimination runs on optical chunks (~1k coefficients, dense, clean doraemon targets). TPC keeps exactly one mechanism control: rounds 0 vs 1 (does cross-level coupling help at all).

**IT justification:** the expected information gain of an experiment is the prior uncertainty it can resolve. At L=4, the hypothesis class nearly collapses: any topology gives ≤4-hop paths, attenuation over ≤4 residual hops is negligible, and C0/C1/C2 differ by at most 2 hops — so "V-cycle ≈ column at L=4" is the *expected null result*; a test that cannot plausibly fail carries ~0 bits. Likewise weight sharing: at L=4 sharing saves 4× on tiny stem parameters — nothing — and the gradient-pooling argument for sharing is void on TPC because the bands are measured co-equal (A4/D4/D3 within 30%; no starved band). Sharing is load-bearing only at depth 10 (optical D1: 9 survivors/chunk) and for cross-modality transfer. The discriminating signal lives where the statistics gradients are steep (survival 36%→0.05%, lift 2×→57×) — optical. And that is also where experiments are cheapest: one chunk ≈ 1k active coefficients (vs ~50k/plane), dense (62.5% occupancy — no sparse machinery), with a feasible full-attention oracle.

**What is lost:** the possibility that wire-axis × tree interactions on TPC produce a surprise the 1-D optical case can't show; direct TPC evidence for the W7/C0 defaults. Mitigated by the retained TPC rounds-0/1 control and the one-point tokenizer verification (P6) — if those pass, the TPC pre-attention stage is validated end-to-end without ladder runs.

**Feasibility:** immediate. Needs the small doraemon `label_N` loader (already flagged) — that loader is now on the critical path and should be the first engineering item.

---

## P2. One ceiling model: a full-attention, per-level-parameter point transformer at chunk scale, jointly serving as the C4 oracle *and* the W7 ceiling

**Replaces:** the separate W7 ceiling in T-A4, the C4 oracle in T-B3, and the full W×C factorial ("run as one experiment grid" — currently up to 8×5 = 40 cells).

**Simpler alternative:** train **one** ceiling: full attention over all active coefficients of a chunk, per-level parameters, positional encoding = (physical time, log-scale). Every structured candidate (any W rung, any C topology) is literally this model with attention masks and weight ties imposed. Then run a **star design**, not a factorial: ceiling + one-factor-off arms {C0+per-level, C2+per-level, C0+FiLM, C0+fully-shared}, each scored as degradation from the ceiling.

**IT justification:** by the data-processing inequality, masking edges and tying weights can only remove representational channels — the ceiling upper-bounds every cell of the factorial, so every decision of the form "adopt cheapest within ε of ceiling" needs only the candidate's distance to *one* reference, not the full grid. The factorial's off-diagonal cells answer interaction questions nobody's decision rule consumes. The star measures the main effects (topology marginal, sharing marginal) at the ceiling's operating point, which is what the pre-registered rules actually use.

**What is lost:** W×C interaction effects (e.g., "FiLM suffices under C2 but not C0"). If a one-factor-off arm lands in the ambiguous band, add the single crossed cell then — a contingency, not a plan. Also one confound: an arm missing ε doesn't say *which* restriction hurt — the star structure resolves this by construction (each arm changes one factor).

**Feasibility:** trivially cheap — full attention at N≈1k is nothing; the ceiling is arguably the *easiest* model in the plan to implement (no gather/scatter, no index maps).

---

## P3. Collapse the W ladder 8 → 3 rungs: {W0, W2, W7}; everything else is bisection or a transfer-only candidate

**Replaces:** W1, W3, W5, W6 as planned arms; the W-axis of T-A4.

**Simpler alternative:** test W0 (fully shared), W2 (shared+FiLM), W7 (per-level = the P2 ceiling). Pre-register: if W2 within ε of W7 → adopt W2; if W0 within ε → adopt W0; only if W2 fails *and* W7 passes, bisect once (W3 or W6 — and W6's grouping is already fixed by the measured regime table, T-A1, so it needs no design work). W4 (hypernet) is evaluated **only** in the transfer test (P8), never in the capacity ladder. W5 is cut.

**IT justification:** the ladder is not 8 distinct hypotheses; it is one **rate axis** — bits of level-conditional parameterization — and the decision rule ("smallest capacity within ε of ceiling") needs only the envelope of a monotone rate-distortion curve: two endpoints and one cheap midpoint locate the knee; bisection refines it iff ambiguous. On expressivity: W1 ⊂ W2 strictly (bias ⊂ affine); W2, W3, W6 ⊂ W7 strictly; W4 and W5 (with adequate capacity, distinct level embeddings) can *emit* arbitrary per-level weights, so they are expressivity-equivalent to W7 — their distinct content is inductive bias (continuity in scale; learned regime boundaries), not capacity. So as capacity-ladder rungs, W4/W5 measure nothing W7 doesn't; W4's unique claim (a 4-level and 11-level model are the same function) is a *transfer* property, testable only by transfer. One honesty note: FiLM is a *diagonal* weight delta and LoRA a *low-rank* one — neither contains the other, so "FiLM ⊂ LoRA ⊂ hypernet" is a partial order, not a chain; this doesn't change the recommendation, since the decision needs the curve's envelope, not its interior ordering.

**What is lost:** W5's ability to *discover* regime boundaries (its value was interpretive — the measured T-A1 table already supplies the grouping); resolution of the knee to one rung (recovered by at most one bisection).

**Feasibility:** pure deletion; the three retained rungs are the easiest to implement (W7 is the P2 ceiling; W2 is one FiLM layer; W0 is W2 with conditioning removed).

---

## P4. Collapse the topology menu 5 → 3: {C0, C2, ceiling}; C1 and C3 are dominated

**Replaces:** C1 and C3 as arms of T-B3.

**Simpler alternative:** test C0 (adjacent, 1–2 rounds) and C2 (hierarchical up + ancestor-chain attention down) against the P2 ceiling, optical chunks only.

**IT justification:** C1 (dilated scale edges) and C2 answer the same latent question — "do direct long-reach edges beat relayed hops?" — and C2 dominates: it achieves 1-hop reach in the down direction at O(N·L), and at L=10 the cost difference vs C1's O(N log L) is a factor of 3 on an op that is ~nothing at chunk scale. C1's only distinct content is long-reach in the *up* direction, where the doc's own structural argument (aggregation must summarize anyway; 2^Δ fan-in) says direct long edges are the wrong primitive. C3 (pool-then-attend per cell) imposes within-cell pooling of fine bands *before* the cross-scale exchange — i.e., it inserts a lossy bottleneck exactly at the saturated cells the plan elsewhere identifies as the highest-information, most-at-risk regions; its compensating virtue is cost (O(cells·L²)), which is irrelevant at the scale where the question gets decided. A dominated option needs no experiment. Note also the measured shift-sweep flatness (conditionals vary <2% over ±3 shifts) says edge placement is forgiving — the topology choice is about *reach*, a 1-bit question (relay suffices / it doesn't), and {C0, C2, ceiling} brackets it.

**What is lost:** if C2 wins but is too expensive at TPC production scale, C1/C3 were the fallbacks — keep them as named contingencies in the cost benchmark (T-B7 can still microbench their index maps for ~zero extra effort, without training arms).

**Feasibility:** deletion + the note that T-B7 keeps all four index maps (benching is cheap; training arms are what's cut).

---

## P5. Replace T-B1/T-B2 probe training with direct CMI estimation — and convert every existing lift to bits first

**Replaces:** T-B1 (ancestor-ablation predictability curves), T-B2 (descendant-truncation curves), and the practice of arguing from lift ratios.

**Simpler alternative:** (i) Activity side: compute I(child ; ancestor_Δ | parent, …) and I(child ; within-band neighbors | tree context) **exactly** from contingency tables on the existing scans — binary variables, closed form, no training; half the joint counts already exist in `optical_tree_stats.json` (the `markov` table is one marginal short of the full triple joint). (ii) Value side: I(asinh-value_child ; ancestor values | parent value) via Gaussian-copula MI or a 2-bin/4-bin plug-in on the dumped npz files — small conditioning sets, which is exactly the question being asked. Pre-register in bits: an edge of reach Δ is justified iff CMI_Δ × (population mass) exceeds ε bits/event.

**IT justification:** lift is a *pointwise* MI statement (log lift = i(child=1; anc=1)) and systematically over-weights rare events. Concretely: the headline non-Markov residual at D1 — 21× lift given parent inactive — operates on a population where H(child | parent=0) = H(0.000195) ≈ **3 milli-bits per slot**. The entire D1 non-Markov drama is bounded by milli-bits; meanwhile D8 (p_noparent = 0.099 → 0.293 given grandparent) sits inside H(0.099) ≈ 0.46 bits/slot — the *absolute* long-range information lives at coarse/mid levels, the opposite end from where the lift table draws the eye. This single unit conversion will likely shrink the case for long-reach edges to a small, localized correction — and it costs an afternoon of counting, not probe training. T-B1's planned "small predictors" are themselves just plug-in CMI estimators; the simplification is to recognize that and skip the gradient descent for the activity half entirely.

**What is lost:** CMI measures information *existence*, not *learnability by the production operator class* — a real gap when the dependency is high-order. But that gap is exactly what the P2 star design measures; the data-only CMI parameterizes (edge reach, locality of summarization) and the AE arms validate. Also nonparametric value-MI estimators are biased upward in high dimensions — keep conditioning sets ≤3 variables (which matches the actual design question).

**Feasibility:** highest of anything in the plan — extends `measure_coeffs_optical.py` by a few dozen lines; both dumped events plus the 20k-event doraemon set for tight error bars.

---

## P6. Bit-count the bottlenecks: turn T-C2's sweep and T-B5's d_s sweep into one-point verifications of an arithmetic prediction

**Replaces:** the T-C2 N×d×anchor 3-axis sweep (most of it), T-B5 (d_s capacity sweep), and open #7 (invertible lifting) as live questions.

**Simpler alternative:** compute the source rate per token/node and compare to channel capacity; train **one** calibration AE to fix the single unknown (effective bits per latent dimension, empirically ~4–8 for trained nets); then verify the predicted knee ± one factor of 2 instead of sweeping.

The counts, from the measured tables:

- **Global:** TPC typical event ≈ 315k surviving coeffs × ~13 bits (≤10-bit value entropy post-asinh + ~3 bits position-within-footprint) ≈ **4 Mbit/event**. Token budget: 25.4k × d=512 × a conservative 4 effective bits/dim ≈ 50 Mbit. **The token stage is over-provisioned ~10–100× globally.** Fluctuation preservation is *not* rate-limited by any fixed token budget under consideration — the question dissolves from "rate-distortion of the tokenizer" into "does optimization use the capacity," which is a trainability question one AE answers. This retires a whole anxiety thread in Block 4.
- **Local (the real constraint — saturated cores):** TPC 8×4 p95 token load ≈ 81 coeffs ≈ 1.0 kbit; max ≈ 240 coeffs ≈ 3.1 kbit. Predicted knee: d ≈ 256–512 for p95 fidelity; only the <1% max-load cores press d=1024. **Prediction: the T-C2 curve is flat in N and knees in d around 256–512, with all residual loss concentrated in the top core-load percentile.** Test that point, stratified by core load, done.
- **d_s on the up-path (T-B5, answered by the same arithmetic):** optical anchor cell p95 fan-out = 206 coeffs ≈ 2.5 kbit, vs d_s=64 ≈ 0.5 kbit even at 8 bits/dim. **d_s=32–64 cannot make the anchor a sufficient statistic of the p95 subtree for reconstruction** — the sweep's outcome is foreseeable (fine-band recon saturates at d_s ≈ load×bits/8), and the design response is already on the menu: drop the anchor to level 9/8 (T-C1's cheap npz recompute) rather than widen d_s. So run T-C1 first, size d_s by arithmetic, verify once.
- **Invertible lifting (open #7):** a zero-information-loss guarantee on the lift is worthless while a later stage is the binding bottleneck and that bottleneck is over-provisioned — close open #7 permanently rather than carrying it.

**IT justification:** straight source–channel separation. When capacity exceeds the source rate by an order of magnitude, distortion is an optimization artifact, not an information-theoretic necessity — sweeping capacity measures your optimizer, not your design.

**What is lost:** the entropy estimates are order-of-magnitude (conditional value entropy given neighbors is below 10 bits; effective bits/dim is architecture-dependent). Hence one calibration run plus a ±2× bracket rather than zero training. If measured loss wildly exceeds the bit-count prediction, that itself is the most informative possible result (pure optimization failure — different fix than widening d).

**Feasibility:** the arithmetic is a notebook cell on existing JSON; the calibration AE is the first arm of P2's star anyway.

---

## P7. Cut T-A2 (CKA filter similarity) and T-A3 (gradient conflict) as decision tests

**Replaces:** T-A2, T-A3, and their gating role in the priority list ("choose ladder rungs worth running").

**Simpler alternative:** delete T-A2; if a no-training substitute is wanted, compare per-band *second-order statistics* (autocorrelation / cross-spectra of asinh-normalized active neighborhoods) directly — for the tiny 1–2-layer denoisers proposed, optimal filters are functions of signal/noise spectra, so spectra similarity ⇒ filter similarity without training anything (Wiener argument). Demote T-A3 to free telemetry: log per-band gradient cosines during the P2/P3 star runs (a callback, not an experiment).

**IT justification:** the pre-registered decision rule consumes only one quantity — distance-to-ceiling of each sharing rung (P3). T-A2 and T-A3 are upstream *proxies* for that quantity, with known failure modes (CKA is insensitive to functionally important low-variance directions; gradient cosine is noisy and layer-dependent), and the plan already schedules the direct measurement. A proxy that is strictly upstream of an already-planned sufficient statistic adds risk and cost, not decision-relevant bits — by sufficiency, condition on the statistic and the proxy is independent of the decision.

**What is lost:** mechanistic narrative ("*why* sharing fails, and between which bands") — genuinely useful for a paper and for designing the W6 bisection grouping if needed. The telemetry version of T-A3 retains most of this for free; the regime grouping is anyway already fixed by T-A1's measured table.

**Feasibility:** pure deletion + ~20 lines of logging in the star runs.

---

## P8. Shrink T-A5: the within-modality arm is vacuous; the rest reduces to zero-shot evaluations of already-trained models

**Replaces:** T-A5 as a standalone training experiment (train with each conditioning variable on one modality, evaluate frozen on the other).

**Simpler alternative:** note that *within one modality*, level-index and physical-scale are bijectively related (scale = index + log2(tick)); a model conditioned on either is the same function class — so the within-modality comparison cannot distinguish them even in principle. What remains: (i) zero-shot evaluate the optical-trained P3 winner at TPC depth (and vice versa) under each conditioning convention — frozen-model evaluations, no new training; (ii) the SER 5 µs vs 10 µs pair (light_output vs doraemon) as the controlled response-conditioning probe — again an evaluation across two existing datasets, not a new training axis.

**IT justification:** sufficiency under bijection: conditioning variables related by an invertible map induce identical conditional models; only the *transfer* setting, where the map differs between train and test domains, makes them distinguishable. So all of T-A5's information is in the transfer evaluations, which are free once P2/P3's models exist.

**What is lost:** training-dynamics differences between the two parameterizations (real, second-order, and detectable in the P3 runs' training curves if large).

**Feasibility:** evaluation scripts only; gated on the doraemon loader (same dependency as P1).

---

## P9. Defer Group D (T-D1 masking geometry, T-D2 fluctuation retention) out of the design-pinning phase entirely

**Replaces:** T-D1, T-D2 as part of the current battery.

**Simpler alternative:** nothing now. The masking-geometry choice does not gate any encoder-architecture decision (the encoder must handle arbitrary visible subsets regardless), and the measured conditionals already predict the difficulty ordering T-D1 would "verify" (P(child|parent) 0.65 → 0.03 across depth *is* the difficulty curve of scale-prediction; a learnable-but-nontrivial regime is already established by the non-Markov residual). T-D2's question is answered structurally by P6: capacity is over-provisioned, so fluctuation retention is an objective-design question for the trunk phase, where it belongs.

**IT justification:** value-of-information ordering — experiments whose outcome changes no near-term decision have zero VoI now and full VoI later (when the trunk exists and masking is the live knob). Verifying a prediction the measurements already make with high confidence buys ~0 bits.

**What is lost:** early warning if subtree masking is degenerate (trivially easy or impossible). The measured conditional band (0.03–0.65) makes both extremes unlikely.

**Feasibility:** pure deferral.

---

## P10. Make T-B4 (gradient reach) and T-B6 (scale telegraph) contingent diagnostics, not battery members

**Replaces:** their status as pre-registered tests.

**Simpler alternative:** decision tree: run the P2 star; **iff** C0's gap to ceiling exceeds ε on fine bands, run T-B4 (distinguishes "topology can't carry it" from "task didn't need it") and, if still ambiguous, T-B6 (synthetic forced-communication micro-task). If C0 ≈ ceiling, neither ever runs.

**IT justification:** T-B4/T-B6 measure *cause*, not *choice* — they are diagnostic of a failure that, on the priors (gradual measured Δ-decay; residual connections; only the milli-bit-scale fine-level non-Markov residual from P5), is more likely absent than present. Conditioning expensive measurements on the cheap measurement that determines their relevance is the optimal experiment ordering.

**What is lost:** if the gap does appear, total latency increases by one round-trip. Acceptable for hours-scale chunk experiments.

**Feasibility:** a paragraph edit in mechanism_tests.md.

---

## P11. Adopt the point-set formulation as the unifying design language: everything is points in (space, physical-time, log-scale, value); tree/wire/plane structure = positional encodings + attention masks

**Replaces:** no single test, but several framing-level questions at once: "is unification real?", "must the operator be support-agnostic?", the separate vocabulary of stems/sweeps/topologies as different *mechanisms*.

**Simpler alternative:** state once that every architecture under discussion is one model family — a transformer over the active-coefficient point set — with (a) a positional code over (s, t_phys, log-scale) and (b) sparsity/tying patterns imposed for cost: C0–C3 are attention masks; the V-cycle is a masked transformer with level-sequential scheduling; W0–W7 are weight-tying patterns over the scale coordinate; per-band convs are local masks on the native grids; the tokenizer is local pooling in point space. The P2 ceiling is then literally the unrestricted family member, and "TPC vs optical unification" reduces to one question: *does the same positional code (physical time + log-scale) with the same masks work at both depths?* — which P1–P4's experiments already answer with no extra runs.

**IT justification:** sufficient-statistics view — the input is fully described by the point set; every structural device either restricts the channel (masks/ties, justified by cost and by measured priors like the tree lift) or re-parameterizes the index (PEs). Framing structure as *restrictions of one family* makes the entire design space a single rate(cost)–distortion(gap-to-ceiling) trade, with the ceiling measured once. Several "open questions" stop being architecture forks and become mask-pattern choices on a common substrate.

**What is lost:** the production system still needs the structured implementations for MFU (the measured cost analysis stands — full attention over 300k TPC coefficients is out of the question); the unification claim must be honestly scoped to "unified family, per-modality masks," not "one network." Risk: language slippage into overclaiming.

**Feasibility:** documentation-level change plus it dictates the P2 implementation (which is needed anyway).

---

## P12. Measurement-driven closures: declare these questions answered, remove their tests/flags

**Replaces:** residual hedging spread across the docs.

Already effectively answered, needing no test:

1. **Alignment / open #18** — measured: naive τ>>1 within 1–2% everywhere; the coarse-level drift is absorbable by the 3-tap conv. Close it (the docs say "mostly retired"; make it fully retired).
2. **TPC topology and weight structure** — closed by P1's argument + measurement (flat bands, L=4): W7 + C0, no test.
3. **Drop-D1** — both modalities settled (noise floor / 0.9% budget). Remove from open lists.
4. **A-band handling** — measured lateral lift 1.9× (vs 2–57× tree lifts): one lateral edge type, low stakes, no ablation needed.
5. **Cross-volume coupling** — measured weak (+3.4 pts over chance) and direct-aligned: the summary-token + boundary-attention design follows; remove "full token-level" from the menu rather than carrying it as a rejected option to re-justify.
6. **Masking difficulty ordering** (T-D1's target) — predicted by the measured conditionals; see P9.
7. **Invertible lifting / open #7** — closed by the P6 capacity count.
8. **C3's motivation** — its cost advantage is moot at the decision scale; see P4.

**IT justification:** MDL on the *plan itself* — every live fork carried in the docs costs coordination and re-litigation; forks whose posterior is already concentrated should be collapsed and their description length reclaimed.

**What is lost:** optionality, in the cheap sense; each closure names its measured basis, so reopening requires new data, which is the correct bar.

**Feasibility:** an editing pass.

---

# The minimal test set that still pins the design

Five items replace the ~17-test battery and both full ladders:

| # | What | Subsumes | Cost |
|---|---|---|---|
| **M1** | Cost microbench on dumped events: tree-op index maps (all C0–C3, benched not trained), batched stems | T-B7, opens #1/#2 | hours, no training |
| **M2** | Data-only information audit in **bits**: exact activity-CMI from contingency tables + copula/plug-in value-CMI, both corpora; + optical anchor-level recompute | T-B1, T-B2, T-C1, retires the lift-based reach argument | days, no training |
| **M3** | **One** chunk-scale star design on optical (doraemon, clean targets): ceiling (full attention, per-level params) + arms {C0+W7, C2+W7, C0+W2, C0+W0}, d_s at the P6-predicted knee ±2×, pre-registered ε per band | T-A4, T-B3, T-B5, both ladders | hours/arm, 1 GPU |
| **M4** | Zero-shot transfer evaluations of M3 winners: scale- vs index-conditioning across depth; SER 5 µs↔10 µs pair | T-A5 | evaluation only |
| **M5** | TPC one-point verification: pre-attention stack (W7, C0), rounds 0 vs 1 control, tokenizer at the P6-predicted d, AE stratified by core saturation | T-C2, T-C3, TPC arms of everything | days, 1 GPU |

**Contingencies (run only on trigger):** T-B4/T-B6 iff C0's fine-band gap > ε (P10); one W-bisection arm iff W2 fails while W7 passes (P3); C1/C3 training arms iff C2 wins but its production cost benches badly (P4); T-A3 lives on as free telemetry inside M3 (P7). Group D defers to the trunk phase (P9).

**The one new critical-path item this creates:** the doraemon `label_N` loader + forward optical noise model — M2 (re-measurement on clean data), M3, and M4 all gate on it. It should be promoted from a footnote to the first task.

**Net effect:** the plan keeps every decision it was going to make, makes most of them from counting rather than training, concentrates all training-based discrimination into one star design at the scale where it is cheapest and most discriminating, and converts the two big ladders from enumeration problems into a rate–distortion knee search with pre-registered bisection.
