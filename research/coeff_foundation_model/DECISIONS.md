# DECISIONS — operator-pinning phase log

Every gated decision with the numbers that made it. Convention: a decision is
either OPEN, GATED (named test pending), or CLOSED (numbers cited).

## CLOSED

### D-01 Batched stems amortize (handoff open #2) — CLOSED 2026-06-11
M1 T1.3: TPC SubMConv2d, 24 per-(plane,band) calls = 12.1 ms vs 4 band-batched
calls = 2.5 ms (d=32) / 4.2 ms (d=64) → **4.8× / 2.9× amortization. Batch
stems across planes per band-type.** (`artifacts/microbench_treeops.json`)

### D-02 Tree-op cost does NOT discriminate topologies (handoff open #1) — CLOSED 2026-06-11
M1 T1.1, typical events (optical 369k coeffs / TPC 321k), A100 fwd ms:

| d_s | C0 | C1 (Δ=1,2,4,8) | C2 (anc-attn) | C3 (cell-axial) |
|--:|--:|--:|--:|--:|
| 32 | 6.9 | 11.0 | 8.2 | 4.9 |
| 64 | 4.1 | 8.0 | 10.3 | 5.5 |
| 128 | 6.0 | 11.6 | 18.3 | 7.6 |

fwd+bwd: C0 15–21 ms, C2 25–61 ms. Against a ~200 ms/event trunk, even the
most expensive topology is ≤15% of step. **Topology will be chosen on quality
(M3) alone; cost is immaterial at these scales.** The handoff's pre-attention
cost estimate (~10 ms tree op) is confirmed for C0.

### D-03 Optical within-level ops: packed/gather, NOT dense — CLOSED 2026-06-11 (reversal)
M1 T1.2, per-band depthwise Conv1d on dense grids vs gather(±1)+linear on
active sites, true doraemon occupancies: gather wins every band except D10
(17.9% occ, tie 0.09 vs 0.10 ms); fine bands are not close (D3: 6.1 vs
0.17 ms; D2: 12.2 vs 0.10 ms; D1: 24.3 vs 0.10 ms). **The earlier
"dense conv for optical" choice (based on 62.5% cell-union occupancy — a
category error, coefficient occupancy is 0.001–18%) is reversed: ONE packed
sparse implementation serves both modalities.** Skeptic finding S6 confirmed.

### D-04 Alignment / index-map offsets (handoff open #18) — CLOSED 2026-06-11
M2 T2.4 impulse probe (coif3 periodization, exact): δ(A10) = −2.4 coeffs;
δ(details) ∈ {0, +0.5, +1}. Value-level corr(|child|,|parent@s|) peaks at
s ∈ {0, ±1} for all detail bands (TPC and optical) — naive τ>>1 is correct
within one slot for detail↔detail edges; the A-band lateral edge gets the
−2 offset baked in. (`artifacts/info_audit.json: group_delay_coif3`)

## CLOSED (continued)

### D-08 Cross-level op: small positive, optional; C2 unneeded — CLOSED 2026-06-11 (short battery)
Test A (masked-value prediction, 500 steps, fold 0, doraemon clean support):

| arm | primary | vs rounds0 | orphan | parented |
|---|--:|--:|--:|--:|
| rounds0 (no cross-level) | 6.613 | — | 6.542 | 8.366 |
| c0w7 (V-cycle, per-band) | 6.172 | **−6.6%** | 6.229 | 7.850 |
| c0w0 (V-cycle, fully shared) | 6.335 | −4.2% | 6.371 | 7.896 |
| ceiling (full attention) | 7.161 | +8.3% | 6.685 | 8.769 |

- **Tree op effect is real but below the pre-registered 10–20% large-effect
  bar** (−6.6% primary; per-band it concentrates at D10 −13% and D2 −7%).
  Consistent with M2's bits audit (small unique cross-scale information).
  → C0 kept as default (cheap: 31 ms = 15% of ViT-L step), explicitly marked
  DROPPABLE; revisit only at scaling if fine-band quality matters downstream.
- **Prediction P3 partially refuted:** the c0w7-vs-rounds0 gap does NOT
  concentrate in the orphan stratum (−4.8% orphan vs −6.2% parented) — no
  structural-rescue signal → **C2's ancestor attention has no population to
  win on; C2 is dropped without a run.**
- **Conditioning (D-07 resolved):** c0w7 vs c0w0 = 2.6% — weight structure is
  NOT a large effect; fully-shared weights cost ~3% at this scale. Best news
  possible for unification: adopt shared+FiLM (W2-class) as default.
- **Ceiling is optimization-bound at short budgets** (worst arm) — exactly
  the skeptic's S5 warning; "ε-of-oracle" rules are void at this scale;
  structured arms ARE the envelope here.

### D-09 Bottleneck not binding at short scale — Test B, CLOSED for this phase
rounds0 AE, 500 steps: d_tok 256 → 4.835; d_tok 64 → 5.026 (+4%). A 4×
channel-rate cut costs 4% — distortion is optimization-dominated, not
rate-dominated (consistent with the registered global over-provisioning,
P1). The core-stratified rate-distortion knee is only measurable near
convergence → deferred to the scaling phase. (Reference: ceiling AE @3000
steps = 3.49 vs ~4.8 @500 — runs are ~30% from converged.)

### D-10 Tokenizer variation matrix: NO knob passes the large-effect filter — CLOSED 2026-06-11
Full matrix on the on-the-fly noisy→clean task (500 steps, doraemon, classical
baseline 0.0388):

Routing (masked): default c0w2 r1 K2 = 6.443. reach ±2/±4: −0.6/−1.1%;
K1/K3: +2.8/−0.8%; rounds0 (no tree op): +3.7%; c0w0 (no conditioning): +1.6%.
Bottleneck (AE): attn 4.957; **attn4 4.744 (−4.3%)**; sum 5.261 (+6.1%);
dtok64 5.312 (+7.2%).

**Every variation is within ±7% — the tokenizer is structurally insensitive.**
Consistent sub-filter trends, recorded not acted on: reach helps
monotonically (matches the audit's information, tiny functionally); the tree
op's edge shrinks under noise (6.6% clean → 3.7% noisy); attn4 > attn > sum
(mean-like pooling worst, as designed); d_tok matters more under denoising
than clean (7.2% vs 4%).

**Adopted config for the convergence run:** c0w2, reach ±1, K=2, **attn4
pooling** (equal-cost short-scale winner on primary, orphan, and bce — the
filter blocks added complexity, not free choices), d_tok 256, anchor 1024.
Everything else: defaults stand; variations deferred to the scaling phase.

### D-11 Optical tokenizer-only denoising AE @20k steps — phase deliverable — CLOSED 2026-06-12
Config: c0w2, ±1, K=2, attn4, d_tok=256, anchor 1024; on-the-fly noisy→clean;
held-out events. primary 1.272 (500 steps: 4.74; 3000: ~3.5). Per band vs the
classical baseline (production's kept noisy coeffs vs clean):
A10 0.086/0.085 (**1.0×**), D10 2.3×, D9 3.6×, D8 4.5×, D7 8.7×, D6 12.5×,
D5 16.2×, D4 35×, D3 **117×**, D2 **142×**.
**Reading: lossless at coarse scale (A10 ≈ baseline exactly), progressively
rate-limited fine-ward** — distortion grows monotonically with
slots-per-token (1 A10 slot → 256 D2 slots per cell): the registered P1/P2
structure observed within-cell. The 256-d token cannot carry fine-band slot
multiplicity at anchor 1024. Orphan ≈ parented (1.11/1.16) — D-08 holds at
convergence. Occupancy bce 0.032.
**Targeted follow-ups (one run each, not a sweep):** (a) anchor 512 (halves
slots/token), (b) d_tok at the fine-band knee, (c) per-band decoder heads.
The binding constraint is now localized and measurable.

### D-12 TPC tokenizer: tree op IS load-bearing; AE far from baseline with mid-band/core-concentrated loss — CLOSED 2026-06-12
Masked (500 steps): c0w2 6.047 vs rounds0 7.135 — **−15.3%, PASSES the
large-effect filter** (vs +3.7% optical). The modality asymmetry matches the
audits exactly: TPC lateral reach dies at ±1 while its cross-scale value
coupling is the strongest measured (I_v(c;p|L)=0.139) → the scale axis does
real work on TPC where the lateral axis can't. **C0 is required for TPC,
optional for optical.**

TPC AE @20k (production geometry, 25.4k occupied 8×4 patches): primary 5.685
vs classical baseline 0.564 (≈10×). Per band: A4 4.9×, **D4 24×, D3 21×**,
D2 5.6× — NOT slot-multiplicity-ordered (unlike optical): the mid bands,
where TPC's bipolar signal structure lives, dominate the loss; parented ≫
orphan (6.55 vs 1.96) → distortion concentrates in dense signal cores.
Note TPC's classical baseline is 15× optical's (0.564 vs 0.038): real
denoising headroom exists (coherent residual + intrinsic noise).
**Candidate causes, one run each:** (a) NO wire-direction mixing in v1 (the
design's anisotropic 7-wire reach is absent — wire context only enters via
pooling); (b) token rate at the cores; (c) under-training (TPC content is
richer at equal steps). (a) is the prime suspect and the cheapest test.

### D-13 TPC tokenizer v1 was improper: the wire axis was a container, not an operator — FIXED 2026-06-12
v1 defects (caught on user push, confirmed by measurement):
1. **No wire-direction mixing** — the design's anisotropic 7-wire×3-time stem
   was absent; wire info entered only via embeddings + pooling. Adding ±1
   wire gathers: masked 6.047 → 3.433 (**−43%, the largest effect in the
   program**), even crippled by (2).
2. **Hard walls every 8 wires** (unit = 8-wire block was an implementation
   convenience): tracks cross blocks, wire-neighbors couldn't. Fixed: unit =
   whole plane (6 units/event, ~1 plane per batch), full-plane wire keys.
3. **Wire reach never audited** (the reach audit was time-only). Now measured
   (`wire_reach_audit.py`): **r(c, wire±1) = 0.685 at D4 — the strongest
   value coupling measured anywhere** (vs 0.27 time, 0.51 tree parent);
   incremental beyond ±1 small (r(W2|W1)=0.05–0.16; activity I(W2|W1)≈0.003)
   → ±1–2 wire reach captures the bulk; the wire axis dominates TPC locality.
4. **Normalization wobble**: per-event signal-contaminated MAD σ scaled the
   regression targets. Fixed: σ table per (plane, band) calibrated once
   (2 fixed events); production threshold itself unchanged (per-event MAD,
   as production specifies).
Token-count validation preserved after rebuild: 25,401 occupied patches on
ev33.

**v2 results (plane units + fixed norm, 2026-06-12):**
- masked 500: wiremix 3.527 vs nowire 6.664 → **−47%** (plane-level
  confirmation; norm+units fix alone gave ~7%: nowire 7.135→6.664).
- AE@20k: wiremix **4.387** vs nowire 5.685 (**−23% at convergence**);
  per-band ratios: A4 4.9→3.4×, D4 24→17×, D3 22→16×, D2 5.6→4.9×;
  parented 6.55→5.24.
Wire mixing is adopted as a REQUIRED TPC tokenizer component. Residual gap
(still ~7.6× classical overall, mid-band dominated): remaining suspects =
token rate at cores (p95 ~80 coeffs ≈ 1 kbit vs d_tok 256 — marginal, the
optical d_tok-1024 run informs by analogy), wire reach ±2 / K=3, training
length. These are scaling-phase questions, not structural defects.

### D-14 Optical fine-band limit is NOT channel rate — CLOSED 2026-06-12
d_tok 256 → 1024 (4× rate) at 20k steps: primary 1.272 → 1.056 (−17%);
fine bands improve only 12–30% (D3 117×→91×, D2 142×→125× of classical).
**Quadrupling the token rate barely moves the binding bands** → the fine-band
limit is pooling structure / decoder head / intrinsic value predictability,
not d_tok. Notable: A10 goes sub-classical (0.0806 vs 0.0852) — the first
band where the learned tokenizer beats production's kept coefficients
(denoising win). D10 wiggle (−6%) ≈ seed noise scale (~5%).
Remaining optical knobs (scaling phase): anchor 512 (geometry), more pool
queries (n_q × d_tok/n_q structure), per-band decoder heads.

### D-15 Systematic bottleneck protocol (closed-form PCA) — replaces full-run sweeps; attribution REVISED — 2026-06-12
`cell_rd.py`: the bottleneck is a per-cell problem → its LINEAR rate-
distortion is the PCA eigenspectrum of cell slot-vectors — every (anchor, d)
point in seconds, no training. Cross-event validated (fit ev A, eval ev B:
cross-event residual ≤ in-event floor — subspace generalizes).

Per-ACTIVE-coefficient linear floor, optical anchor 1024 / d=256, cross-event,
vs the trained AE (D-11):

| band | linear floor | trained | trained/floor |
|---|--:|--:|--:|
| A10–D4 | 0.000–0.003 | 0.08–0.96 | **30–300×** |
| D3 | 0.391 | 3.96 | **10×** |
| D2 | 4.680 | 6.46 | **1.4×** |

**Revised attribution (supersedes D-11's "fine bands rate-limited"):**
- **D2 is genuinely compression-limited** — even the optimal linear 256-dim
  code leaves 4.7/coeff (≈100× classical); the trained model is near its
  floor. Options: hybrid (carry D2 coeffs alongside tokens, skip the
  bottleneck), more tokens for fine bands only, or accept D2 loss.
- **D3 and ALL mid/coarse bands are architecture/optimization-limited** —
  the trained model sits 10–300× above a floor achievable by a plain
  per-cell linear map. The attention-pool + shared-MLP decode is leaving
  enormous headroom; a learned LINEAR cell projection might beat it.
- N-vs-d (linear, fixed total bits): bigger cells at higher d beat smaller
  cells at lower d (1024/256: 0.0047 < 512/128: 0.0086 < 256/64: 0.0141
  per-slot) — shared structure compresses.
- Trained N-axis point agrees directionally: anchor 512 @20k = 0.853 vs
  1024 = 1.272 (−33%; D3 117→73×, D2 142→94×).

**TPC N-axis trained point (completes the picture, 2026-06-12):** pw 8→4
(slots 256→128, occupied tokens 25.4k→34.6k, +36%): primary 4.39 → **2.48**
(−43%); per band A4 3.4→**1.9×**, D4 17→9.3×, D3 16→8.8×, D2 4.9→3.1×.
TPC trajectory: v1 5.68 → +wiremix 4.39 → +pw4 2.48. Both modalities respond
strongly to slots-per-token; with PCA floors ≈0 on these bands, the residual
1.9–9.3× remains decode/optimization headroom — the free direction (per-cell
micro-AE) should be exhausted before buying more tokens.

**Protocol going forward (replaces 20k-step sweeps for bottleneck questions):**
1. PCA grid over (anchor, d) — seconds, cross-event validated;
2. tiny per-cell AEs (pool+decode only, cells sampled from dumps) for the
   nonlinear/optimization gap at selected points — minutes;
3. ONE full-model confirmation run at the chosen operating point.
Full runs are reserved for routing/integration questions only.

### D-16 Tokenization design: per-band patchify favored over column tokens — 2026-06-12
The user's ORIGINAL formulation (per-level patchify, level-specific patches)
re-examined after the column design's founding statistic (cone-of-influence /
31× lift) was debunked as envelope confound. `tokenizer_compare.py`: both
designs, both modalities, PCA/linear/MLP frontiers on dump-sampled patches,
held-out split, per-active per-band MSE.

**Token cost (measured):** TPC per-band 8×8: 41k (+62% vs 25.4k column);
8×16: 33k (+30%). Optical P=64: 48.8k (+44%, per-interaction basis).

**Frontier results (matched total dims/event):**
- TPC: per-band DOMINATES — 8×8@d32 (1.31M dims) primary 0.075 vs column
  8×4@d64 (1.63M dims) 0.154; D2 9× better (0.042 vs 0.396).
- Optical fine bands: per-band decisive — D2@d32 = 1.24 vs column@d256 =
  4.10 (D2's "incompressibility" was partly a shared-capacity artifact).
- Optical mid bands (D7–D5): mixed cells pack BETTER at high d (cross-band
  shared structure) → pure uniform-d per-band underuses coarse tokens.
- In the real trunk all tokens share d_model, so capacity allocation =
  TOKEN allocation — which per-band does natively (fine bands get more
  tokens). Non-uniform per-band token granularity (or hybrid column-coarse +
  perband-fine) is the refinement axis.
- Calibration: even linear micro-models (TPC column d64: 0.154) crush the
  full trained AE (4.39) — the production AE's plumbing remains the gap.

**Confirmation @20k (optical, full pipeline): per-band primary 0.777 — best
yet** (column-1024: 1.272; column-512: 0.854). The profile FLATTENS exactly
as the micro-frontier predicted: fine bands 2.7–3.5× better (D3 3.96→1.46,
D2 6.46→1.84), coarse/mid worse (D8 0.14→0.59, D7 0.22→0.75 — mixed cells
pack shared structure better there). Band-wise best-of ⇒ **HYBRID: column
tokens for bands A10–D4 + per-band P=64 tokens for D3/D2** — predicted
primary ≈ 0.5 at similar token count. Implemented (`--tokmode hybrid`,
column type id = NBANDS in cell_band); 20k confirmation launched.
Trunk implications unlocked either way: (position, log-scale) PE
unification, scale-axis masking SSL, within-band→cross-band→cross-plane
hierarchy.

### D-17 Simplified architecture (post step-back): ViT-style patch tokenizer — built 2026-06-12, acceptance pending
`vit_model.py`: asinh → hybrid patchify → **LINEAR patch embedding**
([values, occupancy bits] → d_model) + token-type embedding + physical-time
PE → optional full-attention blocks (within chunk) → linear decode. NO deep
encoder, NO attention pooling, NO tree ops — justified by the frontier
result that linear per-patch maps reach the floors. 0.20M params at d=256
blocks=0 (vs 1.4M deep substrate).
**Acceptance battery (running):** (a) blocks=0 AE @10k — must match/beat the
deep substrate hybrid (0.434); (b) blocks=2 AE — does within-chunk attention
context improve denoising; (c) blocks=4 MAE (mask 30% of TOKENS, predict
clean slots from context) — the first trunk-objective pilot on this
tokenization. Early signal: blocks=0 at 50 steps already at 3.13 (deep
substrate needed ~500 steps for similar) — the linear path optimizes far
faster, consistent with D-15's optimization-gap diagnosis.
**TPC twin RESULT (vit_tpc.py, per-band 2D 16×8, 10k steps) — PART ONE CLOSED:**
| arm | primary | vs classical 0.618 | vs deep-sub 2.48 |
|---|--:|--:|--:|
| linear (b=0, 0.23M) | **0.533** | 0.86× (sub-classical) | 4.7× better |
| +2 attn blocks (1.81M) | **0.164** | 0.27× | 15× better |
Per-band b2: A4 0.354, D4 0.138, D3 0.080, D2 0.084 — all far below
classical (0.96/0.35/0.36/0.81), flat across bands. **Linear TPC tokenizer
is sub-classical** (mirrors optical 0.0351<0.0388); +2 blocks → 3.8× below
classical. Wire-kill aug (deadfrac=0.1): primary 1.29 — robustness costs ~2.4×
on clean eval (expected; trains dead-channel inference, evaluated WITH dead
rows present) — a tunable robustness/fidelity dial, not a fidelity result.
Both modalities: tokenizer = patchify + ONE linear layer, sub-classical,
provably not the limiting stage. Attention (trunk's job) adds 3–6× on top.

### D-18 Encoder-drop (true MAE) + structured masking — fm/, 2026-06-13
**Encoder-drop replaces in-place mask tokens.** Original `fm/model.py` was
BERT-style: masked tokens → shared `mask_tok` but KEPT in the sequence, so all
enc+dec blocks ran on full N≈30k tokens/event → 316 ms/step (d=256, 8+2 blocks,
batch=1). Cost is pure O(N²) attention (4·N²·d ≫ the 24·N·d² linear term; the
8M params are irrelevant — sequence length is everything). **Switched encoder to
run on VISIBLE tokens only; mask tokens (= `mask_tok` + band/plane emb + FiLM +
RoPE, i.e. full position/response conditioning, NO content) are inserted only
before the 2-block decoder** (`index_copy` reassembly). Measured d=256/8+2:
mask 0.50 316→**153 ms** (2.1×), mask 0.75 →**115 ms**. Consequence that
matters: **harsher masking is now CHEAPER, not equal-cost** (fewer visible
tokens → smaller encoder N²) — removes the only reason not to push mask ratio up.
Tradeoff: masked-token recon now flows only through the thin decoder, so OVERFIT
memorization is weaker (2k steps → 2.24× classical vs the in-place model's
0.022); expected MAE asymmetry, not a bug — the encoder holds the representation,
decoder is a shallow read-out. If value-regression fidelity needs it, deepen
`dec_blocks` (orig MAE used 8 for pixels).
**Structured masking (`make_mask`, modes random|plane|block):** per-band patch
tokens let `random` masking CHEAT — a masked D2 patch is interpolable from the
visible A4/D4/D3 patches at the same wire/time. `block` = wire-slab tube across
ALL bands in a plane (VideoMAE-style; tracks run along wires) kills the cross-band
shortcut; `plane` = hide whole plane(s) = direct cross-plane mutual-information
probe. Sweep (1000 ev, 5000 steps, d=256/8+2): random@0.5, random@0.75,
plane n=1 (1/6), plane n=3 (whole volume), block@0.6 — RESULTS pending.
NOTE: recon-MSE measures pretext difficulty + cross-plane MI, NOT representation
quality — the adjudicated eval backbone (frozen probe on tpc_de + label-efficiency
curve) is still the gating measurement.

### D-19 random pretext settled + first real scaling run — fm/, 2026-06-13
**Masking axis decided: plain `random` (cross-everything).** One event = all 6
planes concatenated into ONE token set; `random` masks uniformly over it with
global attention, so it already reconstructs from cross-plane + cross-band +
cross-wire context jointly — it IS "fully random cross-plane." `plane`/`block`
were measurement instruments, not better pretexts. Measured cross-plane MI
(forced via `plane` mode, 6k steps): lives almost entirely in the **A4/bulk band
(29% var-explained); detail bands D4/D3/D2 ≈ 1-3%** (= flat at predict-mean).
Physics: U/V/Y are 3 projections of the same 3D ionization — total charge (A4)
is shared, fine detail is projection-specific. Cross-band is 4-5× richer at every
band, so random wins.
**Masking unit = TOKEN (patch), supervision unit = COEFFICIENT.** Each token =
(plane, band, 16-wire×8-tick) patch ≈128 coeff slots → one linear-embed vector;
masking hides the whole patch, loss predicts clean value at each active slot.
Per-coeff tokens (N 30k→320k) are O(N²)-infeasible; within-token slot masking
leaks. Token granularity is the right unit.
**Metric: % target variance explained** (predict-mean MSE per band =
{A4 14.0, D4 13.83, D3 11.51, D2 6.11}; ×noise ratios are misleading because
surviving coeffs have ~18× the noise variance).
**FIRST SCALING RUN (25M: d=384 enc10 dec4, mask 0.75, 6k ev, warmup1000+cosine,
24k steps≈5 ep):** held-out var-explained **18.5%(2.5k)→39→46→49→51→53→54.3→55.1
→55.6→55.5%(24k)**, train/test gap **0.033** (NOT overfitting). Per-band test:
A4 74%, D4 55%, D3 51%, D2 42% — every band up vs 8M@0.5 (66/41/39/25); fine
bands gained most. Vs 8M@0.75/5k=27%, 8M@0.5/6k=46%: bigger+schedule at the
HARDER mask beats the old easier setup. **Soft plateau ~55.5% (last 4k flat) but
LR-decay-driven + tiny gap ⇒ this is the ceiling for THIS config, not the
architecture — still scaling, no data/capacity wall hit.** Levers untested:
more width/depth, longer schedule, more data. Artifacts: `fm/run_scale.log`,
`fm/fm_curve.jsonl` (tag=scale25M), `fm/scaling_curve.png`.
**STILL the gating measurement: the frozen probe on `tpc_de`** — recon
var-explained is rising cleanly but says nothing about whether the encoder
features are downstream-useful. Build before declaring the representation good.

### D-05 Cross-level operator (C0 vs C2 vs none) — superseded by D-08
M2 evidence (typical events; multi-event CIs pending):
- **The envelope control guts the activity-level tree story**: I(child;
  parent | within-band L,R) ≈ 0.004 bits/slot (TPC D3) vs raw I(child;
  parent) = 0.058 — ~93% of the parent's activity information is already in
  within-band context. Optical: I(c;p|LR) ≈ 0.03–0.04 vs I(c;p) 0.15–0.23.
  Within-band neighbors alone out-inform the parent everywhere
  (I(c;LR) > I(c;p), every band, every corpus).
- Total cross-scale activity info beyond local context ≈ 50 kbit/event (TPC)
  / ~60 kbit (optical) vs ~4.7 Mbit source — ~1%.
- Value-level coupling among co-active pairs is substantial (copula r:
  TPC D3 0.51; optical D9 0.72→D2 0.26; partial grandparent r≈0.2–0.34), and
  — the completing measurement — **it survives the lateral control**:
  I_v(c;p|L) = 0.139 bits at TPC D3 (63% of raw I_v(c;p)=0.222), 0.09–0.36
  bits at optical coarse/mid bands. **Asymmetry established: activity (where)
  is ~fully local within-band; values (how much) carry genuine cross-scale
  information beyond lateral context.** The tree operator's job — if it earns
  its place in M3 — is value-context propagation among co-active sites, not
  support prediction.
- Orphans: 25–39% of active children lack an active parent; grandparent
  rescue 50–80% — C0's missing-relay population is real but its information
  stake is bounded by the small CMI numbers.
**Registered prediction (pre-M3):** rounds-0 (no tree op) will be much closer
to C0 than previously assumed; any C0-vs-C2 gap concentrates in the orphan
stratum (prediction P3); if the star shows all three within ε, the tree
operator is dropped in favor of within-band convs + trunk attention.

### D-06 Anchor level / d_s (optical) — GATED on M3 one-point verification
M2 T2.5 (doraemon, per-interaction chunks): cell 1024 → 21.3k cells/ev,
fan-out p95 110; cell 512 → 42.7k cells, p95 59; cell 256 → 85.4k cells,
p95 32. Bit-count prediction P2: d_s_min ≈ fanout_p95 × 12 / bits_per_dim.
light_output (summed-readout proxy): 5.8k/11.5k/23.1k cells, p95 204/109/58.
Token-count × trunk-cost tradeoff to be set with M3's d_s verification.

### D-07 Weight structure (W0/W2/W7) — resolved by D-08 (≈3% effect; shared+FiLM default)

## Registered predictions (M2 T2.6, before any M3 training)
- **P1**: bottleneck recon flat in N, knees in d ≈ 256–512; residual loss in
  the top core-load percentile. (TPC source ≈ 4.7 Mbit/event vs 26–104 Mbit
  token capacity → 6–22× over-provisioned globally.)
- **P2**: d_s_min(anchor) ≈ fanout_p95 × 12 / bits_per_dim.
- **P3**: any C0-vs-C2 quality gap concentrates in the orphan stratum;
  parented-stratum gap ≈ 0 at TPC depth, small at optical depth.

## Follow-ups queued
- Re-run `info_audit.py` on the 200-event doraemon dump when the scan lands
  (current numbers: typical single events; add event-bootstrap CIs via a
  multi-event rescan mode).
- Value-CMI conditioned on lateral *values* (I(v_c; v_p | v_L)) — the last
  uncontrolled channel in D-05's evidence.
- M3 star substrate (`star_model.py`).
- M5 TPC verification after M3 geometry is fixed.
