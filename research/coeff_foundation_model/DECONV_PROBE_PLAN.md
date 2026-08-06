# Deconvolution probe — FUTURE plan (not active)

> **Note (research thinning).** The scripts named below (fm/probe.py) were removed from this
> branch; the measurements they produced stand. Recover any of them with
> `git show main:research/coeff_foundation_model/fm/<name>` — `main` keeps the
> full research tree.


**Status: PARKED.** We continue with the **classic masked-MAE methodology** (predict clean
*wire* coeffs at masked/visible slots; metric = per-band asinh-MSE / var-explained). This
document records the deconvolution evaluation we will build *later*, once the MAE program is
further along. Nothing here changes the current training.

## Why (the real downstream objective)

The thing we ultimately care about is **deconvolution** — recovering the true ionization
**charge** from the wire signal — not denoising. Our current target `val_clean` is the
*denoised wire signal*, which is still **convolved** with the field+electronics response. The
deconvolution target is the true **diffuse charge** `Q(wire, tick)` (post-drift/diffusion,
**pre-response**), which on induction planes (U/V) is genuinely different from the wire signal
(bipolar wire → unipolar charge).

So the FM's value should eventually be measured in **deconvolution quality**, via a probe whose
target is the charge — not abstract labels, not wire-recon-R².

## Data (verified 2026-06-14)

Under `/sdf/data/neutrino/omara/JAXTPC_Wire/test_00_00_02/` there are sibling modalities:
- **`sensor`** — the **wire signal** (`delta_wire/delta_time/values`, response-convolved ADC).
  This is what we train on now; `val_clean` = its denoised DWT.
- **`hits`** — the **true diffuse charge** (the deconvolution target). Per
  `event_NNN/volume_{0,1}/{U,V,Y}`:
  - `center_wires (Ng) int16`, `center_times (Ng) int16` — per hit-group center (same 0–4321
    tick grid as the wire signal → aligns).
  - `group_ids (Ng) int32`, `group_sizes (Ng) uint8`, `peak_charges (Ng) float32`.
  - `charges_u16 (Nd) uint16`, `delta_wires (Nd) int8`, `delta_times (Nd) int8` — per-deposit
    charge + offset from its group center (Nd = sum of group_sizes).
  - event-level `deposit_to_group`, `group_to_track`, `qs_fractions (Nd) float16`.
  - `config/num_wires = [[1969,1969,1443],[1969,1969,1443]]`.
- `step` — (Geant step-level; not needed for this).

We currently load only `modalities=("sensor",)`.

## Building the target Q(wire,tick) → charge coeffs

1. For each `(volume, plane)`: scatter-add deposit charges onto the wire×tick grid at
   `(center_wire + delta_wire, center_time + delta_time)` → `Q(wire, tick)` (dense per plane,
   same shape as the wire image).
2. **coif3 L4 DWT** of `Q` (the *same* transform as the model) → **charge coefficients** =
   the probe/target, per band (A4/D4/D3/D2).

**Caveats to resolve when building it:**
- **Absolute charge assembly**: reconstruct true charge from `charges_u16` (quantized) +
  `peak_charges` (float scale) + `qs_fractions` + group structure. **Check for a pimm-data
  `hits` decoder/modality first** rather than hand-rolling the dequantization.
- **Response group-delay alignment**: there may be a tick offset between charge-arrival
  (`center_times`) and the convolved wire-signal peak; align before comparing coefficients
  (cf. the coif3 group-delay table already characterized for the wire path).
- **Induction vs collection**: Y unipolar (easy); U/V bipolar (the hard, ill-posed case).

## The probe

Freeze the MAE-pretrained encoder → **linear probe** from the per-token representation to the
charge coefficients `DWT(Q)` at that token's location → **R² per band**. (Also a small MLP
probe as a non-linear reference.) Run the **label-efficiency curve**: probe-from-frozen vs
from-scratch as a function of # labeled events.

## Ceiling — two reference lines (this is "measure the ceiling", deconv version)

1. **Classical per-wire Wiener oracle (baseline to beat).** Deconvolve the *clean* wire signal
   per wire with the known response, DWT, compare to `DWT(Q)` → R² per band. High on Y,
   **capped well below 1 on U/V** (bipolar = frequency zeros = information loss). That cap is
   the room a cross-plane model can claim.
2. **Information ceiling (joint, all planes).** Best recovery of `Q` from all planes' data
   jointly. Bracket empirically: **clean-input oracle** (feed clean wire coeffs → strips the
   noise penalty) and a **mask-ratio sweep** (more visible context → asymptote).

Place the model's probe R² between the two lines. **Landing above classical-Wiener, especially
on U/V, is the FM earning its keep** — because attention can deconvolve *jointly* across U/V/Y
(same 3-D charge), which per-wire classical cannot.

## Probe-first vs retrain (decision for later)

- **Probe-first (cheap):** freeze current denoising-pretrained encoder, linear-probe to charge.
  If it deconvolves well → the MAE pretext learned charge-relevant structure (good).
- **If weak → switch the training target to charge coeffs** (masked *deconvolution*: predict
  `DWT(Q)` at masked+visible slots). Since we only care about deconv and sim truth is free, this
  aligned objective is the likely endpoint; the denoising/MAE pretext then serves as
  initialization for **label-efficiency** and **real-data transfer** (no charge truth on real
  data → can't supervise directly there).

## Concrete steps (when un-parked)

1. Check pimm-data for a `hits` decoder; prototype `Q(wire,tick)` for one event; visualize the
   charge image + its coefficients vs the wire-signal coefficients.
2. Build the classical per-wire Wiener deconv baseline (needs the field response) → ceiling
   line 1.
3. Linear deconv probe on a frozen MAE checkpoint → R² per band; place vs the two ceiling lines.
4. Label-efficiency curve (frozen vs scratch).
5. If probe weak: retrain with charge-coeff target (masked deconvolution).

Related: [[tpc-coeff-foundation-model]] memory; `DECISIONS.md` D-18/D-19 (MAE program).

---
## Empirical analysis of the parts (2026-06-14, event 0; deconv_analysis.py + deconv_plot.py)

Built true charge Q(wire,tick) from hits (charge = charges_u16/65535 * peak_charges[group],
scatter at center+delta) and clean wire S(wire,tick) from sensor (ds.get_data -> decoded
wire/time/value, pedestal-subtracted). Both per-wire coif3-L4 DWT.

KEY: the response is a low-pass smear that shifts coeff energy COARSE-ward and the wire signal
retains almost no fine-band detail => that lost detail is exactly the deconvolution target & the
ceiling source.

| plane | charge occ | wire occ | ticks/wire charge->wire (spread) |
|---|--:|--:|--:|
| Y (collection) | 0.109% | 0.293% | 9.6 -> 25.5 (2.7x) |
| U (induction)  | 0.030% | 0.374% | 13.4 -> 153.8 (11.5x) |

Per-band coeff energy % (charge vs wire):
- Y: A4 44/73, D4 25/21, D3 24/5.7, D2 6.7/0.1, D1 0.2/0.0
- U: A4 45/64, D4 29/29, D3 22/6.2, D2 4.0/0.2, D1 0.1/0.0

Implications for the probe:
1. Deconv = restoring D3/D2 detail the response attenuated; wire D2 is ~0.1-0.2% (nearly gone),
   so fine charge detail is LARGELY LOST in the wire signal -> caps achievable deconv R² (esp U).
2. Induction (U) spreads 11.5x vs collection (Y) 2.7x and is bipolar (S range [-588,227]) =>
   far more ill-posed; this is where cross-plane (U/V/Y) joint deconv must help vs per-wire Wiener.
3. Charge is SPARSER than wire (response disperses it) -> charge coeffs are sharper/more concentrated.
4. Charge & wire are in different units (eâ» vs ADC) -> probe target should be normalized per band.

---
## RESULTS (2026-06-14/15): probe + supervised limit

**Frozen MAE probe (linear, rep vs raw -> charge, per-band R²):**
- rep: overall 6.3% | A4 43.6  D4 -4.2  D3 -6.0  D2 -8.3
- raw: overall -0.1% | A4 7.6  D4 -5.2  D3 -2.2  D2 -0.7
=> denoising-MAE rep linearly exposes BULK charge (A4, beats raw 5.7x) but NOT detail.

**Coverage / correlation check:** charge nonzero at wire-token slots = A4 92/D4 94/D3 92/D2 82%
(NOT an alignment artifact); corr(clean-wire-coeff, charge-coeff) at same slot ~0 for detail
(A4 0.18, D4 0.04, D3 -0.10, D2 0.05). Same-slot correlation is the WRONG measure (deconv is
non-local).

**Fully-supervised end-to-end (wire->charge, all visible, 44M scratch, 240 train ev):**
peak ~5k steps: overall 65.6% | A4 82.1  D4 67.2  D3 66.0  D2 47.1.
=> DETAIL IS RECOVERABLE. Representation-limited, NOT information-limited; the frozen-probe
corr~0 was misleading because deconvolution is non-local + non-linear (inverse response over a
neighborhood). Architecture deconvolves ~as well as it denoises (~66% vs recon ~70%).

**BUT overfits** after ~5k steps (test overall 65.6->60.0, D2 47->33; A4 robust ~82). Deconv is
a harder/more-specific target than MAE recon (which didn't overfit at 1k ev). => deconv needs
more data and/or pretraining. Motivates the MAE-init arm (FM value = data/label efficiency).
Checkpoint: fm/ckpt_deconv_scratch.pt. Scripts: fm/probe.py, fm/deconv_train.py, fm/charge_cache.py.

---
## BREAKTHROUGH (2026-06-15): MAE pretraining transforms deconvolution

Supervised deconv (wire->charge, 240 train ev, per-band R²), scratch vs MAE-init:
| band | scratch @5k (peak) | MAE-init @5k | MAE-init @7.5k |
|---|--:|--:|--:|
| A4 | 82.1 | 92.9 | 93.1 |
| D4 | 67.2 | 89.2 | 90.0 |
| D3 | 66.0 | 87.6 | 88.4 |
| D2 | 47.1 | 60.7 | 56.9 |
| overall | 65.6 | 82.6 | 82.1 |

MAE-init = +17 overall, +22 on detail bands, AND overfits far less (scratch decayed 65.6->60
by 15k; MAE-init detail still climbing at 7.5k, only D2 mild). => THE foundation-model value
proposition, confirmed and large: denoising-MAE features transfer to deconvolution; detail is
representation-limited (recoverable to ~88-90%), not information-limited. Frozen LINEAR probe
fails on detail (rep A4 44, detail ~0) but FINE-TUNING the same pretrained net is excellent.
Lever for deconv: pretrain (MAE) -> fine-tune; train end-to-end, not frozen-probe.
ckpts: ckpt_deconv_mae_240.pt (240ev), ckpt_deconv_mae.pt (2000ev, running). Viz: fm/viz_recon.png.
Next: 2000-ev MAE-init (pretrain + more data) for best ceiling, esp D2.

## 2000-event MAE-init (best deconv ceiling) — 2026-06-15
MAE-init + 2000 events (8x data) @step4000 (run interrupted, ckpt saved): overall **86.5%** |
A4 94.1  D4 91.5  D3 90.5  D2 69.9. More data lifted D2 60->70 and detail to ~91% (still climbing).
Ladder: scratch 65.6 -> MAE-init@240ev 82.6 -> MAE-init@2000ev 86.5. Pretrain + data both help;
detail bands ~91% => deconvolution is highly recoverable end-to-end. ckpt_deconv_mae.pt (step4000).
Viz: fm/viz_recon.png (wire linthresh=2 ADC, charge linthresh=100 e-; ringing floor ~15-30 e- = waverec
artifact from sparse-coeff reconstruction). Run stopped ~step5500 (infra); deconv_train has no --resume.

## FINAL (2000-ev MAE-init, step 8000, converged) — 2026-06-15
overall **90.2%** | A4 96.1  D4 94.9  D3 93.6  D2 76.2 (D2 still climbing).
LADDER: scratch240 65.6 -> MAEinit240 82.6 -> MAEinit2000@4k 86.5 -> @8k 90.2.
=> deconvolution ~90% recoverable; detail D4/D3 ~94-95% (≈ bulk); D2 finest 76% (data-limited, lifts
with more events/steps). Frozen-probe failure was REPRESENTATION-limited, not info-limited. FM recipe
(MAE pretrain -> fine-tune + data) is the lever. deconv_train.py now has --resume (model+opt+step).
ckpt_deconv_mae.pt = step8000.

## PUSH (2026-06-15): more steps + more data
2000-ev MAE-init extended to 16k steps: 90.2(8k)->91.4(12k)->92.0(14k)->**92.2%(16k)** |
A4 97.1 D4 96.5 D3 95.3 D2 79.9. More steps help (no overfit, MAE-init). D2 76->80.
6000-ev MAE-init (16k steps) launched to push data-limited D2 further. ckpt_deconv_mae6k.pt.

## FINAL PUSH (6000-ev MAE-init, 16k steps) — 2026-06-15
overall **93.9%** | A4 97.7  D4 97.2  D3 96.3  D2 84.2.
LADDER: scratch240 65.6 -> MAEinit240 82.6 -> 2000ev16k 92.2 -> 6000ev16k 93.9.
More data lifts D2 (data-limited): 76(240)->80(2000)->84(6000); detail D4/D3 ~96-97% solved.
Deconvolution ~94% recoverable via MAE-pretrain->fine-tune + data. ckpt_deconv_mae6k.pt step16000.
Viz: fm/viz_recon.png (image space), fm/deconv_scatter.png (per-band pred-vs-true).

## P0 SCALE TEST (2026-06-16, NLL deconv, 6k events, converged 16k) — REVERSES earlier "scale is dead"
Reference=noisy input, Oracle=clean input (noise-free ceiling). per-band charge R²:
| config | overall | A4 | D4 | D3 | D2 |
|---|--:|--:|--:|--:|--:|
| d512/44M ref    | 89.1 | 97.3 | 94.9 | 93.7 | 70.6 |
| d512/44M oracle | 90.1 | 97.5 | 95.6 | 94.7 | 72.6 |
| d768/114M ref   | 93.9 | 98.5 | 98.2 | 97.1 | 81.6 |
| d768/114M oracle| 93.7 | 98.5 | 98.0 | 97.2 | 81.2 |
CONCLUSIONS (confound-free: converged, capacity varied, oracle ceiling, NLL):
1. SCALE HELPS: d512->d768 = +4.8 overall, +11 on D2 (the hard band). Earlier "info ceiling / scale won't
   help" was an artifact of fixed small model size (the audit's central criticism) -> FALSIFIED.
2. D2 was MODEL-capacity-limited, NOT response/information-limited: bigger model raised the NOISE-FREE
   oracle D2 ceiling 72.6->81.2. The info was there; 44M couldn't extract it.
3. NOISE not the limit at any scale: d768 ref(93.9) ~= oracle(93.7). Clean input buys ~0.
4. NLL adopted: calibrated (cov ~68/96); costs ~5pt R² at d512 (down-weights hard coeffs) but d768
   recovers it. Distributional metrics (NLL/calibration) tracked alongside R².
Bottleneck = MODEL CAPACITY (scale), not data-quantity, not noise, not objective. Next: keep scaling
(d768->bigger; joint d768@12k+), the curve has NOT flattened.

## PRETRAIN SCALING (2026-06-20, MAE d512, mask 0.75, proper dataloaders: 8 workers + pinned)
Answering the 3 pretrain questions (convergence / data+epochs / masking), confound-free.

### Q1+Q2 DATA SCALING at MATCHED COMPUTE (6k reused vs 20k fresh, masked var_expl):
| steps | 6k(reuse) | 20k(fresh) | gap | 6k epochs |
|--:|--:|--:|--:|--:|
| 24000 | 58.3 | 58.2 | ~0  | 5  |
| 32000 | 58.8 | 60.4 | +1.6| 6.7|
| 40000 | 60.8 | 61.9 | +1.1| 8.3|
| 64000 | 63.6 | 64.6 | +1.0| 13.3|
- <=5 epochs: reuse == fresh (data quantity IRRELEVANT).
- >5 epochs: fresh holds a STABLE ~+1pt lead (not widening). Data repetition is benign (13x reuse
  costs only ~1pt vs fresh).
- Compute is the big lever: +6pts from 24k->64k steps. Earlier "data-saturated" was UNDER-TESTED
  (fixed-step runs never passed 5 epochs); the real effect is small but real past 5 epochs.

### Q3 MASKING (downstream deconv R2, NLL, 6k, converged 16k):
| pretrain mask | deconv R2 | D2 | pretrain recon |
|--|--:|--:|--:|
| random | 88.8 | 72.2 | 58.3% |
| block/span | 59.9 | 39.3 | 23.3% |
- RANDOM >> BLOCK by 29pts downstream. Block (contiguous wire-slabs) destroys the LOCAL wire
  correlations the task needs -> weaker rep, transfers terribly, fine-tune can't recover.
- Random masking is near-optimal; the pretext is NOT an untapped lever.

### NET: pretrain is COMPUTE- and CAPACITY-bound. Data quantity ~1pt (minor), masking already
optimal (random). Levers that matter: model size (d512->d768 = +11 D2) and compute (steps).

## FLUCTUATION EVAL (2026-06-21, deconv_fluct.py, CPU) — gates the distributional head
var(pred)/var(true) per band (1.0=fluctuations preserved, <1=mean blurs them); CRPS dist vs point.
| model (R2) | A4 | D4 | D3 | D2 | mean vr | CRPS gain |
|--|--:|--:|--:|--:|--:|--|
| block-init (60%) | .628 | .425 | .419 | .287 | .440 | -27% |
| d512 (89%) | .892 | .855 | .848 | .617 | .803 | -28% |
| d768 (94%, best) | .902 | .898 | .890 | .716 | .852 | -27% |
FINDINGS: (1) mean-prediction under-disperses; variance-capture rises with training/scale
(.44->.80->.85) but does NOT reach 1.0 — best model still loses ~15% var overall, ~28% on D2.
(2) D2 always worst. (3) Gaussian-NLL CRPS beats point forecast by ~27% at every quality level.
=> distributional/flow head is WARRANTED (measured, not hypothesized); targets the scale-resistant
D2 variance shortfall. R2 hides this entirely.

## RoPE + AdaLN A/B (2026-06-21, MAE d512 6k, vs old mae6k=58.3%@24k) [@9600 partial]
- new per-axis RoPE vs old base=10000: 50.9% vs 50.8% @9600 -> NEUTRAL (positional waste wasn't
  binding; bitter-lesson/ViT-ablation confirmed: PE details barely matter). Keep (free correctness).
- AdaLN-Zero vs FiLM: 42.7% vs 50.9% @9600 -> AdaLN BEHIND (66M vs 44M, likely undertrain like FFNx16).
  Final @24k pending.

## FLOW-MATCHING HEAD A/B (2026-06-21, deconv, 6k, 16k steps) — flow head ADOPTED
Per-token flow-matching head (flow_head.py) vs NLL mean head. var-ratio=var(pred)/var(true).
| metric | NLL d512 | FLOW d512 | NLL d768 | FLOW d768 |
|--|--:|--:|--:|--:|
| R2 (point/sample-mean) | 89.1 | 94.1 | 93.9 | 95.3 |
| var-ratio (mean) | 0.80 | 0.968 | 0.85 | 0.972 |
| D2 R2 | 70.3 | 84 | 81.2 | 86 |
| D2 var-ratio | 0.62 | 0.91 | 0.72 | 0.92 |
| D2 CRPS | 0.121 | 0.105 | 0.095 | 0.096 |
WIN on every axis: (1) fluctuations recovered (var-ratio ~0.97 vs 0.80-0.85; D2 0.62->0.91);
(2) R2 UP not down (+5 at d512) — NLL baseline was crippled by Seitzer 1/sigma^2 pathology; flow
has none, recovers the lost R2 AND preserves fluctuations; (3) CRPS <=. Runtime: +18% train step
(596 vs 503ms); sampling k*steps linear (k=16,4steps=2s/event), tunable. => ADOPT flow head.
