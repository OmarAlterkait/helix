# Coherent noise in wavelet-coefficient space — structure & removal

Question: with NO coherent removal, decompose signal / coherent / intrinsic
separately (no thresholding), understand the per-level structure, and ask whether
the within-block coherent coefficients can be exploited to remove coherent in
coefficient space (a "helix in coefficient space"). Production transform: coif3 L4
periodization; deep-level study uses sym4 (supports L9).

Components (from `research/wire_denoise/common.py`):
- signal    = doraemon clean truth (noise-free, pedestal-subtracted)
- coherent  = `tools.coherent_noise`: per-64-wire-block waveform, neighbors
              anti-correlated via beta=0.15; spectrum 1/(1+f/20kHz)^0.75 (low-freq)
- intrinsic = per-wire independent colored+white (JAXTPC)

## 1. Within a block: coherent coefficients are EXACTLY identical (all levels)
`broadcast_to_wires` gives every wire in a block the *same* waveform; the DWT is
linear and per-wire, so the coherent coefficients are bit-identical across the 64
wires — confirmed `max|coeff-coeff_wire0| = 0.000e+00` at every level L1..L10, every
plane. Coherent is an exact rank-1 (constant-across-wires) common-mode per block.

## 2. Across blocks: beta "bleeding", confined to immediate neighbors
Cross-block correlation of the coherent coeff vector: lag-1 = **-0.29**, lag-2 ~ +0.02,
lag>=3 ~ 0 — at every level. Matches the model exactly: waveforms[g]=base[g]-beta(base[g-1]
+base[g+1]) gives corr1 = -2beta/(1+2beta^2) = -0.287, corr2 = +beta^2/(1+2beta^2)=+0.02.
=> The within-block estimate already captures the full waveforms[g] (bleed included);
no deconvolution needed. The bled component is NOT recoverable separately (base[g] are
independent) — bleeding helps nothing for removal, just sets the local correlation.

## 3. Where each component lives (per-level energy, fig5)
- signal: coarse (Y A4 78%, kurtosis ~3000 -> sparse peaky pulses). Induction V splits A4/D4.
- coherent: coarse, Gaussian (A4 58%, kurtosis ~3). MAD-sigma A4=7.6 >> intrinsic 2.6
  -> coherent dominates the noise scale exactly where signal lives -> inflates per-band
  thresholds (the leakage source).
- intrinsic: mid-band (D3 30%), flatter.

## 4. The separability crossover (fig7) — and why MORE LEVELS DON'T HELP
Per level, ratio = (signal common-mode)/(coherent common-mode):
- Y (collection): >1 (signal swamps) for all coarse bands D5..A9 (A9=4.7); <1 only D1-D3.
- U (induction):  mid-bands D6-D8 swamped (1.3-1.7); deepest A9/D9 <1 (bipolar = less DC).
- V (induction):  <1 at EVERY level (max 0.67) — bipolar signal least common-mode.
k-sigma masking lowers all, but cannot clear Y/U coarse bands (signal genuinely common-mode).

**9-level answer: deeper does NOT change the picture.** Splitting the low-frequency
region into more bands just yields more coarse bands each swamped by signal — the
coarsest A9 is the *worst* for Y (4.7). The crossover is a physical property of broad,
multi-wire pulses, not the level count.

## 5. Removal experiments (6 events, F0 on signal, coh_left = RMS leftover coherent)
WITH SIGNAL (realistic), Y:
| method                | F0     | noise_rms | coh_left |
|-----------------------|--------|-----------|----------|
| raw (no removal)      | 0.948  | 3.01      | 2.54     |
| **helix (sample)**    | **0.962** | **1.67** | **0.57** |
| coeff median (all)    | 0.850  | 2.15      | 3.28     |
| coeff masked (detail) | 0.928  | 2.60      | 2.30     |
| helix + coeff detail  | 0.937  | 1.78      | 1.25     |
| cone-of-influence     | 0.852  | 2.14      | 3.23 (worse at L7/L9) |

NOISE-ONLY (signal=0), Y:
| method            | noise_rms | coh_left |
|-------------------|-----------|----------|
| helix             | 1.64      | 0.43     |
| **coeff masked**  | **1.60**  | **0.21** |

## 6. SMART removal — level-aware common-mode gating BEATS helix (the result)
Naive/cone removal failed because (a) they subtract EVERY block common-mode, injecting
the signal common-mode, and (b) one k-sigma threshold ignores the ~10x per-level
amplitude differences. The fix (smart.py), per band:
  1. across wires (within block): robust common-mode m[blk,pos] = k-sigma masked mean.
  2. across blocks (LEVEL-AWARE): coherent scale sigma_coh = MAD over (blk,pos) of m
     (robust to sparse signal). This is the per-level amplitude the naive mask ignored.
  3. GATE (reverse threshold): coherent_est = m where |m| < k*sigma_coh, else 0.
     Coherent is SMALL+DENSE (present every block ~sigma_coh) -> kept & subtracted;
     signal is LARGE+SPARSE common-mode -> dropped -> never injected. Subtract; IDWT.

Pure coefficient-space (no multi-pass, no temporal dilation), fuses with sparsify.
6 events, k=4 (k=3 maximizes F0; k=4 minimizes noise/coh_left):
| plane | raw F0/nrms/cohL | helix F0/nrms/cohL | **smart F0/nrms/cohL** |
|-------|------------------|--------------------|------------------------|
| Y | 0.948 / 3.01 / 2.54 | 0.962 / 1.67 / 0.57 | **0.965 / 1.61 / 0.35** |
| U | 0.869 / 3.04 / 2.55 | 0.890 / 1.79 / 0.80 | **0.892 / 1.67 / 0.53** (k3: 0.899/1.71/0.61) |
| V | 0.860 / 3.04 / 2.55 | 0.891 / 1.78 / 0.72 | **0.904 / 1.66 / 0.32** |
Smart wins on ALL THREE metrics on ALL THREE planes; induction (U/V) gains most F0.

## 6b. Coefficient count — does smart removal compress better? (50 events, count.py)
Each path -> SAME production sparsify (per-band sigma + threshold-approx, coif3 L4, k=1).
Fewer kept coeffs at equal/higher F0 = better. (raw keeps FEWER but at worse F0: un-removed
coherent inflates the coarse-band sigma -> over-thresholds real signal.)
| plane | helix kept/F0 | smart kept/F0 (best k @ matched F0) | gain |
|-------|---------------|-------------------------------------|------|
| Y | 50543 / 0.9566 | 47371 / 0.9570 (k4) | **-6.3% coeffs, +F0** (200x vs 177x) |
| V | 53818 / 0.8835 | 46276 / 0.8877 (k3.5) | **-14.0% coeffs, +0.004 F0** (231x vs 187x) |
| U | 62776 / 0.8972 | 61949 / 0.8968 (k3) | tie (-1.3%, -0.0004 F0) |
Pushing k higher trades F0 for more compression (Y k4 -6%, U/V k3.5 -13/-14%). U is the
hard plane: its bipolar A4 signal common-mode ~ coherent scale (ratio_mk 0.99) so no
within-block method separates them there.

## 6c. HEADROOM — smart already matches the no-coherent ORACLE on count (oracle.py)
The decisive diagnostic. oracle = sparsify(signal+intrinsic), coherent NEVER added (the
absolute target). 30 events:
| plane | smart kept/F0 | ORACLE kept/F0 (no coherent) | gap |
|-------|---------------|------------------------------|-----|
| Y | 49514 / 0.9574 | 49173 / 0.9654 | kept +0.7%, F0 -0.008 |
| U | 54213 / 0.8910 | 54762 / 0.9344 | kept -1.0%, F0 -0.043 |
| V | 45872 / 0.8876 | 45909 / 0.9129 | kept -0.1%, F0 -0.025 |
=> smart's KEPT-COUNT EQUALS the no-coherent oracle (within +/-1%). For compression there
is ZERO headroom: smart removes coherent so completely you keep exactly as many coeffs as
if coherent never existed. The only residual gap is F0 (fidelity), from coherent/signal
OVERLAP in dense regions (worst on U's bipolar A4) — fundamentally unseparable, NOT a
method deficiency. (idealmask with a perfect signal mask but no gate over-subtracts ->
ignore it; the no-coherent oracle is the clean bound.)
k-frontier (kfine.py, 20 ev): F0 PEAKS at k~2.5-3 (coherent gone, signal untouched) then
declines as the gate eats signal; count drops monotonically with k. Best F0>=helix points:
Y k3.5 (-4%), V k3.5 (-15.5%, +0.005 F0), U k3 (tie, F0-limited).

## 6d. Recovering coherent BEHIND signal — detect-then-estimate (de), the F0 lever
Goal shift: count is maxed (=oracle); the residual gap to oracle is F0 (coherent left in
signal regions). Human insight: coherent is constant across wires + sharp block boundaries,
so it's readable off the signal-free wires even behind a track.
Understanding (explore_sc.py, 1 event): per-tick occupancy is LOW (busiest Y block max 17/64);
at a signal tick the clean wires form a tight dense cluster at coherent, signal wires are far
outliers -> masked-mean recovers coherent to ~floor 0.21 EVEN BEHIND SIGNAL. (But truly DENSE
blocks exist: a track parallel to the wires hits up to 62/64 -> few clean wires.)
UPPER BOUND (oracle clean-wire mask): sample-space removal with the TRUE signal mask BEATS
smart on every plane (Y F0 0.969, U 0.912, V 0.899) -> the lever is clean-wire DETECTION.
de method (detect_estimate.py): robust DETECTION baseline (smart's coherent estimate) ->
flag signal wires per (block,tick) vs baseline (k-sigma, dilated) -> ESTIMATE coherent as
masked-mean over clean wires, temporally interp the few all-signal ticks -> subtract from all.
KEY KNOB: the detection threshold. ksig=3 (default) was too conservative -> missed weak
bipolar signal -> contaminated estimate -> hurt U. Lowering to ksig=1.5 catches it; the extra
false-flagged clean wires (~13% at 1.5sigma) are harmless (plenty remain).
12-event result, de at ksig=1.5 (F0_recon / coh_left / kept):
| plane | smart | de_k1.5 | oracle |
|-------|-------|---------|--------|
| Y | 0.9587 / 0.384 / 37558 | **0.9638 / 0.261 / 37569** | 0.9658 / 0 / 37225 |
| U | 0.8864 / 0.537 / 42900 | 0.8866 / 0.552 / 42900 (tie) | 0.9333 / 0 / 43421 |
| V | 0.8855 / 0.357 / 36539 | **0.8898 / 0.336 / 36604** | 0.9120 / 0 / 36417 |
=> de_k1.5 is >= smart on EVERY plane: big near-ORACLE win on COLLECTION (Y, F0_recon
0.964 vs 0.959, ~half the leftover coherent, same #coeffs), modest win on V, TIE on U.
Never worse than smart. (A single-event U gain at ksig~1.5-2 did NOT generalize over 12
events -> U detection genuinely hard; its large oracle headroom 0.933 stays OPEN.)
Tried, didn't help U / didn't generalize: wire-axis dilation, median estimate, clamp-to-smart,
low-pass smoothing (coherent NOT band-limited, 1/f^0.75), interp-only (variance), no-alpha
full subtraction (dense-block garbage). RECOMMENDATION: de (ksig 1.5) replaces smart as the
default (>= on all planes); U's residual headroom needs better bipolar-signal detection (open).
Figs 13 explore, 14 hist, 15 de-recover, 16 de-summary.

## 6e. Pushing U/V toward the oracle — de2 (induction-capable), and the fundamental limit
Goal: approach the no-coherent oracle on induction (U 0.919, V 0.908 F0_recon vs smart
0.883/0.889). The lever (diagnosis induction.py): clean-wire DETECTION; de's amplitude
threshold over-flagged isolated INTRINSIC spikes (precision 0.25). Fix = hysteresis (Canny):
strong SEEDS grown along the contiguous track -> precision 0.86, recall 0.95; iterate
(refine baseline) + temporal dilation (catch bipolar tails). "de2" = iterate hysteresis-amp,
klo0.7 khi3.5 dilate15, 4 iters. Optional +joint refinement (estimate signal per-wire via
wavelet-denoise, subtract from ALL wires, common-mode of residual) adds a hair.
de3 = de2 + joint refinement (estimate signal per-wire by wavelet-denoise, subtract from ALL
wires, take common-mode -> sidesteps signal-majority). 30-event F0_recon / coh_left / kept:
| plane | smart | de3 (FINAL) | oracle |
|-------|-------|-------------|--------|
| Y | 0.9567 / 0.430 / 47682 | **0.9602 / 0.341 / 47271** | 0.9643 / 0.238 / 47457 |
| U | 0.8874 / 0.611 / 53057 | **0.8957 / 0.549 / 53312** | 0.9201 / 0.359 / 54144 |
| V | 0.8887 / 0.376 / 43421 | **0.8940 / 0.351 / 43463** | 0.9069 / 0.257 / 43272 |
=> de3 >= smart on F0 ALL planes, lower coh_left, equal-or-fewer coeffs. V within ~0.013 of
oracle, Y within ~0.004, U within ~0.024.

FINAL METHOD = de3 CLAMPED to smart's coherent +/- 4 ADC (induction.final_removal). The clamp
prevents over-removal in dense regions (keeps behind-track recovery within +/-4) -> IMPROVES U
further AND makes it per-event SAFE (>= smart on 100% Y, 100% U, 95% V events). 30-ev F0_recon:
Y 0.9603 / U 0.8981 / V 0.8953 (smart 0.9567/0.8874/0.8887; oracle 0.9643/0.9201/0.9069).
Gains vs smart: Y +0.0036, U +0.0107, V +0.0066; gaps to oracle Y 0.004, V 0.012, U 0.022.
Lowest coh_left, equal/fewer coeffs. This is the best safe induction-capable coherent remover.
FUNDAMENTAL LIMIT (figs 17,18): de2 MATCHES the oracle in non-dense blocks (occ<32: 0.21 vs
0.21); the entire residual gap is DENSE blocks where a track runs PARALLEL to the wires for
>50 ticks (U has such runs up to 167t). There the signal is the MAJORITY of wires with no
temporal anchor (coherent corr length ~100t) -> the coherent is statistically unrecoverable:
EVERY robust estimator (median, trimmed, mode, IRLS, consensus, run-consensus) locks onto the
majority signal; matched-filter (shape-based, baseline-free) detection has too-low recall to
fix it. The oracle only wins there by KNOWING the true mask. So U's residual 0.029 is an
information-theoretic floor, not a method deficiency. de2 captures all the recoverable headroom.

## 6f. ABLATIONS — simplest thing that works (ablate.py, ablate2.py, fig19)
F0-gain vs smart, 15 events, ablating each component:
- JOINT step: de2_clamp ~= de3_clamp everywhere (U +0.0099 vs +0.0104) -> DROP joint (pure overhead).
- CLAMP (clip estimate to smart +/-4): the cheap KEY ingredient for induction (U: hysteresis
  +0.0068 -> +0.0099). Harmless/tiny-negative for collection. Always include for U/V.
- DETECTION: collection (Y) needs only a PLAIN amplitude threshold (+0.0059, BEST) — hysteresis
  & clamp slightly HURT Y (easy detection; clamp pulls back Y's already-good estimate). Induction
  needs HYSTERESIS (U plain-thr+clamp +0.0050 vs hysteresis+clamp +0.0099).
- ITERATION: i4 > i2 for U (+0.0099 vs +0.0081); ~2-4 iters.
=> SIMPLEST PER-PLANE (best): collection = plain-amplitude-threshold iterate, NO clamp; induction
   = hysteresis iterate + clamp. SIMPLEST UNIVERSAL (near-best, one method) = hysteresis iterate
   + clamp ("de2_clamp", joint dropped): Y +0.0035 / U +0.0099 / V +0.0053. The joint step and
   the de3 complexity buy nothing -> the production-worthy method is de2_clamp.

## 6g. MEASURED limits for U/V (not assumed) — gls.py, template_mca.py, validate_gls.py, fig20
Decomposed the gap with two oracles (12 ev): oracle_pt = true-mask + per-tick masked-mean
(the reachable mask ceiling); nocoh = sparsify(signal+intrinsic), coherent never added (true ceiling).
| plane | de3c | oracle_pt | nocoh | DETECTION gap | FLOOR gap |
|-------|------|-----------|-------|---------------|-----------|
| Y | 0.9613 | 0.9645 | 0.9667 | 0.003 | 0.002 |
| U | 0.8917 | 0.9201 | 0.9317 | 0.028 | 0.012 |
| V | 0.8951 | 0.9075 | 0.9119 | 0.012 | 0.004 |
- DETECTION gap (de3c->oracle_pt): RECOVERABLE in principle (the true mask reaches it) but hard:
  dense-region clean-wire ID requires detecting signal at SNR~1 (|signal|~2 ADC ~ clean-wire noise),
  where signal and clean wires statistically overlap. Tried to close it: signal-template matched
  filter (recall OR precision, not both: rec 0.95/prec 0.05 or pruned rec 0.85/prec 0.75 -> F0~de3c);
  MCA template signal-subtraction (WORSE, single template incomplete); dense-MF union (tie). Caps ~de3c.
- FLOOR gap (oracle_pt->nocoh): FUNDAMENTAL — any removal injects sigma_int/sqrt(n_clean) common-mode
  noise; even the true mask cannot reach nocoh. ~0.004 (V) / 0.012 (U).
- COHERENT TEMPLATE (its covariance/spectrum) does NOT help: GLS/Wiener with the coherent prior is
  WORSE than the per-tick masked-mean (oracle_GLS < oracle_pt on every plane) — the coherent has a
  SHORT correlation (~5-10 ticks; autocorr<0.5 by lag5, <0.1 by lag30) and the per-tick estimate is
  already high-SNR, so the prior over-smooths and biases. So the coherent's "consistency" gives nothing.
NET: how close we get = de3c (Y 0.961/U 0.892/V 0.895). Reachable ceiling oracle_pt (Y 0.965/U 0.920/
V 0.908); the recoverable part is the detection gap (U 0.028, V 0.012) limited by SNR~1 detectability;
the rest (floor) is fundamental. A learned detector is the only untried lever for the detection gap.

## 6h. The kappa knob — trade coefficients for F0 (kappa_knob.py, fig21)
Q: reach the (oracle) F0 by keeping more noise? YES via sparsify kappa. de3c residual coherent
in dense regions (a) inflates per-band sigma -> over-thresholds signal (recoverable by lower
kappa) and (b) rides on the kept signal coeffs (NOT recoverable by kappa). 10-ev:
- V: de3c F0 reaches the oracle's production F0 at kappa~0.8 (0.905 vs oracle 0.907, ~2x coeffs);
  kappa0.6 exceeds it (0.910, ~7x). V loss is mostly (a) -> knob recovers it.
- U: lower kappa recovers +0.008 (0.894->0.902 @k0.6) then PLATEAUS below oracle 0.919 -> channel
  (b) caps it (residual coherent on signal in long parallel-track regions). Knob ~1/3 of gap.
Cost: global kappa keeps noise EVERYWHERE (2-7x coeffs). Smarter = spatially/locally-adaptive
threshold (lower only in residual-coherent dense regions) or sigma from clean regions (un-inflated)
-> same F0 recovery at a fraction of the coeff cost. (untested refinement)

## 6i. LOCAL keep-near-signal knob = wavelet HYSTERESIS sparsify (hyst_sparsify.py, fig22)
Global low-kappa recovers F0 but keeps noise EVERYWHERE (5-6x coeffs). Local fix: per band,
SEEDS=|c|>k_seed*t, GROW=|c|>k_grow*t (k_grow<k_seed=1); keep connected (wire x position)
components of GROW containing a SEED -> recovers the signal's weak shoulders near tracks, drops
ISOLATED weak coeffs (noise far from signal). 8-ev (de3c-cleaned), best k_grow=0.5:
| plane | std kappa1 | HYST grow0.5 | global kappa0.6 |
|-------|-----------|--------------|-----------------|
| U | 0.8970 / 43k | 0.9077 / 70k | 0.9045 / 277k |
| V | 0.8928 / 37k | 0.9094 / 65k | 0.9068 / 272k |
| Y | 0.9595 / 39k | 0.9660 / 56k | 0.9654 / 191k |
=> hysteresis-grow0.5 DOMINATES global kappa0.6: HIGHER F0 (U +0.011, V +0.017, Y +0.007 over
std) at ~4x FEWER coeffs than global kappa (70k vs 277k). k_grow=0.35 adds a hair more F0 but
4x the coeffs (171k) -> 0.5 is the sweet spot; k_grow<=0.2 percolates (blows up). Coefficient-
efficient knob to buy back the channel-(a) F0 loss (over-thresholded signal near tracks). General
(helps any cleaned image). [NOTE: earlier table with grow0.35=55k was a pre-completion misread.]

## 6j. VALID frontier comparison: hysteresis vs standard sparsify (frontier_compare.py, fig23)
The "4x fewer than global-kappa" claim was INVALID (global-kappa is a strawman). Correct comparison
= standard sparsify (kappa-sweep) vs hysteresis (grow-sweep) on the SAME de3c image, matched coeffs:
10-ev, matched coeffs:
U: ~45k std 0.8945 / hyst(g0.95,46k) 0.8960 (+0.0015); ~66k std(k0.85,65k) 0.8988 / hyst(g0.55) 0.9049 (+0.006).
V: ~38k std 0.8962 / hyst 0.8983 (+0.0021); ~58k std(k0.85) 0.9027 / hyst(g0.55) 0.9122 (+0.0095).
MATCHED F0: hyst uses ~2-3x FEWER coeffs than standard, e.g. V F0~0.908: hyst 49k vs std 132k (2.7x);
U F0~0.902: hyst ~57k vs std saturates ~0.902 only near 139k+. Standard SATURATES (U caps ~0.902 even
at 139k — its extra low-kappa coeffs are noise reconstructing onto signal); hysteresis keeps climbing.
=> hysteresis is a strictly better frontier. At the production budget (~45k) the gain is small (~+0.001-
0.002); it grows to +0.006-0.0095 at ~+30-45% coeffs, OR equivalently ~2-3x fewer coeffs at matched F0
(valid vs the PRIOR standard sparsify, not the global-kappa strawman). de3c removal remains the larger lever.

## 6k. CLEAN build-up ablation across planes (ablation_full.py, fig24) — the per-plane understanding
F0_recon, 12 ev, add one component at a time (-> standard sparsify k=1):
| step | Y | U | V |
|------|---|---|---|
| raw | 0.9370 | 0.8397 | 0.8360 |
| smart | 0.9586 | 0.8828 | 0.8890 |
| helix | 0.9586 | 0.8946 | 0.8872 |
| amp1 (plain detect, 1 pass) | 0.9623 | 0.8812 | 0.8901 |
| amp4 (iterate) | 0.9631 | 0.8667 | 0.8880 |
| hyst4 (hysteresis detect) | 0.9634 | 0.8818 | 0.8916 |
| hyst4_d15 (+dilation = de2) | 0.9638 | 0.8879 | 0.8973 |
| +clamp (= de2_clamp) | 0.9635 | 0.8926 | 0.8975 |
| +joint (= de3c) | 0.9633 | 0.8926 | 0.8974 |
PER-PLANE STRUCTURE:
- smart (coeff gate) is the big jump from raw (+0.02 Y / +0.04 U / +0.05 V); the baseline.
- COLLECTION (Y): simple sample-space PLAIN amplitude detection (amp1) already ~optimal (+0.004);
  iterate/hysteresis/dilation add ~+0.0015; clamp/joint slightly HURT. Y is easy (large unipolar signal).
- INDUCTION (U,V): PLAIN amplitude detection HURTS (U amp4 0.867, worse than smart 0.883! bipolar signal
  mis-detected; iterating plain makes it worse). HYSTERESIS is ESSENTIAL (recovers U 0.867->0.882).
  Dilation helps both (+0.006). Clamp helps U (+0.005), neutral V.
- JOINT adds ~0 on every plane -> DROP it. de2_clamp == de3c.
- U IS THE HARDEST: even full de2_clamp (0.8926) does NOT beat HELIX (0.8946) on U; the sample-space
  approach only matches helix there. For V, de2 clearly beats helix (0.8973 vs 0.8872). For Y all ~equal.
=> Algorithm by plane: Y -> smart+plain-detect (simplest); V -> de2 (hysteresis+dilation); U -> helix or
   de2_clamp (tie). Hysteresis-vs-plain is the key plane-dependent switch (induction needs hysteresis).

## 6l. FOUR-metric ablation (ablation_metrics.py, fig25) — F0, kept, noise IN vs OUT
12-ev, cleaned-image RMS residual on-signal (nz_in) and off-signal (nz_out):
| plane | method | F0 | kept | nz_in | nz_out |
|-------|--------|----|------|-------|--------|
| Y | smart | .9586 | 40k | 2.32 | 1.62 |
| Y | de2_clamp | .9635 | 40k | 1.93 | 1.60 |
| U | smart | .8828 | 45k | 3.01 | 1.67 |
| U | helix | .8946 | 55k | 2.72 | 1.79 |
| U | de2_clamp | .8926 | 45k | 2.83 | 1.66 |
| V | smart | .8890 | 37k | 2.38 | 1.66 |
| V | helix | .8872 | 46k | 2.50 | 1.78 |
| V | de2_clamp | .8975 | 37k | 2.15 | 1.66 |
KEY: nz_OUT hits the intrinsic FLOOR (~1.6) for ALL removers -> off-signal coherent removal is SOLVED;
the discriminator is nz_IN (coherent residual ON the signal, from dense regions). Findings:
- smart GATES dense regions -> leaves ~all coherent ON U signal (nz_in 3.01 ~ raw 3.06!); F0 hides this.
- de2_clamp lowers nz_in best on Y (1.93) and V (2.15) AND uses fewest coeffs; on U (2.83) it trails
  helix (2.72) but helix costs +22% coeffs (55k) and worse nz_out (1.79).
- plain-amp on U INJECTS on-signal noise (amp4 nz_in 3.53 > raw). joint ~0 on all 4 metrics -> drop.
- V: de2_clamp dominates helix on ALL FOUR. U: F0/nz_in tie helix at fewer coeffs vs more. Y: de2_clamp best.

## 6m. THOROUGH knob sweep of de2_clamp (sweep_knob.py, tuned_config.py, figs 26) — what's needed
One-at-a-time sweep, 4 metrics, 10 ev, per plane. Sensitivity (F0 range over sweep) + best value:
| knob | sensitive? | best / behaviour |
|------|-----------|------------------|
| kgate (smart gate) | HIGH (U +0.007) | U/V prefer 3 (removes more coherent, lower nz_in); Y 4. BIGGEST U lever. |
| detector | HIGH (plane) | Y: plain ('ampthr') best; U/V: 'amp+mf' (hysteresis+MF) >= hysteresis > plain (plain HURTS induction). |
| dilate | MED (plane) | Y 1-5, V 11-21, U 21-31 (induction wants more). |
| baseline | HIGH for U/V | smart ESSENTIAL (interp/median -> nz_in 3.0 on U); Y indifferent. |
| n_iter | MED (U) | U gains to 6; Y/V plateau ~4. |
| clamp | MED (U) | optimum 3-4 ALL planes; no-clamp worse (U -0.0036). |
| klo (grow) | floor | <=0.5 PERCOLATES (kept explodes, nz_out up); use 0.7-0.9 (Y tolerates 1.1). |
| khi (seed) | LOW | flat 2.5-4.5 -> 3.5. |
| minc | LOW | 2-4 (low). reducer LOW: mean (best nz_out). |
INSENSITIVE (keep default): khi, minc, reducer. SENSITIVE & PLANE-DEPENDENT: kgate, detector, dilate, n_iter.
COMBINED tuned-per-plane (gains STACK), 14 ev (F0/kept/nz_in/nz_out):
  Y plain,dil3,klo0.9,kg4   : 0.9630 / 39k / 1.82 / 1.60  (default 0.9609; oracle 0.9644; ~at oracle, fewer coeffs)
  U amp+mf,kg3,dil25,it6,cl3 : 0.8991 / 48k / 2.56 / 1.69  (default 0.8958; helix 0.8950; +6% coeffs, beats helix)
  V amp+mf,kg3,dil15,cl3     : 0.8947 / 38k / 2.14 / 1.67  (default 0.8927; helix 0.8844)
=> tuning buys +0.002-0.003 F0 and lower nz_in (the right target); off-signal nz_out is already ~floor.
The efficient tuning surface = {kgate, detector, dilate, n_iter} (induction); everything else stays default.
Single biggest lever for U = kgate=3. NOTE 'amp+mf' (hysteresis UNION matched-filter) marginally best for U/V.

## 6n. REALITY CHECK — do the small F0 gaps matter? (fig27) -> SIMPLIFY to smart
1-F0 = fraction of signal CHARGE mis-reconstructed (L1). 20-ev (1-F0 %; per-event std):
  Y: smart 4.2% / de2cl 3.9% / floor(no-coh) 3.4%   (de2-smart +0.3%, scatter +-0.5%)
  U: smart 11.0% / de2cl 10.0% / floor 6.5%          (de2-smart +1.0%, scatter +-1.3%)
  V: smart 11.1% / de2cl 10.5% / floor 8.5%          (de2-smart +0.6%, scatter +-1.3%)
Grounding: (1) de2_clamp over smart = ~1% charge (U), tuning +0.2-0.3% -> BELOW LArTPC charge/
calorimetric resolution (several %); won't move energy/dE/dx/PID. (2) the gain is WITHIN the
per-event F0 scatter (+-1.3% U/V > the ~1% mean gain). (3) most of 1-F0 is the IRREDUCIBLE floor
(intrinsic+sparsify: 6.5-8.5% of the ~10-11%); coherent residual is the small remainder and de2
shaves ~1% off it. Physics-relevant axes already solved by smart: off-signal floor (hit-finding,
nz_out~1.6 for all) and COMPRESSION (= no-coherent oracle count). de2's edge is only on-signal
charge precision (nz_in), the least critical axis.
=> SIMPLIFICATION: use SMART ALONE for production (count-optimal, one DWT pass, within ~1% charge
of the whole de2_clamp stack). DROP the sample-space detect/iterate/hysteresis/dilation/clamp/joint
machinery and per-plane tuning -> sub-1% gains, below resolution + inside event scatter. Keep
de2_clamp only for a dedicated charge-precision study / clean displays (set once, don't per-plane-tune).

## 6o. CONSOLIDATED knob sensitivity, ALL 4 metrics (fig28, knob_consolidate.py, 12 ev)
Range each OAT knob induces, Y/U/V (F0 milli | kept count | nz_in ADC | nz_out ADC):
  kgate    0.6/15.5/2.3 | 1323/1872/1443 | .06/.41/.06 | ~0     <- MASTER lever (sets smart base; U-only)
  dilate   3.1/9.2/4.0  | 1034/1758/1226 | .25/.28/.04 | ~0     <- biggest COEFF lever, all planes
  detector 2.8/7.5/3.2  |  336/467/106   | .23/.19/.04 | ~0     <- per-plane SEED switch (Y=amp,U/V=hyst/mf)
  minc     1.8/2.5/3.2  |  422/1050/671  | .14/.08/.11 | ~0     mild
  n_iter   0.6/7.7/4.1  |   16/275/224   | .05/.21/.09 | 0      induction-only; ZERO for Y; 4 saturates
  klo      2.1/2.0/0.5  |  295/875/528   | .18/.07/.08 | ~0     floor only (>=0.7 or mask percolates)
  baseline 0.6/7.9/2.5  |  123/174/63    | .05/.23/.05 | 0      binary: USE SMART (range=how bad interp/median)
  clamp    0.4/3.6/1.5  |  690/1286/584  | .05/.16/.08 | 0      flat over [2-8]; swing is the no-clamp endpoint
  reducer  0.2/0.7/1.3  |  336/466/392   | ~.02        | .03    INERT (mean fine; only nz_out mover, wrong way)
  khi      0.1/1.1/1.6  |  136/176/150   | ~.03        | 0      LEAST important; flat [2.5-4.5]
DEF: Y .9617/40203/2.09/1.61  U .8940/45789/2.77/1.66  V .8969/37234/2.25/1.66.
TIERS: T1(the algorithm)=kgate,dilate,detector. T2(set-once/induction)=n_iter,baseline,minc,klo,clamp.
T3(inert)=reducer,khi.
CROSS-CUTTING: (1) nz_out range <=0.03 ADC for EVERY knob -> off-signal floor untouchable by tuning
(hit-finding noise is locked regardless). (2) Y insensitive to all but detector/dilate -> tuning lives on U.
Consistent with 6n: kgate's whole sweep spans ~1.5% charge -> knobs reshuffle within a sub-resolution band.

## 6p. SMART ADDITIONS tested (check_smart_v2.py, 12 ev): keep only the reliability output
#1 NOISE-ANCHORED sigma (anchor sigma_coh on low-occupancy positions): NO-OP. Numerically identical
  to legacy global MAD (U k3.0: 54786 vs 54606 kept; F0 0.894 vs 0.893). smart's median sigma was
  ALREADY robust to sparse signal (median dominated by noise-only positions even on dense U).
  -> reverted default to 'global'; option kept for record. Side result: a single kgate~3.0-3.5
  already works across Y/U/V (per-plane kgate tuning was never needed).
#2 RELIABILITY MAP (signal-occupancy, sm.smart_removal(...,return_info=True) -> info['occ_map']):
  WORKS, ~free (gate already computes clean-wire counts). fig29: lights up exactly the dense track
  cores (long parallel tracks) = where smart is least reliable AND where de2's edge lives. Dense
  fraction (occ>0.5) ~1-2% of (block,tick) for all planes. KEEP.
#3 MASK_DILATE (dilate signal-outlier mask along position before masked-mean; cheap coeff-space
  port of de2's spatial mask): WEAK. md2 recovers only ~+0.001/+0.0016 F0 (U/V) AND COSTS coeffs
  (47098->47519 U) -- wrong direction. de2_clamp gets +0.005 F0 at FEWER coeffs (45822). So the cheap
  coefficient-space spatial port does NOT capture the U/V edge; the edge genuinely needs de2's
  sample-space detect->masked-estimate->clamp (better clean-wire ID, not excluding more wires).
NET: kept ONLY the reliability map (return_info -> occ_map; purely additive, default smart path
byte-identical to before). #1 and #3 code REMOVED from smart.py (not carried as dead options);
the check script was deleted (it called the removed params). smart.py = original method + occ_map.
MINIMAL way to add the U/V edge: cheap port fails -> keep smart as anchor + run de2_clamp ONLY on
the ~2% reliability-flagged dense (block,tick) regions (#2 as the trigger) -> ~all the de2 edge at
~2% of the extra cost. Still sub-1%/sub-resolution (6n) -> opt-in for charge-precision use only.

## 6q. DE2+ KEEP/REMOVE audit (systematic; what actually worked)
KEPT = the validated de2_clamp opt-in (induction.py): smart_baseline; detectors mask_amp(Y plain,
seed 'ampthr') + mask_hysteresis(U/V, seed 'amp'); estimate (masked-MEAN) + dilate_t; iterate
(detect->estimate loop); final_removal = de2 + clamp (NOW joint-free); scorers full_metrics/
mask_quality; mask_true (oracle ref); diag.
REMOVED NOW (dead-ends, zero dependents): _run_consensus + iterate_runaware (run-aware consensus
for long dense runs -- locks onto majority signal, the fundamental U limit, did NOT recover it);
irls (Tukey IRLS -- same majority-lock failure); rem_metrics (dead stub); the de3/joint step inside
final_removal (ablated to ZERO -> de2_clamp == de3_clamp).
REMOVED (full cleanup, DONE + verified): joint_removal fn; mask_mf/mask_mf_multi/_dog/mask_energy;
the median/trim branch of _reduce (estimate now mean-only, no reducer arg); sweep_knob trimmed
(detector -> [amp,ampthr]; reducer knob dropped); DELETED 13 probe scripts (gls, validate_gls,
template_mca, template_detect, ablate, ablation_full, ablation_metrics, validate_clamp,
validate_final, kappa_knob, frontier_compare, panels_de3, tuned_config). hyst_sparsify KEPT and
re-pointed to de2_clamp (was using joint). All remaining .py compile; final_removal F0 unchanged
(U 0.902); no dangling refs. induction.py = smart_baseline + mask_amp/mask_hysteresis + estimate
(mean) + dilate_t + iterate + final_removal(de2+clamp) + full_metrics/mask_quality + mask_true/diag.

## 7. Why it works / earlier-method post-mortem
- It exploits BOTH structural facts: within-block identity (rank-1 common-mode across
  wires) AND the coherent's small-dense vs signal's large-sparse contrast across blocks,
  with per-level scaling. It does NOT need to recover coherent inside signal-saturated
  coarse blocks (those it leaves, preserving F0) — yet still removes coherent everywhere
  else (most of the plane), beating helix's leftover.
- Noise-only check: coeff common-mode estimator beats helix (coh_left 0.21 vs 0.43).
- Failed predecessors: naive median/masked subtraction (F0 0.85, injects signal),
  cone-of-influence time+level mask (worse, and worse with depth — coarse coeffs lose
  time-locality, signal detector blind to common-mode signal). 9 levels do NOT help.

## 8. Pushing the estimate — iteration / cross-block / multi-level (all tested, none beat mag)
The magnitude gate ("mag") is near the theoretical floor, so refinements give ~nothing:
- THEORETICAL FLOOR: the per-block estimate averages 64 wires -> intrinsic-mean residual
  sigma_int/sqrt(64) ~ 1.58/8 = 0.20 ADC = the measured noise-only coh_left 0.21. Per band
  coherent dominates that residual 8-24x (high SNR), so spectral/Wiener filtering of the
  common-mode cannot help; neighbor averaging cannot help (neighbors have DIFFERENT coherent).
  smart's clean-block estimate is already AT this floor; its 0.35 (Y) gap is residual coherent
  in SIGNAL blocks, which overlaps signal positions (kept anyway) -> little count benefit.
- iteration (iter2, 2-pass mask refine): BIT-IDENTICAL to 1-pass — the signal-wire mask is
  unchanged by coherent removal, so it converges immediately.
- trimmed/interquartile-mean common-mode: WORSE (discards clean coherent wires).
- cross-block (per-position gate from MAD-over-blocks + beta linear-fill of signal blocks):
  slightly WORSE; neighbor coherent predictability is low (lag1 -0.29) so beta-fill recovers ~nothing.
- multi-level cross-scale persistence gating: WORSE (over-flags broad pulse tails -> leaves coherent).
- DEEP decomposition (sym8 L8 / coif3 L7) + cross-level per-wire signal detection (flag a
  wire as signal by persistence of large coeffs across scales, exclude it from the common
  mode): WORSE on all planes (Y coh 0.99-1.19 vs 0.43). Over-flags wires (intrinsic
  fluctuations aggregate across levels -> fewer clean wires -> noisier estimate).
  => MORE LEVELS do not help the estimate (consistent with the sec-4 "9 levels" finding).
Conclusion: the level-aware magnitude gate already optimally uses the structure
(small+dense=coherent, large+sparse=signal, per-level scale). It is near-optimal.
Scripts: push.py (iter/trim/xblock), multilevel.py (cross-scale), deep.py (cross-level per-wire).

## 8b. More levels / other wavelets (level_wav_sweep.py, count2.py)
Removal quality vs (wavelet, level), k=4, 8 events:
- Y, V: L4 OPTIMAL; deeper monotonically worse (coh_left up, F0 down). Wavelet barely
  matters at L4 (coif3 ~ sym8 ~ bior4.4; sym8 marginally best for V).
- U: small genuine gain from L6 — db4 L6 / bior4.4 L6 reach F0 0.8984 vs coif3 L4 0.8936
  (+0.005), similar coh_left. Deeper spreads the bipolar signal so the gate preserves it.
End-to-end count, sparsify @ coif3-L4 vs bior4.4-L8 on cleaned images (fixed kappa=1):
- bior4.4-L8 sparsify keeps MORE coeffs (different operating point: higher F0, more kept),
  so it does NOT beat coif3-L4 at production kappa. The prior "bior4.4-L8 best" was
  frontier-matched (varying kappa), not a fixed-kappa win.
- smart removal beats helix at BOTH sparsify wavelets. The win is the removal, not the transform.
Bottom line: coif3 L4 + smart level-aware gated removal is the operating point; deeper
levels and alternate wavelets do not improve the coefficient count.

Figures: fig1 within/across, fig2 signal-adjacent, fig3 per-level bands, fig4 coeff maps,
fig5 level energy, fig6 cross-block corr, fig7 crossover-per-level, fig8 full-plane maps
both cases, fig9 smart-vs-helix bars, fig10 smart panel (stripes removed, signal kept),
fig11 coeff-count bars, fig12 F0-vs-coeff frontier (smart reaches oracle count, dominates helix).
FINAL: smart level-aware gated removal at k~3.5 (Y,V)/k~3 (U) is count-optimal (= no-coherent
oracle) and beats helix; remaining F0 gap is the irreducible coherent/signal overlap. Done.
Scripts: probe.py, figs.py, per_level.py, crossover.py, planes_levels.py, coeff_removal.py,
multiscale_mask.py, smart.py (the method), smart_figs.py.
