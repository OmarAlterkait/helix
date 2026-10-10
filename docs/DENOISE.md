# Denoising: noisy coefficients -> pre-response charge

The goal: a model that takes the noisy, sparsified wavelet coefficients of an
event -- exactly what the foundation model sees -- and returns the charge that
arrived at each wire before the detector response: denoised AND deconvolved.
Two questions: how low a detection floor can such a model reach when trained
with every label we have (the supervised ceiling), and how few labelled events a
pretrained encoder needs to get close to it (label efficiency).

## Target and grid

The target is the simulation's `hits` (pre-response charge per wire and tick,
per particle group), summed over groups onto the fine grid of
`helix.probe.resolution`: 2 wires x 16 ticks per cell, predicted as
`log1p(q / Q0)` with `Q0 = 2043.26` (median positive cell charge of the floor
evaluation's probe-train events).

Why not clean coefficients: they still contain the field response (bipolar
induction lobes) and the wavelet basis -- not physical charge, and not what the
floor measures. Why not single pixels: hits and sensor pixels do not map 1:1 --
the response lags the charge by 2 (U) to 6-7 (V, Y) ticks and induction spreads
it over neighbouring wires; a 16-tick cell absorbs that, and it is the grid every
floor number so far is scored on, so every model here is directly comparable to
every probe result.

## Model

Encoder: the FM encoder (`SerialFMModel.encode`), all tokens visible. Head
(`helix.model.denoise.CellHead`): for each queried cell, the features of the token
covering it in each of the 4 bands (LayerNorm, project to 256, zeros if absent)
plus the cell's offset inside those tokens, Fourier-encoded
(`helix.probe.resolution.cell_inputs`, shared with the frozen probe), then an MLP
(1024 -> 512 -> 1). Loss: MSE on `log1p(q/Q0)` -- the probe's loss. So frozen
probe, fine-tuned and from-scratch differ only in what is trained.

Query cells per event follow the evaluation's row recipe: every charge-carrying
cell (cap 20k), as many cells near them (+-8 cells, restricted to cells under a
kept coefficient), and as many random cells under kept coefficients -- the
noise-only cells that teach the model to stay at zero, which is where the floor
is lost. Cells no token covers are predicted 0 by construction, as in evaluation.

## Data

Truth: `scripts/build_denoise_truth.py` decodes `hits` with the floor truth's own
decoder (`helix.probe.truth.decode_hits_plane`) for every event of the 8-run
corpus: 790 shards, ~38k charge-carrying cells per event, 71 GB at
`$HELIX_SCRATCH/denoise_truth`. Splits: each run's `holdout.json` (train 0.95, val
0.03, probe 0.02, blake2b of the simulation identity -- the FM's own split;
`scripts/write_holdout.py` reproduces run 1's file identically and wrote the
other seven). Training uses `train` only (~150k events); validation 32-64 `val`
events of run 1; evaluation the probe split's test events (truth_v2, 128 events)
via `scripts/eval_denoise.py`, scored exactly as `scripts/eval_resolution.py`.

## Plan

1. LR sweep, 3k steps, one node each: from scratch {3e-4, 1e-3, 3e-3, 1e-2};
   fine-tuned from the noisy-recipe FM (nz_pw16_s1) {1e-4, 3e-4, 1e-3, 3e-3}.
2. Supervised ceiling: d768 from scratch on all ~150k train events, 4 nodes (B=16),
   ~40k steps (~4 epochs); then the same initialised from the FM (does pretraining
   still help with all labels?); width (d1536) if time allows.
3. Label efficiency: N = 64, 256, 1k, 4k, 16k labelled events, from scratch vs
   fine-tuned from the FM, best checkpoint by val MSE. The frozen probe (260 events,
   nz_pw16/_s1) is one more point.

## Results (2026-10-07)

All scored on the floor evaluation's 128 probe-split test events (truth_v2), the
same cells, windows and Q0 as every probe number; interval widths and seed spread
as in docs/SCIENCE.md s10-11 (1%-FPR efficiency ~ +-0.03-0.05).

**LR.** From scratch (3k steps, B=4): 1e-2 best (val MSE 0.060), 3e-3 0.063, 1e-3
0.083, 3e-4 0.143. Fine-tuned from nz_pw16_s1 (encoder LR; head 1e-2): 3e-3 best
(0.024), 1e-3 0.026, 3e-4 0.030, 1e-4 0.033.

**Sampling matters more than anything else found so far.** The first ceiling run
(M1) sampled empty cells only under kept coefficients; at 6k steps it had per-cell
presence AUC 0.93 but window floor AUC 0.675 (<0.1 MeV) -- the floor's noise windows
score the MAX over cells of noise tokens, most of them coefficient-free, a cell type
the model never saw. Adding empty cells drawn uniformly inside token footprints
(`cov_per_sig=2`, M1b) gave floor AUC 0.862 at the same step.

**Supervised ceiling** (M1b: d768 from scratch, all ~150k train events, 40k steps x
16 events, lr 1e-2 cosine): map r 0.995, presence AUC 0.948, floor AUC 0.886 /
0.963 (<0.1 / 0.1-0.2 MeV), 5%-FPR efficiency 0.893, 1%-FPR efficiency 0.224 / 0.689
/ 0.984 (<0.1 / 0.1-0.2 / 0.2-0.5 MeV), localisation 1.60 wires / 6.6 ticks. The
frozen probe on the same noisy-recipe FM (260 events) gives, for its two seeds
nz_pw16 / nz_pw16_s1: map r 0.944 / 0.946, floor AUC 0.829-0.875 / 0.912-0.940, 5%-FPR
efficiency 0.712 / 0.799, 1%-FPR efficiency at 0.1-0.2 MeV 0.447 / 0.577, localisation
~2.0 wires / 11.5 ticks, and 1%-FPR efficiency below 0.1 MeV 0.219 / 0.347.
Supervision wins on every metric except that faintest bin, where it is within the
probe's seed range (0.224). (Against the clean-target probes of s10 -- 0.40-0.44 there
-- it would look like a deficit; that is the wrong baseline for a noisy-input model.)
In charge units, regression to log1p(q/Q0) does put faint deposits close to noise
(window-score medians 360 vs the clean probe's 586), which is what the presence-head
ablation tests.

**Label efficiency** (4k steps x 4 events for every arm, best val checkpoint; the
ceiling saw 640k events, so from-scratch arms are compute- as well as label-limited):

| labels | fine-tuned: eff 1% (0.1-0.2) / eff 5% / floor AUC <0.1 / map r | from scratch: same |
|---|---|---|
| 64 | 0.641 / 0.813 / 0.865 / 0.974 | 0.359 / 0.605 / 0.779 / 0.940 |
| 256 | 0.678 / 0.875 / 0.885 / 0.982 | 0.445 / 0.727 / 0.844 / 0.961 |
| 1,024 | 0.710 / 0.899 / 0.891 / 0.988 | 0.528 / 0.831 / 0.869 / 0.977 |
| 4,096 | 0.643 / 0.903 / 0.890 / 0.991 | 0.547 / 0.811 / 0.867 / 0.983 |
| 16,384 | 0.656 / 0.904 / 0.890 / 0.991 | 0.499 / 0.758 / 0.858 / 0.984 |
| ceiling (150k, scratch) | 0.689 / 0.893 / 0.886 / 0.995 | |

With the pretrained encoder, 64 labelled events reach 93% of the ceiling's 1%-FPR
efficiency and 256-1,024 match it on every floor metric; from scratch, the same
counts reach 52-77%. Fine-tuned runs with few labels also keep more of the faintest
bin (1%-FPR <0.1 MeV 0.29-0.32 at N <= 1,024, against 0.20-0.22 at N >= 4,096 and the
ceiling's 0.224): more labels pull them toward plain charge regression.

**M3** (whole noise windows + presence head, lr 1e-2) produced a non-finite gradient
at step 1,150 and trained on NaN weights; the training script now skips such steps
and stops after 20 in a row (isolated non-finite gradients recur occasionally --
about one per 600 steps in one ablation, and once at step 10 of a run without
windows -- and are absorbed).

**Floor-objective ablation** (fine-tuned, N=1,024, 4k steps; two baseline seeds):

| variant | windows scored by | eff 1% FPR 0.1-0.2 / <0.1 MeV | eff 5% | floor AUC <0.1 / 0.1-0.2 | loc wires / ticks |
|---|---|---|---|---|---|
| baseline, seeds 0 / 1 | charge | 0.710 / 0.307, 0.641 / 0.219 | 0.899 / 0.897 | 0.891 / 0.962 | 1.56-1.59 / 6.4 |
| + whole noise windows (48/event) | charge | **0.784 / 0.413** | **0.917** | **0.903 / 0.966** | **1.50 / 6.1** |
| + presence head | charge | 0.675 / 0.266 | 0.868 | 0.880 / 0.953 | 1.75 / 6.6 |
| + presence head | presence | 0.729 / 0.373 | 0.874 | 0.889 / 0.954 | -- |
| + both | charge | 0.763 / 0.425 | 0.898 | 0.887 / 0.960 | 1.67 / 6.9 |
| + both | presence | 0.768 / 0.499 | 0.890 | 0.899 / 0.958 | -- |

Training on every cell of charge-free windows -- the unit the floor scores -- is the
largest improvement found: +0.11 (0.1-0.2 MeV) and +0.15 (<0.1 MeV) in 1%-FPR
efficiency over the baseline mean, beyond the seed spread (0.07, 0.09), with every
other metric better too. The presence head helps only the faintest bin, only when
windows are scored by presence. With windows, 1,024 labelled events on the
pretrained encoder (0.784) beat the windowless full-label ceiling (0.689); the
ceiling is being re-run with windows from scratch (M2w) and fine-tuned (M4w).

**Other diagnostics.** Top-scoring noise windows concentrate on U planes (22 of 25 at
M1b 12k steps). The hits-to-coefficient time offset is small on every plane (A4, in
the coefficient frame: U -4, V -4, Y -9 ticks), so the cell mapping is not the cause.

## First principles: noise vs hits, and the bound (`scripts/noise_vs_hits.py`)

128 test events, 6,265 isolated deposits; numbers on the 3,595 whose surroundings
(+-8 wires, +-64 ticks) hold <= 25% foreign charge, measured on each deposit's own
wires/ticks of the CLEAN sensor image.

**Units.** `hits` is electrons: 20,400-20,600 e-/MeV deposited on every plane
(42,400/MeV at W = 23.6 eV x recombination ~0.48). Q0 = 2,043 e- is ~0.1 MeV.

**Gain and noise.** Peak clean signal per 1,000 e-: U 2.0, V 1.65, Y 4.5 ADC; the
forward model's incoherent noise is 2.0 ADC/tick, so ENC ~ 1,000 (U), 1,230 (V),
440 (Y) e- -- MicroBooNE-like. The optimal (whitened matched filter, exact noise
spectrum, coherent noise perfectly removed, shape known) SNR is 1 at 460 / 540 /
250 e- of deposit charge. An ideal detector searching a window needs 4.4-4.5 sigma
for 1% FPR, so its 50% point is ~2,000-2,400 e- (~0.1-0.12 MeV) on U/V and
~1,100 e- (~0.055 MeV) on Y.

**The stored sensor is zero-suppressed.** It is integer ADC with every |v| < 2
removed, before noise is added. 36% of the deposits below 0.1 MeV (42% U, 47% V,
18% Y) have NO clean signal left: no detector can find them, yet they are in the
floor's <0.1 MeV denominator. Ideal 1%-FPR efficiency there is 0.22 (U/V/Y 0.13 /
0.09 / 0.45); at 0.1-0.2 MeV 0.85 (0.85 / 0.74 / 0.98).

**The model against the bound** (ab_ft1024_win, 1%-FPR efficiency by the deposit's
optimal SNR): no signal 0.30 (ideal 0), SNR < 3 0.29 (0.02), 3-5 0.37 (0.37), 5-7
0.54 (0.90), 7-10 0.87 (1.00), >= 10 1.00 (1.00). It reaches the bound above SNR
10 and is ~1.5 sigma short at threshold. The 0.29-0.30 it scores on deposits with no
signal at all is not sensitivity: noise windows are drawn >= 16 wires / 128 ticks
from any charge, isolated deposits sit near activity, and the model predicts a
haze of 100-300 e- per cell around activity -- so the floor metric credits the haze
as detections. Training rarely sees those cells: `near` cells are restricted to
kept coefficients and noise windows to charge-free surroundings.

### Why 3-10 sigma is lost: stage by stage (`scripts/snr_stages.py`)

The corpus chain replayed on the clean sensor of 48 test events with fresh noise
(emulated kept coefficients / corpus = 1.000 median over 264 planes), 775 isolated
deposits; known-location 1%-FPR efficiency of a matched filter on each stage (an
upper bound for any detector reading it), against the model's window efficiency:

| SNR | raw | gate | gate, no D1 | corpus (kappa 1, no D1) | flat 4 sigma | kappa 0.75 | kappa 0.5 | model |
|---|---|---|---|---|---|---|---|---|
| 3-5 | 0.92 | 0.91 | 0.74 | **0.19** | 0.10 | 0.34 | 0.53 | 0.34 |
| 5-7 | 0.99 | 0.99 | 0.97 | **0.48** | 0.32 | 0.68 | 0.83 | 0.50 |
| 7-10 | 1.00 | 1.00 | 1.00 | **0.86** | 0.70 | 0.95 | 1.00 | 0.87 |

The coherent gate loses nothing; dropping D1 loses a little at threshold; the
sparsification threshold (kappa x MAD sigma x sqrt(2 ln n_band) = 3.35-3.9 sigma per
band) erases faint deposits whose energy is spread over coefficients each below it
-- median signal response at SNR 3-5 goes 4.2 -> 0.0. The model is at the limit of
its input (above it at 3-5: context/haze). Kept coefficients: kappa 0.75 2.1x, 0.5
9.0x, flat 4 sigma 0.9x the corpus. The corpus is unchanged; this is measurement.

### Evaluation and training near activity

`scripts/dump_near_windows.py` writes, beside truth_v2, charge-free windows WITH
charge within 16 wires / 128 ticks (`helix.probe.resolution.near_windows`, kind
`bgn`): the surroundings the floor's own noise windows exclude. `eval_denoise.py
--near <dir> --annot <noise_vs_hits.npz>` adds `near_fpr_at_far{1,5}pct`,
`near_eff{1,5}pct_<bin>` (threshold set near activity) and
`snr_eff1pct_{far,near}_<snr bin>` (incl. `nosignal`); every existing key is
unchanged. Training: `--near-any-per-sig` (cells within +-8 of charge, coefficient
or not) and `--win-near-per-event` (every cell of near-activity windows).

### Results on the near-activity evaluation (2026-10-07 evening)

42% of the floor's isolated deposits have foreign charge in the very next cell and
93% within 8 cells; only 7% sit where the floor's noise windows are drawn. Charge-free
windows near activity hold no clean signal (94-100%), yet the models score them up
to 2-3k e- (99th pct U/V/Y 1,948 / 2,361 / 2,881 vs 554 / 237 / 227 far away): a
context prior, not a response to waveforms. With the 1%-FPR threshold set near
activity (`near_eff1pct`; a distance-matched threshold gives the same: 0.21 vs
0.15 for ab_ft1024_win at 0.1-0.2 MeV):

| model | near eff 1% 0.1-0.2 [16-84%] | <0.1 | by SNR 7-10 / 10-15 | far eff 0.1-0.2 (old metric) |
|---|---|---|---|---|
| every earlier model (ab_*, le_ft_*, M1b, M2w) | 0.13-0.16 | <= 0.006 | 0.25-0.32 / 0.52-0.59 | 0.53-0.78 |
| + near-activity sampling, seeds 0 / 1 | 0.298 / 0.299 [0.26-0.34] | 0.011 | 0.42 / 0.78 | 0.70 / 0.76 |
| + near sampling + presence head, scored by presence | **0.465** [0.45-0.48] | 0.055 | 0.68 / **0.977** | 0.72 |

Full labels do not fix it (M1b 0.14, M2w 0.13): the haze is a sampling/objective
problem, not a data-quantity one. The old metric ranks the new arms LOWER -- it
credited the haze. Near-activity eff of signal-free deposits is 0.002-0.006, as
it must be at 1% FPR.

### Label efficiency on the near-activity evaluation (le3: near sampling + presence head)

Every arm 4k steps x 4 events, fine-tuned from nz_pw16_s1 (lr 3e-3) or from scratch
(1e-2), whole noise windows + near-activity cells (1/sig) and windows (24/event),
presence head; windows scored by presence. 1%-FPR efficiency, threshold set near
activity (charge-score rows in brackets; le2 = the same grid without near sampling
or presence, charge-scored):

| labels | fine-tuned 0.1-0.2 MeV | scratch 0.1-0.2 | fine-tuned SNR 10-15 | scratch SNR 10-15 | le2 fine-tuned 0.1-0.2 |
|---|---|---|---|---|---|
| 64 | **0.428** [0.278] | 0.253 [0.118] | 0.954 | 0.647 | 0.172 |
| 256 | 0.450 [0.337] | 0.309 [0.169] | 0.957 | 0.812 | 0.114 |
| 1,024 | 0.465 [0.330] | 0.364 [0.273] | 0.977 | 0.934 | 0.177 |
| 4,096 | 0.485 [0.426] | 0.407 [0.335] | 0.990 | 0.974 | 0.188 |
| 16,384 | **0.498** [0.395] | 0.405 [0.348] | 0.993 | 0.974 | 0.175 |

Fine-tuned with 64 labelled events beats from-scratch with 16,384 (0.428 vs 0.405):
pretraining is worth more than 256x the labels on this task. Seeds reproduce
(1,024 fine-tuned: 0.465 / 0.463; nr_ft1024_pres repeats le3_ft_N1024 exactly).
Doubling near sampling (nr2: 2/sig, 48 windows) gives 0.482 at 1,024. The full-label
windowed runs without near sampling stay at 0.13-0.25 (M2w 0.134, M4w 0.255). Map r
of presence-scored rows (~0.65) is a probability against charge and not meaningful;
the charge rows keep 0.97-0.99.

**Supervised ceiling with the near-activity recipe** (M4n: fine-tuned from
nz_pw16_s1, all ~150k train events, 40k steps x 16, near sampling + presence head;
near-activity threshold, presence-scored): 0.1-0.2 MeV **0.558** [0.549-0.577],
<0.1 MeV 0.065, 0.2-0.5 MeV 0.975; by deposit SNR 5-7 0.32, 7-10 **0.81**, 10-15
0.98; charge-head map r 0.995. Against the matched-filter bound on the corpus input
(known location, no window search): 0.48 at SNR 5-7, 0.86 at 7-10 -- the ceiling is
close to what the sparsified coefficients still hold. M2n (the same from scratch)
is the remaining comparison.
M2n (the same recipe from scratch, all labels, 40k steps): 0.1-0.2 MeV 0.524
[0.509-0.538], SNR 5-7 0.27, 7-10 0.79, 10-15 0.99 -- below M4n's 0.558
[0.549-0.577]: pretraining still helps with every label, by ~0.03.

### A real classical detector: whitened matched-filter bank with window search (`scripts/mf_detector.py`)

The simulation's fitted 2D response (baselines/uboone_sp) convolved with 9 charge
boxes (1-4 wires x 4-48 ticks), whitened by the exact noise spectrum; a window
scores the max z over templates and positions. 48 test events, 2,384 deposits;
1%-FPR efficiency, threshold from far | near-activity noise windows:

| input | 0.1-0.2 MeV | SNR 5-7 | SNR 7-10 | threshold z far / near |
|---|---|---|---|---|
| no neighbours: deposits' own charge x response, float, + incoherent | 0.68 | 0.53 | 0.98 | 5.6 / 5.6 |
| + stored like the sensor (zero-suppressed) | 0.55 | 0.37 | 0.89 | 5.6 |
| + coherent noise + gate | 0.52 | 0.32 | 0.86 | 5.5 |
| + corpus threshold, no D1 | 0.41 / 0.36 | 0.20 / 0.12 | 0.64 / 0.48 | 4.2 / 4.7 |
| real planes (neighbours), raw + incoherent | 0.32 / 0.02 | 0.14 / 0.01 | 0.29 / 0.01 | 10.8 / 260 |
| real planes, corpus input | 0.34 / 0.02 | 0.15 / 0.01 | 0.33 / 0.01 | 9.3 / 262 |
| M4n (real corpus input) | 0.81 / **0.57** | 0.67 / 0.33 | 0.91 / **0.82** | -- |

Without neighbours the classical loss is the sim's zero suppression (-0.13) and the
corpus threshold (-0.11); the gate costs ~0. With real neighbours a matched filter
collapses: other charge's response doubles the threshold even far from activity
and raises it 50x near it -- the job MicroBooNE's ROI / 2D deconvolution machinery
exists for. M4n, on real data near activity, beats the matched filter on a
neighbour-FREE corpus-type input (0.57 vs 0.36-0.41) and matches it on the
uncompressed gated input (0.52); it stays below the known-location bound (SNR 5-7
0.48, 7-10 0.86 on corpus input). The synthetic deposits use the same fitted kernel
as the templates, which favours the matched filter.

### Output model: hurdle loss, cell-resolution decoder, presence gating (2026-10-09)

Fast protocol (fine-tuned, 1,024 labels, 4k steps, near-activity recipe, presence
head), scored with --near/--annot. `_pres` = windows scored by presence; `_gated` =
the charge map kept only where presence > 0.5. haze = fraction of truth-empty map
rows above 0.05 Q0; res68 / med = 68% half-width and median of pred/true on 2-16k e-
map rows; loc = eval localisation (wire / tick) for 0.2-0.5 MeV deposits.

| model | near eff 1% 0.1-0.2 (presence) | charge-scored 0.1-0.2 | SNR 7-10 (presence) | haze | res68 | med | loc |
|---|---|---|---|---|---|---|---|
| baseline: MLP head (le3_ft_N1024), charge map | 0.465 [0.454-0.486] | 0.330 | 0.686 | 0.204 | 0.482 | 0.862 | 0.96 / 4.19 |
| baseline, gated | | 0.330 | | 0.084 | 0.483 | 0.862 | 0.25 / 1.32 |
| hurdle loss (charge on charged cells), gated | 0.459 | 0.282 | 0.680 | 0.080 | 0.420 | 0.868 | 0.27 / 1.36 |
| decoder | 0.472 [0.458-0.489] | **0.413** | **0.732** | 0.240 | 0.486 | 0.863 | 0.97 / 3.97 |
| decoder, gated | | 0.419 | | 0.077 | 0.486 | 0.863 | **0.19 / 0.92** |
| decoder + hurdle, gated (seeds 0 / 1) | 0.467 / 0.467 | 0.307 / 0.295 | 0.720 / 0.705 | 0.077 / 0.081 | **0.429 / 0.434** | 0.885 / 0.898 | 0.26 / 1.1 |

- Gating the charge map by presence removes ~60% of the haze on empty map rows and
  fixes localisation (0.96 / 4.2 -> 0.19-0.27 wire / 0.9-1.4 ticks; the tuned
  classical chain: 0.17 / 1.5): the haze, not the representation, made the charge
  head localise badly.
- The decoder makes the charge map itself a far better detector (0.330 -> 0.413 at
  0.1-0.2 MeV, SNR 10-15 0.82 -> 0.91) and gives the sharpest gated map.
- The hurdle loss tightens per-cell charge (res68 0.48 -> 0.42-0.43, median 0.86 ->
  0.87-0.90) but its gated map drops faint detections.
- No variant moves presence-scored detection (0.459-0.472, within the interval):
  the faint floor is set by the input, as the stage replay showed.

### Full-label output models (M5d decoder, M5dc decoder + hurdle; 40k steps x 16, as M4n)

Trained through the OST-61 overlay (identical data; scripts/study/denoise_full.sh),
scored with --near/--annot/--q0. Detection near activity at 1% FPR, 0.1-0.2 MeV;
haze / res68 / med / loc as in the ablation above.

| model | presence-scored | charge map | gated map | SNR 7-10 (pres.) | haze (gated) | res68 | med | loc (gated) |
|---|---|---|---|---|---|---|---|---|
| M4n (MLP head) | 0.558 [0.549-0.577] | 0.420 | 0.420 | 0.812 | 0.162 -> 0.037 | 0.211 | 0.964 | 0.13 / 0.59 |
| **M5d (decoder)** | **0.592 [0.579-0.607]** | **0.564** | **0.572** | **0.846** | 0.162 -> 0.037 | 0.198 | 0.956 | 0.15 / 0.63 |
| M5dc (decoder + hurdle) | 0.582 [0.569-0.598] | -- | 0.429 | 0.828 | 0.038 | **0.180** | **0.970** | 0.20 / 0.80 |

At full scale the decoder is the first change to move the faint floor (+0.034,
intervals disjoint) and makes the charge map nearly as good a detector as the
presence head (0.420 -> 0.564). Presence gating alone delivers the haze (0.16 ->
0.04) and localisation (0.82 / 3.4 -> 0.13 / 0.6 wire / tick; tuned classical 0.17 /
1.5) gains, for M4n as for M5d. The hurdle loss buys the best per-cell charge
(res68 0.18, median 0.97) at the cost of faint detections in its gated map.
Recommended product: M5d, detection by presence, charge map gated by presence.

### Dense rerun with gated outputs, and the decoder's charge cap (fixed: M6d / M6dc)

baselines/uboone_sp/dense/results/dense2.md (all 128 test events, every cell):
presence gating fixes the faint-region over-estimate (16x8-cell regions with 1-4k /
4-16k e-: raw 3.0x / 1.8x -> M5dc gated 0.94 / 1.00; classical 0.36 / 0.56) and cuts
dense-region haze 8-10x (empty cells > 100 e-: 0.20-0.43 -> 0.03-0.11; classical
0.07-0.12 above 1M e- of neighbourhood charge, cleaner only below 100k), at the cost
of 0.75% of all true charge zeroed (classical tuned 1.47%), almost all in sparse
cells < 2k e-. Every dense-activity win survives (moves <= 0.01). The decoder's
window-level detection gain does not appear at the cell level (same-density 1%-FPR
efficiency equal to M4n within 0.01).

The run found that CellDecoder's output (LayerNorm -> Linear) capped the charge
logit: M5d / M5dc saturated at 775k / 853k e- per cell. Fixed by a linear readout of
the residual stream (87b4caa) and retrained (M6d, M6dc; same recipe). Eval metrics are
unchanged within intervals (presence 0.1-0.2 MeV: M6d 0.591, M6dc 0.579; gated haze
0.036-0.037; res68 0.206 / 0.183), and bright cells are now right: median pred/true
1.00 / 1.00 at 0.2-1M e- and 0.97 / 0.98 above 1M (M4n 0.98 / 0.93), maximum
prediction 2.2M e-. Recommended product: M6d (better or tied on detection, gated-map
detection 0.566 vs 0.416 and localisation 0.16 / 0.60 vs 0.22 / 0.84; M6dc only on
per-cell resolution 0.183 vs 0.206), detection by presence, charge gated. The
calorimetry comparison (dense rerun) is pending for M6d vs M6dc.
