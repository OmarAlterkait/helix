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
