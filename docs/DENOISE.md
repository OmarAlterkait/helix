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

## Results

(filled in as runs complete)
