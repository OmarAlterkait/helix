# M7dc high-charge error: test plan (written BEFORE running; predictions fixed in advance)

Claims from m7_diagnostics.md to be tested, not assumed. Each test states what each outcome means.
Events: floor events ev260-307 (48 test events with floor.py inversions) unless stated; calibration
events ev000-063 minus 41 for anything fitted; training events for the train-set test.

## T0. Independent recomputation (own code, from raw files; no m7diag_* code)
- T0a alignment: correlate M7dc calo with the floor's complete truth at wire shifts -2..2 and time
  shifts -3..3 cells. Pass: maximum at (0, 0) on every plane. Fail: every comparison so far is misaligned.
- T0b per truth bin (16-64k, 64-256k, >=256k), per plane: median and hw68 of pred/truth for M7dc calo
  (bf16 dump + products_maps with params_M7dc.json) and the corpus Tikhonov maps at ALL three lambdas
  (no lambda choice). Pass: agent's numbers reproduced within sampling error (Y: 0.037 / 0.033 / 0.039
  vs 0.040 / 0.0175 / 0.0086).
- T0c the same with cells selected by the INVERSION's value and by the MODEL's value instead of the truth
  (selection on truth is common to both, but check that the conclusion does not depend on the binning).

## T1. What the model receives (plumbing)
- T1a every kept corpus coefficient (band < n_bands) appears in the token set with its exact value
  (assemble -> denormalize round trip; counts per event). Fail = the model sees less than the inversion.
- T1b the dtype of the token values at the model's first layer under the eval's bf16 autocast (hook).
- T1c input quantisation: run the corpus Tikhonov inversion on coefficients round-tripped
  value -> arcsinh(v/sigma) -> bf16 -> sinh*sigma. Prediction if input precision limits the model:
  the inversion's hw68 at >= 64k rises to ~0.03-0.04. If it stays at ~0.009-0.018, input quantisation
  is not the cause.
- T1d floor input = model input: the floor's corpus image uses bands A4-D2 (D1 dropped). Confirm it is
  built from the same shard/event and the same bands the model tokenizes (n_bands from the checkpoint).

## T2. The model as a function
- T2a gain response: scale every coefficient value by (1+e), e in {-0.02, -0.01, +0.005, +0.01, +0.02, +0.05};
  per bright cell the change in y should be log((Q0 + q(1+e)) / (Q0 + q)) (~e at high q). Report the
  mean slope dy/dlog(1+e) and its spread per truth bin. A spread comparable to 0.035/e... i.e. jitter
  of the order of the error floor => the map is not smooth at the percent level. A linear inversion
  has slope 1 and zero spread.
- T2b fp32 vs bf16 per truth bin (whole model fp32), recomputed from the agent's fp32 maps with own code.
- T2c local sensitivity: add tiny independent perturbations (1e-3 relative) to the input values,
  5 repeats; spread of the output y per bright cell. Measures how much of the per-cell error could be
  numerical jitter rather than a smooth bias.

## T3. Fit vs generalisation
- T3a train-set error: M7dc on 48 TRAINING events (from its own training split), complete truth from the
  training labels (denoise_truth_complete), per truth bin hw68. Prediction: if train ~= test (~0.035)
  the model underfits high charge (capacity / optimisation / objective), if train << test it is
  generalisation.
- T3b training-scale trend: hw68 per truth bin on the floor events for models trained with less:
  fast-protocol arms (1,024 labels, 4k steps: lb_dec_ct, sp_dec_c25, sp_dec_cond_c25) vs M7d/M7dc
  (40k x 16). Flat at high charge across a 150x compute range => not optimisation-limited.
- T3c training support above 64k / 256k / 1M: fraction of training charged cells, and the share of the
  training loss they carry.

## T4. Top end (> 1 M e-)
- T4a T3a's train events above 1 M: is the model low on cells it was trained on?
- T4b log-space slope of y_pred vs y_truth on unsaturated cells per decade, test and train.

## T5. Presence calibration in the faint bins
- T5a reconstruct the training-row sampler on calibration events (firstp_sampling machinery) and
  compute the train/deploy odds ratio conditioned on (plane, pdens, distance to the nearest charged
  cell). Prediction if position-based sampling breaks the single offset: the OR varies with distance
  within a pdens stratum by more than the 0.03-0.05 miscalibration requires.
- T5b apply the distance-conditioned OR (deployable distance = to the nearest cell with predicted
  charge, not truth) on test: does the 102-500 e- truth/pred move from 1.12 toward 1?

## T6. M7d (non-hurdle) conditional charge 5.6% low
- T6a decompose: raw y on truly charged cells vs log1p(Tf/Q0); the Jensen factor per bin; the p_cal/p
  factor. Which term carries the 5.6%.

## Layers, bins, compression (written before running, 2026-10-13)
- L1 heads on the frozen PRETRAINED FM at layer k = 3, 6, 9, 12 (fresh M7dc-shape CellDecoder, charge-only, 2500 x 4
  events, as P2; P2 = same head on the FINE-TUNED encoder, layer 12: Y 0.0386 / 0.0374 / 0.0506). If deeper pretrained
  layers discard amplitude precision, an earlier k is better than k = 12.
- L2 nonlinear (MLP) probe of each token's OWN bright coefficient values (arcsinh) from layer-k features, k = 0..12,
  pretrained and fine-tuned encoder; fitted on calibration events, scored on test events. Bin-quantisation level
  (uniform within a bin, hw68 = 0.34 w): A4 0.028, D4 0.024, D3 0.021, D2 0.017. Plateau near these => representation
  at bin resolution; error growing with depth => processing loses it; bad already at k = 0 => input encoding.

## Round 2 (2026-10-21, after the independent audit): what limits a learned network at high charge?

The fundamental chain: (1) is the precise answer in the input?  (2) is it a LEARNABLE function of the input from
(input, truth) pairs, with no physics knowledge?  (3) can a network of ours learn it, and what stops it -- the input
encoding, the loss's priorities, or the representation the encoder/FM provides?  (4) does the FM's objective have
a blind spot?  (5) can cross-volume attention stay without the coupling?

| test | question | confirms / kills |
|---|---|---|
| C14 overflow bins | what fraction of bright cells' input coefficients lie beyond the FM's last finite bin edge? | large => a concrete FM blind spot at exactly the charges in question; ~0 => not a cause |
| T13 learned linear filter | does a linear map LEARNED from (corpus image, truth) pairs, no kernel, match the exact-kernel Tikhonov? | matches => the bright answer is a learnable linear function of the input; a network failing it is an optimisation / encoding / loss problem, not missing information or missing physics |
| H1 / H2 no-encoder heads (arcsinh / + linear amplitudes of the 3x3 slots) | can the head alone learn it, and does the input encoding decide? | H2 << H1 => the log encoding limits learning; H2 ~ inversion => no encoder needed at bright |
| H3 frozen M7dc + 3x3 + amplitudes | the audit's missing control | H3 << P5_long => the plateau was the encoding of the direct inputs |
| H4 / H6 loss on truth >= 16 k only | does the loss's allocation (bright rows 0.07%) hold the bright error up? | lower bright error => the objective's priorities matter |
| H5 frozen PRETRAINED FM + 3x3 + amplitudes, vs H2 | do the FM's features add to the raw inputs? | H5 < H2 => information in the features beyond the inputs; H5 ~ H2 => no measurable contribution |
| M8a augmentation only, original layouts (vs M8g layout + aug, M8c layout only) | can cross-volume attention stay? | cross-volume response ~0 and gain ~0.94 => the layout change is unnecessary; cathode Y better than M8g => cross-volume attention is useful |
| M12 w0 / w1 / w2: M10lin fine-tuned 6k steps with the charge loss weighted (1 + q/Q0)^k, k = 0 (control) / 1 / 2 | does the objective put the flat ~3% relative floor where it is? (T13: networks flat in charge, the learned linear map -- least squares in LINEAR charge, i.e. k ~ 2 -- falls ~1/q) | bright error falls with k toward the linear map (Y 0.0175 / 0.0085) while faint bins worsen => the floor is the objective's allocation, a choice of output representation; no change => the objective is ruled out (next: fp32 training, linear output parameterisation). Checkpoints every 2000 steps give the bright error's convergence |
| M14 g05 / g0 / hl: M10lin fine-tuned 6k steps with the conditional charge as a DISTRIBUTION -- Gaussian with learned per-cell width (beta-NLL 0.5 / plain likelihood) or 1024 fine bins with Gaussian-smoothed targets (HL-Gauss) -- each starting exactly at M10lin's head | can both regimes be good at once? A point head with squared error assumes equal uncertainty for every cell (the flat floor); weighting by the TRUE charge (M12, M13) reallocates capacity but biases / costs the faint side (M13 products: detection 0.548 vs 0.585). A likelihood with input-dependent width weights each cell by its achievable precision without biasing the mean | bright bins near M12w1's AND faint bins near M12w0's, medians ~0, widths calibrated (c15: |z| < 1 for ~68%) => the distribution is the fix; which parameterisation wins answers "bins vs parametric" |
