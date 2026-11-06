# M7dc high-charge error: tested causes (2026-10-12)

Each claim from `m7_diagnostics.md` and each candidate mechanism was tested directly, with the prediction written
first (`m7_tests_plan.md`). Own code only (`t0_verify.py`, `t1_inputs.py`, `t1f_local.py`, `t2t3_model.py`,
`t2t3_summary.py`, `t2_fp32_summary.py`, `t2d_local.py`, `t3b_t1g_summary.py`, `t7_fluct.py`, `t8_draws.py`,
`t8_event.py`, `t4_top.py`, `t2d_t8_summary.py`); numbers in `/global/cfs/cdirs/m5238/users/oalter/helix_runs/m7tests/`.
Error = hw68 of pred/truth (or of log pred - log truth), bright cells, complete truth. Nothing fitted on test events.

## Result in one paragraph

Above ~64 k e- M7dc's per-cell error is flat at ~3-4% while a linear exact-kernel inversion of the same input
improves as 1/q. That error is **a deterministic approximation error of the model**: 85-92% of it repeats in every
noise draw of the same event (T8), the model is exactly as wrong on its own training events (T3a), and the error
shrinks 4-5x with training scale at every charge (T3b). It is **not** missing information, input rounding, the
decoder's local view, numerical jitter, smoothing of real fluctuations, or event size. Separately, the model's
output depends on the amplitude of the **whole event** (T2d): scaling any part of the input up lowers the
predicted charge everywhere else, including the other TPC volume, with an exponent ~0.3. In simulation (fixed gain)
this coupling costs ~1% of the error variance, but it makes the global gain response 0.67 instead of 1.

## Tests

| test | question | prediction that would confirm / kill | outcome |
|---|---|---|---|
| T0a | are model and floor maps aligned? | max correlation at (0, 0) | **pass**: r 0.996 at (0, 0), next 0.84, every plane |
| T0b | agent's per-truth-bin numbers | reproduce Y 0.037 / 0.033 / 0.039 vs 0.040 / 0.0175 / 0.0086 | **reproduced exactly**; the inversion wins above 64 k at every lambda |
| T0c | does it depend on binning by truth? | same when binned by model or inversion value | **same** (model 0.033-0.043 above 64 k in all three binnings) |
| T1a | does the model receive every coefficient, exactly? | kept count = token slots, max diff 0 | **pass** on 48 events; inversion code reproduces floor.py to 6e-8 |
| T1c | input rounding (arcsinh -> bf16) | inversion degrades to ~0.03-0.04 at >= 64 k | **killed**: Y >= 256 k 0.0081 -> 0.0104, 64-256 k 0.0175 -> 0.0187 |
| T1e/f | how much input context does an exact inversion need? (16 events) | — | Y (64-256 k / >= 256 k): covering token only 0.066 / 0.057, +-1 time token 0.022 / 0.011, 3x3 tokens 0.018 / 0.008 = full image. U / V need the 3x3 neighbourhood (0.062-0.072 / 0.030-0.037 vs full 0.043-0.065 / 0.017-0.034). The decoder's direct view alone is not enough; its error does not show that limit (T1g) |
| T1g | is the model worse where its direct view is cut (token edges)? | edge cells worse if locality limits it | **killed**: hw68 flat across wire position (0.032-0.034) and time position in A4 / D3 / D2 tokens |
| T2b | bf16 inference | — | fp32 Y 64-256 k 0.0327 -> 0.0306; sqrt(0.0327^2 - 0.0306^2) = 0.0115 = the bf16 output grid (0.031 step above y = 4): **~12% of the variance** |
| T2a | gain response (fp32) | slope 1 for a linear measurement | **slope 0.66-0.71** (0.60-0.63 above 256 k), same for e = -2% .. +5%, residual 0.06-0.3% |
| T2c | numerical jitter (1e-3 input noise, fp32) | jitter comparable to the error => noise-like map | **killed**: output sd 0.0004, 100x below the error |
| T7 | does the model smooth real cell-to-cell fluctuations (Landau)? | within-track slope ~0.67 confirms; ~1 kills | **killed**: slope 0.996-1.000 at >= 16 k and >= 64 k, error uncorrelated with the true fluctuation (|r| <= 0.03); error variance 85-90% within track, 10-14% between tracks, ~1% between events |
| T2d | where does the 0.67 come from? | local scaling ~1 and global ~0.67 => context normalisation | **context coupling**: scaled region 0.81-0.92, unscaled cells **-0.03 to -0.15** (other planes, other TPC volume); inside - control 0.93-0.98; per band A4 0.41-0.54, D4 0.12-0.25, D3 / D2 < 0.07, additive |
| T3a | fit vs generalisation | train ~= test => underfit | **train = test**: Y 64-256 k 0.0326 vs 0.0327, >= 256 k 0.041 vs 0.038; U, V the same |
| T3b | training scale | flat across compute => not training-limited | error **falls 4-5x** from fast arms (1,024 labels, 4k steps: 0.12-0.19) to full (0.032-0.053), at every charge, flat in charge in both |
| T8 | noise-driven or deterministic? | inversion v_sys << v_noise; approximation error v_sys >> v_noise | **model deterministic**: >= 64 k sys / noise 0.028 / 0.012 (Y 64-256 k), 0.035 / 0.010 (Y >= 256 k), 0.038 / 0.012 (U); inversion 0.014 / 0.008, 0.007 / 0.003, 0.014 / 0.008 |
| T8b | attention dilution with event size? | deterministic error grows with token count | **no**: 0.028-0.035 in each of 24 events, corr 0.22-0.29 (n.s.) |
| T4a | top-end deficit (> 1 M e-): generalisation? | low only on test => generalisation | **underfit**: 1-2 M median 0.974-0.978 on training events, 0.978-0.981 on test; 67-89 cells per 48 events |

## T9: is M7dc converged? (`t9_converge.py`, final weights, exact training loss, training events)

Gradients over 256 training events, the loss split exactly into bright rows (y >= log1p(64k/Q0); 10.9% of charged
rows, 0.07% of the loss: 0.000186 of 0.256, which BCE dominates at 93%) and the rest:

| | signal G^2 | noise per 16-event batch | SNR_B16 |
|---|---|---|---|
| bright, raw | 4.3e-4 | 1.9e-5 | 22 |
| other, raw | 5.9e-3 | 1.03e-2 | 0.57 |
| total, raw / Adam-preconditioned | | | 0.66 / 0.088 |

cos(G_bright, G_other) = 0.15 raw, 0.09 preconditioned (no first-order conflict). Batch-mean gradient norm 0.13 vs
clip 1.0 (clipping inactive). Line search on 48 held-out training events (fp32), step sized for a first-order
decrease c * L0: along the preconditioned total direction the best is -0.04% of the loss (c = 0.003), along the
preconditioned bright direction -0.5% of the bright term (c = 0.03); larger steps increase both; raw directions give
nothing. **The weights are at a first-order minimum of the training objective, and the bright term at its own local
minimum**: the bright gradient is statistically real but the curvature along it leaves <= 0.5% to gain. Continuing
at the final learning rate would not lower the bright error. **Correction (after T11):** this is a LOCAL statement
about the end of an annealed schedule; it does not bound what a LONGER schedule (more steps at high learning rate,
then annealed) would reach. Short head probes also end first-order converged (P5_nbr: line search -0.04% total,
-1.1% bright) at worse errors than M7dc (0.040 vs 0.031), so "converged" here means "at the bottom of the basin the
schedule reached". Whether a longer schedule reaches a better one is measured by P2_long vs P2_fresh (T11 long runs).

## What is established, and what is not

Established [V]: the high-charge error is a model approximation error (deterministic, on training data too,
training-scale dependent), not an input / information / noise / locality problem; bf16 output rounding is ~12% of
the variance at 64-256 k; the model couples every cell to the event-wide amplitude (exponent ~0.3) through the
encoder's global self-attention (the only cross-token path; the conditioning has no event pooling — code, not a
test of which layers carry it).

Established by T9: not a convergence problem (first-order minimum; <= 0.5% of the bright loss reachable along its
own gradient), and not a first-order trade-off with the rest of the objective. Not established: whether a different
representation (e.g. linear-amplitude inputs to the head), a larger model, or a differently weighted objective
trained from the start would reach a better minimum — those change the function class or the landscape, not the
convergence. Each is a training experiment. Not yet tested from the plan:
T3c (training support / loss share at high charge), T5 (presence calibration mechanism in the faint bins), T6
(M7d's 5.6% conditional-charge deficit).

## T10: linear readouts of log T along the network (`t10_readout.py`)

Ridge readouts fitted on calibration events (ev000-063 minus 41), lambda chosen on ev050-063, scored on the 48 test
events; hw68 of the log residual, Y plane, 16-64 k / 64-256 k / >= 256 k (U, V alike):

| stage | readout |
|---|---|
| raw coefficient slots, arcsinh / amplitude | 0.38 / 0.37 / 0.34, 0.39 / 0.33 / 0.33 |
| encoder features (covering + 3x3 mean) / z of the 3x3 neighbourhood | 0.39 / 0.31 / 0.28, 0.39 / 0.31 / 0.27 |
| decoder query after its input MLP (q0) | 0.38 / 0.37 / 0.22 |
| after decoder attention block 1 | 0.080 / 0.091 / 0.117 |
| after block 2 (= model output, affine) | 0.035 / 0.031 / 0.036 |

lambda went to the top of the grid for every stage before the decoder's attention: log charge is not linearly
readable from the encoder features or the raw slots at all, so a linear probe cannot say whether precision is lost
before or in the head (absence of a linear readout is not absence of information; T1f shows the raw neighbourhood
carries it to 0.008-0.018 via the exact kernel). The reimplemented decoder reproduces the model (block-2 readout =
y). Next: nonlinear probes — heads retrained on the frozen encoder (T11).

## T11: heads retrained on M7dc's FROZEN encoder (`t11_head_probe.py`)

Charge-only hurdle loss (no presence BCE), same training rows as M7dc, 2500 steps x 4 events (~1.5% of M7dc's
event budget), AdamW cosine; scored in fp32 on the 48 test events. hw68 of y - log1p(T/Q0), Y plane, 16-64 k /
64-256 k / >= 256 k (U, V show the same ordering):

| probe | step 1000 | step 2000 | step 2500 |
|---|---|---|---|
| M7dc as trained | 0.0333 / 0.0306 / 0.0368 | | |
| P1 M7dc's head continued (lr 3e-4) | 0.0330 / 0.0303 / 0.0367 | 0.0324 / 0.0297 / 0.0362 | 0.0322 / 0.0295 / 0.0359 |
| P2 fresh CellDecoder (M7dc's shape) | 0.0564 / 0.0554 / 0.0695 | 0.0405 / 0.0396 / 0.0547 | 0.0386 / 0.0374 / 0.0506 |
| P3 P2 + amplitude inputs sinh(value) | 0.0578 / 0.0595 / 0.0783 | 0.0415 / 0.0410 / 0.0569 | 0.0391 / 0.0377 / 0.0501 |
| P4 larger decoder (proj 256, 4 blocks) | 0.0588 / 0.0671 / 0.0843 | 0.0363 / 0.0341 / 0.0450 | 0.0346 / 0.0325 / 0.0425 |

- The head's input encoding is not the limit: amplitude inputs (P3) = arcsinh only (P2) at every step [V].
- Removing the presence term from M7dc's own head's loss moves it by 3% of its error (P1) [V] (as T9 predicts).
- A larger head trains faster and is better than P2 at equal budget, but at 1.5% of the budget is still at or above
  M7dc (0.0325 vs 0.0306 at 64-256 k) [V]. Every head approaches the same ~3% from above with slowing gains; whether a
  converged larger head goes below it is NOT established (the fresh heads are not converged).
- P5_nbr (P2 + raw slots of all 3x3 neighbourhood tokens per band, 2500 steps, same schedule as P2): 0.0411 / 0.0402 /
  0.0527 vs P2 0.0386 / 0.0374 / 0.0506 (step 1000: 0.0666 / 0.0663 / 0.0847 vs 0.0564 / 0.0554 / 0.0695). Direct access
  to the coefficients an exact inversion needs does not beat the encoder path at equal schedule [V]. Head-only
  convergence test at its end: SNR_B total 2.9, bright 16; line search best -0.04% total, -1.1% bright: locally
  converged.
- (superseded by P5) What the probes had not tested: a head with DIRECT access to the raw coefficients of the 3x3 token neighbourhood.
  T1f shows that is what an exact inversion needs; the decoder's direct view is only the covering token per band,
  and everything from the neighbours reaches it through the (frozen) encoder features.

## T11 long runs (10k steps, frozen M7dc encoder, each ending first-order converged)

Y plane 16-64 k / 64-256 k / >= 256 k: P2_long (M7dc-shape head) 0.0341 / 0.0317 / 0.0403; P5_long (+ raw 3x3 slots)
0.0349 / 0.0328 / 0.0402; P4_long (larger head) 0.0324 / 0.0299 / 0.0370; M7dc 0.0333 / 0.0306 / 0.0368. Line searches at
the end: total -0.02% / -0.01% / +0.03%, bright -0.5% / -0.5% / -0.2%. Every converged head lands at M7dc's error
(the larger one 2-3% better below 256 k): the ~3% bright floor is set by the encoder's representation, not by the
head, its inputs, or the schedule.

## The event-wide amplitude coupling: cause and removal (C1-C7, M8)

- C1/C1b (simulation): the 8 runs have identical corpus configuration; per-event exact-inversion / complete truth on
  bright Y cells varies by 0.09% between events (0.06% statistical): no event-level scale for a model to estimate.
- C3 (simulation): event totals vary +-45%; the two volumes' charge correlates +0.17 (planes of a volume 1.0). Nothing
  rewards a NEGATIVE coupling.
- C2: the cross-volume response (volume 0 x1.05 -> volume-1 cells) is -0.073 for le3_ft vs -0.017 for le3_scratch
  (same recipe, 16k labels); -0.073 M4n_ft vs -0.038 M2n_scratch; already -0.117 after 4k fine-tuning steps; -0.11 M7dc.
- C4: encoding each volume as its own event leaves bright accuracy unchanged (hw68 +-0.001) while moving each cell
  ~1%: the other volume carries no useful information. Within a volume the response fits normalisation by the
  VISIBLE amplitude, exponent ~0.27.
- C5: the PRETRAINED FM's own masked predictions already respond -0.043 cross-volume (data correlation +0.17),
  +0.056 cross-plane (physical).
- C6: the encoder attends within groups of a serialised order; its two DRIFT layouts sort by time alone, so a group
  mixes all planes of both volumes. Bright volume-1 tokens give 28% of their drift-layer attention to volume-0
  tokens; scaling volume 0 changes that share by only -0.07% (the coupling travels through the attended VALUES, not
  reallocated weights -- competition for attention weight is ruled out). Sorting the drift layouts by (volume, time),
  weights unchanged: cross-volume coupling -0.106 -> -0.0004, accuracy unchanged.
- C7: the charge-scaling augmentation reproduces each region's gain on signal-dominated coefficients (median ratio
  1.0000, 16-84% 0.998-1.002) with the noise untouched.
- Fix (e68fe7e): SerialFMModel(drift_planes_per_volume=3) + DenoiseEvents gain_sigma_plane / gain_sigma_block
  (signal and truth scaled per (plane, 64-wire block) region). M8g = M7dc fine-tuned 6k steps with both:

| scaled by 1.05 | M7dc inside / unscaled | M8g inside / unscaled |
|---|---|---|
| whole event | 0.68-0.72 | 0.93-0.95 |
| one volume | 0.80-0.83 / -0.10 to -0.13 | 0.94-0.95 / -0.001 |
| one plane | 0.89-0.92 / -0.03 to -0.05 | 0.97 (U, V), 1.00 (Y) / -0.01 |
| 64-wire strips | 0.83-0.85 / -0.13 to -0.15 | 0.97 / -0.03 |
| 512-tick blocks | 0.81-0.82 / -0.12 to -0.14 | 0.97-0.98 / -0.04 |

  Bright accuracy unchanged (hw68 Y 0.033 / 0.030, U 0.042 / 0.035, V 0.045 / 0.037; val MSE 0.01231 vs 0.01230); raw
  median +0.3-0.6% (M7dc -0.1 to +0.3%). Residual: whole-volume gain response 0.94-0.95. M8c (same fine-tuning, layout
  only, no augmentation) separates the augmentation from the extra training.
- M8c (same fine-tuning, volume-aware layout, NO augmentation): whole event 0.73-0.76; one volume inside 0.73-0.76,
  other volume -0.001; one plane inside 0.88, other planes -0.03; accuracy and val MSE (0.01231) unchanged; raw median
  +0.5-0.6%. Attribution: the layout removes the cross-volume coupling; the augmentation lifts the within-volume
  response (whole volume 0.74 -> 0.94, plane 0.88 -> 0.97-1.00, leakage -0.03 -> -0.01); the +0.5% raw-median shift
  comes from the fine-tuning (present without augmentation), to be checked under the calibrated products.

## Layers, bins, compression, cathode (L1, L2, C8-C12; 2026-10-13/14)

- C8 (measured): the FM's 128 categorical bins are uniform in arcsinh, i.e. constant RELATIVE amplitude width at
  bright values: A4 8.5%, D4 7.4%, D3 6.5%, D2 5.1%.
- C9a/geometry (code): JAXTPC wire index = f(y, z) only (r = y sin(theta) + z cos(theta), same y-z extent and angles in
  both volumes): the same index is the same y-z line in both volumes. Volumes share the cathode at x = 0 (x < 0 drifts
  to -x, x > 0 to +x): equal drift time = mirror positions +-d, adjacent only at the cathode (tick ~3510 for t0 = 0).
- C11 (paired, 128 test events, ticks 3400-3560): per-volume encoding is worse than full near the cathode on Y
  (mean |e| +0.0006 [+0.0002, +0.0010] at 16-64 k, +0.0005 [+0.0000, +0.0011] at 64 k+; hw68 x1.017-1.020, CI touching 1),
  U +1% (n.s.), V none. Over the whole event no difference (C4, +-0.3%). Cross-volume attention helps a little exactly
  where the volumes are adjacent; separating volumes (e68fe7e) gives that up. A signed-x position (sort and RoPE by
  x = +-d) would keep it and remove the far-field coupling (needs re-pretraining; not tested).
- L2 (MLP probe of each token's own amplitudes): the probe's own floor at layer 0 (0.06-0.13 log) exceeds the bin
  level (0.017-0.028), so it cannot decide the bin question [inconclusive]. Own-token amplitude fades with depth in both
  encoders (A4: 0.066 -> 1.13 pretrained, 0.057 -> 1.55 fine-tuned at layer 12).
- L1 (heads on the frozen PRETRAINED FM, 2500 x 4 events): Y 64-256 k error 0.240 (layer 3), 0.183 (6), 0.175 (9), 0.173
  (12) vs 0.037 for the same head on the FINE-TUNED encoder. Earlier layers are worse, not better (prediction killed);
  the pretrained representation carries ~17% charge precision at any layer; fine-tuning the encoder builds it (5x).
- C12 (matched pairs): full scale, pretrained-then-fine-tuned M4n vs from-scratch M2n: Y 0.044 / 0.042 / 0.045 vs
  0.045 / 0.043 / 0.049, U 64-256 k 0.047 vs 0.047. 16k labels: le3_ft 0.095 vs le3_scratch 0.182 (Y 64-256 k).
  Pretraining helps label efficiency, NOT the asymptotic floor: the floor is reached from scratch, so the FM's
  objective / bins do not set it. What sets it: the supervised encoder (architecture + objective at full training);
  the arcsinh input encoding inside the encoder is not yet tested causally.

## Encoder ablations (fine-tunes of M7dc, 6k steps, one change each vs M8c; c13_eval_models.py, paired, 48 test events)

- M9lin (+ linear-amplitude input channel sinh(value)/30 beside arcsinh, zero-initialised): hw68 ratio vs M8c, Y
  0.973 [0.968, 0.978] / 0.903 [0.892, 0.911] / 0.80 [0.76, 0.85] (16-64 k / 64-256 k / >= 256 k); U 0.982 / 0.948 / 0.898;
  V 0.993 / 0.968 / 0.904. Y >= 256 k 0.0371 -> 0.0297. Gain response 0.747 -> 0.798. Val MSE 0.01225 vs 0.01231.
  M7dc vs M8c: 0.99-1.015 (the control is stable). The log-compressed input limits high-charge precision, more so
  the higher the charge; feeding the amplitude recovers 10-20% at >= 256 k in 6k steps. [V; seed replicate pending]
- M9grp (attention groups doubled, gp 2048 / gd 4096): hw68 ratio vs M8c 0.977-1.007, every 95% interval contains 1
  (Y 0.994 / 0.993 / 1.007); cathode region unchanged; gain 0.755. Attention context size is not the limit.
- M9phys (physical drift coordinate x = +-(3510 - t), drift layouts sorted by x; fine-tune of M7dc whose weights were
  trained on t, so volume 1's RoPE direction flips -- step-50 loss 0.45 vs ~0.26 for the others): hw68 ratio vs M8c
  1.008-1.026, several intervals excluding 1 (Y 64-256 k 1.025 [1.016, 1.036]); cathode region no gain (1.006-1.074, all
  intervals include 1); val MSE 0.01240 (M8c 0.01231). Under the FINE-TUNE protocol the coordinate change costs ~2% and
  does not recover the cathode gain; this does not test the representation from (re)pretraining. Its training log
  stopped at step 200 (CFS write); checkpoints and val.jsonl run to 6000.
- M8c_s1 (M8c, seed 1): ratio vs M8c 0.98-1.004, every interval includes ~1 (Y 1.003 / 0.993 / 0.981; U 1.004 / 1.002 / 0.98);
  fine-tune noise is <= +-0.7% below 256 k and ~+-2% above. So M9lin (0.80-0.97) is far outside it; M9phys (+1.8-2.6%
  below 256 k) is a small real degradation under the fine-tune protocol; M9grp is null.
- M10lin (M7dc's full recipe from the pretrained FM, + amplitude channel from step 0; control = M7dc): running.
  Val MSE at step 2000 / 4000: 0.0271 / 0.0216 vs M7dc 0.0305 / 0.0226.
- **M10lin** (M7dc's full recipe, 40k x 16 from the pretrained FM, amplitude channel from step 0; control M7dc): hw68
  ratio vs M7dc, Y 0.920 [0.913, 0.927] / 0.815 [0.806, 0.823] / 0.729 [0.693, 0.763] (Y 0.0306 / 0.0249 / 0.0268), U 0.943 /
  0.864 / 0.791, V 0.991 / 0.940 / 0.817; cathode Y 64 k+ 0.78; gain response 0.70 -> 0.87 (no augmentation); val MSE
  0.01210 vs 0.01230. Larger than the 6k fine-tune (M9lin) and growing with charge on every plane. The gap to the
  corpus inversion (Y 64-256 k 0.0175, >= 256 k 0.0086) closes by ~45% / ~36%. The log-compressed input is the main
  encoder cause of the high-charge floor found so far; the amplitude channel is the fix. Full products standard pending.
- M10lin, full products standard (5 draws, complete truth; dense/results/products_draws.md) vs M7dc: bright calo median /
  res68 0.9989 / 0.0374 vs 0.9967 / 0.0413; clean / calo L1 0.0488 / 0.0541 vs 0.0529 / 0.0582; plane totals 0.995 / 0.993 /
  0.999 vs 0.994 / 0.992 / 0.998; calibrated presence LL / ECE 0.0773 / 0.0030 vs 0.0774 / 0.0031; isolated near detection
  (p_cal) 0.563 vs 0.555 (within errors); clean haze 0.0014 vs 0.0012. Cost: calo linearity at 1-2 k / 102-500 e- predicted
  1.022 / 1.147 vs 1.007 / 1.129. Better or equal everywhere else.
- **M11** (M7dc's full recipe + amplitude channel + volume-aware drift layouts + charge-scaling augmentation, 40k x 16;
  val MSE 0.01213 vs M10lin 0.01210, M7dc 0.01230): bright hw68 vs M10lin 0.978-1.010 (Y 16-64 k +1% [1.003, 1.019], the rest
  intervals contain 1) -> keeps the amplitude-channel gain (M7dc vs M10lin 1.06-1.37). Gain response (t2d, 24 events):
  whole event 0.97-0.98 (M10lin 0.87, M7dc 0.70); one volume 0.97-0.98 / other volume 0.000; one plane 0.99-1.00 / other
  planes >= -0.002; 64-wire strips 0.99 / -0.02; 512-tick blocks 0.98 / -0.015. Raw bright median +0.6-1.0% (training-
  label truth). Products standard running.
- M11, full products standard (5 draws, complete truth) vs M10lin / M7dc: bright calo median / res68 1.0023 / 0.0382 vs
  0.9989 / 0.0374 vs 0.9967 / 0.0413; plane totals 0.999 / 0.997 / 1.002 (closest to 1 of the three); clean L1 0.0492 vs
  0.0488 vs 0.0529; calibrated presence LL / ECE 0.0773 / 0.0028 (best ECE); isolated near detection (p_cal) 0.574 vs
  0.563 vs 0.555 (within errors); haze 0.0013; calo linearity 1-2 k / 102-500 1.022 / 1.145 (as M10lin). Keeps the
  amplitude-channel precision (res68 -7.5% vs M7dc; paired bins within +-1% of M10lin) and removes the coupling
  (gain response 0.97). Candidate to replace M7dc as the reference.

## Independent audit (2026-10-21; claims re-checked against code and stored outputs; key items re-verified by hand)

- Head vs encoder: L1 heads on the frozen pretrained FM (0.24 / 0.18 / 0.175 / 0.173) are worse than a linear
  inversion of their OWN covering-token input (T1f 0.066), so they do not bound what the features contain; and
  P5_long (raw 3x3 slots, frozen M7dc) = P2_long (0.0328 vs 0.0317 at Y 64-256 k) although T1f says those
  coefficients allow 0.018. "The encoder's representation sets the floor" does not follow. Supported: learned
  networks reading arcsinh inputs plateau near 3% at these budgets (encoder or head). Missing control: a 3x3 head
  with linear amplitudes.
- The FM: pretrained on NOISY inputs and noisy targets, mask 0.75 (+ whole-plane 10%), vis_w = 0, 128-bin CE + BCE
  on masked tokens. Input distribution matches the denoiser's; visibility does not (L1/L2 run it at 100% visible).
  FM top bins open-ended (edges end at arcsinh 7.14 / 6.40 / 5.92 / 5.12, then 1e18). C8's "bin width < 17%"
  compared a full bin width to an hw68 (uniform quantisation hw68 ~ 0.34 w ~ 2.9% for A4).
- C12 (scratch = pretrained) used CellHead models (M4n / M2n), not M7dc's recipe; single seeds.
- Inversion: the quoted 0.0175 / 0.0086 are Tikhonov lam 1e-3, Y only; U 0.0415 / 0.0167, V 0.0650 / 0.0249 vs the
  model 0.0377 / 0.0419, 0.0395 / 0.0432 -- the model is better on U / V at 64-256 k. lam reported on test events.
- Coupling: "seeded by pretraining" overstated (present from scratch, C2). The data reward ZERO cross-volume
  coupling (C4), not a positive one. Exponent ~0.27 and C3's +0.17: no stored script output (UNVERIFIED in the
  stored record). Mechanism (LayerNorm division vs log-space subtraction) untested; values-not-weights is partly
  forced by QK-norm.
- Attention structure: rope_split = False in the FM artifact (verified), so drift layers use axial (t, wire) RoPE,
  not time-only. Groups are consecutive chunks of the sorted order and straddle plane / volume boundaries; the
  rolled drift layout wraps volume 1 onto volume 0. M11 is not volume-separated by construction.
- M11 vs M10lin near the cathode, Y 16-64 k: 1.041 [1.004, 1.073] (degradation). One-plane gain 0.982-0.998.
- M7dc val.jsonl has two step-40000 rows (0.012303 / 0.012345, duplicate loop; verified); the gap exceeds M11 -
  M10lin. Which final.pt the M7dc comparisons used: unverified.
- T3b confounds labels (1k vs 150k) with steps (4k x 4 vs 40k x 16).

## Round 2 results (plan: m7_tests_plan.md "Round 2")

- **C14 overflow bins** (48 test events; m7tests/c14.json): the FM's finite bin edges end at arcsinh 7.14 / 6.40 /
  5.92 / 5.12 (bands A4, D4, D3, D2). Over all occupied slots 0.02-0.23% lie beyond. For cells >= 256 k the covering
  token has at least one slot beyond in 79% (Y, A4) / 59% (U) / 13-42% (V, A4 / D3) of cells (slot fraction 1-5%);
  for 64-256 k in 16% (Y A4), 9% (U), 8% (V D3); below 64 k in <= 5.5%. So the masked objective's TARGET saturates
  for most >= 256 k cells -- a real blind spot of pretraining exactly there -- but not for most 64-256 k cells,
  whose error is also ~3%. The input itself is not clipped (the encoder reads the continuous value).
- **T15 tokenisation phase** (products calibration dumps ev000-063, complete truth; m7tests/t15.json; pw 16, pt 8):
  the per-cell error barely depends on where the cell sits in its token. Across the 8 time phases (band 0) hw68
  varies by +-3% (Y >= 64 k: M10lin 0.0259-0.0275, M7dc 0.0325-0.0341), with no edge pattern. Across wire phases the
  two token-EDGE phases (0, 14) are worse by ~5% (Y, V) to ~10% (U): U >= 64 k M10lin 0.0352 / 0.0351 vs 0.0306-0.0324
  inside; M7dc 0.0408 / 0.0401 vs 0.0370-0.0376. Same pattern in M7dc, M10lin, M11. Patch tokenisation costs a few
  per cent at token wire edges; it is not the floor.
- **T13 learned linear filter** (fit ev000-127, 128 events, test ev260-307; m7tests/t13.json; hw68 of log(pred/truth)):
  a translation-invariant linear filter LEARNED from (corpus image, complete truth) pairs, with no kernel, equals
  the exact-kernel Tikhonov on Y: 0.0404 / 0.0175 / 0.0085 (16-64 k / 64-256 k / >= 256 k) vs 0.0408 / 0.0176 / 0.0086
  (lam 1e-3); converged (fit on 64 events: 0.0402 / 0.0176 / 0.0086). On U / V the learned minimum-MSE filter is worse
  than the best Tikhonov at bright charge (U >= 256 k 0.037 vs 0.017 at lam 1e-4; V 0.042 vs 0.025) and every linear
  map is worse than the networks at 64-256 k (U 0.042-0.053, V 0.066-0.073 vs M11 0.0305 / 0.0353). So: on Y the
  precise bright answer is a learnable linear function of the input (no physics needed); the networks' Y error is
  FLAT in charge (M11 0.031 / 0.025 / 0.027) while any linear map's falls ~1/q (0.040 / 0.0175 / 0.0085) -- the network
  error behaves as a constant RELATIVE floor, the linear one as noise. On U / V the networks are already better than
  every linear map below 256 k and within the linear range above.
- **T16 labels vs rebuild truth** (48 test events; m7tests/t16.json): the networks' training label (corpus complete
  truth) and the rebuild truth the inversions / linear filter are scored on are IDENTICAL per cell (hw68 0, median 0,
  no missing labels, every plane and bin). The network-vs-linear comparison is on one truth; label disagreement is
  not part of the floor.
- **H6 nofeat_amp_bright** (no encoder; raw 3x3 slots + linear amplitudes; loss on truth >= 16 k; 10k steps;
  m7tests/t14/): Y 0.112 / 0.090 / 0.126 (16-64 k / 64-256 k / >= 256 k) vs the learned linear filter's 0.040 / 0.0175 /
  0.0085 on the same information -- this head (CellDecoder, slots flattened into its aux vector) does not learn the
  deconvolution from raw slots at this budget; a statement about the head's design, not the encoding. Its end point
  is first-order converged (line search best -0.1% total) and shows a TRADE-OFF: the bright-row (>= 64 k) and
  16-64 k gradients point in opposite directions (cos -0.98); stepping along the bright gradient lowers bright loss
  by 12% while raising 16-64 k by 32%. Precision across charge ranges competes for the same parameters.
- **M12 objective weighting** (M10lin fine-tuned 6k x 16, conditional-charge loss weighted (1 + q/Q0)^k, output
  representation unchanged; 48 test events, paired vs the k = 0 control M12w0; m7tests/c13_m12.json). hw68 ratio vs
  control [95% CI], truth bins 1-4 k / 4-16 k / 16-64 k / 64-256 k / >= 256 k:

| | U | V | Y |
|---|---|---|---|
| k = 1 | 0.984 / 1.018 / 0.959 / 0.848 / 0.725 | 0.979 / 1.000 / 0.958 / 0.856 / 0.716 | 0.982 / 1.042 / 0.955 / 0.832 [0.823, 0.838] / 0.692 [0.662, 0.728] |
| k = 2 | 0.985 / 1.147 / 1.083 / 0.905 / 0.684 | 0.975 / 1.106 / 1.072 / 0.898 / 0.668 | 1.007 / 1.260 / 1.133 / 0.906 / 0.683 |
| M10lin (no extra training) | 0.998-1.006 ... 0.994 | 0.997-1.005 ... 0.963 | 0.999-1.013 |

  Absolute Y hw68 16-64 k / 64-256 k / >= 256 k: control 0.0303 / 0.0248 / 0.0265 (flat in charge); k = 1 0.0290 /
  0.0206 / 0.0183 (now FALLING with charge); k = 2 0.0344 / 0.0225 / 0.0181; learned linear filter 0.040 / 0.0175 /
  0.0085. U / V >= 256 k k = 1: 0.0235 / 0.0248 vs the best linear map 0.0168 / 0.0251. Bright medians: control -0.30
  to -0.44% (>= 64 k), k = 1 -0.14 / -0.16% (Y), ~0 (U, V); k = 2 Y >= 256 k -0.66%. Gain response 0.82 / 0.81 / 0.76.
  Cathode window: same pattern. Validation MSE (uniform weighting) 0.01214 / 0.01314 / higher -- the metric k moves
  away from.
  Reading: the OBJECTIVE sets a large part of the floor. With architecture, input and output representation fixed,
  re-allocating the loss toward charge cut the bright error by 15-17% (64-256 k) and 28-31% (>= 256 k) on every plane
  in a 6k-step fine-tune (seed spread <= 0.7% / ~2%), turned the flat relative floor into an error that falls with
  charge, and removed the bright median bias. k = 1 costs 2-4% at 4-16 k (Y, U) and gains 2% at 1-4 k; k = 2 buys
  little more at >= 256 k and costs 7-26% at 4-64 k: precision across charge ranges competes for the same capacity
  (as the head probes' cos -0.96 / -0.98 between bright and fainter gradients said). Remaining gap on Y >= 256 k:
  0.018 vs 0.0085 (linear map); at 64-256 k 0.0206 vs 0.0175.
- M12 convergence (kept checkpoints, k = 1 vs k = 0 at the same step; m7tests/c13_m12_s{2000,4000}.json): Y hw68 ratio
  64-256 k / >= 256 k 0.944 / 0.794 (step 2000), 0.813 / 0.687 (4000), 0.832 / 0.692 (6000); U, V alike. The RELATIVE
  gain of the weighting is established by step 4000 and stable to the end. Absolute errors of both arms still fall
  with the annealing learning rate (Y >= 256 k k = 1: 0.0325 / 0.0227 / 0.0183; k = 0: 0.0409 / 0.0330 / 0.0265), so
  the absolute floor a full-length k = 1 run reaches is not determined by this fine-tune.
- **H4 m7_bright** (frozen M7dc encoder, M7dc-shape head, loss on truth >= 16 k only, 10k steps): Y 0.0326 / 0.0303 /
  0.0382 vs P2_long (all rows) 0.0341 / 0.0317 / 0.0403 -- 4-5% better. **H3 m7_nbramp** (frozen M7dc + 3x3 slots +
  linear amplitudes): Y 0.0351 / 0.0332 / 0.0437 vs P5_long 0.0349 / 0.0328 / 0.0402 -- no better. On frozen features
  neither the head's objective nor linear direct inputs moves the floor much; the same reweighting applied to the
  whole model (M12) gave 17-31%. The precision lives in what the ENCODER is trained to represent.
- **M8a** (M7dc fine-tuned 6k steps, charge-scaling augmentation only, original volume-mixing layouts; vs M8g =
  layout + augmentation; m7tests/c13_m8a.json, t2d_M8a_summary.log): accuracy = M8g (paired ratios 0.975-1.011, V
  >= 256 k 1.039 [1.008, 1.066]); whole-event gain response 0.93-0.94 (M8g 0.93-0.95, M8c 0.75, M7dc 0.70); one volume
  scaled: inside 0.97, other volume -0.035 (M8g -0.001, M8c -0.001, M7dc -0.10 to -0.13); 64-wire strips 0.97 / -0.035,
  512-tick blocks 0.96 / -0.035, one plane 0.99-1.00 / -0.006 to -0.014. With cross-volume attention KEPT, the
  augmentation cuts the cross-volume coupling by ~70% and leaves -0.035 -- the same size as the leakage into any
  other distant region of the same volume (strips, blocks): what remains is a generic ~3.5% non-local normalisation,
  not a volume-specific one. Cathode window: no significant difference vs M8g at this length (Y 64 k+ 1.035
  [0.965, 1.089]). The volume-aware layout is not needed to remove the volume-specific coupling.
- **H1 / H2 / H5** (same head -- 3x3 raw slots flattened into CellDecoder's aux -- all rows, 10k steps), Y 16-64 k /
  64-256 k / >= 256 k: no encoder, arcsinh slots only 0.255 / 0.279 / 0.325; no encoder, + linear amplitudes 0.168 /
  0.149 / 0.197; frozen PRETRAINED FM + the same slots and amplitudes 0.128 / 0.127 / 0.145; frozen M7dc encoder + the
  same (H3) 0.035 / 0.033 / 0.044. For this head, linear amplitudes make the raw slots ~1.9x more usable; the
  pretrained FM's features add ~15% over the raw inputs; a fine-tuned encoder's add ~4x. The pretrained features
  carry some charge information beyond the inputs, far less than what supervised training builds.

### Round 2 in one paragraph
On Y the precise bright answer is a learnable linear function of the input (T13) and the labels are exact (T16).
The networks' error is a flat ~3% RELATIVE floor; tokenisation phase (T15) and the FM's target saturation (C14,
>= 256 k only) are minor. The floor is mostly the OBJECTIVE's allocation of precision: weighting the conditional-
charge loss by charge (M12, k = 1) cut the bright error 15-31% on every plane in 6k steps, made the error fall with
charge, and removed the bright bias, at a 2-4% cost at 4-16 k; heavier weighting (k = 2) trades too much faint
accuracy. The effect acts through the ENCODER's representation (the same reweighting at the head on frozen features:
4-5%). Precision across charge ranges competes for the same capacity (bright vs fainter gradients cos -0.89 to
-0.98 in three probes). Cross-volume attention can stay: augmentation alone removes the volume-specific coupling
(M8a), leaving a generic ~3.5% non-local normalisation that the volume split does not remove either.

## M13 = M7dc recipe + amplitude channel + charge-weighted loss (k = 1) + charge-scaling augmentation, original
## volume-mixing layouts (40k x 16 from the pretrained FM; 2026-10-22; m7tests/c13_m13*.json, t2d_M13_summary.log)

- Val MSE (uniform weighting) 0.01405 vs M10lin 0.01210 / M11 0.01213 (+16%; the metric k moves away from).
- Paired vs M11 (48 test events), hw68 ratio [95% CI] 1-4 k / 4-16 k / 16-64 k / 64-256 k / >= 256 k:
  U 1.005 / 1.106 / 0.983 / 0.841 / 0.669 [0.638, 0.708]; V 1.007 / 1.102 / 1.002 / 0.843 / 0.661;
  Y 1.030 / 1.141 / 0.967 / 0.784 [0.775, 0.794] / 0.581 [0.551, 0.611].
  Absolute Y 16-64 k / 64-256 k / >= 256 k: 0.0299 / 0.0197 / 0.0157 (M11 0.0309 / 0.0251 / 0.0271, M7dc 0.0333 / 0.0306 /
  0.0368; learned linear filter 0.040 / 0.0175 / 0.0085). U >= 256 k 0.0211, V 0.0219 (best linear map 0.0168 / 0.0251).
  Bright medians within +-0.2% (M11 +0.5 to +1.0%). The Y error falls with charge.
- Costs: 4-16 k 10-14% worse (the 6k fine-tune: 2-4%); Y 1-4 k +3%.
- Gain response (t2d): whole event 0.97-0.98 (M11 0.97-0.98); one volume scaled: inside 0.98-0.99, other volume -0.009
  to -0.015 (M8a -0.035, M11 0.000, M7dc -0.10 to -0.13); strips 0.98-0.99 / -0.012 to -0.021; blocks 0.97-0.98 /
  -0.008 to -0.015; one plane 0.99-1.00 / <= -0.006. Cross-volume attention kept, volume coupling ~1%.
- Convergence (kept checkpoints vs final, Y 64-256 k / >= 256 k): step 10k 4.37 / 4.59x, 20k 1.96 / 2.06x, 30k 1.31 /
  1.36x the final error: the last quarter of the schedule (learning-rate anneal) still cuts the bright error ~25%;
  whether a longer schedule would go lower is not determined.
- Products standard: running.
- M13 products standard (5 draws, complete truth; dense/results/products_draws.md) vs M11 / M7dc:
  bright calo median / res68 0.9997 / 0.0352 vs 1.0023 / 0.0382 vs 0.9967 / 0.0413 (best); plane totals 1.001 / 1.001 /
  1.000 (best); clean L1 0.0474 vs 0.0492 / 0.0529 (best). Faint side worse: calibrated presence LL / ECE 0.0785 / 0.0036
  vs 0.0773 / 0.0028; isolated-deposit detection on the clean map (published def.) 0.548 [0.538, 0.558] vs 0.585 [0.574,
  0.595] (p_cal 0.609 vs 0.621); calo haze 0.0164 vs 0.0121; calo linearity 1-2 k 0.940 (M11 1.022, M7dc 1.007), 4-8 k
  0.979, region 4-16 k 0.966 (102-500 e- 0.994 vs 1.145). The charge weighting moves capacity from faint to bright in the
  shared encoder, presence included. M13 is the best charge model above ~16 k and not a drop-in replacement for the
  faint products.

## M14: the conditional charge as a distribution (6k fine-tunes of M10lin, = M12's schedule; 2026-10-23;
## m7tests/c13_m14.json, c15_m14.json)

Paired vs the k = 0 control M12w0 (48 test events), hw68 ratio, bins 1-4 k / 4-16 k / 16-64 k / 64-256 k / >= 256 k:

| | U | V | Y |
|---|---|---|---|
| M14g0 gauss, plain NLL | 1.008 / 0.941 / 0.800 / 0.710 / 0.713 | 1.008 / 0.956 / 0.828 / 0.738 / 0.674 | 1.011 / 0.879 / 0.709 / 0.610 [0.603, 0.618] / 0.586 [0.562, 0.611] |
| M14g05 gauss, beta-NLL 0.5 | 0.996 / 0.940 / 0.852 / 0.787 / 0.790 | 0.996 / 0.948 / 0.874 / 0.814 / 0.769 | 0.988 / 0.900 / 0.792 / 0.727 / 0.711 |
| M14hl 1024 bins (HL-Gauss) | 1.011 / 0.960 / 1.029 / 1.184 / 1.397 | 1.011 / 0.972 / 0.999 / 1.100 / 1.322 | 1.024 / 0.936 / 1.102 / 1.321 / 1.429 |
| M12w1 (k = 1, for reference) | 0.984 / 1.018 / 0.959 / 0.848 / 0.725 | 0.979 / 1.000 / 0.958 / 0.856 / 0.716 | 0.982 / 1.042 / 0.955 / 0.832 / 0.692 |

- M14g0 absolute Y 4-16 k / 16-64 k / 64-256 k / >= 256 k: 0.0596 / 0.0215 / 0.0151 / 0.0155 (control 0.0679 / 0.0303 /
  0.0248 / 0.0265; learned linear filter - / 0.040 / 0.0175 / 0.0085): at 64-256 k BETTER than the linear map; >= 256 k
  still 1.8x. Medians +0.13 to +0.26% (g05: +0.5 to +0.65%; control -0.2 to -0.4%).
- Both sides improve: the Gaussian heads are better than the control at every bin from 4 k up on every plane, 4-16 k
  included (k = 1 cost 2-4% there), at a ~1% cost at 1-4 k (g0) or none (g05). Val MSE (uniform) g05 0.01196 / g0
  0.01242 / hl 0.01310 vs control 0.01214.
- Calibration (c15): the Gaussian widths match the errors from 4 k up -- |z| < 1 for 0.67-0.70 (4-64 k; 0.683 expected),
  < 2 for 0.94-0.96; median width = error hw68 within ~5%; slightly conservative >= 256 k (0.73-0.79). At 1-4 k
  under-covered (0.50-0.62 within 1 sigma): faint posteriors are heavier-tailed than a Gaussian (presence ambiguity)
  -- where a mixture would help. The binned head's widths stay near their 0.05 initialisation at >= 16 k (0.059
  vs errors 0.033-0.045): at this budget fine bins do not sharpen, and its bright error is 18-43% WORSE than the
  control -- the parametric distribution wins clearly here.
- Gain response (no augmentation in this schedule): control 0.82, g05 0.77, g0 0.76, hl 0.73 -- the likelihood heads
  follow a whole-event 5% scaling LESS; to be checked with the augmentation in the full recipe.
- Reading: the flat floor was the point head's assumption of equal uncertainty everywhere. A head that predicts each
  cell's width and is trained by likelihood allocates precision where it is achievable without the faint cost of
  weighting by the true charge: Y 64-256 k error 39% below the control (k = 1: 17%), 4-16 k 12% below (k = 1: +4%).
