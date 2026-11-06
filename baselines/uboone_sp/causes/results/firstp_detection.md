# First principles: sampling prior, the 25 e- cut, and deposit-level detection

Post-hoc diagnostics only (no training, no new losses or heads). Every number is [V] (computed from the files named; scripts in `baselines/uboone_sp/causes/firstp_*.py`, outputs under `helix_runs/firstp/`) unless marked [A]. Intervals: 16-84 % over 200 event-bootstrap resamples of the 128 test events (ev260-387).

## A. The training prior (details: `firstp_sampling.md`, `firstp_sampling.json`)

200 train events (25 per run, shard 0010, 8 runs), 4 draws each, with the checkpoints' exact recipe (identical `meta['args']` for M6d, M6dc and the four lb_* arms: sig_cap 20000, near 1, neg 1, cov 2, win 48, near_any 1, win_near 24; rows asserted equal to `DenoiseEvents.rows`).

| density (hits truth, +-8 cells) | hits r_train | hits r_deploy | hits OR | complete r_train | complete r_deploy | complete OR |
|---|---|---|---|---|---|---|
| iso | 0.0005 | 0.00003 | 16.1 | 0.0004 | 0.00004 | 9.6 |
| <=10k | 0.2430 | 0.01882 | 16.7 | 0.2960 | 0.04297 | 9.4 |
| 10k-100k | 0.3618 | 0.05608 | 9.5 | 0.4602 | 0.12675 | 5.9 |
| 100k-1M | 0.3538 | 0.12006 | 4.0 | 0.4875 | 0.23005 | 3.2 |
| 1M-4M | 0.4706 | 0.29185 | 2.2 | 0.6438 | 0.46899 | 2.0 |
| >4M | 0.6811 | 0.58329 | 1.5 | 0.8496 | 0.78127 | 1.6 |
| all | 0.3633 | 0.05643 | 9.5 | 0.4761 | 0.09866 | 8.3 |

* The presence target is trained at odds 9.5x the deployment odds overall (hits), up to ~16x where activity is sparse and ~1.5x in the densest cells: a presence probability calibrated on training rows over-states P(charge) by log(OR) = 0.4-2.8 nats depending on density, i.e. the operating point is density-dependent [V]. The complete truth lowers the sparse-region OR to ~9.5 because its charged population is 1.8x larger while the row budget is fixed [V for the ORs; A for the reason].
* Complete truth: 70,995 charged cells/event vs 40,064 (hits); 26,612/event are below 25 e- and 5,417/event below 1e-3 e-, all of them presence positives (y > 0) [V]. With sig_cap = 20000 binding in 195/200 events (hits: 177), the uniform cap spends ~38 % of the charged-row budget on sub-25 e- cells, so bright cells (> 16k e-) trained per event fall from 5,172 to 3,002 (-42 %) [V].
* The training noise windows are unaffected: under either truth, 0.000 (win) / <= 0.001 (win_near) of them hold any complete charge; median distance to activity 29 fine cells (win) and 5 (win_near) [V].

## B. The 25 e- cut is per (group, wire, tick) sample, not per cell (`firstp_cut.py`, `firstp_cut.json`)

Train events (200): of the cells with complete charge > 102 e-, 1.01% have hits label 0 and 4.44% have < 50 % of their charge; for > 500 e- it is 0.017% and 0.30% [V].

| density (complete, +-8 cells) | cells > 102 e- | hits = 0 | hits < 50 % | median hits/complete | charge lost |
|---|---|---|---|---|---|
| iso | 232 | 0.43% | 3.02% | 0.982 | 1.16% |
| <=10k | 343,395 | 0.85% | 5.35% | 0.917 | 3.94% |
| 10k-100k | 984,326 | 1.00% | 5.14% | 0.926 | 2.72% |
| 100k-1M | 1,952,013 | 1.05% | 4.74% | 0.951 | 1.75% |
| 1M-4M | 2,907,791 | 1.13% | 4.44% | 0.967 | 1.47% |
| >4M | 1,461,014 | 0.78% | 3.38% | 0.975 | 1.42% |

| cell charge (e-) | cells | hits = 0 | hits < 50 % | median hits/complete | charge lost |
|---|---|---|---|---|---|
| 102-250 | 1,066,459 | 6.66% | 25.43% | 0.694 | 34.54% |
| 250-500 | 913,392 | 0.60% | 5.68% | 0.838 | 18.98% |
| 500-1k | 966,655 | 0.09% | 1.40% | 0.903 | 11.46% |
| 1-2k | 971,702 | 0.01% | 0.30% | 0.946 | 6.95% |
| 2-4k | 697,047 | 0.00% | 0.05% | 0.966 | 4.70% |
| 4-8k | 485,209 | 0.00% | 0.02% | 0.973 | 3.55% |
| >8k | 2,548,307 | 0.00% | 0.00% | 0.987 | 1.27% |

Mechanism, test events (re-simulated with group identity; the cut reproduces the hits truth by construction, validated at rel. L2 0.4 % in the corpus build): cells > 102 e- by their LARGEST single per-group sample and by the number of deposit groups sharing them [V]:

| largest per-group sample (e-) | cells | hits = 0 | hits < 50 % | charge lost |
|---|---|---|---|---|
| <=25 | 45,036 | 100.00% | 100.00% | 100.00% |
| 25-50 | 332,185 | 0.00% | 41.50% | 48.95% |
| 50-100 | 566,879 | 0.00% | 3.03% | 24.47% |
| 100-250 | 909,520 | 0.00% | 0.09% | 11.83% |
| >250 | 2,927,495 | 0.00% | 0.00% | 1.37% |

| groups in the cell | cells | hits = 0 | hits < 50 % | charge lost |
|---|---|---|---|---|
| 1 | 1,246,025 | 0.51% | 3.94% | 4.35% |
| 2 | 694,188 | 1.01% | 5.46% | 4.37% |
| 3-4 | 519,058 | 1.26% | 5.77% | 3.47% |
| 5-8 | 317,556 | 1.82% | 6.72% | 2.23% |
| >8 | 2,004,288 | 0.97% | 3.13% | 1.40% |

Reading: the old label of a cell is a function of how its charge is SPLIT (over 32 samples per group and over groups), not of its charge. A cell is zeroed whenever no single group puts > 25 e- in any one of its samples, whatever its total (up to ~800 e- for one group spread evenly) [V for the counts, A for the 800 e- bound = 32 x 25]. The label noise is set by the cell's charge: among 102-250 e- cells 6.7 % are zeroed, 25 % keep < 50 % and 35 % of their charge is dropped; 250-500 e-: 0.6 % / 5.7 % / 19 %; above 1k e- nothing is zeroed but 1-7 % of the charge is still dropped [V]. At fixed charge threshold it rises with the number of sharing groups (0.5 % -> 1.8 % zeroed from 1 to 5-8 groups) and the dropped share is largest at low density (3.9 % of > 102 e- charge where the +-8-cell neighbourhood holds < 10k e-) [V]. So the hits label is noisy for faint charge (a 100-250 e- cell is labelled at 0.69 of its charge on median, and 0 one time in fifteen), grouping-dependent, and only mildly affected in dense charge, where many groups each exceed 25 e- [V].

## C. Deposit-level detection (`firstp_windows.py`, `firstp_detection.py`, `firstp_hitless.py`)

### C1. The floor's noise windows under the complete truth, with particle identity

| | bg (far), n | any complete charge | max cell > 25 / 102 / 500 e- | own particle (f_in >= 0.5) | from own / from tails (> 25 e-) |
|---|---|---|---|---|---|
| bg all | 7680 | 0.0001 | 0.0000 / 0.0000 / 0.0000 | 0.0000 | 0.0000 / 0.0000 |
| bg U | 3368 | 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.0000 | 0.0000 / 0.0000 |
| bg V | 3579 | 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.0000 | 0.0000 / 0.0000 |
| bg Y | 733 | 0.0014 | 0.0000 / 0.0000 / 0.0000 | 0.0000 | 0.0000 / 0.0000 |
| bgn all | 7680 | 0.0017 | 0.0010 / 0.0003 / 0.0000 | 0.0008 | 0.0008 / 0.0003 |
| bgn U | 2560 | 0.0020 | 0.0016 / 0.0008 / 0.0000 | 0.0012 | 0.0012 / 0.0004 |
| bgn V | 2560 | 0.0016 | 0.0008 / 0.0000 / 0.0000 | 0.0004 | 0.0004 / 0.0004 |
| bgn Y | 2560 | 0.0016 | 0.0008 / 0.0000 / 0.0000 | 0.0008 | 0.0008 / 0.0000 |

* The noise windows ARE empty under the complete truth: 1 of 7680 far windows holds any charge (4e-7 e-), none has an own particle; 0.17% of near windows hold charge, 0.026% above 102 e-, and 0.078% (6 windows) contain an own particle, all of 0.011-0.019 MeV [V]. Diffusion tails reach ~2 wires / ~6 ticks beyond a hits pixel (K_w = 2, K_t = 6 in JAXTPC), far less than the 16 wire / 128 tick padding of a bg window or the 1-cell margin of a bgn window [A, from the kernel extent].
* Completeness of the complete truth itself: particles with NO hits are left out (out of window). Re-simulated at an in-window t0, 2,767 of them per event would ALSO leave no hit (largest sample <= 25 e-) and could therefore be in the window, unseen by both truths; together they carry 158,774 e- per event (median 46 e-, 99.9 % below 210 e- per particle) [V] -- below the single-wire noise and irrelevant for detection [A]. Their step energy is large for their charge (25.8 MeV/event; heavily quenched deposits) [V].

**E_min.** Since no far window and 6 of 7680 near windows contain an own deposit (all < 0.02 MeV), every E_min in [0, 0.1] MeV gives the same noise set; E_min = 0.02 MeV is the primary choice because isolated deposits below it are not detected at all at 1 % FPR (0/5; 0.02-0.05 MeV: 2-12 % of 41, C3), i.e. they are noise-like and an own deposit that small does not make a window signal [V]. The published floor and near efficiencies are therefore UNCHANGED by the deposit-level noise definition, for every model (table C2) [V].

### C2. Detection at 1 % FPR per plane, 0.1-0.2 MeV (stored corpus)

Positives: the `iso` windows (6265 in total, 1522 at 0.1-0.2 MeV). Their surroundings: 93.1% have complete charge > 25 e- in the same 16-wire / 128-tick padding a bg window must keep empty (75.5% > 1 Q0; median ring charge 23,728 e-) [V]: the iso deposits are isolated from FOREIGN CHARGE IN THE WINDOW (< 5 %), not from activity. 'Isolated positives' keep only those whose padding holds no cell > 25 e- (107 at 0.1-0.2 MeV), the deposit-level analogue of a bg window.

| model | published (far) | noise = no own deposit >= 0.02 MeV | isolated positives | non-isolated positives | published (near) | isolated (near) |
|---|---|---|---|---|---|---|
| M4n_charge | 0.811 [0.796, 0.826] | 0.811 | 0.664 [0.625, 0.713] | 0.823 | 0.385 | 0.430 |
| M4n_pres | 0.851 [0.833, 0.859] | 0.851 | 0.692 [0.648, 0.740] | 0.863 | 0.579 | 0.579 |
| M4n_gated | 0.777 [0.769, 0.790] | 0.777 | 0.673 [0.622, 0.712] | 0.785 | 0.385 | 0.430 |
| M6d_charge | 0.790 [0.779, 0.807] | 0.790 | 0.654 [0.619, 0.709] | 0.800 | 0.534 | 0.514 |
| M6d_pres | 0.842 [0.828, 0.855] | 0.842 | 0.692 [0.651, 0.737] | 0.853 | 0.609 | 0.561 |
| M6d_gated | 0.795 [0.786, 0.805] | 0.795 | 0.682 [0.634, 0.723] | 0.804 | 0.578 | 0.542 |
| M6dc_charge | 0.258 [0.243, 0.274] | 0.258 | 0.252 [0.198, 0.280] | 0.258 | 0.001 | 0.000 |
| M6dc_pres | 0.830 [0.814, 0.844] | 0.830 | 0.692 [0.633, 0.731] | 0.841 | 0.611 | 0.570 |
| M6dc_gated | 0.793 [0.781, 0.804] | 0.793 | 0.654 [0.605, 0.706] | 0.804 | 0.472 | 0.439 |
| lb_dec_hits_charge | 0.694 [0.674, 0.713] | 0.694 | 0.561 [0.510, 0.602] | 0.705 | 0.284 | 0.299 |
| lb_dec_hits_pres | 0.701 [0.684, 0.723] | 0.701 | 0.579 [0.532, 0.624] | 0.710 | 0.486 | 0.439 |
| lb_dec_hits_gated | 0.701 [0.678, 0.718] | 0.701 | 0.570 [0.515, 0.611] | 0.711 | 0.309 | 0.318 |
| lb_dec_ct_charge | 0.697 [0.680, 0.721] | 0.697 | 0.570 [0.523, 0.608] | 0.707 | 0.403 | 0.393 |
| lb_dec_ct_pres | 0.742 [0.724, 0.756] | 0.742 | 0.570 [0.530, 0.607] | 0.755 | 0.524 | 0.449 |
| lb_dec_ct_gated | 0.738 [0.721, 0.749] | 0.738 | 0.570 [0.527, 0.609] | 0.751 | 0.429 | 0.393 |
| lb_dec_cond_hits_charge | 0.454 [0.433, 0.476] | 0.454 | 0.449 [0.391, 0.500] | 0.454 | 0.039 | 0.037 |
| lb_dec_cond_hits_pres | 0.734 [0.713, 0.752] | 0.734 | 0.542 [0.491, 0.587] | 0.748 | 0.463 | 0.411 |
| lb_dec_cond_hits_gated | 0.720 [0.698, 0.732] | 0.720 | 0.542 [0.499, 0.591] | 0.734 | 0.227 | 0.262 |
| lb_dec_cond_ct_charge | 0.298 [0.264, 0.354] | 0.298 | 0.346 [0.296, 0.406] | 0.294 | 0.001 | 0.000 |
| lb_dec_cond_ct_pres | 0.701 [0.679, 0.721] | 0.701 | 0.551 [0.504, 0.599] | 0.712 | 0.507 | 0.439 |
| lb_dec_cond_ct_gated | 0.698 [0.669, 0.721] | 0.698 | 0.551 [0.505, 0.602] | 0.710 | 0.340 | 0.355 |

Per plane, published vs isolated positives (far threshold, 0.1-0.2 MeV):

| model | U pub | U iso | V pub | V iso | Y pub | Y iso |
|---|---|---|---|---|---|---|
| M4n_pres | 0.832 | 0.714 | 0.781 | 0.500 | 0.951 | 0.914 |
| M4n_gated | 0.729 | 0.643 | 0.670 | 0.500 | 0.953 | 0.914 |
| M6d_pres | 0.800 | 0.714 | 0.787 | 0.500 | 0.949 | 0.914 |
| M6d_gated | 0.752 | 0.714 | 0.701 | 0.477 | 0.949 | 0.914 |
| M6dc_pres | 0.790 | 0.714 | 0.763 | 0.500 | 0.951 | 0.914 |
| M6dc_gated | 0.745 | 0.679 | 0.707 | 0.455 | 0.945 | 0.886 |
| lb_dec_hits_pres | 0.633 | 0.464 | 0.565 | 0.409 | 0.932 | 0.886 |
| lb_dec_hits_gated | 0.611 | 0.464 | 0.581 | 0.409 | 0.936 | 0.857 |
| lb_dec_ct_pres | 0.683 | 0.464 | 0.617 | 0.409 | 0.949 | 0.857 |
| lb_dec_ct_gated | 0.681 | 0.464 | 0.610 | 0.409 | 0.947 | 0.857 |

### C3. Efficiency vs deposit energy (far threshold): the low-energy 'detections' are activity nearby

| model | positives | 0.01-0.02 | 0.02-0.05 | 0.05-0.1 | 0.1-0.2 MeV |
|---|---|---|---|---|---|
| M6d_pres | all (published) | 0.281 (n=263) | 0.290 (n=905) | 0.464 (n=1449) | 0.842 (n=1522) |
| M6d_pres | isolated | 0.000 (n=5) | 0.073 (n=41) | 0.250 (n=76) | 0.692 (n=107) |
| M6d_gated | all (published) | 0.202 (n=263) | 0.215 (n=905) | 0.386 (n=1449) | 0.795 (n=1522) |
| M6d_gated | isolated | 0.000 (n=5) | 0.024 (n=41) | 0.171 (n=76) | 0.682 (n=107) |
| lb_dec_hits_pres | all (published) | 0.198 (n=263) | 0.225 (n=905) | 0.342 (n=1449) | 0.701 (n=1522) |
| lb_dec_hits_pres | isolated | 0.000 (n=5) | 0.049 (n=41) | 0.118 (n=76) | 0.579 (n=107) |
| lb_dec_ct_pres | all (published) | 0.373 (n=263) | 0.380 (n=905) | 0.462 (n=1449) | 0.742 (n=1522) |
| lb_dec_ct_pres | isolated | 0.000 (n=5) | 0.122 (n=41) | 0.079 (n=76) | 0.570 (n=107) |
| lb_dec_hits_gated | all (published) | 0.243 (n=263) | 0.228 (n=905) | 0.349 (n=1449) | 0.701 (n=1522) |
| lb_dec_hits_gated | isolated | 0.000 (n=5) | 0.049 (n=41) | 0.118 (n=76) | 0.570 (n=107) |
| lb_dec_ct_gated | all (published) | 0.342 (n=263) | 0.358 (n=905) | 0.458 (n=1449) | 0.738 (n=1522) |
| lb_dec_ct_gated | isolated | 0.000 (n=5) | 0.122 (n=41) | 0.132 (n=76) | 0.570 (n=107) |

A 0.01-0.02 MeV deposit (~300-600 e- before response, at or below one wire's noise) is 'detected' 20-37 % of the time at a 1 % false-positive rate when it sits near activity, and 0 % (0/5) when it is isolated [V]: the far threshold credits the score that nearby activity lends the window (haze, or the decoder's receptive field), not the deposit [A for the mechanism; V for the numbers]. The near-window threshold (bgn) removes most of it (near efficiencies at 0.01-0.05 MeV are 1-8 %) [V].

### C4. The label arms' gain (complete-truth label minus hits label), paired bootstrap

| pair | score | published (far) | noise redefined | isolated positives (far) | non-isolated (far) | published (near) | isolated (near) | draws-mean far: published / isolated |
|---|---|---|---|---|---|---|---|---|
| lb_dec_ct - lb_dec_hits | charge | +0.003 [-0.013, +0.026] | +0.003 | +0.009 [-0.028, +0.042] | +0.002 [-0.016, +0.026] | +0.119 [+0.083, +0.146] | +0.093 [+0.033, +0.146] | +0.007 / +0.015 |
| lb_dec_ct - lb_dec_hits | pres | +0.041 [+0.018, +0.057] | +0.041 | -0.009 [-0.042, +0.018] | +0.045 [+0.020, +0.061] | +0.039 [+0.033, +0.062] | +0.009 [-0.021, +0.051] | +0.009 / +0.007 |
| lb_dec_ct - lb_dec_hits | gated | +0.037 [+0.016, +0.054] | +0.037 | +0.000 [-0.025, +0.031] | +0.040 [+0.017, +0.057] | +0.120 [+0.069, +0.141] | +0.075 [+0.021, +0.126] | +0.016 / +0.007 |
| lb_dec_cond_ct - lb_dec_cond_hits | charge | -0.156 [-0.188, -0.101] | -0.156 | -0.103 [-0.146, -0.054] | -0.160 [-0.191, -0.102] | -0.038 [-0.064, -0.030] | -0.037 [-0.056, -0.015] | -0.128 / -0.080 |
| lb_dec_cond_ct - lb_dec_cond_hits | pres | -0.033 [-0.050, -0.011] | -0.033 | +0.009 [-0.023, +0.050] | -0.036 [-0.054, -0.014] | +0.044 [+0.024, +0.057] | +0.028 [+0.000, +0.045] | +0.004 / -0.004 |
| lb_dec_cond_ct - lb_dec_cond_hits | gated | -0.022 [-0.042, +0.000] | -0.022 | +0.009 [-0.026, +0.046] | -0.024 [-0.045, +0.000] | +0.113 [+0.064, +0.153] | +0.093 [+0.029, +0.146] | +0.007 / +0.006 |

### Verdict

1. **Noise windows (C1):** under the complete truth the floor's far and near noise windows are empty of charge and of own deposits (no far window, 0.08 % of near windows with a 0.01-0.02 MeV deposit). Redefining 'noise' as 'no own deposit above E_min' changes no number: 100 % of every detection result, and of the label arms' gain, survives THIS change [V].
2. **Positives (C2-C3):** the real ambiguity is on the positive side. 93 % of the `iso` positives sit within 16 wires / 128 ticks of other charge, while the noise windows by construction do not; at the far threshold a window's score includes what nearby activity lends it. On deposit-level positives (as empty around as a noise window) every useful score loses 0.10-0.19 of efficiency at 0.1-0.2 MeV (M6d pres 0.842 -> 0.692, M4n pres 0.851 -> 0.692), with only 107 such windows (+-0.045) [V]. The ranking M4n ~ M6d ~ M6dc > label arms is preserved [V].
3. **Label arms (C4):** the published far-threshold gain of lb_dec_ct over lb_dec_hits (+0.041 pres, +0.037 gated, stored corpus) (a) shrinks to +0.009 / +0.016 as a mean over the five fresh noise draws, before any redefinition [V]; (b) is entirely carried by the non-isolated positives (+0.045 / +0.040) and is -0.009 / +0.000 (intervals spanning 0) on isolated ones [V]; (c) is largest at 0.01-0.05 MeV (+0.15-0.18 on all positives), where isolated deposits are detected at <= 12 % by every model [V]. So the far-threshold gain is consistent with the complete label teaching presence in the diffuse surroundings of activity, which that threshold credits, not with better detection of an isolated deposit [A: inference; the 107-window sample cannot exclude a true gain of ~0.04]. What does survive is the NEAR-threshold gain of the charge and gated outputs (+0.12 published; on isolated positives +0.09 charge / +0.08 gated): with the hits label those outputs leave a haze next to activity that the near noise windows punish, and the complete label removes part of it [V for the numbers; A for the reading]. The cond arms lose with the complete label at the far threshold (-0.02 to -0.16) [V].
4. **Sampling (A) and the cut (B):** the presence head is trained at 9.5x the deployment odds (16x in sparse regions); switching to the complete label adds 31k sub-25 e- positives per event, lowers that ratio in sparse regions to ~9.5x and, through the fixed sig_cap, removes 42 % of the bright training cells. The hits label's own noise is concentrated in faint cells (102-250 e-: 6.7 % zeroed, 35 % of the charge dropped) and grows with the number of deposit groups sharing a cell [V].

