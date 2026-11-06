# The ROI side of the MicroBooNE chain: audit, diagnosis, faithful port, ROC, decomposition

128 test events (truth_v2 ev260-ev387). The ROI-finding input is R2/R3's input: digitize(clean + incoherent + coherent), then our gate. The noise realisation is the same as everywhere else in this baseline (seed 1000 + ev).

The scoring is `eval_denoise`'s:
- **Windows:** 18,994 floor windows plus the 7,680 near-activity `bgn` windows of `truth_v2_near`.
- **Annotations:** `snr`/`clean` from `noise_vs_hits.npz`.
- **Metrics:** `helix.probe.resolution.window_stats` / `scalars` / `near_scalars`.

The model compared is **M4n_ft_full_near, presence-scored**.
- **Window metrics:** from its own near-scored windows file. These are our windows, in the same order (verified). The scorer reproduces its dn_near.jsonl numbers: near eff@1% 0.1-0.2 MeV = 0.558, SNR 7-10 = 0.81.
- **Per-cell ROI decomposition:** from a CPU replay of its per-cell outputs (`model_presence.py`).

Labels: **[V]** verified (run, or code line read); **[A]** assumed / not checked.

## Headline

| | far eff@1% <0.1 / 0.1-0.2 / 0.2-0.5 MeV | near eff@1% <0.1 / 0.1-0.2 / 0.2-0.5 | near_fpr_at_far1pct | SNR 7-10 far / near |
|---|---|---|---|---|
| example ROI chain, as shipped (ex_s1) | 0.006 / 0.004 / 0.007 | 0.014 / 0.013 / 0.023 | 0.004 | 0.01 / 0.02 |
| example chain, best threshold scale (post hoc) | 0.015 / 0.012 / 0.059 | 0.024 / 0.026 / 0.191 | - | - |
| **Wire-Cell port, MicroBooNE config (wc_s1)** | 0.010 / 0.096 / 0.390 | 0.016 / 0.116 / 0.397 | 0.006 | 0.21 / 0.23 |
| Wire-Cell port, U-plane fake removal applied (wc_fixU) | 0.021 / 0.019 / 0.124 | 0.021 / 0.019 / 0.124 | 0.001 | 0.00 / 0.00 |
| Wire-Cell port, no fake-signal removal (wc_nofake) | 0.008 / 0.101 / 0.832 | 0.010 / 0.141 / 0.861 | 0.009 | 0.17 / 0.21 |
| Wire-Cell port, best setting per bin (post hoc) | 0.026 / 0.118 / 0.832 | 0.026 / 0.141 / 0.861 | - | - |
| hybrid: U,V wc_fixU + Y wc_nofake (post hoc) | 0.073 / 0.278 / 0.397 | 0.073 / 0.278 / 0.397 | - | 0.22 / 0.22 |
| **model M4n_ft_full_near (presence)** | **0.403 / 0.827 / 0.993** | **0.065 / 0.558 / 0.975** | 0.141 | **0.92 / 0.81** |

"Best" and "hybrid" rows are chosen on these same test events, so they are optimistic for the classical chains.

Per plane, each plane with its own 1% far threshold, at 0.1-0.2 MeV:

| | U | V | Y |
|---|---|---|---|
| Wire-Cell port, no fake removal | 0.21 | 0.04 | 0.85 |
| model | 0.83 | 0.78 | 0.95 |

So on the **collection plane** the faithful Wire-Cell ROI finder, without MicroBooNE's absolute fake-signal cut, comes within ~0.1 of the model at the far threshold. Near activity the model's Y efficiency is 0.90 and the classical one stays at 0.85. The classical chain's ROI is *defined* away from neighbouring charge only by its thresholds, so it has almost no near-activity haze: its near_fpr_at_far1pct is ≤ 0.01, against the model's 0.141.

The large gap is on the **induction planes and at <0.1 MeV everywhere**. There the model is far ahead at the far threshold, and still ahead near activity except below 0.1 MeV: at the near threshold the model gets 0.065, the classical chains 0.01-0.07.

## 1. Audit (full table with file:line citations: `roi_audit.md`)

Parameter by parameter, the example's ROI code against Wire-Cell Toolkit master @ b7bd2a1a with the MicroBooNE configuration (sources in `reference/wirecell/`) and against arXiv:1802.08709. The decisive differences [V], all with noise in mind:

1. **Filters.** The example's numbers were "read off" the paper's figures and are wrong by large factors. Both the code (`sp-filters.jsonnet:41-61`) and the figures say:
   - Wiener_tight σ ≈ 0.144-0.150 MHz with p ≈ 4.1-4.2; the example has a = 0.28 / 0.35, about 2× too wide.
   - Gaus_wide σ = 0.111 MHz; the example has 0.24 / 0.30.
   - Wire_ind a = 0.395 cycles/wire; the example has 0.11, 3.6× more cross-wire smoothing.
   - Wire_col a = 0.846; the example has 0.30.
   - ROI LF τ = 0.02 (tight), 0.1 (tighter, missing in the example) and 0.0025 MHz (loose); the example has 0.035 / - / 0.010.
2. **Noise RMS.**
   - Wire-Cell: per wire, on the tight-LF output (collection: Wiener_tight). It takes a 16/50/84-percentile seed, then the RMS of samples with |x| < 5·seed (`ROI_formation.cxx:364-405`). That one RMS is reused for every refinement threshold.
   - The example: the seed only, recomputed on every array.
   - Measured on our noise, both estimators agree with the true per-wire noise std within 1-5% (§2). **This is not a source of fakes.**
3. **Tight threshold.** Wire-Cell uses a **signed** `3·rms + 1` on induction (`OmnibusSigProc.h:135`; the paper's 3.5 is not what the code does) and `5·rms + 1` on collection. The example uses `|x| > 3.5·rms`, which doubles the noise crossings.
4. **Loose ROIs.** Wire-Cell rebins by 6 ticks, cuts at 21·rms (capped at 10k), extends edges to 0.7·th and then a local minimum, applies prominence and ±3-bin rules and merges gaps (`ROI_formation.cxx:705-957`). The example applies `|x| > 3.5σ` on unbinned data.
5. **Refinement.** Wire-Cell runs:
   - a 3σ re-check on baseline-subtracted no-LF contents;
   - cluster CleanUp;
   - 2 × {BreakROI (TSpectrum-style peaks, ≥1200 e separation), CheckROIs (link validation), CleanUp};
   - Shrink;
   - **fake-signal removal: an ROI must have a bin > 1000 e or a mean > 500 e, or be linked to such an ROI** (`ROI_refinement.cxx:1289-1612`);
   - Extend.

   The example has none of the fake-signal removal and **copies every tight ROI onto the neighbouring wires**.
6. **Executed vs intended.** At master the **U-plane fake-signal removal is a no-op**: the bad ROIs are collected and never removed (`ROI_refinement.cxx:1460-1520`, against V at 1580-1606). The faithful port reproduces this; `wc_fixU` applies the removal.
7. **FFT padding.** Wire-Cell **zero-pads the FFT** in ticks and wires (`OmnibusSigProc.cxx:863-891`). That is the readout-edge fix found empirically earlier (README Addition 1). The example does not pad.

The paper's published text describes only tight/loose formation and Clean/Break/Shrink. The rebinning, the local-maximum rules, CheckROIs and the fake-signal thresholds appear only in draft text commented out in the arXiv LaTeX source, where the thresholds are 1200 / 1500 e.

## 2. Diagnosis (the example's chain; 32 events; full tables below)

**Fake induction ROIs are created at the very first step, and nothing removes them.**
- **Raw tight ROIs** (|tight-LF| > 3.5·RMS, pad 10) already put an ROI pixel in **54% (U) / 17% (V) of far windows** and 50% / 21% of near windows.
- **Later steps add more.** Loose ROIs and reconciliation take U to 0.56. The tight-ROI copy onto w±1 ("connect") roughly doubles the number of ROIs on charge-free wires: 1,712 → 3,318 per U plane. Clean, Break and Shrink remove almost nothing: U 0.576 → 0.564.

**The thresholds are not miscalibrated.**
- **RMS used vs true noise:** in each filtered domain the RMS the example uses (robust RMS of signal + noise) is 1.01-1.05 × the true noise std measured on noise-only planes from the same realisation.
- **Crossing rates:** the per-sample crossing rate of the noise alone is 3.5e-4 (U), 1.9e-4 (V) and 5.9e-7 (Y), against the Gaussian expectation of 4.7e-4 / 4.7e-4 / 5.7e-7.
- **Why that still floods the windows:** a window holds 4,096 samples, so a ~3.5σ two-sided per-sample cut on independent-ish samples must put a fake in about half the windows.

**The collection-plane misses are a threshold vs per-tick peak problem.** The Y ROI threshold is 5 × 252 = 1,280 e per tick on the example's Wiener output. A deposit's noise-free per-tick peak in that domain is:

| deposit | median peak (e per tick) | peak / threshold |
|---|---|---|
| <0.1 MeV | 193 | 0.15 |
| 0.1-0.2 MeV | 335 | 0.26 |
| 0.2-0.5 MeV | 911 | 0.72 |
| SNR 7-10 | 260 | 0.21 |

Only 1% of 0.1-0.2 MeV deposits peak above threshold. The per-tick cut throws away the matched-filter gain: the SNR in `noise_vs_hits` integrates the whole deposit.

**With Wire-Cell's own filters the Y-plane Wiener noise drops from 252 to ~35 e per tick** [V, one event]. The narrower time filter removes the high-frequency noise that dividing by the response amplifies. That is why the faithful port finds 84% of 0.1-0.2 MeV Y deposits when the fake-signal cut is off.

## 3. Faithful port and sweep

`sp/wirecell_sp.py` is a Python port of OmnibusSigProc's ROI path. Its docstring cites every function's file:line.
- **Ported:** the filters and restore_baseline, tight/tighter/loose formation, the i/i+2 connectivity, load_data's 3σ re-check and link graph, CleanUpROIs, generate_merge_ROIs, BreakROI/BreakROI1 (with a line-by-line port of PeakFinding::SearchHighRes), CheckROIs, ShrinkROI, CleanUp{Induction,Collection}ROIs, ExtendROIs, and apply_roi with per-sample negative zeroing.
- **Not ported:** bad/noisy-channel masks (we have none), multi-plane protection (off for MicroBooNE) and L1SP.
- **[A]:** the FFT pad sizes (200 ticks, 20 wires), and the 1e-6·max|R| floor in the division (Wire-Cell divides raw).
- **[A], not checked:** whether the port is bit-identical to running Wire-Cell itself. It was checked only against the code by reading, plus unit checks of the peak finder on synthetic peaks.

The example's chain is kept as its own family (`ex_*`).

**Sweep.**
- **Joint scales:** threshold factors × {0.67, 0.83, 1, 1.25, 1.5, 2, 3}, jointly for tight, collection and loose. The first 20 events also have ×0.5, which was dropped for the rest to save time; only settings present in every file are scored.
- **Independent settings:** tight ×0.67, loose ×0.67, tight ×1.5, no fake removal, U fake removal fixed.
- **Oracles:** each chain's filters and charge rule with ROIs from truth.

Tables are below, and every setting is in `roi_sweep.json`. Figures: `figs/roi_roc_far.png`, `figs/roi_roc_near.png`.

Reading the sweep:
- **Example chain (ex_s1):** lowering the threshold buys ROI efficiency only at a matching cost in noise windows with ROI charge: at scale 0.83, 0.56 at 0.1-0.2 MeV for a far roi_fpr of 0.68. Raising it to scale 1.5 cuts the far roi_fpr to 0.013 but leaves 0.026 at 0.1-0.2 MeV. Its eff@1% never exceeds 0.06 below 0.5 MeV.
- **Wire-Cell port (wc_s1):** at the default setting U has a 0.44 far roi_fpr (the U no-op) and V and Y have ~0. Below 0.2 MeV, V and Y are killed by the 1000 / 500 e fake-signal cut. Fixing U kills U too: eff@1% 0.019 at 0.1-0.2 MeV.
- **Wire-Cell port, no fake cut (wc_nofake):** the threshold scale hardly matters, because the fake-signal cut, not the σ thresholds, decides below 0.5 MeV. Turning the cut off gives Y 0.85 per plane, but V fakes (far roi_fpr 0.31) and U fakes dominate the pooled threshold.

## 4. ROI-level decomposition (tables below)

- **ROI efficiency vs SNR.** These are clean deposits; "found" means a truth pixel lies inside a found ROI.

  | | SNR 5-7 | SNR 7-10 | SNR 10-15 |
  |---|---|---|---|
  | Wire-Cell port, no fake cut: U | 0.57 | 0.93 | 0.98 |
  | Wire-Cell port, no fake cut: V | 0.45 | 0.86 | 0.99 |
  | Wire-Cell port, no fake cut: Y | 0.13 | 0.55 | 0.96 |
  | example chain: U | 0.10 | 0.22 | 0.63 |
  | example chain: Y | 0.00 | 0.00 | 0.01 |

  The model's 'ROI' is cells with presence above its own window threshold. At the far threshold (presence 0.41) it finds 0.55 at SNR 5-7 and 0.90 at SNR 7-10; at the near threshold (0.96) 0.33 and 0.80 (all planes).
- **Boundary loss.** A found faint deposit (0.1-0.2 MeV) has only a median **72% of its true charge inside the Wire-Cell ROI**. This rises to 94-95% at 0.2-0.5 MeV and 99% above 0.5 MeV. The example's ROIs hold ~100% of it, but on induction they cover 2.4-2.7× the Wire-Cell area (U 92.5M vs 34.3M ROI pixels over the 128 events; V 58.5M vs 24.1M). The model's above-threshold cells hold 100% of a found deposit's truth charge.
- **Charge inside found vs oracle ROIs.** Reconstructed / true charge in the deposit's window box, Wire-Cell port (both chains in the tables below):

  | | found ROIs | oracle ROIs |
  |---|---|---|
  | 0.1-0.2 MeV | 0.83 ± 0.37 | 0.92 ± 0.26 |
  | 0.2-0.5 MeV | 0.93 ± 0.19 | 1.02 ± 0.13 |
  | 0.5-1 MeV | 0.98 ± 0.07 | 1.02 ± 0.07 |

  The example's induction charge is unusable: median -1.1 to 1.5, widths of several hundred percent. This comes from LF wander under its two-point baselines.
- **Purity.** The fraction of ROI pixel area in truth-empty cells:

  | | U | V | Y |
  |---|---|---|---|
  | Wire-Cell port (wc_s1) | 0.29 | 0.15 | 0.11 |
  | example (ex_s1) | 0.65 | 0.53 | 0.05 |
  | oracle (the 10-tick pad) | 0.17 | 0.17 | 0.17 |
  | model, far threshold (covered cells) | 0.37 | 0.43 | 0.23 |
  | model, near threshold (covered cells) | 0.01 | 0.01 | 0.01 |

  For the model at the far threshold, its '1%-FPR' cells are mostly haze next to real charge.

## 5. Figures

- `figs/roi_roc_far.png` and `figs/roi_roc_near.png`: ROI operating points per threshold scale for both chains, score-thresholded ROCs of the default settings, the independent settings, the post-hoc hybrid, and the model's ROC, per energy bin.
- `figs/roi_display_ev300.png`, `figs/roi_display_ev321.png` and `figs/roi_display_ev350.png`: the noisy input after coherent removal (U, V, Y) with oracle ROIs (green), the example's ROIs (blue) and the Wire-Cell port's ROIs (red), plus the iso / bg windows.

## Verified vs assumed (summary)

**[V]:**
- every Wire-Cell parameter and rule quoted (code lines read; `roi_audit.md`);
- the U-plane removal no-op;
- the noise RMS / crossing-rate measurements (noise-only planes from the same realisation);
- the example-chain stage accounting;
- the sweep and its scoring through `helix.probe.resolution` (the model's window file reproduces dn_near.jsonl);
- the model windows being identical to ours.

**[A]:**
- the port matching Wire-Cell output bit for bit (not run against Wire-Cell);
- the FFT pad sizes;
- the deconvolution floor;
- that MicroBooNE production used master's behaviour. The U no-op and the fake thresholds may differ in the version MicroBooNE ran; the paper's draft text says 1200 / 1500 e.
- The "best"/hybrid rows are selected on the test events.
- A single noise realisation per event.

---

# Tables

## 2. Diagnosis tables (example chain, 32 events)

### Noise RMS per filtered domain (median over wires; e- per tick)

| plane | domain | true noise std (noise-only plane) | robust RMS of noise-only | RMS the example uses (signal+noise, robust) | used / true (p10 / p50 / p90 over wires) | threshold (e-) | threshold / true std | per-sample crossing rate, noise alone (Gaussian expectation) |
|---|---|---|---|---|---|---|---|---|
| U | wiener | 82 | 82 | 85 | 0.99 / 1.02 / 1.14 | - | - | - |
| U | tight | 80 | 80 | 83 | 0.99 / 1.01 / 1.09 | 289 | 3.55 | 3.55e-04 (4.65e-04) |
| U | loose | 81 | 81 | 84 | 0.99 / 1.02 / 1.14 | 294 | 3.56 | 3.23e-04 (4.65e-04) |
| U | gauss | 429 | 437 | 444 | 1.02 / 1.05 / 1.09 | - | - | - |
| V | wiener | 134 | 134 | 144 | 0.99 / 1.05 / 1.26 | - | - | - |
| V | tight | 129 | 130 | 137 | 0.99 / 1.04 / 1.19 | 481 | 3.64 | 1.97e-04 (4.65e-04) |
| V | loose | 130 | 131 | 139 | 0.99 / 1.04 / 1.22 | 488 | 3.65 | 1.87e-04 (4.65e-04) |
| V | gauss | 402 | 402 | 413 | 0.99 / 1.02 / 1.09 | - | - | - |
| Y | wiener | 252 | 251 | 255 | 0.99 / 1.01 / 1.05 | 1276 | 5.07 | 5.91e-07 (5.73e-07) |
| Y | gauss | 597 | 600 | 620 | 1.00 / 1.02 / 1.05 | - | - | - |

### Stage-by-stage: fraction of windows with any ROI pixel (U, V pooled per plane type)

| plane | stage | far bg | near bgn | iso <0.1 | iso 0.1-0.2 | iso 0.2-0.5 | iso >0.5 | ROI pixel fraction | ROIs on charge-free wires / event-plane |
|---|---|---|---|---|---|---|---|---|---|
| U | tight_raw | 0.542 | 0.502 | 0.529 | 0.598 | 0.879 | 1.000 | 0.0265 | 1558 |
| U | loose_raw | 0.479 | 0.425 | 0.481 | 0.535 | 0.894 | 1.000 | 0.0327 | 1574 |
| U | reconciled | 0.559 | 0.522 | 0.573 | 0.622 | 0.924 | 1.000 | 0.0349 | 1712 |
| U | connect | 0.576 | 0.542 | 0.592 | 0.654 | 0.939 | 1.000 | 0.0434 | 3318 |
| U | clean | 0.564 | 0.534 | 0.568 | 0.630 | 0.909 | 1.000 | 0.0424 | 3229 |
| U | break | 0.564 | 0.534 | 0.568 | 0.630 | 0.909 | 1.000 | 0.0424 | 3229 |
| U | shrink | 0.564 | 0.534 | 0.568 | 0.630 | 0.909 | 1.000 | 0.0423 | 3229 |
| V | tight_raw | 0.174 | 0.212 | 0.234 | 0.230 | 0.461 | 0.981 | 0.0187 | 619 |
| V | loose_raw | 0.175 | 0.208 | 0.221 | 0.257 | 0.487 | 0.990 | 0.0224 | 642 |
| V | reconciled | 0.187 | 0.236 | 0.260 | 0.257 | 0.487 | 0.990 | 0.0239 | 700 |
| V | connect | 0.197 | 0.247 | 0.277 | 0.277 | 0.500 | 0.990 | 0.0274 | 1240 |
| V | clean | 0.184 | 0.234 | 0.251 | 0.270 | 0.474 | 0.981 | 0.0269 | 1195 |
| V | break | 0.184 | 0.234 | 0.251 | 0.270 | 0.474 | 0.981 | 0.0268 | 1196 |
| V | shrink | 0.184 | 0.234 | 0.251 | 0.270 | 0.474 | 0.981 | 0.0268 | 1196 |
| Y | collection | 0.032 | 0.003 | 0.010 | 0.017 | 0.292 | 0.970 | 0.0098 | 3 |

### Isolated deposits: noise-free peak in the thresholded domain vs the threshold on its wire

(induction: tight-LF Wiener output, |value|; collection: Wiener output)

| plane | energy bin | n | median peak (e-) | median threshold (e-) | median noise std (e-) | peak/threshold p25 / p50 / p75 | frac peak > threshold |
|---|---|---|---|---|---|---|---|
| U | 0-0.1 | 206 | 57 | 319 | 87 | 0.05 / 0.18 / 0.30 | 0.03 |
| U | 0.1-0.2 | 127 | 144 | 316 | 87 | 0.38 / 0.48 / 0.65 | 0.03 |
| U | 0.2-0.5 | 66 | 345 | 312 | 86 | 0.93 / 1.10 / 1.35 | 0.62 |
| U | 0.5-1 | 28 | 816 | 322 | 87 | 2.19 / 2.51 / 3.20 | 1.00 |
| U | SNR 0-1e-09 | 95 | 16 | 313 | 86 | 0.04 / 0.05 / 0.13 | 0.02 |
| U | SNR 1e-09-3 | 37 | 55 | 325 | 88 | 0.15 / 0.17 / 0.25 | 0.03 |
| U | SNR 3-5 | 80 | 90 | 321 | 88 | 0.26 / 0.30 / 0.34 | 0.03 |
| U | SNR 5-7 | 59 | 136 | 313 | 86 | 0.39 / 0.44 / 0.51 | 0.02 |
| U | SNR 7-10 | 44 | 206 | 314 | 85 | 0.55 / 0.66 / 0.74 | 0.00 |
| U | SNR 10-15 | 36 | 270 | 313 | 87 | 0.82 / 0.93 / 1.01 | 0.25 |
| U | SNR 15-1e+09 | 148 | 1205 | 318 | 87 | 1.56 / 3.89 / 6.30 | 0.93 |
| V | 0-0.1 | 231 | 68 | 513 | 133 | 0.08 / 0.13 / 0.19 | 0.00 |
| V | 0.1-0.2 | 148 | 132 | 512 | 135 | 0.20 / 0.26 / 0.35 | 0.02 |
| V | 0.2-0.5 | 76 | 341 | 519 | 132 | 0.49 / 0.62 / 0.89 | 0.18 |
| V | 0.5-1 | 28 | 787 | 509 | 135 | 1.34 / 1.54 / 1.76 | 1.00 |
| V | SNR 0-1e-09 | 105 | 39 | 520 | 133 | 0.05 / 0.08 / 0.12 | 0.00 |
| V | SNR 1e-09-3 | 47 | 69 | 513 | 136 | 0.10 / 0.13 / 0.17 | 0.00 |
| V | SNR 3-5 | 137 | 104 | 509 | 132 | 0.16 / 0.20 / 0.25 | 0.01 |
| V | SNR 5-7 | 65 | 158 | 517 | 136 | 0.24 / 0.31 / 0.34 | 0.02 |
| V | SNR 7-10 | 45 | 217 | 510 | 129 | 0.35 / 0.44 / 0.49 | 0.02 |
| V | SNR 10-15 | 35 | 342 | 527 | 137 | 0.59 / 0.61 / 0.72 | 0.06 |
| V | SNR 15-1e+09 | 125 | 1366 | 507 | 132 | 1.43 / 2.48 / 4.16 | 0.92 |
| Y | 0-0.1 | 204 | 193 | 1288 | 252 | 0.12 / 0.15 / 0.18 | 0.00 |
| Y | 0.1-0.2 | 120 | 335 | 1280 | 251 | 0.21 / 0.26 / 0.36 | 0.01 |
| Y | 0.2-0.5 | 72 | 911 | 1279 | 251 | 0.54 / 0.72 / 0.89 | 0.19 |
| Y | 0.5-1 | 23 | 1871 | 1304 | 254 | 1.17 / 1.50 / 2.01 | 0.78 |
| Y | SNR 0-1e-09 | 41 | 68 | 1304 | 251 | 0.03 / 0.05 / 0.09 | 0.00 |
| Y | SNR 1e-09-3 | 39 | 171 | 1288 | 251 | 0.12 / 0.13 / 0.16 | 0.00 |
| Y | SNR 3-5 | 50 | 188 | 1285 | 252 | 0.13 / 0.15 / 0.17 | 0.00 |
| Y | SNR 5-7 | 50 | 214 | 1282 | 252 | 0.14 / 0.17 / 0.19 | 0.00 |
| Y | SNR 7-10 | 64 | 260 | 1274 | 251 | 0.17 / 0.21 / 0.24 | 0.00 |
| Y | SNR 10-15 | 57 | 386 | 1299 | 254 | 0.23 / 0.31 / 0.36 | 0.02 |
| Y | SNR 15-1e+09 | 195 | 1655 | 1284 | 252 | 0.63 / 1.24 / 3.51 | 0.56 |

## 3-4. Sweep, decomposition and model tables (128 events)

128 events, 26674 windows (floor + near-activity); model windows identical to ours: True.

### Every setting: ROI operating point and score-thresholded efficiency at 1% FPR

`roi_fpr` = fraction of noise windows with ANY ROI charge; `roi_eff` = fraction of isolated-deposit windows with any ROI charge. `eff@1%` uses the 99th percentile of the far (bg) or near (bgn) window scores as the threshold.

| setting | roi_fpr far / near | roi_eff <0.1 / 0.1-0.2 / 0.2-0.5 / 0.5-1 | far eff@1% <0.1 / 0.1-0.2 / 0.2-0.5 | near eff@1% <0.1 / 0.1-0.2 / 0.2-0.5 | near_fpr_at_far1pct |
|---|---|---|---|---|---|
| ex_oracle | 0.000 / 0.000 | 0.824 / 0.903 / 0.955 / 0.984 | 0.824 / 0.903 / 0.955 | 0.824 / 0.903 / 0.955 | 0.000 |
| wc_oracle | 0.000 / 0.000 | 0.987 / 0.999 / 1.000 / 0.997 | 0.987 / 0.999 / 1.000 | 0.987 / 0.999 / 1.000 | 0.000 |
| ex_s0.67 | 0.940 / 0.832 | 0.881 / 0.892 / 0.966 / 0.997 | 0.005 / 0.004 / 0.003 | 0.009 / 0.009 / 0.013 | 0.003 |
| ex_s0.83 | 0.681 / 0.505 | 0.526 / 0.560 / 0.750 / 0.975 | 0.003 / 0.005 / 0.005 | 0.010 / 0.009 / 0.011 | 0.004 |
| ex_s1 | 0.264 / 0.189 | 0.215 / 0.228 / 0.458 / 0.937 | 0.006 / 0.004 / 0.007 | 0.014 / 0.013 / 0.023 | 0.004 |
| ex_s1.25 | 0.024 / 0.014 | 0.038 / 0.039 / 0.227 / 0.869 | 0.006 / 0.003 / 0.026 | 0.024 / 0.022 / 0.191 | 0.002 |
| ex_s1.5 | 0.013 / 0.005 | 0.024 / 0.026 / 0.137 / 0.773 | 0.008 / 0.007 / 0.059 | 0.024 / 0.026 / 0.137 | 0.002 |
| ex_s2 | 0.010 / 0.003 | 0.015 / 0.012 / 0.053 / 0.481 | 0.015 / 0.012 / 0.053 | 0.015 / 0.012 / 0.053 | 0.003 |
| ex_s3 | 0.008 / 0.001 | 0.008 / 0.007 / 0.017 / 0.156 | 0.008 / 0.007 / 0.017 | 0.008 / 0.007 / 0.017 | 0.001 |
| wc_s0.67 | 0.414 / 0.302 | 0.325 / 0.338 / 0.427 / 0.932 | 0.015 / 0.040 / 0.352 | 0.013 / 0.034 / 0.347 | 0.012 |
| wc_s0.83 | 0.353 / 0.252 | 0.272 / 0.314 / 0.426 / 0.932 | 0.014 / 0.065 / 0.372 | 0.017 / 0.073 / 0.375 | 0.008 |
| wc_s1 | 0.193 / 0.135 | 0.157 / 0.260 / 0.424 / 0.932 | 0.010 / 0.096 / 0.390 | 0.016 / 0.116 / 0.397 | 0.006 |
| wc_s1.25 | 0.028 / 0.019 | 0.047 / 0.171 / 0.418 / 0.932 | 0.012 / 0.118 / 0.398 | 0.017 / 0.134 / 0.400 | 0.006 |
| wc_s1.5 | 0.001 / 0.001 | 0.025 / 0.103 / 0.411 / 0.937 | 0.025 / 0.103 / 0.411 | 0.025 / 0.103 / 0.411 | 0.001 |
| wc_s2 | 0.000 / 0.000 | 0.020 / 0.039 / 0.357 / 0.937 | 0.020 / 0.039 / 0.357 | 0.020 / 0.039 / 0.357 | 0.000 |
| wc_s3 | 0.000 / 0.000 | 0.017 / 0.017 / 0.228 / 0.923 | 0.017 / 0.017 / 0.228 | 0.017 / 0.017 / 0.228 | 0.000 |
| wc_tight0.67 | 0.415 / 0.301 | 0.323 / 0.336 / 0.424 / 0.932 | 0.012 / 0.055 / 0.365 | 0.012 / 0.054 / 0.365 | 0.010 |
| wc_loose0.67 | 0.195 / 0.138 | 0.161 / 0.261 / 0.425 / 0.932 | 0.010 / 0.070 / 0.379 | 0.013 / 0.084 / 0.382 | 0.008 |
| wc_tight1.5 | 0.001 / 0.002 | 0.026 / 0.104 / 0.411 / 0.937 | 0.026 / 0.104 / 0.411 | 0.026 / 0.104 / 0.411 | 0.002 |
| wc_nofake | 0.335 / 0.228 | 0.305 / 0.716 / 0.977 / 0.989 | 0.008 / 0.101 / 0.832 | 0.010 / 0.141 / 0.861 | 0.009 |
| wc_fixU | 0.000 / 0.001 | 0.021 / 0.019 / 0.124 / 0.866 | 0.021 / 0.019 / 0.124 | 0.021 / 0.019 / 0.124 | 0.001 |
| model | 1.000 / 0.899 | 0.944 / 0.981 / 1.000 / 1.000 | 0.403 / 0.827 / 0.993 | 0.065 / 0.558 / 0.975 | 0.141 |

### Best efficiency at 1% FPR over settings (chosen on these test events: optimistic for the classical chains)

| family | far <0.1 | far 0.1-0.2 | far 0.2-0.5 | near <0.1 | near 0.1-0.2 | near 0.2-0.5 |
|---|---|---|---|---|---|---|
| example chain (ex_s*) | 0.015 (ex_s2) | 0.012 (ex_s2) | 0.059 (ex_s1.5) | 0.024 (ex_s1.5) | 0.026 (ex_s1.5) | 0.191 (ex_s1.25) |
| Wire-Cell port, all settings (wc_*) | 0.026 (wc_tight1.5) | 0.118 (wc_s1.25) | 0.832 (wc_nofake) | 0.026 (wc_tight1.5) | 0.141 (wc_nofake) | 0.861 (wc_nofake) |
| post-hoc hybrid hyb_UVfixU_Ynofake (U,V: wc_fixU; Y: wc_nofake) | 0.073 | 0.278 | 0.397 | 0.073 | 0.278 | 0.397 |
| model M4n_ft_full_near (presence) | 0.403 | 0.827 | 0.993 | 0.065 | 0.558 | 0.975 |

### Efficiency by deposit SNR (clean isolated deposits) at the far / near 1%-FPR threshold

| setting | SNR nosignal | SNR 1e-09-3 | SNR 3-5 | SNR 5-7 | SNR 7-10 | SNR 10-15 | SNR 15-inf |
|---|---|---|---|---|---|---|---|
| ex_s1 | 0.00 / 0.02 | 0.01 / 0.03 | 0.01 / 0.02 | 0.00 / 0.01 | 0.01 / 0.02 | 0.00 / 0.01 | 0.27 / 0.35 |
| wc_s1 | 0.00 / 0.01 | 0.00 / 0.01 | 0.01 / 0.02 | 0.07 / 0.08 | 0.21 / 0.23 | 0.26 / 0.27 | 0.81 / 0.81 |
| wc_fixU | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.01 / 0.01 | 0.00 / 0.00 | 0.01 / 0.01 | 0.74 / 0.74 |
| ex_s1.5 | 0.00 / 0.01 | 0.00 / 0.00 | 0.00 / 0.01 | 0.01 / 0.01 | 0.00 / 0.02 | 0.01 / 0.02 | 0.62 / 0.72 |
| wc_nofake | 0.01 / 0.02 | 0.01 / 0.01 | 0.01 / 0.02 | 0.03 / 0.05 | 0.17 / 0.21 | 0.46 / 0.51 | 0.97 / 0.98 |
| hyb_UVfixU_Ynofake | 0.00 / 0.00 | 0.00 / 0.00 | 0.01 / 0.01 | 0.04 / 0.04 | 0.22 / 0.22 | 0.46 / 0.46 | 0.89 / 0.89 |
| model | 0.28 / 0.01 | 0.33 / 0.01 | 0.39 / 0.04 | 0.65 / 0.32 | 0.92 / 0.81 | 1.00 / 0.98 | 1.00 / 1.00 |

### Per plane (each plane's own thresholds): far eff@1% / near eff@1% at 0.1-0.2 MeV, roi_fpr far / near, roi_eff

| setting | U | V | Y |
|---|---|---|---|
| ex_s1 | 0.00 / 0.01; fpr 0.44 / 0.41; eff 0.62 | 0.01 / 0.01; fpr 0.14 / 0.15; eff 0.35 | 0.00 / 0.01; fpr 0.04 / 0.00; eff 0.23 |
| wc_s1 | 0.21 / 0.21; fpr 0.44 / 0.40; eff 0.70 | 0.02 / 0.02; fpr 0.00 / 0.00; eff 0.21 | 0.01 / 0.01; fpr 0.00 / 0.00; eff 0.23 |
| wc_fixU | 0.02 / 0.02; fpr 0.00 / 0.00; eff 0.22 | 0.02 / 0.02; fpr 0.00 / 0.00; eff 0.21 | 0.01 / 0.01; fpr 0.00 / 0.00; eff 0.23 |
| wc_nofake | 0.21 / 0.21; fpr 0.44 / 0.40; eff 0.70 | 0.04 / 0.01; fpr 0.31 / 0.28; eff 0.60 | 0.85 / 0.85; fpr 0.00 / 0.00; eff 0.60 |
| ex_s1.5 | 0.00 / 0.06; fpr 0.02 / 0.01; eff 0.28 | 0.02 / 0.02; fpr 0.01 / 0.01; eff 0.19 | 0.00 / 0.00; fpr 0.03 / 0.00; eff 0.18 |
| wc_nofake | 0.21 / 0.21; fpr 0.44 / 0.40; eff 0.70 | 0.04 / 0.01; fpr 0.31 / 0.28; eff 0.60 | 0.85 / 0.85; fpr 0.00 / 0.00; eff 0.60 |
| model | 0.83 / 0.50; fpr 1.00 / 0.92; eff 0.98 | 0.78 / 0.38; fpr 1.00 / 0.91; eff 0.96 | 0.95 / 0.90; fpr 1.00 / 0.86; eff 0.97 |

### ROI-level decomposition per isolated deposit (classical): ROI efficiency (any truth pixel inside a found ROI), fraction of the deposit's truth charge inside the ROI (if found), reconstructed / true charge in its window box (if found)

| setting | plane | <0.1 | 0.1-0.2 | 0.2-0.5 | 0.5-1 |
|---|---|---|---|---|---|
| ex_s1 | U | 0.04; in 0.96; q -6.67 (n=867) | 0.14; in 1.00; q 1.16 (n=501) | 0.76; in 1.00; q 1.52 (n=287) | 0.99; in 1.00; q 1.41 (n=139) |
| ex_s1 | V | 0.01; in 0.65; q 14.62 (n=869) | 0.03; in 1.00; q -3.48 (n=549) | 0.31; in 1.00; q -1.14 (n=283) | 0.95; in 1.00; q 0.29 (n=119) |
| ex_s1 | Y | 0.00; in 0.02; q 0.65 (n=893) | 0.01; in 0.95; q 0.46 (n=472) | 0.22; in 0.79; q 0.55 (n=301) | 0.93; in 0.87; q 0.68 (n=108) |
| ex_s1 | all | 0.02; in 0.92; q -4.45 (n=2629) | 0.06; in 1.00; q 0.39 (n=1522) | 0.43; in 1.00; q 0.52 (n=871) | 0.96; in 1.00; q 0.74 (n=366) |
| wc_s1 | U | 0.05; in 0.71; q 1.08 (n=867) | 0.60; in 0.72; q 0.83 (n=501) | 0.98; in 0.94; q 0.93 (n=287) | 0.99; in 0.99; q 1.00 (n=139) |
| wc_s1 | V | 0.00; in 0.02; q 1.46 (n=869) | 0.00; in -; q - (n=549) | 0.05; in 0.97; q 0.87 (n=283) | 0.86; in 0.98; q 0.97 (n=119) |
| wc_s1 | Y | 0.00; in 0.02; q 1.93 (n=893) | 0.00; in -; q - (n=472) | 0.16; in 0.96; q 0.94 (n=301) | 0.94; in 0.99; q 0.98 (n=108) |
| wc_s1 | all | 0.02; in 0.70; q 1.15 (n=2629) | 0.20; in 0.72; q 0.83 (n=1522) | 0.39; in 0.95; q 0.93 (n=871) | 0.93; in 0.99; q 0.98 (n=366) |
| wc_fixU | U | 0.00; in -; q - (n=867) | 0.00; in -; q - (n=501) | 0.06; in 0.99; q 1.01 (n=287) | 0.82; in 0.99; q 0.98 (n=139) |
| wc_fixU | V | 0.00; in 0.02; q 1.46 (n=869) | 0.00; in -; q - (n=549) | 0.05; in 0.97; q 0.87 (n=283) | 0.86; in 0.98; q 0.97 (n=119) |
| wc_fixU | Y | 0.00; in 0.02; q 1.93 (n=893) | 0.00; in -; q - (n=472) | 0.16; in 0.96; q 0.94 (n=301) | 0.94; in 0.99; q 0.98 (n=108) |
| wc_fixU | all | 0.00; in 0.02; q 1.70 (n=2629) | 0.00; in -; q - (n=1522) | 0.09; in 0.97; q 0.94 (n=871) | 0.87; in 0.99; q 0.97 (n=366) |
| ex_s1.5 | U | 0.00; in -; q - (n=867) | 0.01; in 1.00; q 4.06 (n=501) | 0.29; in 1.00; q 1.10 (n=287) | 0.99; in 1.00; q 1.21 (n=139) |
| ex_s1.5 | V | 0.00; in 0.02; q 4.40 (n=869) | 0.00; in -; q - (n=549) | 0.02; in 1.00; q -0.21 (n=283) | 0.72; in 1.00; q 0.10 (n=119) |
| ex_s1.5 | Y | 0.00; in 0.02; q 0.65 (n=893) | 0.00; in -; q - (n=472) | 0.04; in 0.93; q 0.60 (n=301) | 0.64; in 0.69; q 0.51 (n=108) |
| ex_s1.5 | all | 0.00; in 0.02; q 2.53 (n=2629) | 0.00; in 1.00; q 4.06 (n=1522) | 0.12; in 1.00; q 0.70 (n=871) | 0.80; in 1.00; q 0.55 (n=366) |
| wc_nofake | U | 0.05; in 0.71; q 1.08 (n=867) | 0.60; in 0.72; q 0.83 (n=501) | 0.98; in 0.94; q 0.93 (n=287) | 0.99; in 0.99; q 1.00 (n=139) |
| wc_nofake | V | 0.03; in 0.75; q 1.07 (n=869) | 0.39; in 0.71; q 0.68 (n=549) | 0.94; in 0.92; q 0.85 (n=283) | 0.99; in 0.98; q 0.97 (n=119) |
| wc_nofake | Y | 0.13; in 0.76; q 0.80 (n=893) | 0.84; in 0.72; q 0.70 (n=472) | 0.98; in 0.94; q 0.90 (n=301) | 0.98; in 0.99; q 0.98 (n=108) |
| wc_nofake | all | 0.07; in 0.75; q 0.85 (n=2629) | 0.60; in 0.72; q 0.72 (n=1522) | 0.97; in 0.94; q 0.90 (n=871) | 0.99; in 0.99; q 0.98 (n=366) |

### ROI efficiency vs deposit SNR (clean deposits), per plane

| setting | plane | nosignal | 1e-09-3 | 3-5 | 5-7 | 7-10 | 10-15 | 15-inf |
|---|---|---|---|---|---|---|---|---|
| ex_s1 | U | 0.02 | 0.04 | 0.06 | 0.10 | 0.22 | 0.63 | 0.99 |
| ex_s1 | V | 0.00 | 0.02 | 0.02 | 0.03 | 0.08 | 0.29 | 0.95 |
| ex_s1 | Y | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.63 |
| ex_s1 | all | 0.01 | 0.02 | 0.03 | 0.05 | 0.09 | 0.26 | 0.84 |
| wc_s1 | U | 0.00 | 0.03 | 0.15 | 0.57 | 0.93 | 0.98 | 1.00 |
| wc_s1 | V | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.87 |
| wc_s1 | Y | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.62 |
| wc_s1 | all | 0.00 | 0.01 | 0.05 | 0.20 | 0.29 | 0.28 | 0.81 |
| ex_s1.5 | U | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.07 | 0.89 |
| ex_s1.5 | V | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.82 |
| ex_s1.5 | Y | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.53 |
| ex_s1.5 | all | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.02 | 0.73 |
| wc_nofake | U | 0.00 | 0.03 | 0.15 | 0.57 | 0.93 | 0.98 | 1.00 |
| wc_nofake | V | 0.00 | 0.01 | 0.11 | 0.45 | 0.86 | 0.99 | 1.00 |
| wc_nofake | Y | 0.00 | 0.00 | 0.02 | 0.13 | 0.55 | 0.96 | 1.00 |
| wc_nofake | all | 0.00 | 0.01 | 0.11 | 0.40 | 0.76 | 0.97 | 1.00 |
| model (presence > 0.413, its far1pct threshold) | all: 0.02, 0.05, 0.18, 0.55, 0.90, 1.00, 1.00 |
| model (presence > 0.959, its near1pct threshold) | all: 0.00, 0.00, 0.03, 0.33, 0.80, 0.98, 1.00 |

### Charge inside found ROIs vs inside oracle ROIs (deposits the default setting found): reco/true in the window box, median and 68% half-width

| chain | plane, energy | n | found ROI | oracle ROI | truth charge inside found ROI (median frac) |
|---|---|---|---|---|---|
| ex | U_E0-0.1 | 32 | -6.67 ± 28.96 | 0.12 ± 10.31 | 0.96 |
| ex | U_E0.1-0.2 | 68 | 1.16 ± 10.69 | 0.86 ± 4.90 | 1.00 |
| ex | U_E0.2-0.5 | 217 | 1.52 ± 4.09 | 0.66 ± 1.82 | 1.00 |
| ex | U_E0.5-1 | 137 | 1.41 ± 1.92 | 0.67 ± 0.93 | 1.00 |
| ex | V_E0-0.1 | 12 | 14.62 ± 130.63 | -2.57 ± 9.50 | 0.65 |
| ex | V_E0.1-0.2 | 16 | -3.48 ± 13.20 | 0.36 ± 4.45 | 1.00 |
| ex | V_E0.2-0.5 | 88 | -1.14 ± 3.07 | 0.58 ± 1.92 | 1.00 |
| ex | V_E0.5-1 | 113 | 0.29 ± 1.72 | 0.73 ± 0.81 | 1.00 |
| ex | Y_E0.2-0.5 | 67 | 0.55 ± 0.14 | 0.86 ± 0.24 | 0.79 |
| ex | Y_E0.5-1 | 100 | 0.68 ± 0.18 | 0.86 ± 0.17 | 0.87 |
| ex | all_E0-0.1 | 45 | -4.45 ± 39.79 | -0.37 ± 10.18 | 0.92 |
| ex | all_E0.1-0.2 | 88 | 0.39 ± 11.03 | 0.70 ± 4.77 | 1.00 |
| ex | all_E0.2-0.5 | 372 | 0.52 ± 3.41 | 0.70 ± 1.58 | 1.00 |
| ex | all_E0.5-1 | 350 | 0.74 ± 1.39 | 0.80 ± 0.68 | 1.00 |
| wc | U_E0-0.1 | 44 | 1.08 ± 0.67 | 1.12 ± 0.46 | 0.71 |
| wc | U_E0.1-0.2 | 301 | 0.83 ± 0.37 | 0.92 ± 0.26 | 0.72 |
| wc | U_E0.2-0.5 | 280 | 0.93 ± 0.23 | 1.01 ± 0.16 | 0.94 |
| wc | U_E0.5-1 | 137 | 1.00 ± 0.09 | 1.01 ± 0.08 | 0.99 |
| wc | V_E0.2-0.5 | 15 | 0.87 ± 0.12 | 0.95 ± 0.11 | 0.97 |
| wc | V_E0.5-1 | 102 | 0.97 ± 0.09 | 1.01 ± 0.08 | 0.98 |
| wc | Y_E0.2-0.5 | 49 | 0.94 ± 0.05 | 1.04 ± 0.05 | 0.96 |
| wc | Y_E0.5-1 | 101 | 0.98 ± 0.04 | 1.03 ± 0.03 | 0.99 |
| wc | all_E0-0.1 | 46 | 1.15 ± 0.66 | 1.18 ± 0.49 | 0.70 |
| wc | all_E0.1-0.2 | 301 | 0.83 ± 0.37 | 0.92 ± 0.26 | 0.72 |
| wc | all_E0.2-0.5 | 344 | 0.93 ± 0.19 | 1.02 ± 0.13 | 0.95 |
| wc | all_E0.5-1 | 340 | 0.98 ± 0.07 | 1.02 ± 0.07 | 0.99 |

### ROI purity (all ROI pixels of all test events)

| setting | U: area truth-empty / +charge truth-empty / truth charge in ROI | V | Y |
|---|---|---|---|
| ex_oracle | 0.17 / 0.05 / 1.00 | 0.17 / 0.05 / 1.00 | 0.17 / 0.05 / 1.00 |
| wc_oracle | 0.17 / 0.01 / 1.00 | 0.17 / 0.01 / 1.00 | 0.17 / 0.01 / 1.00 |
| ex_s1 | 0.65 / 0.30 / 0.98 | 0.53 / 0.23 / 0.97 | 0.05 / 0.02 / 0.96 |
| wc_s1 | 0.29 / 0.02 / 0.98 | 0.15 / 0.01 / 0.97 | 0.11 / 0.01 / 0.97 |
| wc_fixU | 0.16 / 0.01 / 0.97 | 0.15 / 0.01 / 0.97 | 0.11 / 0.01 / 0.97 |
| ex_s1.5 | 0.37 / 0.11 / 0.97 | 0.28 / 0.09 / 0.94 | 0.03 / 0.01 / 0.95 |
| wc_nofake | 0.29 / 0.02 / 0.98 | 0.25 / 0.02 / 0.98 | 0.11 / 0.01 / 0.99 |

### The model's 'ROI' (cells with presence above its own 1%-FPR window threshold)

| threshold | plane | ROI eff <0.1 / 0.1-0.2 / 0.2-0.5 / 0.5-1 | truth charge in ROI if found (0.1-0.2) | covered cells truth-empty | window fpr far / near |
|---|---|---|---|---|---|
| far1pct (0.413) | U | 0.08 / 0.65 / 0.98 / 0.99 | 1.00 | 0.37 | 0.010 / 0.144 |
| far1pct (0.413) | V | 0.05 / 0.57 / 0.98 / 0.99 | 1.00 | 0.43 | 0.010 / 0.144 |
| far1pct (0.413) | Y | 0.27 / 0.95 / 0.98 / 0.99 | 1.00 | 0.23 | 0.010 / 0.144 |
| far1pct (0.413) | all | 0.14 / 0.71 / 0.98 / 0.99 | 1.00 | - | 0.010 / 0.144 |
| near1pct (0.959) | U | 0.01 / 0.44 / 0.97 / 0.99 | 1.00 | 0.01 | 0.000 / 0.010 |
| near1pct (0.959) | V | 0.00 / 0.36 / 0.97 / 0.99 | 1.00 | 0.01 | 0.000 / 0.010 |
| near1pct (0.959) | Y | 0.15 / 0.90 / 0.98 / 0.99 | 1.00 | 0.01 | 0.000 / 0.010 |
| near1pct (0.959) | all | 0.06 / 0.55 / 0.97 / 0.99 | 1.00 | - | 0.000 / 0.010 |
