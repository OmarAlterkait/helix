# Causes of the denoiser's failings: hypotheses A, D, E

2026-10-10. Data: the 128 test events (ev260-387), every isolated-deposit (`iso`) window and every
near-activity noise window (`bgn`). Statistics are event bootstraps (200 resamples, 16-84%). Labels:
- **[V]** = verified by running;
- **[A]** = assumed or inferred.

Files:
- tables and intervals: `causes_ade.json` and `causes_ade_shares.json`;
- per-window arrays: `/global/cfs/cdirs/m5238/users/oalter/helix_runs/causes2/windows.npz`;
- figures: `../figs/causes_A_D.png` and `../figs/causes_E.png`.

## What was built for these tests

1. **Exact corpus noise.** I tried to regenerate the exact noise each corpus event carries (`dense/corpus_noise.py`), from
   the recorded `ident/noise_seed` with the corpus builder's own torch-GPU noise path, run on this node's torch 2.10 on an
   A100. **It does not reproduce** [V]:
   - with the recorded seed, 0.797 of stored coefficients are re-found and 0.02% of values are equal;
   - with a wrong seed, the figures are 0.793 and 0.02%;
   - so the corpus's torch RNG stream (2080 Ti, older torch) is not reproducible here.

   The model's exact input therefore exists only as the stored coefficients.
2. **Model on the classical chain's waveforms** (`dense/model_dump_cn.py`).
   - Inputs: the classical chain's exact digitized waveforms (`v2/classical.py` noise, seed 1000+ev).
   - Processing: passed through the corpus DSP (gate, κ=1 hard threshold) and tokenized with each shard's own
     `norm_sigma` and band lengths, then fed to M6d (**M6d-cn**).
   - The two methods now read **identical waveforms**.
3. **Second fresh draw, both methods** (`model_dump_cn.py --noise tn`, `cls_tn.py`).
   - Noise: a fresh draw from the corpus's own torch-GPU generator.
   - It is fed to both M6d (**M6d-tn**) and the classical chain (**cls-tn**).
   - Its noise statistics equal the stored corpus's: per-band σ within 0.3%, kept coefficients within ±1-2% [V].
4. **Per-deposit matched-filter evidence** (`dense/snr_deposits.py`).
   - Statistic: the deposit's own clean template, DWT'd and band-whitened, as in `scripts/snr_stages.py`.
   - It is evaluated on three inputs:
     - `gate`: the classical input;
     - `k1`: the same realisation through the corpus threshold, with D1 dropped (exactly M6d-cn's input);
     - `corp`: the stored corpus (M6d's actual input).
   - Each input gets its own null: 40 charge-free placements of the template per deposit.
   - "Evidence" z is in units of the gated-noise σ.
   - "KL" is known-location detection at that input's pooled 1% null: an upper bound for any detector reading that input.
5. **Window scores.** Every window is scored by every method: the max over its cells, exactly as in `eval_denoise` and
   `v2/score.py`. The 1% threshold is set per plane on the near-activity noise windows ("near pp").
   - The M6d maps reproduce the eval's window scores exactly [V, dense3].
   - Classical tuned reproduces the benchmark's 0.516 [V].

Headline efficiency, 0.1-0.2 MeV, near pp (`causes_ade.json → headline`):

| | U | V | Y | pooled |
|---|---|---|---|---|
| classical tuned (seed 1000+ev) | 0.365 | 0.302 | **0.924** | 0.516 [0.506-0.529] |
| classical tuned, fresh torch draw | 0.365 | 0.326 | **0.924** | 0.524 |
| M6d presence, stored corpus | **0.535** | **0.421** | 0.907 | **0.609** [0.597-0.622] |
| M6d presence, classical's draw (M6d-cn) | 0.499 | 0.417 | 0.894 | 0.592 |
| M6d presence, fresh torch draw (M6d-tn) | 0.519 | 0.413 | 0.867 | 0.589 |
| M4n presence, stored corpus | 0.499 | 0.377 | 0.898 | 0.579 |
| known-location bound, full gated waveform | 0.988 | 0.969 | 0.987 | 0.981 |
| known-location bound, stored corpus | 0.661 | 0.557 | 0.956 | 0.715 |

The eval's 0.591 for M6d is its pooled-threshold number; with per-plane thresholds it is 0.609.

## (A) Faint isolated deposits: where and why the classical chain leads

**Where the lead is.** On identical waveforms the classical chain leads only on **Y** (paired differences,
classical − model):

| | 0.1-0.2 MeV | SNR 3-10 |
|---|---|---|
| Y, classical's draw (cls − M6d-cn) | +0.030 [0.016, 0.043] | +0.079 [0.063, 0.093] |
| Y, fresh torch draw (cls-tn − M6d-tn) | +0.057 | +0.121 |
| U / V (cls − M6d-cn) | −0.13 / −0.12 | −0.10 / −0.09 |

The model is ahead on U and V at every SNR (`eff_vs_snr`, `causes_A_D.png` top row).

**The dense2/3 cell-level lead is an operating-point effect.** dense2/3 found the classical chain ahead for U/V 2-4 k
and Y 1-2 k cells in sparse surroundings at 1% cell FPR. At the cell FPRs that set the window thresholds (1-2e-4) it
reverses or vanishes [V, `E_cell_roc`]:

| cells (neighbourhood) | classical @ 1e-2 | M6d @ 1e-2 | classical @ 1e-4 | M6d @ 1e-4 |
|---|---|---|---|---|
| 2-4 k (≤ 10 k e-) | 0.867 | 0.808 | 0.650 | **0.680** |
| 2-4 k (isolated) | 0.816 | 0.769 | 0.582 | **0.620** |
| 1-2 k (≤ 10 k e-) | 0.483 | 0.446 | 0.278 | 0.276 |

**Decisive test** (`causes_A_D.png` bottom row; `causes_ade_shares.json → A`; same noise draw for both methods):

| Y deposits, SNR 3-10 | n | classical | M6d-cn |
|---|---|---|---|
| corpus-input evidence z_k1 < 2σ | 364 (53%) | 0.170 | **0.022** |
| z_k1 ≥ 2σ | 322 | 0.736 | **0.736** |

- **The whole Y lead sits in deposits the corpus input barely contains.** At z_k1 < 2σ the model finds 2%, against
  17% for the classical chain from the full waveform. At z_k1 ≥ 2σ the two tie.
- **At equal full-waveform evidence z_gate the model trails** (Y: 5-6σ 0.30 vs 0.41; 6-8σ 0.67 vs 0.84).
- **At equal corpus-input evidence it matches or beats the classical chain**, except in the 2-3σ bin.

| explanation | share of the Y lead (SNR 3-10, identical waveforms, gap 0.079 [0.061-0.093]) | evidence |
|---|---|---|
| (i) **information**: sparsified corpus input (κ=1 hard threshold 3.35-3.9σ, D1 dropped) | **100%** carried by deposits with z_k1 < 2σ: 0.079 [0.066-0.088]. Under the strict criterion "not detectable even at known location on the corpus input" (KL_k1 fails): 0.039 [0.032-0.047] = 49% | known-location bound drops 0.95 → 0.56 (Y, SNR 3-10) from the gated waveform to the corpus input; at 5-7σ, 0.99 → 0.55; deposits with z_k1 ≥ 2σ: classical = model (0.736 vs 0.736) [V] |
| (ii) **modelling / training** | **0% of the lead**, but it decides who wins the low-evidence deposits: a large Y-specific headroom of +0.117 [0.106-0.126] if Y deposits followed the model's U+V evidence→efficiency curve | Y low-evidence deposits score presence 0.29 (median) / 0.87 (90th pct), all below Y's near threshold of 0.95, which is set by confident false presences near activity. At the far threshold the model finds 44% of them (classical 22%). They sit near foreign charge (median foreign/q 0.33, vs 0.02 on U). So the model cannot tell a faint real Y deposit from its own near-activity haze [V numbers; mechanism A]. Strong-evidence misses are rare: 34 of 2,304 deposits with z_corp ≥ 6 and KL_corp (1.5%), mostly < 0.1 MeV deposits lying under much brighter foreign charge (median foreign/q 224) [V] |
| (iii) **noise realisation** | **0% of the identical-waveform lead.** It biased the published comparison in the model's favour: the stored corpus realisation scores M6d +0.020 [0.007, 0.031] above a fresh draw of the same generator (Y +0.040 [0.027, 0.054]) | classical is draw-invariant: 0.516 vs 0.524, Y 0.924 vs 0.924. M6d on two fresh draws: 0.592 / 0.589. Noise σ and kept counts match the stored corpus to 0.3% / 1-2%, so the stored-corpus advantage is not a noise-level mismatch; its cause (this realisation, or build code / GPU of the 2080 Ti DSP) is not isolated [A]. It hid two thirds of the Y gap at 0.1-0.2 MeV (+0.017 published vs +0.057 on a fresh draw) |
| operating point (dense cell-level claim) | the whole cell-level "classical leads" result at 1e-2 | reverses at 1e-4 (table above) [V] |

**(A) bottom line.**
- The classical chain's remaining lead (Y, SNR 3-10) is **information** the corpus threshold removed. The model
  matches the classical chain wherever the corpus keeps at least ~2σ of the deposit's evidence.
- Recovering it needs the waveform or a lower threshold. `docs/DENOISE.md` measured that the stage bound at κ=0.5
  recovers SNR 3-5 from 0.19 to 0.53 [A, not re-run].
- A model fix (better near-activity presence calibration on Y) would address the separate +0.12 Y headroom at low
  evidence. It would not by itself exceed the classical chain where the information is gone.
- On U and V the model is ahead in every SNR bin, despite reading ~4x less evidence (median z_k1 1.2-1.6σ vs z_gate
  5.5σ). It extracts far more per unit evidence: 50% efficiency at z_k1 ≈ 2.3σ (U, V), against z_gate ≈ 6.5σ for the
  classical chain.

## (D) Induction weaker than collection

0.1-0.2 MeV per-plane gaps, Y − U and Y − V:

| method | U gap | V gap |
|---|---|---|
| M6d | 0.372 | 0.486 |
| classical | 0.558 | 0.621 |

The test re-weights Y's efficiency to the plane's distribution of deposit SNR (analytic optimal SNR, `noise_vs_hits`)
and reports the residual (`causes_ade_shares.json → D`).

| explanation | share of the U/V-vs-Y gap | evidence |
|---|---|---|
| (i) **physics: lower SNR per electron** (gain + bipolar response + noise) | **M6d 96-100%** (residual U 0.014 [−0.11, 0.03], V 0.007; clean deposits U −0.003, V −0.001). **Classical 86-96%** (residual U 0.078 [0.05-0.10], V 0.024). M6d on both fresh draws: 94-107% | median deposit SNR at 0.1-0.2 MeV: U 6.5, V 5.4, Y 11.6. The analytic SNR-1 charge is 460 / 540 / 250 e-; peak ADC per 1000 e- is 2.0 / 1.65 / 4.5 (`DENOISE.md`) [V]. The matched filter recovers part of the bipolar loss (SNR ratio Y/U 1.8 < peak ratio 2.25) [A] |
| (ii) zero-suppression | **~0 at 0.1-0.2 MeV**: deposits with no clean signal are 0.8% (U), 0.7% (V), 1.3% (Y) [V] | dominant only below 0.1 MeV: U 43%, V 47%, Y 18% [V] |
| (iii) corpus sparsification hits bipolar signals harder | **~0 beyond SNR.** Known-location bounds on the corpus input are equal at matched SNR (KL_k1, SNR 5-7: U 0.56, V 0.59, Y 0.55); the residual Y−U/V gap of the corpus bound at matched SNR is 0.002-0.023 | a smaller fraction of induction evidence survives (median resp_k1 / resp_gate: U 0.29, V 0.23, Y 0.67) only because induction deposits sit at lower SNR [V] |
| (iv) modelling | **≤ 0.02** (the M6d residual above) | at matched corpus-input evidence the model is MORE efficient on U/V than on Y (residual −0.13 to −0.18): the opposite of an induction modelling deficit [V] |

**(D) bottom line.** The induction deficit is the deposit's signal-to-noise per electron. At equal SNR M6d is as good
on U and V as on Y. The classical chain keeps a U-specific residual of about 0.08, which comes from its own U statistic.

## (E) Why the decoder's window gain is invisible at the cell level

Gain in near-pp efficiency at 0.1-0.2 MeV, M6d − M4n presence: U +0.036, V +0.044, Y +0.009.

Both presence maps are used, on the same stored corpus and the same windows. The window 1% threshold corresponds to a
per-cell FPR of 1.0-1.9e-4 on the empty cells of noise windows; one noise window ≈ 53-97 independent cells
(`E`: n_eff = ln 0.99 / ln(1 − f_cell)).

The gain is decomposed exactly (`E_decomposition`): correlation part + tail-ranking part + spread part = total gain.

| explanation | U | V | Y | evidence |
|---|---|---|---|---|
| (i) **tail effect**: better ranking of charged cells at cell FPR ~1e-4 | +0.024 (67%) | +0.015 (33%) | +0.009 (100%) | at equal cell FPR, charged-cell detections M6d − M4n at cell FPR 1e-4 / 1e-3 / 1e-2: U +0.030 / +0.018 / +0.016; V +0.016 / +0.024 / +0.009; Y +0.008 / +0.013 / 0.000. Over all cells (dense3 records), M6d − M4n on 1-2 k cells at 1e-4 / 1e-2: +0.004 to +0.016 / ≤ +0.006 by density. The advantage lives in the deep tail and fades by 1e-2, the only FPR the dense2/3 comparison probed [V] |
| (ii) **spatial correlation of false positives** | +0.014 (39%) | +0.022 (50%) | 0.000 | M6d's false positives are more clustered: n_eff U 72 vs 87, V 53 vs 65, Y 91 vs 97. At the same 1% window FPR it runs at 21% higher cell FPR (U 1.40e-4 vs 1.16e-4; V 1.89e-4 vs 1.56e-4) [V]. No cell-level metric can show this |
| (iii) **true-positive spatial spread** (window fired only by an empty cell beside the deposit) | −0.002 | +0.007 (17%) | 0.000 | empty-only detections at matched cell FPR: V 2.2% vs 1.5% of windows [V] |

Further evidence [V]:
- 0.7-7% of M6d's iso-window detections fire only on empty cells (U 7.1%, V 6.9%, Y 0.7% of detections).
- Detected windows have 2.2-2.5 cells above threshold for both models.
- Paired: deposits found only by M6d vs only by M4n are U 4.4% vs 0.8%, V 5.8% vs 1.5%, Y 1.5% vs 0.6%.

**(E) bottom line.** The window gain is real and comes from two things:
- **Tail ranking** (U, Y mostly): the 1e-4 tail of the per-cell score distribution, invisible to a 1%-FPR cell metric.
- **False-positive clustering** (half of V's gain): clustered false positives make the window max over ~100 cells less
  noisy at a fixed cell FPR. A cell-level ROC cannot see this by construction.

Only a small part on V comes from detections by cells next to the deposit.

## Caveats

- **Evidence index.** z is a template matched filter in gated-noise units. For thresholded inputs its null is
  non-Gaussian, so cross-input comparisons use known-location decisions, each against its own null, plus the evidence
  bins. The `z < 2σ` cut is a convention; the strict known-location criterion is reported alongside.
- **Templates.** Templates are the stored clean sensor in the deposit's box, so foreign charge inside the box enters
  non-clean deposits. "Clean" rows (foreign ≤ 25% of deposit charge) are reported throughout and agree.
- **Fresh draws.** M6d-cn / M6d-tn are fresh draws processed with this node's DSP (A100). The DSP device differs from
  the 2080 Ti in ~0.016% of coefficients (builder note) [A].

## Reproduce (node 59625958; outputs in /global/cfs/cdirs/m5238/users/oalter/helix_runs/causes2/)

```bash
H=baselines/uboone_sp/dense; C=/global/cfs/cdirs/m5238/users/oalter/helix_runs/causes2
scripts/helix_run.sh python $H/corpus_noise.py --events 260-261          # exact-noise reproduction test (fails, see above)
$H/run_dump_cn.sh                                                         # M6d on the classical noise draw -> $C/model_cn
$H/run_dump_tn.sh                                                         # M6d on a fresh torch draw -> $C/model_tn (+ planes)
scripts/helix_run.sh python $H/cls_tn.py --planes-dir $C/tn_planes --out $C/cls_tn
scripts/helix_run.sh python $H/snr_deposits.py --out $C/snr               # per-deposit evidence
scripts/helix_run.sh python $H/causes_ade.py --out $H/results/causes_ade.json
scripts/helix_run.sh python $H/causes_ade_shares.py --out $H/results/causes_ade_shares.json
scripts/helix_run.sh python $H/figs_causes.py
```
