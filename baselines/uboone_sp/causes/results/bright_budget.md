# Bright-charge error budget: M6d / M6dc (charge gated by presence > 0.5)

2026-10-10. This is a diagnostic only. No training objective, head or refinement was added. The only changes were inference variants (fp32) and a diagnostic calibration fit on non-test events.

**Data.**
- All 128 test events (ev260-387), every plane.
- The calibration events are probe-train events ev000-063, excluding ev041, whose truth file is on OST 61.
- The floor uses 48 test events: ev260-307.

**Truth.**
- Every accuracy number is against the **complete truth Tf**: all step deposits, attenuated and diffused with JAXTPC 1ea2282's own hits code, with no 25 e- cut (as `causes_FBC.md`).
- The **hits truth Th**, which the models were trained on, is shown alongside.

**Bright** means:
- cells with Tf ≥ 16 k e-, on the 2-wire × 16-tick grid;
- tiles of 2 × 2, 4 × 4 or 16 × 8 cells with Tf ≥ 64 k e-;
- particles with E ≥ 1 MeV.

**Metrics.** r = pred / truth − 1. The tables give the median and the half-width of the 16-84% spread (hw68). Variance shares refer to var(r).

**Sources.**
- Code: `causes/bright_dump.py` (precision variants), `bright_build.py` (records), `floor.py` (exact-kernel inversions), `saturation.py`, `bright_analyze.py`.
- Numbers: `causes/results/bright_budget.json` and `saturation.json`.
- Intermediates: `/global/cfs/cdirs/m5238/users/oalter/helix_runs/bright/`.

Labels: [V] verified by running; [A] assumed.

## Headline per cell (Tf ≥ 16 k e-; 0.43 / 0.43 / 0.49 M cells on U / V / Y) [V]

| plane | M6d gated: vs complete / vs hits truth | M6dc gated: vs complete / vs hits | M4n gated vs complete | classical tuned vs complete | hits truth vs complete |
|---|---|---|---|---|---|
| U | −0.71% (0.052) / +0.63% (0.052) | −1.35% (0.043) / −0.02% (0.042) | −2.16% (0.055) | −11.6% (0.141) | **−1.27% (0.009)** |
| V | −0.87% (0.054) / +0.47% (0.054) | −1.61% (0.045) / −0.27% (0.045) | −2.03% (0.056) | −11.9% (0.151) | **−1.27% (0.009)** |
| Y | −0.44% (0.044) / +0.93% (0.043) | −1.36% (0.035) / +0.00% (0.034) | −1.80% (0.046) | −5.7% (0.081) | **−1.26% (0.009)** |

Each entry is the median, with hw68 in brackets.

## Error budget per plane (per cell; Tf ≥ 16 k e-)

Each cause is listed with its share of the per-cell **variance** of r, and its contribution to the **bias** (median r). The variance shares are not additive: each is the part one correction would remove.

| cause | U: M6d / M6dc | V: M6d / M6dc | Y: M6d / M6dc | evidence |
|---|---|---|---|---|
| **total** (median, hw68) | −0.71%, 0.052 / −1.35%, 0.043 | −0.87%, 0.054 / −1.61%, 0.045 | −0.44%, 0.044 / −1.36%, 0.035 | [V] |
| **0. Truth definition** (Th lacks the per-group sub-25 e- tails): bias | **−1.27%** of −0.71% / of −1.35% | **−1.27%** of −0.87% / of −1.61% | **−1.26%** of −0.44% / of −1.36% | M6dc against Th is −0.02 / −0.27 / 0.00%: the model reproduces its training target. M6d sits +0.5-0.9% above Th, which partly cancels. The deficit grows linearly with drift, from −0.3% at 0-27 cm to −2.1% at 189-216 cm, because longer diffusion leaves more of the charge in sub-threshold tails. Against Th the models show **no drift dependence** (M6dc flat within ±0.1%) [V] |
| 0. Truth definition: variance | 3% / 5% | 3% / 4% | 5% / 8% | var(Th/Tf) relative to the model's. Scoring against Th leaves the variance unchanged: the extra spread is mostly correlated with what the model learned [V] |
| **1. Numerical precision** (bf16 → fp32, whole model) | variance 1.5% / 2.7%; bias +0.37 / +0.24 pt | 1.3% / 1.6%; +0.28 / +0.29 pt | 1.6% / 4.1%; +0.48 / +0.30 pt | rms log-charge change 0.012-0.016. Error correlation bf16 vs fp32 is 0.95-0.985. Head-only fp32 (`head32`) gives about the same as full fp32 for M6dc, and for M6d moves the bias −0.2 pt instead. Precision is **not** the dominant term; the bf16 step (2.4% at log1p(q/Q0) ≈ 6) mostly averages out [V] |
| **2. Systematic scale**: calibration in predicted charge, fit on calibration events | MSE −0.9% / −4.5%; variance ≈ 0 | −1.3% / −5.6%; ≈ 0 | −0.9% / −7.7%; ≈ 0 | the fit removes the global offset: the median goes to −0.1%. The residual charge trend across 16 k-1 M is within ±0.5% (M6d) / ±0.8% (M6dc) [V] |
| 2. + drift (oracle drift in the calibration) | MSE −1.5% / −5.3% | −1.9% / −6.6% | −1.7% / −9.2% | the between-drift-bin share of variance is 0.6-1.7%, entirely the truth-definition trend above [V] |
| 2. Readout edges (first / last 10 cells), wire position | < 0.1% / < 0.25% | < 0.1% | < 0.3% | last 10 cells 0.3 pt more negative [V] |
| 2. Above 1 M e- | −3.6% / −2.6% at 1-2 M (U, 103 cells); V −4.4% at 1-2 M and **−18%** at > 2 M (11 cells) | | −4.0% (28 cells) | real, but these are 0.03% of bright cells (< 0.3% of the MSE) [V]. Mechanism: the log-target cap or saturation, see 4 [A] |
| **3. Topology** (cell): dE/dx, majority share, number of particles | ≤ 0.4% each | ≤ 0.5% each | ≤ 0.4% each | Bragg-like cells (dE/dx > 4 MeV/cm) are +0.5-1 pt **higher** than MIP cells, with a smaller spread. Mixed cells (majority < 0.9) are 0.5 pt lower [V] |
| **4. ADC saturation** | 0.06% of bright cells; 0.6% of MSE | 0.015%; 0.1% | 0.016%; 0.3% | **saturation exists.** 775 / 122 / 167 clean-sensor samples sit at the 12-bit limit (U / V / Y, 128 events). Affected cells have hw68 0.11-0.17 vs 0.05 and are 1-6 pt lower. Classical is worse there (−20 to −25%) [V] |
| **5. Information limit of per-wire truth** (floor: exact kernel, clean sensor, oracle support) | 14% / 21% of the variance (floor hw 0.019) | 25% / 37% (0.028) | 9% / 14% (0.013) | the same inversion on a sensor the per-wire kernel describes exactly gives hw 0.002-0.003. The 0.013-0.028 is the cost of missing sub-wire position and drift, plus integer ADC and zero-suppression [V] |
| **remaining: noise (plus corpus sparsification)** | ≈ 85% / 80% | ≈ 75% / 63% | ≈ 90% / 85% | the rest of the variance above the clean floor. The noisy-input floors do not undercut the model on U/V: oracle-support least squares gives 0.074 / 0.096, best Tikhonov 0.068 / 0.102. On Y, a linear Tikhonov inversion of the corpus input (0.032) does better than M6d (0.044) and M6dc (0.035): **on Y, 48% (M6d) / 18% (M6dc) of the variance is model inefficiency** relative to a linear inversion with the exact kernel [V]. The corpus threshold costs little at high charge: noisy vs corpus floor 0.068 vs 0.074 (U), 0.029 vs 0.032 (Y) [V] |
| **6. Grid / cell boundaries** | not a cause | not a cause | not a cause | errors of the bright cells in a 2 × 2 tile are positively correlated (−9 to −29% "cancellation"), so charge is not shuffled across cell edges. Aggregation narrows hw68 from 0.052 to 0.033 (2 × 2) to 0.026 (4 × 4), slower than √n. The residual is a shared scale over a few cells [V] |

### Region level (tiles with Tf ≥ 64 k e-) [V]

hw68 (median) per tile size; floors use the 48 floor events, models all 128 events:

| plane | tile | M6d | M6dc | classical tuned | floor: clean, oracle support | floor: noisy, oracle / best Tikhonov | floor: corpus, oracle / best Tikhonov |
|---|---|---|---|---|---|---|---|
| U | 2 × 2 | 0.033 (−0.8%) | 0.027 (−1.4%) | 0.120 (−7.7%) | 0.008 | 0.041 / 0.058 | 0.043 / 0.063 |
| U | 4 × 4 | 0.026 (−1.1%) | 0.022 (−1.7%) | 0.101 (−6.2%) | 0.006 | 0.036 / 0.098 | 0.039 / 0.100 |
| U | 16 × 8 | 0.026 (−1.8%) | 0.022 (−2.1%) | 0.098 (−8.3%) | 0.011 | 0.033 | 0.038 |
| V | 4 × 4 | 0.029 (−1.3%) | 0.024 (−2.0%) | 0.110 (−7.1%) | 0.010 | 0.041 / 0.115 | 0.045 / 0.118 |
| V | 16 × 8 | 0.029 (−2.0%) | 0.025 (−2.5%) | 0.111 (−10.2%) | 0.014 | 0.036 | 0.041 |
| Y | 4 × 4 | 0.023 (−0.7%) | 0.020 (−1.6%) | 0.027 (−1.6%) | 0.004 | 0.019 / 0.021 | 0.022 / 0.021 |
| Y | 16 × 8 | 0.021 (−1.2%) | 0.018 (−2.0%) | 0.023 (−1.8%) | 0.005 | 0.015 | 0.020 |

What the region table shows:
- **Bias.** At region level the bias is about the truth-definition deficit (−1.3%), plus the gating loss of the faint cells inside the region (another 0.5-0.8 pt at 16 × 8).
- **Information limit.** It costs hw 0.004-0.010 at 4 × 4 and 0.005-0.014 at 16 × 8, against 0.013-0.028 per cell. Most of it averages out over regions.
- **Model vs linear inversion.** On U/V the model beats every linear inversion of the noisy or corpus input at every scale. On Y it is level with them at 4 × 4 (0.020-0.023 vs 0.019-0.022) and slightly worse at 16 × 8 (0.018-0.021 vs 0.015-0.020).

### Particle level (E ≥ 1 MeV; about 121 k particles per plane) [V]

| plane | M6d vs complete / vs hits | M6dc vs complete / vs hits | fp32 (M6d / M6dc), vs complete | classical tuned | hits truth vs complete |
|---|---|---|---|---|---|
| U | −0.96% (0.032) / +0.60% (0.032) | −1.50% (0.027) / +0.03% (0.026) | −0.59% / −1.30% | −10.5% (0.142) | −1.47% (0.008) |
| V | −1.22% (0.035) / +0.33% (0.035) | −1.79% (0.029) / −0.26% (0.028) | −0.95% / −1.49% | −11.7% (0.149) | −1.47% (0.008) |
| Y | −0.71% (0.030) / +0.86% (0.030) | −1.56% (0.024) / −0.01% (0.024) | −0.18% / −1.23% | −2.1% (0.039) | −1.47% (0.008) |

Per-particle shares:

| cause | share at particle level | evidence |
|---|---|---|
| truth definition | **~100% of the M6dc bias**; MSE share 20-27% (M6dc), 3-8% (M6d) | M6dc against the hits truth is within ±0.3% on every plane [V] |
| precision | bias +0.3-0.5 pt (fp32); hw68 −0.001 | [V] |
| **topology**: angle to the drift axis (0° = across wires at one time, 90° = along the drift) | M6d bias −0.85% → −1.23% (U), −1.11% → −1.54% (V), −0.65% → −0.92% (Y); hw68 0.030 → 0.037; variance share ≤ 0.2% | the classic induction failure is **mild for the model**. Classical goes −9.9% → −14.0% (U) and −11.1% → −15.7% (V), hw 0.12 → 0.21 [V] |
| topology: angle to the wire, Bragg fraction, overlap with other particles, energy | variance shares ≤ 0.4% each; overlap ≤ 0.3% | high-energy particles (> 100 MeV) −0.3% / hw 0.013. Bragg-dominated particles are −0.1 to +0.3%. Classical's overlap share is 5-6% [V] |
| saturation (particles with > 1% of their charge in saturated cells; 0.2-0.6% of particles) | bias −2.3% vs −1.0%, hw68 0.073 vs 0.032 (U); variance share ≤ 0.13% | [V] |

## Summary of causes

1. **The largest bias term is the truth definition, not the model** [V].
   - Against physical (complete) charge the hits truth is 1.27% low per bright cell, 1.47% per particle. The deficit grows with drift, from 0.3% to 2.1%.
   - M6dc reproduces the hits truth to within 0.3%, with no drift dependence.
   - For "exact" physical charge the training target itself must change: complete truth, or the hits truth without the per-group 25 e- cut. This is a label change, not a new loss; it was not done here [A that it would transfer].
2. **Precision is minor** [V]. fp32 inference shifts the bias by +0.3-0.5 pt and removes 1.5-4% of the variance.
3. **Saturation exists but is rare** [V]. 0.01-0.06% of bright cells and < 0.6% of the MSE; those cells are 2-3× noisier.
4. **No exploitable systematic scale** [V]. A charge calibration fitted on calibration events removes only the global offset (1-9% of the MSE, 0% of the variance). Topology, readout edges and wire position each explain < 0.5% of the variance.
5. **The variance is noise-limited** [V].
   - About 75-90% of the per-cell variance (63-85% for M6dc) lies above the clean-sensor floor. The per-wire information limit is the rest: 9-37%, more on V.
   - The model beats exact-kernel linear inversions of the same noisy or corpus input on U and V at every scale.
   - **On Y a linear Tikhonov inversion of the corpus input is better per cell** (hw68 0.032 vs 0.035 M6dc / 0.044 M6d). This is the one place where the model leaves measurable accuracy on the table: about 18% (M6dc) / 48% (M6d) of its Y cell variance.
6. **Cell boundaries are not a cause** [V]. Errors within a 2 × 2 tile are positively correlated, so averaging over regions helps less than √n. The information-limit part averages out to 0.004-0.014 at region level.

## Caveats

- The floors are least-squares or Tikhonov inversions, not information-theoretic bounds. Oracle-support least squares on noisy data is noise-amplifying: its hw68 can exceed the model's. Its rms is inflated by a few outlier cells (support or edge effects), so hw68 is the quantity compared [V].
- The Tikhonov λ (1e-4, 1e-3, 1e-2) is chosen per plane on the same floor events (an oracle choice), which favours the floor [V].
- Particle-level floors were not computed.
- Particle attribution is share-weighted: prediction × the particle's complete-truth share in each cell [V]. Particle directions come from a charge-weighted PCA of each track's deposits per volume [A for curved tracks / showers].
- The saturation mapping from clipped sensor samples to charge cells uses the tokenizer's sensor→drift offset ±1 cell in time on the same wires [A].
- Deposit times are recovered from the hits, as in `causes_FBC.md`. 28.5% of the deposit charge belongs to tracks without hits and lies outside the window [A per deposit].

## Reproduce

```bash
H=baselines/uboone_sp/causes; B=/global/cfs/cdirs/m5238/users/oalter/helix_runs/bright; D=/global/cfs/cdirs/m5238/users/oalter/helix_runs/denoise
CK="--ckpt M6d=$D/M6d_ft_full_dec/final.pt --ckpt M6dc=$D/M6dc_ft_full_dec_cond/final.pt"
CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python $H/bright_dump.py --events 260-387 --precision head32 --out $B/model/head32 $CK
CUDA_VISIBLE_DEVICES=1 scripts/helix_run.sh python $H/bright_dump.py --events 260-387 --precision fp32 --out $B/model/fp32 $CK
CUDA_VISIBLE_DEVICES=2 scripts/helix_run.sh python $H/bright_dump.py --events 0-63 --skip 41 --precision bf16 --out $B/cal_model/bf16 $CK
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=3 scripts/helix_run.sh python $H/floor.py --events 260-307 --out $B/floor
scripts/helix_run.sh python $H/saturation.py --out $H/results/saturation.json
JAX_PLATFORMS=cpu scripts/helix_run.sh python $H/bright_build.py --events 260-387 --out $B/cells
JAX_PLATFORMS=cpu scripts/helix_run.sh python $H/bright_build.py --events 0-63 --skip 41 --cal --out $B/cal_cells
scripts/helix_run.sh python $H/bright_analyze.py --out $H/results
```
