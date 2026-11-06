# Is the stored corpus a lucky draw? Stored corpus vs fresh noise draws

2026-10-10. This follows up `causes_ade.md`. There, M6d scored +0.020 [0.007, 0.031] higher (Y +0.040) on the stored
corpus than on one fresh noise draw with matched statistics, while the classical chain scored the same on both.

Labels:
- **[V]** = verified by running;
- **[A]** = assumed.

Files:
- tables: `draws.json`, `draws_deposit_test.json`, `draws_pt_test.json`;
- intermediates: `/global/cfs/cdirs/m5238/users/oalter/helix_runs/draws/`.

## Verdict

**The +0.02 is a draw fluctuation.** Two things together establish it:
- the comparison against one draw (d0, the second-lowest of ten);
- a realisation-specific excess on the test events' Y plane that does not replicate on 258 more held-out events.

There is no evidence of a generator or code-path difference, and none that the model learned something about the
stored corpus.

1. **Ten fresh draws on the test events** (128 events, each draw through the corpus's own torch-GPU noise path).
   - Pooled efficiency of M6d presence (1% FPR near activity, per-plane thresholds):

     | stored | fresh draws, mean ± sd | range |
     |---|---|---|
     | 0.609 | 0.602 ± 0.009 | 0.587-0.618 |

   - The stored value is z = +0.75, ninth of ten.
   - With the eval's pooled threshold: 0.591 vs 0.586 ± 0.007.
   - M4n: 0.579 vs 0.576 ± 0.011, z = +0.26.
   - The causes_ade +0.020 compared the stored value with d0 (0.589), which sits low in the draw distribution [V].
2. **The one pattern consistent across models was Y and SNR 7-10.** The stored corpus ranks at or near the top of the
   ten draws for all three models there:
   - Y: z = +1.2 to +1.9;
   - SNR 7-10: +0.04 to +0.05, rank 10/10 for all three.

   On the 258 probe-train events (ev000-259, also never used to train the denoisers), the stored realisations score
   **lower** than fresh draws for every model:
   - pooled −0.006 [sd 0.010];
   - Y −0.004 to −0.011.

   Combined over 386 events (inverse-variance; thresholds re-set in every bootstrap resample), stored − fresh is:

   | model | pooled | Y |
   |---|---|---|
   | M6d | −0.001 ± 0.008 | +0.004 ± 0.007 |
   | M6dc | +0.003 ± 0.008 | −0.000 ± 0.007 |
   | M4n | −0.002 ± 0.007 | +0.008 ± 0.007 |

   All |z| ≤ 1.4 [V, `draws_pt_test.json`].

   The test-event Y excess belongs to that one realisation of those 128 events. Three models trained on the same
   corpus respond to a given noise realisation in highly correlated ways, so the three excesses are one observation,
   not three.
3. **Nothing measurable differs between stored and fresh coefficients** [V, `draw_coeff_stats.py`; 128 events, per plane
   and band A4, D4, D3, D2]. Stored corpus vs fresh torch draw vs the classical chain's numpy draw:
   - per-band threshold σ within 0.05%;
   - kept coefficients in signal, near and far regions within 1%;
   - far-noise amplitude tails (|v|/σ > 5 and > 8, 99.9th percentile) within 1-4%, with the two fresh draws deviating
     from the stored corpus by the same amounts;
   - adjacent-wire co-occurrence of kept noise coefficients (a coherent-residual probe) within 0-4%;
   - summed |v| on signal and near-signal coefficients equal to 0.1%.
4. **Bright-charge accuracy is untouched.** Stored vs fresh draws, all three models:

   | metric | difference |
   |---|---|
   | per-cell median and 68% half-width above 16 k e- | ≤ 1e-4 |
   | region sums (16 × 8 cells above 64 k e-) | ≤ 1e-4 |
   | per-particle charge above 1 MeV | ≤ 1e-4 |
   | haze (empty cells near charge above 100 e-) | ≤ 1e-4 |

   These numbers are deterministic functions of the bright signal; the noise realisation does not move them [V].

**Which noise draw future comparisons should use.**
- **Same waveforms for every method.** Re-build the model's input from the same digitized waveforms the classical
  chain reads: `model_dump_cn.py --noise tn` plus `cls_tn.py`, or `--noise cn`.
- **Several draws.** Report the mean over at least 5 fresh draws. The uncertainty should be the draw-to-draw sd combined
  with the event bootstrap.
- **The stored corpus is a legitimate member of the draw distribution,** so it is not biased. But it is one draw, and
  for differences between methods the eval's single-draw event bootstrap **underestimates the uncertainty by about 1.7x**:

  | difference | draw-to-draw sd | within-draw event-bootstrap sd |
  |---|---|---|
  | M6d − M4n | 0.0095 | 0.0058 |
  | M6d − M6dc | 0.0058 | 0.0052 |

## 1. Draw-to-draw spread (test events, 10 fresh draws)

`draw_metrics.py` uses the eval metrics (windows exactly as `eval_denoise`):
- presence detection and gated-charge detection;
- charge accuracy from the raw and gated charge maps.

The draws come from `run_draws.sh` → `model_dump_cn.py --noise tn --seed-offset k`, with k = 0..9 and
seed = recorded noise_seed + k·0x9E3779B1:
- each draw goes through the corpus's own torch-GPU generator and today's DSP;
- each draw is deterministic: d0 reproduces the earlier causes2 draw bit for bit [V].

z = (stored − mean) / (sd · √(1 + 1/10)).

| metric (0.1-0.2 MeV unless stated) | model | stored | draws mean ± sd [min, max] | z | stored rank among 10 |
|---|---|---|---|---|---|
| near pp, pooled | M6d | 0.609 | 0.602 ± 0.009 [0.587, 0.618] | +0.75 | 9 |
| | M6dc | 0.611 | 0.591 ± 0.009 [0.577, 0.604] | +2.15 | 10 |
| | M4n | 0.579 | 0.576 ± 0.011 [0.554, 0.591] | +0.26 | 5 |
| near, eval pooled threshold | M6d | 0.591 | 0.586 ± 0.007 | +0.79 | 7 |
| | M4n | 0.559 | 0.558 ± 0.008 | +0.08 | 5 |
| near pp, U | M6d | 0.535 | 0.510 ± 0.025 | +0.96 | 8 |
| near pp, V | M6d | 0.421 | 0.439 ± 0.019 | −0.93 | 2 |
| near pp, Y | M6d | 0.907 | 0.889 ± 0.014 | +1.22 | 9 |
| | M6dc | 0.917 | 0.886 ± 0.016 | +1.93 | 10 |
| | M4n | 0.898 | 0.874 ± 0.014 | +1.62 | 9 (ties max) |
| SNR 3-5 (clean) | M6d | 0.061 | 0.083 ± 0.016 | −1.24 | 1 |
| SNR 5-7 | M6d | 0.388 | 0.403 ± 0.021 | −0.68 | 3 |
| SNR 7-10 | M6d / M6dc / M4n | 0.831 / 0.837 / 0.815 | 0.788 / 0.785 / 0.772 (sd ~0.02) | +1.9 / +2.6 / +2.2 | 10 / 10 / 10 |
| far pp | M6d | 0.842 | 0.844 ± 0.012 | −0.15 | 5 |
| gated-map detection, pooled threshold | M6d | 0.566 | 0.546 ± 0.014 | +1.36 | 10 |
| near-activity threshold U / V / Y | M6d | 0.885 / 0.925 / 0.964 | 0.888 / 0.913 / 0.962 (sd 0.022 / 0.017 / 0.003) | — | 4 / 8 / 8 |

Charge metrics (all three models): stored − draws ≤ 1e-4 for every metric. Some z values are large (|z| up to 4)
only because the draw sd is ~3e-5.

**The SNR 7-10 / Y pattern is not significant.**
- The exact exchangeability test (rank of the stored corpus among 11 realisations) gives p ≈ 1/11 for each.
- The three models are effectively one observation, and about a dozen detection metrics were inspected per model, so
  one extreme rank is expected.
- A deposit-level test (`draws_deposit_test.py`) gives larger Z (Y: 1.8 / 3.1 / 2.3). It treats deposits as independent
  Bernoullis with p estimated from 10 draws, ignores threshold noise and within-event correlation, and is therefore
  anti-conservative.
- The decisive check is the independent replication in §2.

## 2. Replication on independent stored realisations (258 probe-train events)

`run_draws_pt.sh` and `draws_pt_test.py` cover ev000-259 minus ev041 and ev160 (truth files on the hung OST 61):
stored-corpus maps and two fresh draws, for all three models.

Pooled near-pp detection at 0.1-0.2 MeV, stored − mean(fresh); bootstrap sd re-sets thresholds:

| | M6d | M6dc | M4n |
|---|---|---|---|
| probe-train, pooled | −0.006 ± 0.010 | −0.007 ± 0.010 | −0.006 ± 0.010 |
| probe-train, Y | −0.004 ± 0.009 | −0.011 ± 0.008 | −0.005 ± 0.009 |
| test, pooled | +0.005 ± 0.011 | +0.020 ± 0.013 | +0.002 ± 0.009 |
| test, Y | +0.020 ± 0.012 | +0.026 ± 0.014 | +0.025 ± 0.010 |
| **combined, pooled** | **−0.001 ± 0.008** | **+0.003 ± 0.008** | **−0.002 ± 0.007** |
| **combined, Y** | **+0.004 ± 0.007** | **−0.000 ± 0.007** | **+0.008 ± 0.007** |

The test-set rows here use a weighted-quantile threshold, which differs from `draws.json` in the third decimal.

## 3. What could differ between stored and fresh (tested, none found)

| candidate | test | result |
|---|---|---|
| exact code path of the build (helix c646886, torch DSP on an RTX 2080 Ti, `pimm_data_src` = /sdf/…/pimm-data/src) | look up c646886 locally, in every local helix clone, and on the remote (fresh clone of all branches incl. `legacy-corpus-repro`) | **not recoverable**: the commit exists nowhere [V]. All 38 test shards carry the same provenance (c646886, clean tree, torch, serial, basis_digest 8c4542b6…, the same spectrum sha256 as today's packaged spectrum) [V] |
| torch-wavelet phase (the pre-2026-06-03 torch DWT was PR-exact but not pywt-exact, commit ad188c8) | today's DSP vs stored coefficients | ruled out: rebuilt coefficients overlap the stored ones at the same (band, wire, τ) with value correlation 0.9996 on shared coefficients. A different DWT phase would destroy that [V] |
| noise generator stream | recorded seed vs wrong seed | the corpus RNG stream does not reproduce on torch 2.10 / A100. The recorded seed matches stored coefficients no better than a wrong seed: 0.797 vs 0.793 common, 0.02% equal values [V] |
| noise statistics after the gate and threshold | `draw_coeff_stats.py`, 128 events × 6 planes × 4 bands | identical within 1% (§Verdict 3); the two fresh draws (torch, numpy) deviate from the stored corpus no more than from each other [V] |
| DSP backend (numpy vs torch) | same planes through both | identical coefficient sets; values agree to ≤ 0.24 ADC [V] |
| model memorized the stored corpus | test and probe-train events are held out of denoiser training (probe split); noise seeds are per event | not possible for these events; the stored-vs-fresh difference averages to zero over 386 held-out events [V] |

## 4. Consequences for the margins (fresh-draw means, test events)

| margin | stored corpus | fresh draws mean ± draw sd [min, max] | within-draw event-bootstrap sd |
|---|---|---|---|
| M6d − M4n, presence, near pp | +0.030 | +0.026 ± 0.0095 [+0.011, +0.041] | 0.006 |
| M6d − M4n, eval pooled threshold | +0.033 | +0.028 ± 0.007 [+0.018, +0.039] | 0.005 |
| M6d − M6dc, presence, near pp | −0.002 | **+0.011 ± 0.006** [−0.001, +0.016] | 0.005 |
| M6d − M6dc, gated map, near pp | +0.106 | +0.083 ± 0.016 | 0.008 |
| M6d − classical, same waveforms | — | +0.076 (classical's draw), +0.065 (fresh torch draw) | — |

What this changes:
- **M6d's lead over M4n holds on every draw** (smallest +0.011), but any single draw can show anywhere from +0.01 to
  +0.04.
- **The stored corpus is M6dc's best realisation of eleven.** On fresh draws M6d presence leads M6dc presence by
  +0.011 ± 0.006 (positive on 9 of 10). The earlier "presence heads interchangeable" (dense3) is true at the cell level;
  at the window level M6d is slightly ahead.
- **Model vs classical.** The published stored-corpus comparison (0.609 vs 0.516, per-plane thresholds) mixes two
  different noise draws. On identical waveforms the margin is +0.065 to +0.076; the expected value, fresh-draw mean
  0.602 against classical about 0.520, is about +0.08.

## Reproduce (job 59651328)

```bash
H=baselines/uboone_sp/dense; C=/global/cfs/cdirs/m5238/users/oalter/helix_runs/draws
$H/run_draws.sh; DRAWS="5 6 7 8 9" $H/run_draws.sh                 # 10 fresh draws, M6d/M6dc/M4n -> $C/model_d<k>
scripts/helix_run.sh python $H/draw_metrics.py --draws 0,1,2,3,4,5,6,7,8,9 --out $H/results/draws.json
scripts/helix_run.sh python $H/draws_deposit_test.py --draws 0-9 --out $H/results/draws_deposit_test.json
scripts/helix_run.sh python $H/draw_coeff_stats.py --out $C/coeff_stats
$H/run_draws_pt.sh                                                     # probe-train stored + 2 draws
scripts/helix_run.sh python $H/draws_pt_test.py --out $H/results/draws_pt_test.json
```
