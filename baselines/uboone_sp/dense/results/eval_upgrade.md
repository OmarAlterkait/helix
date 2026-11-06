# The upgraded denoiser evaluation: noise draws, per-plane thresholds, complete truth, classical companion

2026-10-11. This builds the draws findings (`draws.md`) into the standard evaluation. Labels:
- **[V]** = verified by running;
- **[A]** = assumed.

Nothing here is committed; the coordinator reviews and commits.

## What changed (scripts only; no `helix/` package code)

| file | change |
|---|---|
| `scripts/eval_denoise.py` | Refactored into functions, with the original computations moved unchanged. New options: `--draws K`, `--draw-cache`, `--no-stored`, `--complete-truth DIR`. New keys: per-plane thresholds, bright-charge and region keys, and `ct_*` complete-truth keys. The aggregate row `<tag>@draws<K>` is new. |
| `scripts/eval_draws.py` (new) | Fresh noise draw k of an event through the corpus builder's own torch-GPU path. The seed is `noise_seed + k·0x9E3779B1`, then today's DSP. Results are cached per (event, draw) on CFS: the coefficients, the digitised planes (int16) and each plane's sha256. It has `--verify` (regenerates and checks bit identity) and a pre-build CLI. |
| `scripts/eval_classical.py` (new) | Scores the tuned classical chain with the same keys. It imports `baselines/uboone_sp` (v2 picks frozen, fitted kernel). It reads the same cached planes (sha-checked), and with `--v2` also its own benchmark noise draw. |
| `scripts/build_eval_complete_truth.py` (new) | Complete per-event fine-cell truth for the evaluation events: no production thresholds, built with JAXTPC 1ea2282's hits code via `baselines/uboone_sp/causes/build.py:rebuild`. |
| `scripts/study/eval_upgraded.sh` (new) | Runs M6d, M6dc and M4n with `--draws 5 --complete-truth`, one GPU each. |
| `tests/test_eval_denoise_keys.py` (new) | Per-plane thresholds, the draw aggregate and complete-truth lookup. Passes, together with `tests/test_boundary.py` [V]. |

**Keys.** Every existing output key keeps its meaning and value. New keys are appended after the old ones:

- **Per-plane thresholds.**
  - `floor_pp_eff1pct_<E>`: far noise windows.
  - `near_pp_eff1pct_<E>`: near-activity noise windows.
  - Both are pooled over planes, with each plane held at 1% FPR.
  - Per-plane versions: `..._<U|V|Y>`.
  - `snr_pp_eff1pct_{far,near}_<snr>`.
- **Bright charge on the map rows.** `res68_16k` and `med_16k`.
- **Regions** (the 16 × 8-cell eval windows).
  - `region_sum_ratio_*`, `region_med_*`, `region_res68_*`, each for truth window sums of 4k-64k and ≥ 64k e-.
  - `region_empty_gt1Q0`.
- **Complete truth.** `ct_` versions of every charge key, plus `ct_map_r` and `ct_presence_auc`.
- **Aggregate rows** carry:
  - `trained`: the mean over draws. Its interval combines the event bootstrap of the draw mean (the same event resamples in every draw) with the draw-to-draw sd/√K: lo/hi = mean ∓ √(boot² + sd²/K).
  - `draw_sd`: the spread of a single draw.
  - `per_draw`.

**Complete truth is mine, not the other agent's.** `--complete-truth` reads either per-event `evNNN.npz` files or the corpus layout `<run>/truth_<shard>.npz`. The second is the format of the other agent's `/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete`, which was still being produced. `causes/cells` holds no cell coordinates, so it cannot be looked up by cell. I therefore built the complete truth for the test events myself, in `eval_draws/complete_truth`, with the same rebuild code [V].

Machinery check: the rebuild with the production thresholds reproduces the hits truth on charged map rows to a median of 1.0002, but only 69% of rows agree within 2% (94% within 10%). The per-cell scatter comes from the deposit-time recovery [V]. Complete / thresholded total charge is 1.016 [V].

## Validation

| check | result |
|---|---|
| (a) `--draws 0` vs `dn_full.jsonl` for M6d, all three rows | Every one of the 79 / 76 / 79 old keys is identical, in the same order. The top-level fields are identical. The per-window JSON files are byte-identical. The rows now also carry 77 / 54 / 77 new keys, so the rows as a whole are not byte-identical by construction [V]. |
| (a') new stored keys vs `draws.md` | M6d presence near pp 0.609 (U 0.535, V 0.421, Y 0.907), the same as `draws.json`'s stored row [V]. |
| (b) per-draw rows vs `draws.json` | 180 / 180 per-draw values identical, to 4 decimals. That covers near pp pooled + U/V/Y and SNR 5-7 / 7-10, for M6d, M6dc and M4n, presence and gated, d0-d4. `M6d@draws5_pres` = 0.6037, the d0-d4 mean in `draws.json` [V]. |
| bit identity of the draw cache | Regenerating from the seed gives the identical sha256 for all planes (`eval_draws.py --verify`, ev260). The d0 planes are byte-identical to the causes2 fresh draw [V]. |
| (c) classical on d0 vs cls-tn | Near pp U / V / Y / pooled = 0.3653 / 0.326 / 0.9237 / 0.5243, equal to cls-tn (0.365 / 0.326 / 0.924 / 0.524). The tuned_any maps are identical for the 60 (event, plane) pairs compared [V]. The comparison stopped at `causes2/cls_tn/ev270_g1.npz`: that **CFS** file hangs on read (DVS wait, `ipclower_tx_request`). See the note at the end. |
| classical on its own benchmark draw (`--v2`) | tuned_any 0.5158 (U 0.365, V 0.302, Y 0.924) and tuned_q 0.436 — the published 0.516 / 0.436 [V]. |

## Results: 5 fresh draws, the 128 test events

Files:
- model rows: `/global/cfs/cdirs/m5238/users/oalter/helix_runs/eval_draws/results/dn_upgraded_<tag>.jsonl`;
- classical rows: `cls_upgraded.jsonl` (draws) and `cls_upgraded_v2.jsonl` (its own draw).

### Detection: 1% FPR per plane near activity, 0.1-0.2 MeV

| method | stored corpus | 5 draws: mean [combined 16-84] | single-draw sd | U | V | Y | eval's pooled threshold (5 draws) |
|---|---|---|---|---|---|---|---|
| M6d presence | 0.609 | **0.604** [0.591-0.615] | 0.011 | 0.519 | 0.436 | 0.889 | 0.583 |
| M6dc presence | 0.611 | 0.593 [0.582-0.606] | 0.008 | 0.509 | 0.419 | 0.885 | 0.572 |
| M4n presence | 0.579 | 0.576 [0.564-0.588] | 0.009 | 0.493 | 0.397 | 0.873 | 0.559 |
| M6d gated map | 0.578 | 0.558 [0.497-0.605] | 0.022 | 0.482 | 0.432 | 0.786 | 0.552 |
| M6dc gated map | 0.472 | 0.473 [0.414-0.517] | 0.013 | 0.390 | 0.334 | 0.723 | 0.420 |
| classical tuned, best statistic | 0.516 (own draw) | **0.522** [0.512-0.533] | 0.007 | 0.366 | 0.315 | **0.930** | 0.294 (mixed scales) |
| classical tuned, ROI charge | 0.436 (own draw) | 0.441 [0.430-0.452] | 0.004 | 0.335 | 0.117 | 0.930 | 0.437 |

The gated-map rows have wide Y intervals. Most Y noise windows score exactly 0 on a gated map, so its 1% threshold sits on a point mass and the near-pp Y number is unstable. Use presence for detection.

### By deposit SNR (clean deposits), per-plane near-activity thresholds, 5 draws

| method | SNR 3-5 | 5-7 | 7-10 | 10-15 | far pp 0.1-0.2 |
|---|---|---|---|---|---|
| M6d presence | 0.084 | **0.401** | 0.797 | 0.984 | 0.849 |
| M6dc presence | 0.077 | 0.386 | 0.791 | 0.987 | 0.835 |
| M4n presence | 0.061 | 0.366 | 0.778 | 0.982 | 0.841 |
| classical tuned, best statistic | 0.064 | 0.324 | **0.806** | 0.979 | 0.502 |
| classical tuned, ROI charge | 0.033 | 0.202 | 0.656 | 0.959 | 0.426 |

Intervals: ±0.01 at SNR 3-5, ±0.02 at 5-10.

### Bright charge vs the COMPLETE truth (5 draws; the hits-truth value in brackets)

| method | median pred/true, cells > 16k | 68% hw, cells > 16k | region sum ratio > 64k | region 68% hw > 64k | region sum ratio 4-64k | haze (empty rows > 0.05 Q0) |
|---|---|---|---|---|---|---|
| M6d charge | 0.993 (1.007) | 0.050 (0.050) | 1.012 (1.021) | 0.040 (0.043) | 1.370 (1.385) | 0.115 (0.179) |
| M6d gated | 0.993 (1.007) | 0.050 (0.050) | 0.996 (1.006) | 0.028 (0.026) | 0.992 (1.016) | 0.012 (0.040) |
| M6dc gated | 0.986 (0.999) | **0.041** (0.040) | 0.989 (0.998) | **0.026** (0.024) | 1.018 (1.040) | 0.012 (0.041) |
| M4n charge | 0.980 (0.994) | 0.052 (0.052) | 0.997 (1.006) | 0.042 (0.045) | 1.359 (1.373) | 0.119 (0.187) |
| M4n gated | 0.980 (0.994) | 0.052 (0.052) | 0.982 (0.991) | 0.028 (0.027) | 0.982 (1.006) | 0.012 (0.040) |
| classical tuned ROI charge | 0.909 (0.920) | 0.121 (0.125) | 0.965 (0.975) | 0.032 (0.031) | 0.862 (0.889) | 0.033 (0.111) |

What these numbers say:
- **The complete truth moves bright medians by −1.3 to −1.4% for every method.** That is the hits truth's missing threshold charge. M6d goes from +0.7% high against hits to −0.7% low against the complete truth; M6dc from −0.1% to −1.4%.
- **It roughly halves to thirds the "haze"** on empty map rows: 0.04 → 0.012 for the gated maps. Part of what was scored as false charge is real charge below the 25 e- sample threshold [V].
- **Resolution keys barely move** with the truth definition.

## Note for the coordinator: two incidents

1. **CFS hang.** `/global/cfs/cdirs/m5238/users/oalter/helix_runs/causes2/cls_tn/ev270_g1.npz` hangs on read from the compute node. Four of my validation processes are stuck in uninterruptible DVS wait (`ipclower_tx_request`) on nid200436, job 59656440. They cannot be killed and will go with the job. Other CFS reads were fine; that file was read normally the day before.
2. **Overwritten file — please check.** When I created `scripts/build_complete_truth.py` (20:35), the tool reported that it was **updating an existing file**. A file of that name had apparently been created moments earlier, probably by the other agent, and I overwrote it. I renamed mine to `scripts/build_eval_complete_truth.py` within about a minute. The other agent's `scripts/build_complete_corpus_truth.py` is intact (modified 20:37). If the other agent had a `scripts/build_complete_truth.py`, it must be restored from their copy.
