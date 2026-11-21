# Status

Where HELIX stands (2026-11-16): what exists, the decisions in force, what is open. History
is in git (`docs/DECISIONS.md` and `TODO.md` up to a550748) and in the tags `archive-2026-11`
(design notes and study records, `git show archive-2026-11:docs/archive/<NAME>.md`) and
`eval-probes-final` (the frozen floor / labels probes, removed).

## Current state

| | |
|---|---|
| corpus | `coeff_tpc_d1m/j000000`: group j000000 of the JAXTPC campaign, 200k events, 1000 `helix-coeff-2` + 1000 `helix-cells-3` shards. NERSC: `$SCRATCH/coeff_tpc_d1m`, durable copy `$CFS/m5238/users/oalter/coeff_tpc_d1m`; S3DF: `/sdf/data/neutrino/omara/coeff_tpc_d1m` |
| basis | `basis_digest` `0dd4f3a223909fd7d8150f54854111ce6e7c89f030227a20113aabf2a27c9457`: coif3, level 4, periodization, D1 dropped, coherent gate tau 0.05, threshold kappa 1.0 / floor 2.0; built on turing (RTX 2080 Ti) |
| truth | cells-3 `q` = hits truth after the simulation's 25 e- per-sample cut (`q_cut`, ~1.6% of in-window charge). Event 132912 has a plane off its wires (JAXTPC int16 overflow); the dataset rejects it |
| value grid | `coeff_bins_d1m_j000000_noisy`: K = 128 per band, uniform in asinh with one gap bin, from 120 events of j000000 (`helix/data/data/reference_bins.json`, digests recorded) |
| size cap | 1.2M coefficients per event: every d1m event. The S3DF `fm` / `fm_cooldown` trained at 600k and never saw the largest 18.9% of events |
| FM runs | S3DF `fm` (stable, 115,488 steps = 3 epochs, 4 A100, LR 2.2e-3) and `fm_cooldown` (9,386 steps), at NERSC under `$CFS/m5238/users/oalter/exp_s3df/`. NERSC `d1m_lr` arms (`lr2.2e-3_cool`, `lr3.1e-3_cool`, `lr4.4e-3_cool`, `lr4.4e-3_s1_cool`; 16 GPUs) under `$HELIX_EXP` |
| denoiser | full-length d1m runs (below); the default is the categorical head + linear amplitude |
| other corpora (S3DF) | `sbnd_data/coeff_sbnd_v1` (SBND data, production); `coeff_tpc_r2` (superseded simulation: readable, not rebuildable); r1 and older are retired |

**Evaluation of encoders.** ft_eval fine-tune on d1m, one eval version (397ec16), `last` checkpoint, 2 fine-tune seeds each. Presence efficiency at 1% FPR per view, 0.1-0.2 MeV, 2026-11-10:

| encoder | clear | crowded@near |
|---|---|---|
| lr4.4e-3 cooled, training seed 0 | 0.534 / 0.534 | 0.537 / 0.536 |
| S3DF stable phase, EMA | 0.521 / 0.521 | 0.529 / 0.531 |
| lr4.4e-3 cooled, training seed 1 | 0.519 / 0.520 | 0.529 / 0.531 |
| lr3.1e-3 cooled | 0.519 / 0.518 | 0.524 / 0.525 |
| `fm_cooldown` (S3DF) | 0.518 / 0.517 | 0.528 / 0.530 |
| lr2.2e-3 cooled | 0.513 / 0.517 | 0.523 / 0.523 |
| **from scratch** | **0.452 / 0.434** | **0.415 / 0.409** |

- Pretraining adds ~+0.075 clear and ~+0.115 near activity, mostly on the induction planes (U ~0.44 vs 0.35, V ~0.35 vs 0.25; Y 0.80 vs 0.77).
- The fine-tune seed spread is <= 0.004.
- The training-seed spread (lr4.4e-3, s0 vs s1) is 0.015. It is the largest pretraining effect seen, and it holds at `last` with both fine-tune seeds.
- LR orders as uprobe does (2.2e-3 < 3.1e-3 <= 4.4e-3), but by 0.004-0.007, inside the training-seed spread. So the key cannot resolve LR at 1024 labels.
- The knobs swept so far (cooldown vs stable, LR 2.2-4.4e-3, seed) all sit within one training-seed spread.
- Earlier tables (d4a8301, best checkpoint, other windows) do not pair with this one.

Frozen references (removed probes, d4a8301 windows): frozen presence 0.34 / 0.10, plain amplitude 0.26 / 0.06 (Y 0.65).

uprobe (frozen, along-wire R^2): `fm_cooldown` 0.534; NERSC arms 0.445 (2.2e-3) < 0.483 (3.1e-3) < 0.489 / 0.505 (4.4e-3). The fine-tune agrees in direction but cannot resolve it, so uprobe stays as a non-ranking diagnostic of 3D information. The NERSC pipeline reproduces `fm_cooldown` (paired frozen difference -0.005, not significant).

## Standing decisions

| # | decision | why | would overturn it |
|---|---|---|---|
| 1 | Pretraining is noisy-only: the target is the input, no clean modality exists | real data has no noise-free counterpart | a real-data target |
| 2 | Pretraining never reads `cells` (`tests/test_boundary.py`) | truth is simulation-only | nothing on the data side |
| 3 | Basis: coif3 L4, D1 dropped, gate tau 0.05; hysteresis threshold: seeds at kappa 1.0 (3.35 / 3.35 / 3.55 / 3.74 sigma for A4 / D4 / D3 / D2), grown through 8-connected coefficients >= 2 sigma | tau 0.05 won on denoising and support cleanliness (r1) | a d1m measurement that a lower seed or floor recovers faint deposits the model then finds (the r1 "kappa 0.75: 0.19 -> 0.34" was measured WITHOUT hysteresis and with D1 kept: it does not transfer) |
| 4 | One GPU type per corpus (turing for d1m) | turing and A100 differ on 0.016% of kept coefficients | deterministic reductions |
| 5 | One noise realisation per event, seeded from its identity at build | real data has one | a reason to vary noise in pretraining |
| 6 | No charge-scaling augmentation or fresh noise in the d1m denoiser (user decision) | it needs clean shards, which data never has | the user; r1 cost: gain response 0.87 without vs 0.97 with |
| 7 | Raw values plus one frozen `norm_sigma` per corpus, applied at tokenize | the corpus stays reversible, the tokenizer changeable | none |
| 8 | Every shard carries a format stamp; no legacy read paths | a reader that accepts anything cannot say what it read | none |
| 9 | Holdout by identity hash, 0.95 / 0.03 / 0.02 | independent of build and sharding; runs are one distribution (r1) | a need to hold out a whole run |
| 10 | Value grid frozen per corpus, verified by digest, not copied | re-deriving moves it less than its own sampling noise and breaks comparability | the analytic grid (feasible; changes training, the user's call) |
| 11 | Categorical value head + occupancy BCE; MSE / NLL heads removed | the categorical head is calibrated (PIT mean 0.512, sd 0.289) | a continuous target that ranks better on the fine-tune key |
| 12 | One attention path: flash varlen in four orders, sm80+ | 6% faster and 0.005 better cooled loss than padded groups | none |
| 13 | Recipe: d768, 12 + 4 blocks, QK-norm, muP, mask 0.75 random on 90% of steps and one whole view per volume on 10% (cross-plane), K = 128 | measured on r1 (QK-norm: d768 went NaN without; plane masking recorded at plane_frac 0.25 vs 0, not 0.1) | token size, width, decoder depth, plane masking re-tested on the fine-tune key |
| 14 | Peak LR 4.4e-3 at 16 ranks, x sqrt(world / 16) | sqrt(batch) scaling; muP transfers width only | the 6.2e-3 arm, or the fine-tune key ordering LR differently |
| 15 | WSD: the cooldown is a separate run, warm-started by `model.init`, read at `last` | the stable phase commits to no length; resume would skip the anneal | none |
| 16 | No in-training evaluation, no `model_best`, no EMA; the weight is `model/last` | val loss ranks representations at 0.15; EMA and cooldown `last` fine-tune the same (0.530 +- 0.003) | a task where val loss agrees with the fine-tune |
| 17 | Configs are world-size-free; batch = rank count, one event per rank | the FM has no event separation | multi-event batching |
| 18 | Event cap 1.2M coefficients, a guard not a filter | the largest d1m event needs 13.9 GB in a d768 step | a busier corpus |
| 19 | The primary encoder evaluation is the ft_eval fine-tune, keyed on LAUC (efficiency averaged over log FPR 1e-3..1e-1, view-mean) for clear and crowded@near at 0.1-0.2 MeV; per-view curves and eff@1% always reported | an MAE is pretrained to be fine-tuned; frozen probes measured their own extrapolation | the from-scratch arm landing close (then 256 / 64 labels) |
| 20 | Backgrounds uniform in area with >= 1 kept coefficient; thresholds per view | kept-weighted far windows biased every arm; pooled thresholds ran Y at 2-3% | none |
| 21 | uprobe stays as the second, frozen key | the only frozen key that ordered pretraining arms; it measures 3D information | the fine-tune key validated against it |
| 22 | Comparisons are paired (`compare`), seed-averaged, read per view | a pooled gain carried by one view is not a win | none |
| 23 | Presence is q > 25 e- everywhere | it is the truth's own cut | a truth with a lower cut |
| 24 | Train and score on cells-3 `q` as built, recording `q_cut` (user decision) | re-simulating is expensive | r1: complete truth gave +0.035-0.05 detection; re-simulate if that matters |
| 25 | Denoiser defaults: `dist` cat, `lin_amp_scale` 30, `cond_charge`, encoder `fm_cooldown` | full-length d1m: 2-3x tighter bright charge for -0.013 LAUC (the loss is on V) | a presence head beside cat recovering V |
| 26 | Training launches through `pimm submit` (`scripts/submit.sh`) at NERSC, pimm pinned (`helix-min`, pimm-pin2; the image by digest); no training at S3DF for now | multi-node shifter with `nccl-plugin`: 4.9 GiB/s vs podman-hpc 1.23 | none |
| 27 | Reading a run has one owner, `load_run`, from pimm's run directory | an export layer duplicated what the run records | none |

**Re-scored with the 2026-11-16 evaluation** (fbe0d8a; 4113 events offline from the dumps, 95% intervals; /pscratch/sd/o/oalter/helix_work/fixeval/). LAUC view-mean, clear / crowded@near, 0.1-0.2 MeV:

| run | LAUC | eff@1% (old key) |
|---|---|---|
| point + lin, all labels | 0.638 / 0.625 | 0.623 / 0.616 |
| cat + lin, all labels | 0.625 / 0.607 | 0.608 / 0.599 |
| ft_cool, 1024 labels (2 seeds) | 0.551 / 0.544 | 0.528 / 0.537 |
| ft_scratch, 1024 labels (2 seeds) | 0.478 / 0.432 | 0.455 / 0.422 |
| amplitude | 0.293 / 0.137 | 0.284 / 0.066 |

- Pretrained - scratch: +0.072 [0.062, 0.082] clear; larger at 1e-3 FPR (+0.11 / +0.16). Read at `last`; at step 2048 the gap is about twice that, since the pretrained fine-tunes overfit and scratch is still improving. A fair comparison needs checkpoint selection on held-out labels and scratch trained to convergence.
- The six pretrained arms stay within ~0.016 LAUC: not resolvable.
- Charge, binned by the CALLED charge over all covered cells: calls below 1k e- are 35-58% on cells without charge. 43% of 25-1k e- truth cells are detected, and 39% of their charge has no kept coefficient. Closure over all cells 1.02 (haze 0.024).
- Calibration on natural cells: ECE 0.054 raw, 0.003 after one global log-odds offset (a prior shift).

**Fixed in the training code 2026-11-16 (affect new runs only):** the decoder's time groups (above); the muP hidden init, which was 2.45x too small in std (72a82e3; the 4.4e-3 LR was tuned under the old init); `fm_cooldown.py` states the 57,600-event cooldown that was run (327907c). New: whole-location masking (`mask_mode="loc"`; round 1: `configs/sweeps/d1m_r1.yaml`).

## Open work

- **Next pretraining round (d1m, one seed per arm, ~60 GPU-h each):** the current recipe with the fixes, whole-location masking, plane_frac 0 and 0.3; scored on the fine-tune with held-out checkpoint selection and a converged scratch arm.
- **1x4 truth: built (2026-11-17).** All 1,000 d1m shards on the 1 wire x 4 tick grid, `helix-cells-3`, from the campaign hits (25 e- per-entry cut, as cells-3), helix dc16365 clean, JAXTPC f0d2ed39. `$CFS/m5238/users/oalter/truth_1x4_d1m/cells` (621 GiB; scratch copy `$SCRATCH/truth_1x4_d1m/cells`). Summed to 2x16 it equals the stored cells-3 (identical cells in every event, q within 1 float32 ulp) on 10 shards checked. Inputs: j000000 hits/step/labl copied from S3DF by Globus (`HELIX_SIM_ROOT`). To use it: `<root>/coeff` -> the corpus (`$SCRATCH/coeff_tpc_d1m`), `<root>/cells/j000000` -> this directory, `data_root=<root>` with the configs' own `split` (the run list; an empty split cannot be passed through `pimm submit`). Built: `$SCRATCH/helix/data_1x4_d1m`.
- **1x4 denoiser (`dn1x4_cat`, branch denoise-1x4).** The default denoiser (cat, lin_amp 30,
  cond_charge, fm_cooldown) trained directly on the 1x4 truth: the queries are 1x4 cells, same
  recipe and length as cat_l30 (4 epochs, 16 GPUs). Success criterion, written before the run
  finished, on the probe split, per view (U, V, Y each):
  - (a) `python -m helix.eval subcell --fine dn1x4_cat --coarse cat_l30`: on the occupied
    (q > 25 e-) 1x4 sub-cells of bright (q >= 5000 e-, covered) 2x16 parents, the median
    |log(max(pred, 1) / q)| of the 1x4 charge given presence is at most 1/3 of that of cat_l30's
    charge spread evenly over the parent's 8 sub-cells (`even_over_fine` >= 3).
  - (b) `python -m helix.eval dn1x4_cat --corpus $SCRATCH/coeff_tpc_d1m --model-grid 1x4`: the 1x4
    prediction summed to 2x16 (noisy-OR presence; charge = the sub-cells' summed expected charge
    given presence, `helix.eval.denoise.coarsen`) is not worse than cat_l30 scored by the same
    code on the same events and seed: detection -- LAUC headline and eff@1% FPR, per view, from
    `compare` (A = 1x4, B = cat_l30; "not worse" = the difference's 95% interval reaches 0 or
    above); charge -- bias and resolution in bins of the called charge (`cells.charge.<view>.
    pred_bins`) not worse, bin by bin over the bins with n >= 1000 (|bias| and resolution each no
    larger than cat_l30's by more than 0.005; these tables carry no interval).

  What changed for 1x4: nothing in the labels or the sampling (every box is physical; the same
  train events at both grids, truth sums equal). Per event the queries are 152k (85k at 2x16; the
  20k charged-cell cap binds at both grids, 47/48 events), positives 22% (34%): the noise windows
  are the same boxes with 8x the cells, 36% of the rows (8%). The head's activations then ran a
  40 GB A100 out of memory, hence `model.checkpoint_decoder=True` (peak 25.6 GB, 1 GPU). The data
  root is `$SCRATCH/helix/data_1x4_d1m` (`corpus_identity` taught its layout). Pilot
  (`$HELIX_EXP/dn1x4_pilot3`: 1 node, 8,000 events, 2,000 steps): CE 1.45 -> ~0.8, mse 0.74 ->
  0.08, 0.61 s/step, data 0.003 s; on 40 probe events (sub-cell) fine 0.20 against even 0.93
  (`even_over_fine` 4.7; U / V / Y 4.5 / 4.7 / 4.9). The noisy-OR sum in (b) calls many faint parents (its
  100-1k e- bins at bias +2.6 to +3.1 on the pilot, against cat_l30's +0.5 to +1.2): watch it in
  the full run. Full run: job 61541311, `$HELIX_EXP/dn1x4_cat`, frozen tree run-0b81234; its
  evaluations are queued `afterok` (61542978: (b) + compare; 61542979: (a)), outputs in
  `$HELIX_EXP/dn1x4_eval/`, where cat_l30 is re-scored by today's code (`cat_l30.json`).

- **LR:** run the 6.2e-3 arm. The "training-seed noise 0.04" seen on the presence probe is not
  seen on the MSE probe (s0 0.350 vs s1 0.346), so it was likely probe noise.
- **Recipe re-tests on the fine-tune key:** token size (e.g. `pw x pt = 32 x 16`), width,
  decoder depth, plane masking; most were chosen on r1 with metrics later found biased.
- **Value grid:** extend to the observed maximum or switch to the analytic grid (user decision).
- **Corpus threshold:** untested on d1m. The r1 bound (no hysteresis) does not transfer. A d1m bound needs the raw waveforms (at S3DF).
- **Tokenizer time constants (root cause found 2026-11-16; /pscratch/sd/o/oalter/helix_work/align/).**
  The simulation is aligned: truth and wire signal share each plane's clock. The coefficient-to-truth
  offsets are fully explained, to ~1 tick with no fitted parameter, by the wavelet basis (each band's
  atom position) plus each plane's detector response. Two of our constants are wrong:
  - `toff` U = -17.4 (`tokenize.py:88`) is r1's response lag, copied; d1m's is about -1 to -2. It places
    U coefficients ~15 ticks late in every band (V and Y match their responses).
  - `delta` A4 = -2.38 and D2 = 0.50 do not match the basis (-1.859, 1.0; the energy centroid of the
    unit-coefficient reconstruction); D4 and D3 do.
  - Proposed (awaiting decision): `toff` = 0 and `delta` from the basis, i.e. the simulation's time
    taken as given, the response lag left to the denoiser. Old runs record neither constant, and
    `patch_config` fills missing fields from today's defaults, so the old values must be pinned first.
- **d1m compression facts (audit):**
  - 2.05% of the tokenized bands' coefficients are kept (10.1% token-slot occupancy, ~479k per event); 29.9% of them are hysteresis-grown, below the seed threshold.
  - Hysteresis adds only +0.01 to +0.065 to faint-deposit footprint retention, against a +0.035 null.
  - 61-67% of 0-0.1 MeV deposits and 12-33% of 0.1-0.2 MeV deposits leave no kept coefficient in their footprint; only context can find them.
- **Cross-plane reconstruction** (unmeasured until now; audit job 61142279, 200 events): variance explained with one view masked per volume 0.597 (lr4.4e-3) / 0.581 (`fm_cooldown`), against 0.70 random. Two views masked: ~0 (collapse, as on r1). Forced U / V / Y 0.61-0.63 / 0.54-0.56 / 0.52-0.54.
  - These numbers carry a decoder bug, fixed 2026-11-16 (69fc22b): queries and keys were cut into equal-count time groups separately, and a hidden view's tokens sit at different times than the visible views', so 23% of plane-mode queries attended keys from another time range. Same weights, fixed decoder (120 events): one view hidden 0.589 -> 0.614; hidden U / V / Y 0.625 / 0.558 / 0.538 -> 0.638 / 0.583 / 0.612; random 0.701 -> 0.703.
  - Mask queries exist only at occupied patches, so each query knows that its own patch is occupied (one bit; queries never attend each other).
- **d1m denoiser at full length: done (2026-11-13; $CFS/m5238/users/oalter/exp/helix_d1m_v2, run-397ec16, all labels, `last`, encoder fm_cooldown `last`).**
  Presence at 1% FPR per view, 0.1-0.2 MeV; charge hw68 of log(pred/true) at 64-256k e-:

  | run | clear (U / V / Y) | crowded@near | clear <0.1 | hw68 64-256k U / V / Y |
  |---|---|---|---|---|
  | point | 0.618 (0.56 / 0.50 / 0.81) | 0.606 | 0.182 | 0.033 / 0.035 / 0.030 |
  | point + lin_amp 30 | 0.616 | 0.609 | 0.182 | 0.027 / 0.031 / 0.024 |
  | gauss + lin_amp | 0.571 | 0.567 | 0.163 | 0.018 / 0.022 / 0.012 |
  | **cat + lin_amp (now the default)** | 0.601 | 0.591 | 0.179 | **0.015 / 0.019 / 0.010** |

  - All labels lift detection from the 1024-label fine-tune's ~0.52 / 0.53 to ~0.60-0.62.
  - lin_amp tightens charge at no detection cost. cat trades -0.015 detection for 2-3x tighter charge.
  - Charge binned by TRUTH at 1-4k e- is dominated by cells with no visible evidence (crit/denoise audit). Read charge binned by prediction.
- **Hangs: root cause found (2026-11-10, /pscratch/sd/o/oalter/helix_work/hang_diag/).**
  - pimm's NERSC site default gave the step 24 CPUs (`cpus_per_proc 6`), all on NUMA node 0 (63 GB of 251).
  - 4 ranks plus 32 DataLoader workers (the old denoiser: ~72 GB anon+shmem) overflowed it. The kernel evicted the container's code pages and refaulted them through the image's single-threaded squashfs (on Lustre), freezing the job for minutes.
  - Uneven freezes hit NCCL's 600 s timeout; freezes inside a DCP save hit GLOO's 1800 s.
  - Major page faults separate the runs perfectly: hung or stalled steps 0.19-2.1M; every clean step 2.9k-7.8k (FM, ft, v2).
  - Reproduced by squeezing the cgroup: 30-36 of 42 processes in D state in `squashfs_decompress`.
  - Not the evaluator, the data, CFS writes, OOM or NCCL. The FM fit node 0, which is why it never hung.
  - Fix: configs/launch/_common.yaml gives the step the whole node (`cpus_per_proc 32`, `mem "0"`; measured 0.374 vs 0.51 s/step).
  - Verified (2026-11-10, jobs 61094774 / 61094776) under the same 88 GiB squeeze that froze the old shape at step 2304:
    - the whole-node shape ran 4 x 4096 steps with 2.9-6.7k faults and no freeze, ~29 vs 45-48 min per run;
    - the new code's smaller footprint also survived on the old 24-CPU shape (2.3-4.6k faults). That is a margin, not a fix.
  - Keep gating production runs on sacct major faults (TRESUsageInMax `pages`) < 10k per step.
  - Still open: the 600 s NCCL timeout turns a survivable freeze into a lost run; persistent val workers no longer exist (v2).
- **Memory:** the denoiser peaks at 28.0-28.7 GB on d1m's largest events (40 GB A100); it runs
  out before the FM on a busier corpus.
- **Container:** rebuild the NERSC image without jax, repoint `nersc.yaml`'s image (by digest),
  then run the suite there with `HELIX_REQUIRE_PIMM=1 HELIX_REQUIRE_DATA=1`.
- **pimm:** pimm-private has 5 local, never-pushed commits; after the trim only GLOO-DCP,
  external config by path, and preflight `import_custom_modules=False` are needed. Push or
  upstream them; S3DF has no pin.
- **More corpus:** groups j001000-j004000 (~44 turing GPU-hours each). `corpus.sh calibrate` and
  `plan` overwrite the global table and the plan, so build new groups with j000000's table and
  an appended plan (needs a small `corpus.sh` change). NERSC has no campaign copy, so truth is
  built at S3DF.
- **Cooldown subset:** `fm_cooldown`'s `max_len` (57,600 events, the recipe run) takes a
  file-order prefix; replace it with a hashed fraction.
- **RoPE re-test:** a full-width single-axis path, then `wire_rope=False` / `rope_split=True`.
- **Multi-event batching:** design at the archive tag (`MULTI_EVENT_BATCHING`).
- **Optical** is not fed to the model; a joint model needs a light simulation paired with the
  TPC, and `plane_id`'s four roles (FiLM, RoPE axis, mask group, identity) would separate.
- Smaller: a label-efficiency curve driver (the config switches exist); `fm_smoke`'s bin table
  may predate the noisy grid; the geometry JSON is duplicated with pimm-data; the +-1e18
  open-edge sentinels in every checkpoint's bin edges go when the format next changes; the
  SBND-data-helix regression job's simulation reference predates the campaign source; some
  docstrings still narrate history.
- Upstream: JAXTPC f0d2ed39 casts wire indices to int16 unchecked; pimm `run_step` reads
  `coord` whenever an `offset` is present (helix does not hit it).

## Caveats on current numbers

- Event-bootstrap intervals hold the model fixed; the fine-tune's own spread needs seeds.
- The 1% FPR threshold is in-sample per view (the scored events' own background), recomputed
  per bootstrap draw.
- The key is about compact isolated deposits, not deposits in showers.
- The amplitude reference divides by the corpus `norm_sigma`, not the event's threshold sigma.
- Window numbers before 2026-11-09 (frozen probes, kept-weighted backgrounds) are not
  comparable with current ones.
- Anything compared against the S3DF `fm` / `fm_cooldown` carries their 600k size cap.
- Assumed, not measured: the noise model stands in for the detector; the five campaign groups
  are one distribution (one grid and one `norm_sigma` from j000000 describe all); one fixed
  noise realisation per event is enough at 3 epochs.
