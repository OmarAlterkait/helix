# Status

Where HELIX stands (2026-11-09): what exists, the decisions in force, what is open. History
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
| denoiser | reference results on r1 (below); no full-length d1m result yet |
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
| 3 | Basis: coif3 L4, D1 dropped, gate tau 0.05, kappa 1.0 | tau 0.05 won on denoising and support cleanliness (r1) | kappa 0.75 raises the faint bound 0.19 -> 0.34; a rebuild and new digest, the user's call |
| 4 | One GPU type per corpus (turing for d1m) | turing and A100 differ on 0.016% of kept coefficients | deterministic reductions |
| 5 | One noise realisation per event, seeded from its identity at build | real data has one | a reason to vary noise in pretraining |
| 6 | No charge-scaling augmentation or fresh noise in the d1m denoiser (user decision) | it needs clean shards, which data never has | the user; r1 cost: gain response 0.87 without vs 0.97 with |
| 7 | Raw values plus one frozen `norm_sigma` per corpus, applied at tokenize | the corpus stays reversible, the tokenizer changeable | none |
| 8 | Every shard carries a format stamp; no legacy read paths | a reader that accepts anything cannot say what it read | none |
| 9 | Holdout by identity hash, 0.95 / 0.03 / 0.02 | independent of build and sharding; runs are one distribution (r1) | a need to hold out a whole run |
| 10 | Value grid frozen per corpus, verified by digest, not copied | re-deriving moves it less than its own sampling noise and breaks comparability | the analytic grid (feasible; changes training, the user's call) |
| 11 | Categorical value head + occupancy BCE; MSE / NLL heads removed | the categorical head is calibrated (PIT mean 0.512, sd 0.289) | a continuous target that ranks better on the fine-tune key |
| 12 | One attention path: flash varlen in four orders, sm80+ | 6% faster and 0.005 better cooled loss than padded groups | none |
| 13 | Recipe: d768, 12 + 4 blocks, QK-norm, muP, mask 0.75, plane masking 0.1, K = 128 | measured on r1 (QK-norm: d768 went NaN without) | token size, width, decoder depth, plane masking re-tested on the fine-tune key |
| 14 | Peak LR 4.4e-3 at 16 ranks, x sqrt(world / 16) | sqrt(batch) scaling; muP transfers width only | the 6.2e-3 arm, or the fine-tune key ordering LR differently |
| 15 | WSD: the cooldown is a separate run, warm-started by `model.init`, read at `last` | the stable phase commits to no length; resume would skip the anneal | none |
| 16 | No in-training evaluation, no `model_best`, no EMA; the weight is `model/last` | val loss ranks representations at 0.15; EMA and cooldown `last` fine-tune the same (0.530 +- 0.003) | a task where val loss agrees with the fine-tune |
| 17 | Configs are world-size-free; batch = rank count, one event per rank | the FM has no event separation | multi-event batching |
| 18 | Event cap 1.2M coefficients, a guard not a filter | the largest d1m event needs 13.9 GB in a d768 step | a busier corpus |
| 19 | The primary encoder evaluation is the ft_eval fine-tune, keyed on clear and crowded@near at 1% FPR per view | an MAE is pretrained to be fine-tuned; frozen probes measured their own extrapolation | the from-scratch arm landing close (then 256 / 64 labels) |
| 20 | Backgrounds uniform in area with >= 1 kept coefficient; thresholds per view | kept-weighted far windows biased every arm; pooled thresholds ran Y at 2-3% | none |
| 21 | uprobe stays as the second, frozen key | the only frozen key that ordered pretraining arms; it measures 3D information | the fine-tune key validated against it |
| 22 | Comparisons are paired (`compare`), seed-averaged, read per view | a pooled gain carried by one view is not a win | none |
| 23 | Presence is q > 25 e- everywhere | it is the truth's own cut | a truth with a lower cut |
| 24 | Train and score on cells-3 `q` as built, recording `q_cut` (user decision) | re-simulating is expensive | r1: complete truth gave +0.035-0.05 detection; re-simulate if that matters |
| 25 | Denoiser defaults: M6d recipe, `lin_amp_scale` 0, `dist` point, encoder `fm_cooldown` | r1 numbers do not transfer to d1m | a d1m run reproducing M10lin / M11 gains |
| 26 | Training launches through `pimm submit` (`scripts/submit.sh`), pimm pinned per site | multi-node shifter with `nccl-plugin`: 4.9 GiB/s vs podman-hpc 1.23 | none |
| 27 | Reading a run has one owner, `load_run`, from pimm's run directory | an export layer duplicated what the run records | none |

## Open work

- **LR:** run the 6.2e-3 arm. The "training-seed noise 0.04" seen on the presence probe is not
  seen on the MSE probe (s0 0.350 vs s1 0.346), so it was likely probe noise.
- **Recipe re-tests on the fine-tune key:** token size (e.g. `pw x pt = 32 x 16`), width,
  decoder depth, plane masking; most were chosen on r1 with metrics later found biased.
- **Value grid:** extend to the observed maximum or switch to the analytic grid (user decision).
- **Corpus threshold:** kappa 0.75 (user decision; rebuild, new digest).
- **d1m denoiser at full length.** The `d1m_denoise` arms crashed or hung and were initialised
  from `fm_cooldown`'s EMA. The d1m port lacks M11's linear amplitude by default, augmentation,
  complete truth and volume-aware layouts; `dist` is undecided (point / gauss / cat). Its sweep
  scores `last`; the arms in flight (trained at 9f34c74, before the trim) are scored by hand.
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
- **Cooldown subset:** on a one-run corpus `COOLDOWN_RUNS` cannot set the length and `max_len`
  takes a file-order prefix; replace both with a hashed fraction or a length in steps.
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
