# Open work

What is not done, with enough context to pick up cold. Standing decisions are in
`docs/DECISIONS.md` (with the state of record at the 2026-11 handoff); procedures in
`docs/RUNBOOK.md`; the longer records cited here (NEXT, MULTI_EVENT_BATCHING, DENOISE, ...)
are at git tag `archive-2026-11` (`git show archive-2026-11:docs/archive/NEXT.md`).

## Before the next production run

* **The rest of the campaign.** `coeff_tpc_d1m` holds group j000000 (200k events, with cells);
  `corpus.sh` builds j001000-j004000 the same way (~44 turing GPU-hours per group).
* **Confirm the LR at production length.** fm.py scales `nz_pw16`'s 4.4e-3 (29.2k steps,
  16 GPUs) as sqrt(GPUs / 16); the optimum also falls with run length, so a
  production-length stable phase should be checked against a lower LR before it is trusted.
  The denoiser's 3e-3 / 1e-2 (configs/pimm/denoise.py) are M6d's, also at 16 GPUs.
* **The denoiser's amplitude channel on d1m.** `model.lin_amp_scale` defaults to 0; r1's
  M10lin (S = 30) cut the bright error by up to 27% (docs/DECISIONS.md). One d1m run with
  `model.lin_amp_scale=30` against the default, scored by `helix.eval --metrics denoise`
  (`cells.by_view` resolution), before it becomes the default.
* **The denoiser's charge head.** `model.dist` is `point` until one full-length comparison of
  `gauss` and `cat` (both with `model.cond_charge=True`) against it picks one: resolution per
  truth bin and `cells.dist` calibration. Then make the winner the default and remove the
  other (DECISIONS).

## Model and evaluation

* **RoPE re-test.** With an axis disabled, `helix.model.blocks.rope_tables` leaves that
  half of every head unrotated (its docstring calls this a known defect, kept because
  changing it changes the model), so the result behind `rope_split=False` is confounded.
  A full-width single-axis path, then re-run `wire_rope=False` / `rope_split=True` against
  it (NEXT §2, at the archive tag). The two switches stay until then.
* **CRPS, per-slot NLL and PIT in `CoeffFMEvaluator`.** `var_expl` was shown to
  be a Bayes limit (calibrated PIT) under CLEAN targets only, on r1; it has no ceiling for
  the noisy-only target, and loss / var_expl do not rank representations anyway (rank
  correlation 0.15 with the probe over 17 models), so an objective-side change needs these. The open outer bins are
  stored as +-1e18, which a CRPS integral must not trust (NEXT §1).
* **Tokenizer cell geometry.** ~32k tokens per event at 6.87% slot occupancy; try
  `pw x pt = 32 x 16` (`n_slot` 512) at equal wall clock (NEXT §3). A
  config change and a retrain, not a corpus rebuild.
* **Multi-event batching.** The FM has no event separation, so one event per rank;
  MULTI_EVENT_BATCHING (archive tag) has the design and its acceptance criterion.

## Corpus, scripts and CI

* **`fm_smoke`'s bin table** (`$HELIX_ARCHIVE/coeff_bins_K32_smoke.pt`) may predate the
  noisy grid; rederive it from the corpus with K = 32 if the smoke run should resemble production.
* **Optical tokens are not fed to the model.** `helix.optical.tokens` produces 128-slot
  tokens; the model, the dataset and the corpus scripts are TPC-only.
* **The geometry file is duplicated** with pimm-data
  (`helix/tpc/data/cubic_wireplane_geometry.json`; pimm-data's `Densify` ships its own).
  The likely resolution: helix owns the detector description and passes `geom=` to
  `Densify`.
* **History in comments.** Some docstrings and test docstrings still narrate how a bug was
  found rather than the current why (e.g. `helix/data/bins.py`, `helix/paths.py`,
  `tests/test_tokenize.py`). Trim them with the next code change in each file.

## Choices from the 2026-11 handoff

Open at the handoff to helix-nersc (2026-11-06; the state it described is in DECISIONS,
"State at the 2026-11 handoff"). Each with the recommendation made then.

* **C1. The base and what to merge.** Done on `unify` (2026-11-06): consolidated is the base; the
  denoiser's linear-amplitude input and distributional heads (`model.dist` point|gauss|cat) are
  ported; charge-scale augmentation is not (it needs clean shards; one noise realisation, as on
  data -- the user's decision); presence is q > 25 e- everywhere.
* **C2. How much corpus.** One group (200k) or all five (1M: ~176 more turing GPU-hours,
  ~0.9 TB). Calibrate the recipe on j000000 first (C5); build more when a model size or
  length needs it.
* **C3. `norm_sigma` and bins for more groups.** `corpus.sh calibrate` overwrites
  `norm_sigma_global.npy` with the mean of the per-run tables, and `plan` overwrites
  `CHUNKS.txt` / `RUNS.txt`, so `corpus.sh all` over five groups gives the new shards
  another table and the verifier (one global table) fails the corpus. Recommended: build the
  new groups with j000000's table (`--norm-sigma _calib/norm_sigma_global.npy`, plan
  appended, no recalibration) and the frozen bin grid, after checking that per-group tables
  deviate little; `corpus.sh` needs a small change to append to a plan.
* **C4. Where to build.** One GPU type per corpus and j000000 is turing-built; NERSC has
  A100s only. Build the other groups at S3DF and transfer, or rebuild everything at NERSC
  (new digest, grid and truth). Truth needs the campaign's hits / step / labl (S3DF only).
* **C5. The training recipe on d1m**, judged by probes, not loss: a seed pair at d768 (the
  noise floor); LR 1.1e-3, 1.55e-3, 2.2e-3 at ~30k steps with a 10% cooldown at the batch
  NERSC will use; optionally d1024 at equal compute. B = 4 per node until multi-node; the
  critical batch is constant in cells, not events, so recompute it for d1m's event sizes.
* **C6. The cooldown mechanism.** On a single-run corpus `COOLDOWN_RUNS` cannot set the
  length and `max_len` takes a file-order prefix. Replace both with a hashed event fraction
  or a cooldown length in steps over the whole train split.
* **C7. The value target.** The top bins saturate for bright charge (helix-nersc: 79% of
  >= 256k e Y cells past the last finite edge on r1; not measured on d1m's grid). Extend the
  grid to the observed maximum, a continuous distributional target, or both; judged by
  frozen probes with an amplitude metric. The +-1e18 open edges go when the format changes.
* **C8. Evaluation additions.** Implemented on `unify` (63aa42d..6ffced4): near-activity
  background, AUC, per-view and per-event numbers, the raw-input arm, `--probe-seeds`, the
  region-mask recon, the denoiser's charge resolution / linearity / closure / calibration, the
  matched clear/crowded floor, and `basis_digest` / `q_cut` in the JSON. Not done: a
  label-efficiency curve driver (the config switches exist), and the uprobe label's own ceiling.
* **C9. Denoiser defaults.** Adopt linear-amplitude input and charge-scale augmentation
  (settled by helix-nersc; part of the C1 port); NLL head after M15; `cond_charge` stays a
  switch (detection vs calorimetry). The default encoder: above.
* **C11. Optical.** A joint TPC + light model needs a light source paired with the TPC: the
  campaign's step files carry per-deposit `photons` (and JAXTPC f0d2ed39 ships
  `run_light_sim.py`), but there is no propagation or photodetector stage. goop for
  optical-only work, SBND data for real light, or a light stage added to a campaign. If
  optical enters the model, `plane_id`'s four roles must separate.
* **C13. The charge label.** Decided (the user, 2026-11-06): train and score on cells-3 `q` as
  built, with JAXTPC's 25 e- cut (~1.6% of in-window charge, the diffusion tails), recorded as
  `q_cut` in every result; re-simulate with a lower cut later if it turns out to matter. On r1 the
  uncut (complete) truth gave the denoiser +0.035-0.05 detection and removed a bright bias
  (median 0.972 -> 0.995): the size of what is left on the table.

Assumed at the handoff, not measured: the noise model stands in for the detector; the five
groups are one distribution (a grid and a `norm_sigma` from j000000 describe all); one fixed
noise realisation per event is enough at 3 epochs; LR 2.2e-3 at 115k steps is probably above
the optimum; `fm_cooldown` annealed on a file-order subset (`data.train.max_len=46200`, ~8%
of the stable phase).

## Evaluation caveats (line-by-line review, 2026-11-06)

`helix.eval`'s behaviour is unchanged pending these; they qualify every number it gives.

* Intervals understate: the bootstrap resamples scored events with the probe fixed (no
  probe-fit or probe-seed variance; seed pairs, C8).
* The 1% FPR threshold is in-sample: the per-view 99th percentile of the scored events' own
  bg windows, recomputed per draw.
* bg windows are conditioned on a kept coefficient (centred on a random kept noise cell, no
  truth within +-16 w / +-128 t): the FPR is per region where the compressor fired, harder
  than unconditioned noise windows.
* The windows are asymmetric: bg windows need no truth nearby, iso windows only cap foreign
  charge INSIDE the window at 5% (`helix.eval.events.event_truth`), so a probe that hazes
  charge around tracks gains iso efficiency at no FPR cost. main measured an FPR near
  activity (`near_fpr_at_far`); `helix.data.denoise.near_windows` already draws such windows.
* The floor is about compact, isolated particles (iso selection), not deposits in showers.
  Per view and bin (fm_cooldown): ~8.5-8.9k windows < 0.1 MeV, 4.9-5.3k at 0.1-0.2, 2.8-3.0k
  at 0.2-0.5, 4.3-4.9k > 0.5; binomial error ~0.005, event bootstrap ~0.006 below 0.1 MeV.
* The amplitude reference divides by the corpus `norm_sigma`, not the event's
  `sigma_threshold`.
* The labels reference sees no charge (presence bits + offsets, 52 columns), so 0.29 / 0.24
  is a weak floor; a charge-aware reference is the missing control.
* No raw-input arm: the same MLP on the token INPUTS (main's `raw`) separates the probe's
  capacity from what pretraining adds; floor and labels lack it.
* The floor probe's zero-charge fit rows are kept-coefficient cells only, but most scored
  window cells hold no kept coefficient: the probe extrapolates there, and that sets the bg
  tail.
* uprobe scoring is uncapped per event (fit capped at ~364 rows per event), so dense events
  dominate its R^2.
* recon masks randomly at 0.75 with plane masking off (`mode="random"` bypasses
  `plane_frac`), against main's 75% region mask (var_expl ~0.69 against ~0.31-0.39): not
  comparable, and the change has no DECISIONS entry. var_expl pools all bands against the
  noisy target.
* Features are the last encoder block (no final norm), bf16 autocast, gathered as float16.
* `helix.eval.denoise` scores detection only: no charge bias / resolution per bin,
  linearity, presence calibration, near-activity FPR, noise draws or complete truth.

## Upstream

* **pimm `engines/train.py::run_step`** reads `input_dict["coord"]` whenever an `offset`
  is present, so a batch with an offset and no coord raises after the forward. helix does
  not hit it (`CoeffCollect` emits no offset; a test asserts it). One line; report it to
  pimm's author.

## NERSC

The goal is to move to NERSC. `helix/sites/nersc.yaml` holds its roots, container
(shifter; podman-hpc interactive) and scheduler; RUNBOOK §10 has the commands.

* **Stage the corpus.** Copy `coeff_tpc_d1m` with its `_calib/` to `$CFS/m5238/users/oalter/coeff_tpc_d1m`
  (`HELIX_CORPUS_DURABLE`) and stage it to `$SCRATCH/coeff_tpc_d1m`; build or copy its cells
  truth; verify the bin grid there with `python -m helix.data.bins --verify` rather than
  copying it.
* **Rebuild the image without jax** and point `container.image` at the new tag or its
  digest (the profile's `jax-cuda12` tag is stale).
* **Run the suite** (RUNBOOK §2) with `HELIX_REQUIRE_PIMM=1 HELIX_REQUIRE_DATA=1`, and
  `jobs/regress.sh`'s equivalent on NERSC GPUs, which build a different corpus from
  turing's (one GPU type per corpus).
* **Evaluation fits a `shared` GPU job** (32 cores, 56 GB): the floor holds ~25 GB of
  features in host memory at `FIT_ROWS` = 1M.
* **The training size cap (resolved 2026-11-07).** 600k dropped 18.9% of d1m's events (27% of
  coefficients, the busiest); d1m's largest event needs 13.9 GB in a d768 training step, so fm.py
  caps at 1.2M (every d1m event). The S3DF fm / fm_cooldown trained without that fifth of the
  data; anything compared against them carries it. Log max_memory_allocated on the first steps of
  the first uncapped production run (DDP buckets and fragmentation are not in the one-step test).
  The denoiser is the tighter one: 28.0-28.7 GB at d1m's three largest events on a 40 GB A100
  (23.2 GB at a median one; point, gauss and cat alike), its decoder rows growing with the event
  (97k at the top) -- the first thing to run out on a busier corpus, ahead of the FM.
* **JAXTPC: an int16 wire-index overflow (upstream issue).** d1m event 132912 (run j000000, source
  sim_wire_sensor_jaxtpc_wire_j000660-000664.h5, 14,195 tracks, 13.5 GeV) has its whole plane 3
  off the plane: wire centroids spanning +-2^15, charge intact (1.192e8 e-, as its volume-mates).
  f0d2ed39 production/save.py casts center_wires to int16 (and delta_wires to int8) unchecked, and
  the pixel keys were already wrong upstream of the cast. helix rejects such an event (the
  dataset's off-plane share > 1e-3); a busier sample would hit it more often.
* **Denoiser runs hang: one whole node goes silent (2 of 4 d1m_denoise arms, 2026-11-08).**
  - point_l0 (job 60986199): node 0 stalled right after its step-1429 checkpoint (save complete).
  - gauss_l30 (job 60986206): node 1 stalled mid-step near step 5050.
  - Both times the other 12 ranks timed out on the head's gradient bucket after 600 s.
  - The only main-thread frames dumped were `linecache.checkcache`: ranks stat-ing source files
    while formatting an exception. The pimm code snapshot is on CFS (`$HELIX_EXP/<run>/code`).
  - A CFS metadata stall on that node fits both, but it is unproven, and no FM run hung on the
    same layout. What the denoiser adds: it reads the cells shards and does heavier per-event
    CPU work (DenoiseRows).
  - Next time: `TORCH_NCCL_TRACE_BUFFER_SIZE` (flight recorder) and `faulthandler` stacks from
    every rank, so the stuck node's own frames are captured.
  - The chain stops at a failure, so each hang costs a manual resubmit.
* **A data error should fail the job fast.** One DataLoader worker's exception (the off-plane
  truth cells, 2026-11-07) stopped its rank, and every other rank sat in an ALLREDUCE until the
  NCCL watchdog's 600 s timeout: ten minutes of a 4-node allocation, then a crash with the cause
  buried in the log. Options: TORCH_NCCL_ASYNC_ERROR_HANDLING with a shorter collective timeout
  (DCP saves and the evaluator run collectives too -- measure their longest first), or a transform
  that logs and skips a bad event with a counter. Decide before the next long production run.
* **An oracle for the low-energy floor.** Raw and amplitude are weak references; whether the
  model's 5-12% (presence, clear windows below 0.1 MeV, given a kept coefficient) is near what the
  input allows needs a matched filter on the waveform with the known response and noise
  covariance, on the same windows -- as r1's bound (0.48 at deposit SNR 5-7, 0.86 at 7-10; the
  archive's DENOISE). Also: whether "tokonly" detections (covered, no kept coefficient inside)
  are the deposit's own coefficients just outside the window (efficiency falling with distance
  over the level-4 support) or haze (flat).
