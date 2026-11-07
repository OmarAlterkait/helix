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

* **C1. The base and what to merge.** This line is the base (it contains origin/main to
  039f3ee). From helix-nersc's origin `denoise-m14` (tip 8a84597) take 494a8f6, 3dfc02e
  (truth label), 8cfd2cf (linear-amplitude input), e68fe7e (charge-scale augmentation; not
  its volume-aware drift layouts as a default), 23bd7f8 (FM into an encoder with added input
  columns), 2a8533e + 975218f (record, read together); skip 42f6188 (charge-weight power:
  biases faint cells) and the NERSC launchers 5f70a66 / e0eb082. The distributional heads
  (5edce72: `CellDecoder(dist=point|gauss|hlgauss)`, `forward()` returning 3 values) wait
  for M15. **Port:** the denoiser there reads `helix.probe.resolution` (`cell_inputs`,
  `cell_neighbors`, `set_cell`); here those are `helix.data.cells` / `helix.data.denoise` /
  `helix.eval`. 3dfc02e's label is complete truth, not cells-3's `q` (C13).
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
* **C8. Evaluation additions** (agreed with helix-nersc, none implemented): a threshold-free
  companion to the floor (AUC or efficiency at 5% FPR; seed pairs agree to ~0.005 on AUC
  against ~0.04 at 1% FPR); an amplitude metric (median and hw68 of log(pred / truth) per
  view and per truth-charge bin 1-4k, 4-16k, 16-64k, 64-256k, >= 256k e); two seeds per arm,
  and >= 5 fresh noise draws per test event for the denoiser (one draw understates method
  differences ~1.7x); the pretrained-vs-scratch label-efficiency curve (64 ... 16k labelled
  events; `data.train.n_events`, `model.pretrained=False` exist); the corpus `basis_digest`
  in the eval JSON; optionally the bg score tail. Open: which uprobe statistic is primary
  (R^2 here, Fisher r there).
* **C9. Denoiser defaults.** Adopt linear-amplitude input and charge-scale augmentation
  (settled by helix-nersc; part of the C1 port); NLL head after M15; `cond_charge` stays a
  switch (detection vs calorimetry). The default encoder: above.
* **C11. Optical.** A joint TPC + light model needs a light source paired with the TPC: the
  campaign's step files carry per-deposit `photons` (and JAXTPC f0d2ed39 ships
  `run_light_sim.py`), but there is no propagation or photodetector stage. goop for
  optical-only work, SBND data for real light, or a light stage added to a campaign. If
  optical enters the model, `plane_id`'s four roles must separate.
* **C13. The charge label: hits or complete truth.** cells-3 `q` carries JAXTPC's 25 e- cut
  (~1.6% of in-window charge, the diffusion tails). helix-nersc's complete truth
  (`scripts/build_complete_corpus_truth.py` on their branch) gave the denoiser +0.035-0.05
  detection, removed the bright bias (median 0.972 -> 0.995) and cut presence ECE 0.0084 ->
  0.0031, with presence at q > 25 e-. Options: (a) an uncut `q` column in cells-3, presence at
  25 e- (recommended: one truth format); (b) a separate family. Built at S3DF from step + hits
  and a JAXTPC checkout (f0d2ed39), validated against the hits `q` first (`--validate`); the
  script must use `num_s=32` for d1m (the campaign's default) and be ported to `helix.tpc.io` /
  `helix.data.cells`. ~1,700 job-hours (~3,300 CPU worker-hours) for j000000.

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
* **Multi-node training is deferred.** `scripts/train.sh` submits one node (`--nodes=1`);
  multi-node needs shifter's `nccl-plugin` module (4.9 GiB/s across nodes against
  podman-hpc's 1.23) and a launch that spans nodes.
