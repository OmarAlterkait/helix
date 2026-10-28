# Open work

What is not done, with enough context to pick up cold. Standing decisions are in
`docs/DECISIONS.md`; procedures in `docs/RUNBOOK.md`; the longer records cited here are
in `docs/archive/`.

## Before the next production run

* **Derive the noisy r2 bin grid and record its digests.** `helix/data/data/reference_bins.json`
  names `coeff_bins_r2_run0027575715_noisy` with `digests: null`; every config reads
  `$HELIX_ARCHIVE/<name>.pt`, which does not exist yet, and `tests/test_bins.py`'s three
  fingerprint tests fail until the digests are recorded (RUNBOOK §5). While there, check
  that the recorded `corpus.basis_digest` still matches r2: the format migration rewrote
  `removal_json`, and hence `basis_digest`, in every shard; if `derive_coeff_bins.py`
  reports `basis-mismatch`, re-record it.
* **Build r2's cells truth.** `scripts/build_truth.py --corpus <run>` for every r2 run
  (~197 MB per shard). The denoiser and every `helix.eval` metric but `recon` need it.
* **Confirm the LR at production length.** `nz_pw16`'s LR 4.4e-3 was tuned at 29.2k steps
  on 16 GPUs; the optimum falls with run length, so a production-length stable phase
  should be checked against a lower LR before it is trusted.
* **The denoiser's default encoder.** `configs/pimm/denoise.py` defaults `model.fm` to
  `$HELIX_EXP/fm` (the stable phase), not the cooled `$HELIX_EXP/fm_cooldown`; pass it
  explicitly (as the RUNBOOK does) or decide which the default should be.

## Model and evaluation

* **RoPE re-test.** With an axis disabled, `helix.model.layers.rope_tables` leaves that
  half of every head unrotated (its docstring calls this a known defect, kept because
  changing it changes the model), so the result behind `rope_split=False` is confounded.
  A full-width single-axis path, then re-run `wire_rope=False` / `rope_split=True` against
  it (`docs/archive/NEXT.md` §2). The two switches stay until then.
* **Score the `dec_frac` / `n_masks` arms** on the noisy recipe. They are kept as
  switches only until scored; `dec_frac` was measured only on the pooled encoder
  (`docs/archive/ENCODER_POOLING.md` §6). Keep or remove them after.
* **CRPS, per-slot NLL and PIT in `CoeffFMEvaluator`.** `var_expl` is at its Bayes limit
  (the read-back is a posterior mean) and the probe cannot resolve effects below ~0.013,
  so nothing in the code can measure an objective-side change. The open outer bins are
  stored as +-1e18, which a CRPS integral must not trust (`docs/archive/NEXT.md` §1).
* **Tokenizer cell geometry.** ~32k tokens per event at 6.87% slot occupancy; try
  `pw x pt = 32 x 16` (`n_slot` 512) at equal wall clock (`docs/archive/NEXT.md` §3). A
  config change and a retrain, not a corpus rebuild.
* **Multi-event batching.** The FM has no event separation, so one event per rank;
  `docs/archive/MULTI_EVENT_BATCHING.md` has the design and its acceptance criterion.
* **pimm's `MAEEvaluator` does not fit**: it passes `return_pred=` to the model's forward,
  which `FMModel.forward` does not accept, and reads metric names the model does not
  emit. `tests/test_pimm_step_contract.py` pins both; when it fails, delete the test and
  this note.

## Corpus, scripts and CI

* **CI runs a deleted script.** `.github/workflows/docker.yml`'s smoke step calls
  `scripts/smoke_train_fm.py` (retired with the clean targets), and verifies the synthetic
  run with `coeff_verify`'s default `--modalities coeff coeff_clean`, which a build without
  `--keep-clean` now fails. Replace the step with a `configs/pimm/fm_smoke.py` run and
  pass `--modalities coeff`.
* **`corpus.sh` does not forward `--keep-clean`**, and it passes `--partition` from the
  site profile, which NERSC's does not set (so it stops on an unbound variable there).
* **Leftovers of removed options.** `helix.model.denoise.ENCODER_PREFIXES` still lists
  `cond_wire.` and `pool_` (modules the FM no longer has); the `export` extra in
  `pyproject.toml` (safetensors) has no reader; `scripts/write_holdout.py` writes a file
  nothing reads (kept for inspection; `tests/test_corpus_identity.py` pins it).
* **`fm_smoke`'s bin table** (`$HELIX_ARCHIVE/coeff_bins_K32_smoke.pt`) may predate the
  noisy grid; rederive it from r2 with K = 32 if the smoke run should resemble production.
* **Optical tokens are not fed to the model.** `helix.optical.tokens` produces 128-slot
  tokens; the model, the dataset and the corpus scripts are TPC-only.
* **The geometry file is duplicated** with pimm-data
  (`helix/tpc/data/cubic_wireplane_geometry.json`; pimm-data's `Densify` ships its own).
  The likely resolution: helix owns the detector description and passes `geom=` to
  `Densify`.
* **History in comments.** Some docstrings and test docstrings still narrate how a bug was
  found rather than the current why (e.g. `helix/data/bins.py`, `helix/paths.py`,
  `tests/test_regressions.py`). Trim them with the next code change in each file.

## Upstream

* **pimm `engines/train.py::run_step`** reads `input_dict["coord"]` whenever an `offset`
  is present, so a batch with an offset and no coord raises after the forward. helix does
  not hit it (`CoeffCollect` emits no offset; a test asserts it). One line; report it to
  pimm's author.

## NERSC

The goal is to move to NERSC. `helix/sites/nersc.yaml` holds its roots, container
(shifter; podman-hpc interactive) and scheduler; RUNBOOK §11 has the commands.

* **Stage r2.** Copy r2 with its `_calib/` to `$CFS/m5238/users/oalter/coeff_tpc_r2`
  (`HELIX_CORPUS_DURABLE`) and stage it to `$SCRATCH/coeff_tpc_r2`; build or copy its cells
  truth; verify the bin grid there with `derive_coeff_bins.py --verify` rather than
  copying it.
* **Rebuild the image without jax** and point `container.image` at the new tag or its
  digest (the profile's `jax-cuda12` tag is stale).
* **Run the suite** (RUNBOOK §2) and `jobs/regress.sh`'s equivalent on NERSC GPUs, which
  build a different corpus from turing's (one GPU type per corpus).
* **Multi-node training is deferred.** `scripts/train.sh` submits one node (`--nodes=1`);
  multi-node needs shifter's `nccl-plugin` module (4.9 GiB/s across nodes against
  podman-hpc's 1.23) and a launch that spans nodes.
