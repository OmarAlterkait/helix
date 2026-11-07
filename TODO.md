# Open work

What is not done, with enough context to pick up cold. Standing decisions are in
`docs/DECISIONS.md`; procedures in `docs/RUNBOOK.md`; the longer records cited here are
in `docs/archive/`.

## Before the next production run

* **The rest of the campaign.** `coeff_tpc_d1m` holds group j000000 (200k events, with cells);
  `corpus.sh` builds j001000-j004000 the same way (~44 turing GPU-hours per group).
* **Confirm the LR at production length.** fm.py scales `nz_pw16`'s 4.4e-3 (29.2k steps,
  16 GPUs) as sqrt(GPUs / 16); the optimum also falls with run length, so a
  production-length stable phase should be checked against a lower LR before it is trusted.
  The denoiser's 3e-3 / 1e-2 (configs/pimm/denoise.py) are M6d's, also at 16 GPUs.
* **The denoiser's default encoder.** `configs/pimm/denoise.py` defaults `model.fm` to
  `$HELIX_EXP/fm` (the stable phase), not the cooled `$HELIX_EXP/fm_cooldown`; pass it
  explicitly (as the RUNBOOK does) or decide which the default should be.

## Model and evaluation

* **RoPE re-test.** With an axis disabled, `helix.model.blocks.rope_tables` leaves that
  half of every head unrotated (its docstring calls this a known defect, kept because
  changing it changes the model), so the result behind `rope_split=False` is confounded.
  A full-width single-axis path, then re-run `wire_rope=False` / `rope_split=True` against
  it (`docs/archive/NEXT.md` §2). The two switches stay until then.
* **CRPS, per-slot NLL and PIT in `CoeffFMEvaluator`.** `var_expl` was shown to
  be a Bayes limit (calibrated PIT) under CLEAN targets only, on r1; it has no ceiling for
  the noisy-only target, and loss / var_expl do not rank representations anyway (rank
  correlation 0.15 with the probe over 17 models), so an objective-side change needs these. The open outer bins are
  stored as +-1e18, which a CRPS integral must not trust (`docs/archive/NEXT.md` §1).
* **Tokenizer cell geometry.** ~32k tokens per event at 6.87% slot occupancy; try
  `pw x pt = 32 x 16` (`n_slot` 512) at equal wall clock (`docs/archive/NEXT.md` §3). A
  config change and a retrain, not a corpus rebuild.
* **Multi-event batching.** The FM has no event separation, so one event per rank;
  `docs/archive/MULTI_EVENT_BATCHING.md` has the design and its acceptance criterion.

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
* **Run the suite** (RUNBOOK §2) and `jobs/regress.sh`'s equivalent on NERSC GPUs, which
  build a different corpus from turing's (one GPU type per corpus).
* **Multi-node training is deferred.** `scripts/train.sh` submits one node (`--nodes=1`);
  multi-node needs shifter's `nccl-plugin` module (4.9 GiB/s across nodes against
  podman-hpc's 1.23) and a launch that spans nodes.
