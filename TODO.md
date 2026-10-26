# Open work

What is not done, with enough context to pick up cold. Standing decisions are in
`docs/DECISIONS.md`; the longer records cited here are in `docs/archive/`.

## Corpus and format

* **The bin grid's recorded corpus digest.** The format migration of r2 rewrote
  `removal_json` and hence `basis_digest` in every shard. `helix/data/data/reference_bins.json`
  records r2's `basis_digest` from before the migration (it was committed first); if they
  now differ, `helix.data.bins` reports `basis-mismatch` for r2 until the grid's record is
  re-recorded (`scripts/derive_coeff_bins.py --verify`).
* **Optical tokens are not fed to the model.** `helix.optical.tokens` produces 128-slot
  tokens; the model, the dataset and the corpus scripts are TPC-only.
* **The geometry file is duplicated** with pimm-data
  (`helix/tpc/data/cubic_wireplane_geometry.json`; pimm-data's `Densify` ships its own).
  The likely resolution: helix owns the detector description and passes `geom=` to
  `Densify`.

## Model and evaluation

From `docs/archive/NEXT.md`; each is stated as measured / proposed / to test there.

* **CRPS, per-slot NLL and PIT in `CoeffFMEvaluator`.** `var_expl` is at its Bayes limit
  (the read-back is a posterior mean) and the probe cannot resolve effects below ~0.013,
  so nothing in the code can measure an objective-side change. The open outer bins are
  stored as +-1e18, which a CRPS integral must not trust.
* **RoPE bandwidth.** With an axis disabled, `helix.model.layers.rope_tables` leaves that
  half of every head unrotated (its docstring calls this a known defect, kept because
  changing it changes the model), so the result behind
  `rope_split=False` is confounded. A full-width single-axis path, then re-run
  `wire_rope=0` / `rope_split=True` against it.
* **Tokenizer cell geometry.** ~32k tokens per event at 6.87% slot occupancy; try
  `pw x pt = 32 x 16` (`n_slot` 512) at equal wall clock. A config change and a retrain,
  not a corpus rebuild.
* **Multi-event batching.** The FM has no event separation, so one event per rank;
  `docs/archive/MULTI_EVENT_BATCHING.md` has the design and its acceptance criterion.
* **pimm's `MAEEvaluator` does not fit**: it passes `return_pred=` to the model's forward,
  which `FMModel.forward` does not accept, and reads metric names the model does not
  emit. `tests/test_pimm_step_contract.py` pins both; when it fails, delete the test and
  this note.

## Code hygiene

* **Citations of a retired research tree.** Docstrings in `helix/model/mask.py`,
  `helix/model/tokenize.py`, `scripts/feats_rank.py`, `scripts/viz_2x2_corpus.py` and
  some tests cite `research/...` paths that a new reader does not have. Quote the fact
  inline or drop the citation.
* **Stale comments** name retired pieces: the `pimm submit` launch layer
  (`scripts/helix_env.sh` header and line ~121, `helix/paths.py` ~330,
  `helix/data/__init__.py` ~29, `configs/pimm/fm.py`'s usage line), the old
  `coeff_io` note that pimm-data implements the layout, and documents now under
  `docs/archive/` cited at their old paths (`MULTI_EVENT_BATCHING.md`,
  `COEFF_CORPUS_DESIGN.md`, `docs/REVIEW_FIELD.md`, `docs/INVENTORY.md`, `TESTING.md`,
  `DECISIONS.md`). Fix them with the next code change in each file.

## Upstream

* **pimm `engines/train.py::run_step`** reads `input_dict["coord"]` whenever an `offset`
  is present, so a batch with an offset and no coord raises after the forward. helix does
  not hit it (`CoeffCollect` emits no offset; a test asserts it). One line; report it to
  pimm's author.

## NERSC

The goal is to move to NERSC. `helix/sites/nersc.yaml` holds its roots, container
(shifter; podman-hpc interactive) and scheduler. Still to do there: copy or rebuild r2
and its `_calib/`, the bin grid and the archive under the NERSC roots; run the suite
(section 2 of the runbook) and `jobs/regress.sh`'s equivalent on NERSC GPUs, which build
a different corpus from turing's (one GPU type per corpus).
