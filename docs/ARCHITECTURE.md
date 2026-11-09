# Architecture

What the code is and how data moves through it. Procedures are in `docs/RUNBOOK.md`; the
current numbers and standing decisions in `docs/STATUS.md`.

## 1. Data flow

```
JAXTPC wire campaign (HELIX_SIM_ROOT)      SBND extracted files       goop light files
        |  python -m helix.build sim            | ... sbnd                 | ... goop (optical only)
        v                                       v                          v
helix.build: event loop -> TPCChain (gate, DWT, threshold) / OpticalChain -> ShardWriter
scripts/corpus.sh: plan -> calibrate norm_sigma -> build, as Slurm arrays
        v
corpus root  <run>/sim_wire_coeff_NNNN.h5   noisy coefficients        helix-coeff-2
             <run>/sim_wire_cells_NNNN.h5   simulation truth          helix-cells-3  (scripts/build_truth.py)
             <run>/<ds>_optical_NNNN.h5     optical streams           helix-optical-2
             _calib/ CHUNKS.txt RUNS.txt <run>.npy norm_sigma_global.npy
        v
helix.data.CoeffTPCDataset -> helix.model.tokenize.CoeffTokenize -> CoeffCollect
        v
python -m pimm.train + helix.integrations.pimm     (FM pretraining; denoiser fine-tuning)
        v
<save_path> = $HELIX_EXP/<run>  -> helix.model.checkpoint.load_run -> python -m helix.eval
```

## 2. Packages and the boundary

| package | imports pimm-data / pimm | what it is |
|---|---|---|
| `helix.core` | never pimm-data | wavelet transform, robust statistics and hysteresis (`robust`), the coeff shard codec (`coeff_io`), the event and its provenance (`coeff_event`) |
| `helix.tpc` | never pimm-data | the TPC chain (`chain`), coherent gate, threshold, noise model, readouts and plane geometry, the JAXTPC and SBND readers |
| `helix.build` | defers one (`SimSource` uses pimm-data's `densify`) | the build pipeline and its CLI |
| `helix.optical` | no | the optical codec, calibration, optical shards and tokens, the goop source |
| `helix.data` | pimm-data, in `dataset` only | reader and dataset, verifier and corpus identity, bin grid, `cells` truth, the denoiser's labels |
| `helix.model` | defers one (`tokenize` registers into pimm-data) | tokenizer, FM, denoiser, schedule and muP, reading a run |
| `helix.eval` | through `helix.data` when events are read | `python -m helix.eval`: the fine-tune scoring, `uprobe`, `compare` |
| `helix.integrations.pimm` | pimm and pimm-data | registry adapters into pimm's trainer |

**The one real boundary: `helix.core` and `helix.tpc` never import `pimm_data`.** pimm-data
requires torch>=2.5 and hdf5plugin unconditionally; the corpus-builder side must not depend
on it. torch itself is not excluded: `helix.core.robust` and the TPC chain import it at
module scope. `tests/test_boundary.py` checks, importing in clean interpreters: (1) no
submodule of `helix.core` / `helix.tpc` pulls in `pimm_data`; (2) its `LIGHT` list (`helix`,
`helix.core`, `helix.tpc`, `helix.paths`, `helix.data` and its `bins` / `verify` / `cells` /
`denoise`, `helix.model.tokenize`, `helix.model.checkpoint`, `helix.eval`, `helix.eval.events`)
imports with neither torch nor pimm-data, as `pimm submit`'s preflight (a config loaded on a
host without torch) and DataLoader workers need, which is why `helix/data/__init__.py` and
`helix/model/__init__.py` resolve heavy names lazily; (3) `helix.data.dataset` does need
pimm-data; (4) the bundled JAXTPC geometry JSON is byte-identical to pimm-data's; (5) no
pretraining source (configs except `denoise.py`, `helix/model`, `helix/integrations/pimm`)
contains the string literal `'cells'`.

The pimm-data revision is named once, in `pyproject.toml` (`[tool.uv.sources]`);
`container/install.sh` reads it, CI checks it is a full revision, and
`tests/test_pimm_config.py` checks the pimm checkout pins the same one.

## 3. Building a corpus (`helix/build.py`, `scripts/corpus.sh`)

A source is a class that only reads: `SimSource` (`helix/build.py`; the JAXTPC campaign),
`SBNDSource` (`helix/tpc/io_sbnd.py`), `GoopSource` (`helix/optical/goop.py`).
`python -m helix.build SOURCE` (`sim`, `sbnd`, `goop` or `module:Class`) runs it; the source
protocol and every option are in `helix/build.py`'s docstring. Everything after the read is
shared: per `--events-per-shard` events, shard k is written as `PREFIX_coeff_k.h5` (and
`PREFIX_optical_k.h5` when the source has light), each as `.part`, audited
(`coeff_io.audit_shard`) and renamed, so a file under its final name is complete and
`--skip-complete` resumes. `--norm-sigma TABLE` gives every shard the corpus table;
`--calibrate TABLE` writes a table and no shards.

`python -m helix.build SOURCE plan OUT RUN_DIR...` lists the tasks
(`PREFIX FIRST COUNT EVENTS_PER_SHARD SHARD_START INPUT...`). `scripts/corpus.sh` writes the
plan to `_calib/CHUNKS.txt` and the runs to `_calib/RUNS.txt`, calibrates one `norm_sigma`
per run and their mean (`_calib/norm_sigma_global.npy`), then submits arrays over the plan.

**The simulation source** is the JAXTPC wire campaign `doraemon_2026_1M_lartpc` (format 2;
`helix.tpc.io`), laid out `<group>/<task>/{sensor,step,hits,labl}/`: 5 groups `j00X000` of 200
tasks x 1000 events. A corpus run is a group, a build task is one task's sensor file, and shard
k holds campaign events `200k .. 200k+199`. `/ident` is (run = group, source_file = sensor file
name, event = campaign `event_id`); the noise seed (blake2b of the identity) and the holdout
split are functions of it, not of helix's sharding. Format-1 productions are not read.

## 4. The TPC chain (`helix/tpc`)

`helix.tpc.chain.TPCChain`, torch on one device, one chain for simulation and data:

1. pedestal; channel status (dead, noisy and baseline-step rows excluded);
2. DWT: **coif3, level 4**, periodization (`helix.core.wavelet`);
3. **coherent gate** (`helix.tpc.coherent_gate`) on the readout's channel groups: per band and
   pass, the common mode of each (group, position) over non-signal rows is subtracted where it
   is small against its MAD scale `sigc`, and with `tau` also where it is large unless more than
   `tau` of the group's rows were flagged as signal. Simulation (`SIM_BASE`): the noise model's
   64-wire blocks per plane, kgate = ksig = 3, 2 passes, **tau 0.05**, global `sigc`. SBND
   (`SBND_DEFAULT`): front-end boards across planes, per-group `sigc`, no `tau`;
4. **hysteresis threshold** (`helix.tpc.threshold_rows`): sigma per plane and band, seeds at
   kappa = 1.0 times the universal threshold, grown through floor = 2.0 sigma; the finest band
   **D1 is dropped**; kept values rounded to a power-of-two step <= sigma / 16.

**The noise model** (`helix.tpc.noise`): incoherent ENC with the MicroBooNE series spectrum plus
one coherent waveform per 64-wire group (neighbours anti-correlated), then `digitize`. It runs
at build, seeded per event, so the corpus holds exactly one noise realisation per event.

Float32 reductions depend on the GPU generation, so one GPU type builds a corpus.

## 5. Shard formats

**`helix-coeff-2`** (`helix/core/coeff_io.py`, the schema of record; flat columnar):

```
/config  attrs: format, n_events, dataset_name, file_index, global_event_offset, readout_type,
                wavelet, dwt_level, dwt_mode, n_ticks_raw, pad, sigma_norm, basis_digest,
                removal_json, threshold_json, has_coords, noise_json, provenance_json
         datasets: band_lengths (n_bands,), gids (G,), n_wires (G,), norm_sigma (G, n_bands)
/coord   band u8, plane_gid i32, wire i32, tau i32 (M,); event_offset i64 (n_events+1,);
         coord_digest u64 (n_events,); sigma_threshold f32 (n_events, G, n_bands);
         wire_valid u8 (n_events, G, ceil(max n_wires / 8))   absent = all valid
/value   f32 (M,)   raw coefficients
/ident   run, source_file, event (n_events,)  [+ noise_seed for simulation]
```

`format` is required by every reader (no legacy paths). `basis_digest` hashes the basis
(wavelet, level, mode, ticks, pad, band lengths, removal, threshold, sigma_norm): changing any
of them is a rebuild and a new digest, and a checkpoint is refused on a corpus of another
digest. Values are raw; the tokenizer applies `asinh(value / norm_sigma[row of gid, band])`,
rows following the POSITION of a gid in `gids`. `sigma_threshold` (per event, defines the kept
set) and `norm_sigma` (per corpus, normalises) are different tables. `provenance_json` records
the helix `git` commit and `git_dirty`.

**`helix-cells-3`** (`helix/data/cells.py`): simulation truth on the fine grid of 2 wires x 16
ticks, one `{name}_cells_<k>.h5` per coeff shard (same events and order, `/ident` copied),
built by `scripts/build_truth.py` from the campaign's `hits`, `step` and pysupera `labl`.
Four per-event CSR tables: `/cells` (charge `q`, the dominant hits group's share and (y, z)),
`/cell_track`, `/plane_track` (per-track charge centroids), `/tracks` (energy, motif, pid).
`q` is the hits file's electrons after JAXTPC's 25 e- per-sample cut; the cut is recorded
(`label_json` `hits_threshold`) and reported by `helix.eval` as `truth.q_cut`. `Q0` (the charge
scale of every `log1p(q / Q0)`) is defined here once. ~197 MB per 200-event shard.

## 6. Reading the corpus (`helix/data`)

- `CoeffTPCReader` / `CoeffTPCDataset` (`dataset.py`, pimm-data): modalities `coeff` (always)
  and `cells`, joined by identity and checked; one `shard_signature` per corpus (digest,
  removal, gids, n_wires, norm_sigma); the shard tables ride in `sample['coeff']['_meta']`.
  Training reads only `cells/key` and `cells/q` (`cells_columns`).
- **Holdout by identity:** `holdout=dict(seed, fractions)` with `split_role` hashes
  blake2b(run/source_file) + event, so the split survives rebuilds and reshards (fm.py: train /
  val / probe = 0.95 / 0.03 / 0.02). `n_events` + `subset_seed` give a fixed nested label subset.
- **Event rejection:** `max_event_size` drops events above that many coefficients (fm.py:
  1.2M, every d1m event). With `cells`, an event whose truth puts more than 1e-3 of a plane's
  charge off its planes is skipped (d1m event 132912, a JAXTPC int16 wire overflow).
- `verify.py`: `verify(root)` checks a corpus against its plan (planned shards with their
  events, nothing extra, one signature, one clean commit, no identity twice), or a run dir;
  `compare` byte for byte; `corpus_runs`, `corpus_identity`, `check_corpus_matches`, and
  `resolve_corpus`, which finds a run's recorded corpus at another site and holds it to the
  recorded digest.
- `bins.py`: the value grid (section 7); `denoise.py`: `DenoiseRows` (section 8).

## 7. The foundation model (`helix/model`)

- **Tokenizer** (`tokenize.CoeffTokenize`, numpy): coefficient rows -> tokens of `pw x pt =
  16 x 8` slots (`n_slot` 128) per band over A4, D4, D3, D2; per slot the normalised value
  `inp` and occupancy `occ`. The target is the input itself (noisy-only); `dead_frac` is the
  one augmentation.
- **Encoder** (`fm.FMModel`, `blocks.py`): d768, 12 blocks, 12 heads, QK-norm, muP
  (`schedule.py`). Each block attends within fixed-size groups along one of **four token
  orders**: plane x time, time, plane x wire x time, and time shifted by half a group
  (`gp` 1024 / `gd` 2048 tokens). Groups run unpadded through flash varlen attention (sm80+);
  a per-group SDPA reference runs on CPU. `fused_qk` is the Triton QK-norm + RoPE kernel.
- **FiLM** (`ResponseFiLM`) conditions every token on (band, plane, wire feature).
  **RoPE** is axial on (time, wire); `wire_rope` and `rope_split` are switches kept for a
  re-test, because a disabled axis leaves its half of each head unrotated (`rope_tables`).
- **Masking** (`make_mask`): 75% of tokens at random; with probability `plane_frac` (0.1) a
  step hides whole planes instead. Grouped by `plane_id`, the token set is order-free.
- **Decoder:** 4 CrossMAE blocks; masked queries cross-attend the visible encoded tokens in
  drift-time groups.
- **Head and loss** (`loss.py`): per slot an occupancy logit and a **categorical value head**
  over K = 128 bins; loss = occupancy BCE + value cross-entropy, each over its own support
  (masked, valid; the value term on occupied slots only). The grid (`helix.data.bins`) is uniform
  in asinh on each side of zero with one bin over the sub-threshold gap and open outer bins
  (stored as +-1e18); edges and centroids are persistent buffers in every checkpoint. The
  production grid is named in `helix/data/data/reference_bins.json`.
- `lin_amp_scale` S > 0 adds the linear amplitude `sinh(inp) / S` as input columns (zero at
  the start). **One event per rank:** the FM has no event separation (`FMTrainer.check_world`).
- **Reading a run** (`checkpoint.load_run`): pimm's run directory (`resolved_config.json`,
  `model/last`, `provenance.json`) -> the model with its bins, the train split's tokenizer,
  the provenance. `weights="ema"` / `"best"` read an older run's `model_ema.pth` /
  `model_best.pth`. Nothing is exported.

## 8. The denoiser (`helix/model/denoise.py`, `helix/data/denoise.py`)

`DenoiseModel` = an FM run's encoder (`FMModel.encode`: every token visible) + `CellDecoder`.
`build_denoise(fm, pretrained, freeze_encoder, ...)` takes the FM run's architecture and
tokenizer and, with `pretrained`, its `last` weights (`pretrained=False` is the from-scratch
baseline). A pretrained FM loads into a `lin_amp_scale` encoder with the new columns at zero
(`pad_input_columns`). Param groups: the encoder's muP groups plus the head at `lr_head`.

- **`CellDecoder`:** each 2 x 16 fine cell's query (its offsets and the covering tokens' input
  slots) cross-attends the encoder features of the 3 x 3 token neighbourhood in every band, with
  a learned null key. It predicts charge `log1p(q / Q0)` and a **presence** logit (q > 25 e-).
- **`cond_charge`:** False regresses charge on every queried cell (detection, maps); True on
  charged cells only, a charge conditional on presence (a hurdle; calorimetry).
- **`dist`**, the charge head: `point` (squared error, default), `gauss` (+ a per-cell
  log-width, Gaussian NLL), `cat` (one categorical: bin 0 = no charge, 256 bins in y, HL-Gauss
  targets; requires `cond_charge=True`). Every head returns `(charge, presence_logit)` first.
- **Labels** (`DenoiseRows`, a transform after `CoeffTokenize` on a `("coeff", "cells")`
  sample): every charged cell, cells near charge and random cells under kept coefficients,
  coefficient-free cells inside covered tokens, whole charge-free windows far from and near
  activity (`near_windows`). Cells no token covers are not queried (predicted 0).

## 9. Training through pimm (`helix/integrations/pimm`)

helix uses pimm's training engine and launcher; pimm is not modified for helix beyond the
local commits on the site's pinned checkout (`docs/STATUS.md`). A config's
`custom_imports = dict(imports=["helix.integrations.pimm"])` registers:

| registry | names |
|---|---|
| TRANSFORMS | `CoeffTokenize`, `DenoiseRows`, `CoeffCollect` |
| DATASETS | `CoeffTPCDataset` |
| MODELS | `Coeff-FM` (with `init=<run>`: a warm start from that run's `model/last`), `Coeff-Denoise` |
| TRAINERS | `FMTrainer`: the one-event-per-rank and `required_world_size` checks, muP param groups, peak LR x sqrt(world / `lr_ref_world`), `"auto"` warmup and save cadence from the step count, the denoiser's tokenizer from its FM run |
| SCHEDULERS | `WSDStableLR`, `WSDCooldownLR`, `WarmupCosineLR` |
| HOOKS | `HelixProvenance` (provenance.json; refuses a resume on another corpus), `FiniteGuard`, `StepWatchdog` |

`StepWatchdog` (300 s, `faulthandler`) makes a stuck rank dump its stacks and exit before
NCCL's 600 s timeout; `configs/launch/_common.yaml` turns on the NCCL flight recorder and puts
helix and pimm on `PYTHONPATH` on every link. Checkpoints are pimm's standard (DCP), written
by `CheckpointSaverIteration` to `model/last` only. There is no in-training evaluation, no
`model_best`, no EMA.

Configs (`configs/pimm/`) state intent and hold one rank's batch; `scripts/submit.sh` appends
the batch for nodes x GPUs. `fm.py` is the stable phase; `fm_cooldown.py` the WSD cooldown
(warm start `model.init=$HELIX_EXP/fm`, first `COOLDOWN_RUNS` runs, one epoch); `fm_smoke.py`
a wiring check; `denoise.py` the denoiser on `model.fm`; `ft_eval.py` the evaluation protocol.

## 10. Evaluation (`helix/eval`)

**The primary evaluation of a pretrained encoder is a fine-tune**, `configs/pimm/ft_eval.py`:
the denoiser (point head, `cond_charge`, no linear amplitude) on a fixed 1,024-event labelled
subset for 16 epochs on exactly 4 ranks (4,096 steps; `required_world_size`), then
`python -m helix.eval <run>`, which scores the run's `model/last` on every probe-split event of
its corpus (`helix.eval.denoise`):

- **Windows** (`helix.eval.events`), 32 wires x 128 ticks on the fine grid: `iso` windows hold
  one compact isolated particle, `clear` when no other charge is in the +-16 w / +-128 t padding,
  `crowded` when some is but none within a cell. Backgrounds: `bg` (FAR: no charge in the
  padding) and `near` (no charge within a cell, some in the padding), both placed uniformly in
  area and kept only with >= 1 kept coefficient inside.
- **The key:** clear iso windows detected above each view's 99th percentile of its far windows,
  and crowded ones above its near windows' 99th percentile; by deposited energy, per view, with
  an event bootstrap. A window's score is its largest presence log-odds; the fit-free
  `amplitude` (largest kept |coefficient| / sigma) is scored on the same windows. The 1% is of
  noisy windows, not of plane area. A cell no token covers scores as a miss.
- **Charge** on every charged cell: median and hw68 of log(pred / true) per truth bin,
  linearity, per-event closure, presence calibration (ECE, Brier, log-loss).
- The JSON records the run's provenance, the corpus and its `basis_digest`, the FM run and
  `truth.q_cut`; per-window scores go to `<out>.windows.npz`.
- **`compare`** (`python -m helix.eval compare --a ... --b ...`): paired A - B of every window
  number over shared event resamples; several files per side are training seeds, paired seed by
  seed; `views` shows whether a pooled difference is carried by one view. It refuses dumps whose
  windows differ.
- **`uprobe`** (`--uprobe FM_RUN`, `helix.eval.probe`), the one frozen-feature key kept: an
  MLP on the frozen encoder's features (`FMModel.encode`) predicts the along-wire coordinate,
  the 3D information one plane lacks. It fits on the first two thirds of the probe events
  (<= 1M rows) and reports within-(event, plane) R^2 against geometry and raw-input references.
- A run trained at another site is found through `verify.resolve_corpus` and
  `helix.eval.resolve_exp` (or `--corpus` / `--fm`).

## 11. Sites and paths

`helix/paths.py` resolves every external root from `HELIX_<NAME>` in the environment, else
the site profile `helix/sites/<site>.yaml` (`HELIX_SITE`, or detected: `/sdf` -> s3df,
`$NERSC_HOST` -> nersc). Roots: `HELIX_CORPUS_ROOT`, `HELIX_CORPUS` (one run dir; its parent is
the corpus), `HELIX_SIM_ROOT`, `HELIX_ARCHIVE` (bin tables), `HELIX_EXP` (runs),
`HELIX_PIMM_ROOT`, `HELIX_JAXTPC_ROOT`. A root a site lacks is `null`, and asking for it raises
`SiteError`. The profile also holds the container (runtime, image, in-image python, binds,
NERSC's interactive runtime) and scheduler facts per job kind (`train`, `eval`, `corpus`;
`train.pimm_rev` pins pimm).

`scripts/helix_env.sh` (sourced) exports them as `HELIX_*`, `HELIX_CONTAINER_*` and
`HELIX_SLURM_<KIND>_*`. `scripts/helix_run.sh CMD` runs CMD in the site's container (apptainer,
shifter or podman-hpc) with `PYTHONNOUSERSITE=1`, `PYTHONPATH=pimm:helix`, both checkouts bound
and the site's `env:`; podman-hpc on a TTY at NERSC, `HELIX_INTERACTIVE=0` forces the batch
runtime. `container/install.sh` is the one image recipe (`helix-train.def`, `Dockerfile`,
`Dockerfile.nersc`); it installs dependencies only, helix and pimm arrive by bind mount.

| | S3DF (`s3df.yaml`) | NERSC (`nersc.yaml`) |
|---|---|---|
| container | apptainer `helix-train.sif` | shifter `helix-nersc:jax-cuda12` (stale tag); podman-hpc interactive |
| train | `mli:cider-ml`, ampere, QOS normal; pimm not pinned | `m5238_g`, `-C gpu`, QOS preempt (2 h `time_min`); pimm pinned |
| eval | ampere, 1 GPU, 32 cores, 64 GB | QOS shared, 2 GPUs' share (64 cores, 112 GB) |
| corpus | `mli:default`, turing, preemptable, `--requeue` | `m5238_g`, `-C gpu`, QOS shared |

## 12. Optical

`helix.optical` compresses SBND PMT and X-ARAPUCA waveforms and goop's simulated PMTs with one
codec (`codec.py`): per channel the noise and pedestal from light-free samples, the light as a
sparse non-negative deconvolution with the measured single-photon response
(`calibrate.fit_response`), and storage as the integer Haar transform of the raw ADC, keeping a
difference where the light reaches one noise sigma or the data depart from the light model by
5 sigma; decoding is the exact inverse transform. `python -m helix.build sbnd` writes optical
shards (`helix-optical-2`, `helix/optical/shard.py`, joined on identity) beside the TPC ones
(`--no-optical` skips them); `goop` is optical only and
`corpus.sh`, which calibrates a TPC `norm_sigma`, does not serve it. `helix.optical.tokens`
makes 128-slot tokens, but nothing feeds them to the model. The validation numbers are in
`docs/ARCHITECTURE.md` §5 at git tag `eval-probes-final`.
