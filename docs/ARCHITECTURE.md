# Architecture

The modules, how data moves through them, and the on-disk formats. Why each piece
is shaped this way is in `docs/DECISIONS.md`; how to run it is in `docs/RUNBOOK.md`.

---

## 1. Data flow

```
  JAXTPC wire campaign         SBND extracted files            goop light files
  (HELIX_SIM_ROOT/j00X000)     (sbnd_data/extracted/<run>)     (optical only)
        |                              |                              |
  scripts/build_sim.py          scripts/build_sbnd.py           scripts/build_goop.py
  (noise model + digitize)      (TPC readout + PMT/X-ARAPUCA)
        \______________________________|______________________________/
                                       |
                     helix.build: event loop, TPCChain, OpticalChain, ShardWriter
                     scripts/corpus.sh: plan -> calibrate -> build, as Slurm arrays
                                       v
  corpus root  <run>/<dataset>_coeff_NNNN.h5        noisy coefficients (helix-coeff-2)
               <run>/<dataset>_coeff_clean_NNNN.h5  clean values, sim, --keep-clean only
               <run>/<dataset>_optical_NNNN.h5      optical streams (helix-optical-2)
               <run>/<dataset>_cells_NNNN.h5        sim truth (helix-cells-2), scripts/build_truth.py
               _calib/  CHUNKS.txt RUNS.txt norm_sigma_global.npy
                                       |
                     helix.data: CoeffTPCReader -> CoeffTPCDataset (+ AddNoise / Digitize)
                     helix.model.tokenize.CoeffTokenize -> CoeffCollect
                                       v
                     pimm.train + helix.integrations.pimm (FMTrainer, hooks, evaluator)
                                       |
             <save_path>/ (pimm's run dir) -> helix.model.checkpoint.load_run
                                       v
                     helix.eval (python -m helix.eval; truth: scripts/build_truth.py)
```

---

## 2. Packages and the boundary

| package | may import pimm-data / pimm | what it is |
|---|---|---|
| `helix.core` | no | wavelet transform, robust statistics, hysteresis, coeff shard IO, provenance, training cadence |
| `helix.tpc` | no | TPC chain, coherent gate, thresholding, noise model, readouts, readers |
| `helix.optical` | no | optical codec, calibration, optical shards, tokens |
| `helix.model` | no (`tokenize` defers one import) | the FM, tokenizer, masking, loss, muP, reading a run |
| `helix.eval` | no | evaluation: recon, floor, uprobe, denoise |
| `helix.build` | no (sources may) | the build pipeline |
| `helix.data` | pimm-data | corpus reader / dataset / verifier, registered transforms, bins, identity guard |
| `helix.integrations.pimm` | pimm | registry adapters into pimm's trainer |

`helix.core` and `helix.tpc` never import `pimm_data`: the DSP must install with the
base dependencies (numpy, h5py, PyWavelets, scipy, PyYAML) while pimm-data requires
torch and hdf5plugin unconditionally. `tests/test_boundary.py` enforces it with
import probes in clean subprocesses, a source scan, a per-submodule sweep, and a
check that `helix.data.bins` / `helix.data.identity` stay light (`helix/data/__init__.py`
resolves the reader and dataset lazily so they do). torch is optional at `import helix`
time; the DSP modules import it when they run.

**Lockstep with pimm-data.** helix registers the forward-model transforms `AddNoise`
and `Digitize`; pimm-data's registry raises `KeyError` on a duplicate, so a pimm-data
from before the move cannot run with this helix. The pimm-data revision is named once,
in `pyproject.toml` (`[tool.uv.sources]`); `container/install.sh` reads it from there and
`tests/test_lockstep_pin.py` checks that no other build file names one.

---

## 3. The build pipeline (`helix/build.py`)

`helix.build` holds no source. A source is a class in its own script that only reads;
the script ends with `helix.build.main(Source)`, which gives it the command line
documented in `helix/build.py`'s docstring. A source provides `Source(files, device,
optical=True)`, `len()`, `systems` (optical systems; empty for TPC only), `dataset` (the
shards' dataset name), `pattern` (its input files in a run directory),
`events(indices, workers)` (per event: `ident`, the TPC input as `readout` or `planes`,
optional `clean`, `optical`, `truth`), `shard_attrs()`, and for a TPC `chain_config()` /
`bind()`, for optical `optical_chain()`; optionally `file_plan(path, events_per_shard)`.

| script | `dataset` | input | TPC | optical | clean values (`--keep-clean`) |
|---|---|---|---|---|---|
| `scripts/build_sim.py` | `sim_wire` | `<task>/sensor/*_sensor_*.h5` (one run per campaign group) | planes, noised and digitized on the device (`helix.tpc.noise`, seeded per event) | none | yes |
| `scripts/build_sbnd.py` | `sbnd` | `*.h5` (extracted SBND) | readout (adc, pedestal, present) | `pmt`, `xarapuca` | no |
| `scripts/build_goop.py` | `goop` | `*.h5` (goop light) | none | `goop` (+ `truth/pe_counts`) | no |

Shared after the read: per `--events-per-shard` events, shard k (`--shard-start` + 0,
1, ...) is written as `PREFIX_coeff_k.h5`, `PREFIX_optical_k.h5` and, with `--keep-clean`
when the source has noise-free planes, `PREFIX_coeff_clean_k.h5` (k as 4 digits). Nothing
trains on the clean shards; they exist to inspect the DSP. Each file is written as `.part`, audited
(`helix.core.coeff_io.audit_shard` for coeff shards) and renamed whole, so a file with its
final name is complete; `--skip-complete` makes a task whose shards all exist do nothing.
`norm_sigma` is `--norm-sigma TABLE` (one table for the corpus) or else each shard's own;
`--calibrate TABLE` writes the table of the selected events and no shards.

`plan()` (`build_X.py plan OUT RUN_DIR...`) lists a corpus's tasks, one line each,
`PREFIX FIRST COUNT EVENTS_PER_SHARD SHARD_START INPUT...`: one task per input file when
the source defines `file_plan` (sim: shard k = campaign events k*eps .. (k+1)*eps-1), else every
`--events-per-task` events of a run. `scripts/corpus.sh` writes the plan to
`_calib/CHUNKS.txt`, calibrates one `norm_sigma` table per run and their mean
(`_calib/norm_sigma_global.npy`), then submits arrays over the plan on the site's
corpus scheduler. One GPU type per corpus: the site pins the corpus partition.

---

## 4. The TPC chain (`helix/tpc`)

`helix.tpc.chain.TPCChain` is one chain for simulation and data, torch on one device:

1. pedestal (from the input, or a clipped mean per channel);
2. channel status: dead, noisy or baseline-step rows are excluded (zeroed, flagged invalid);
3. DWT: coif3, level 4, periodization, time padded to a multiple of 2^level (`helix.core.wavelet`);
4. coherent removal (`helix.tpc.coherent_gate`) on the readout's channel groups;
5. threshold (`helix.tpc.threshold_rows`): sigma per plane and band on live rows,
   hysteresis seeds at the universal threshold (`kappa`), grown through `floor` sigma; the
   finest band (D1) is dropped; stored values rounded to a power-of-two step <= sigma / q
   (`store.step_input` 16, `store.step_target` 32; `helix.tpc.chain_coeff.quantize`).

`process_event(adc, ped, present)` takes a readout (SBND); `process_planes({gid: image})`
takes plane images (simulation). `helix.tpc.chain_coeff` turns the result into a
`CoeffEvent` and evaluates the clean target at the noisy support (`clean_coeff_event`).
Configuration is a nested dict merged and validated by `make_config`; `SBND_DEFAULT` and
`SIM_BASE` (via `sim_config(num_wires)`) are the two bases.

**The coherent gate** works on integer group labels from the readout
(`helix.tpc.readout.ReadoutMap.labels`; -1 = excluded row). Per band and pass: the common
mode per (group, position) is a clipped mean over non-signal rows; its scale `sigc` is a
MAD over the band (`sigc_scope: global`) or per group (`group`); the mode is subtracted
where `|M| < kgate * sigc`, and with `tau` also where it is larger unless more than `tau`
of the group's rows were flagged as signal in pass 1; signal is re-detected between passes.

    coherent: {groups, cross_plane, kgate, ksig, npass, tau, sigc_scope}

| | groups | cross_plane | kgate / ksig / npass | tau | sigc_scope |
|---|---|---|---|---|---|
| SBND (`SBND_DEFAULT`) | `femb`: front-end boards, 128 channels across planes | true | 3 / 3 / 2 | none | group |
| simulation (`SIM_BASE`) | `block`: the noise model's 64-wire blocks (`readout.group_size`) | false | 3 / 3 / 2 | 0.05 | global |

`groups: null` disables removal. Float32 reductions depend on summation order and a
group's decision covers all its rows, so one GPU generation builds a corpus.

**The noise model** (`helix.tpc.noise`, torch): incoherent ENC
`sqrt(white^2 + (y + z L)^2)` with the measured MicroBooNE series spectrum by default
(`DEFAULT_ENC = (0.90, 0.79, 0.22)`), plus one coherent waveform per group of 64
contiguous wires (adjacent groups anti-correlated by `beta`, renormalised to
`rms_adc` 2.5), then `digitize`. The same functions build the simulation corpus
(`scripts/build_sim.py`) and add fresh noise at load time (`helix.data.transforms.AddNoise`,
`Digitize`; recipe `configs/jaxtpc/sensor_dense_gpu.py`). Realisations come from
per-event torch generators: reproducible on one device type, statistically (not
bit-) equal to JAXTPC's numpy; `tests/_noise_reference.py` is the numpy port the tests
compare against.

Readers: `helix.tpc.io` reads a JAXTPC wire production campaign (format 2,
`doraemon_2026_1M_lartpc`; `sensor_layout`, `event_ids`, `entry`, `read_event_coo`,
`task_files`); `helix.tpc.io_sbnd.SBNDEvents` decodes extracted SBND files in worker processes.

**The simulation source.** The campaign's wire stage (`HELIX_SIM_ROOT` =
`<campaign>/data/jaxtpc_wire`) is `<group>/<task>/{sensor,step,hits}/sim_wire_{kind}_<task>.h5`:
5 groups `j00X000` of 200 tasks, each task 5 generator jobs x 200 = 1000 events in
`entry_NNNNNN` groups. Every file carries `/dprod/events` (`event_id`, `job_id`, `job_event`)
and each entry the same as attrs; `event_id` is unique in the campaign (0..999999) and is
what helix calls the event. A corpus run is a group (`<root>/j00X000/`), a build task is a
task's sensor file, and shard k holds campaign events `k*eps .. (k+1)*eps-1` (eps 200: shard
= generator job id). `/ident` is (run = group, source_file = the sensor file's name, event =
`event_id`); the noise seed and the identity-hash split are functions of it, so neither
depends on helix's sharding. Format-1 productions (`event_NNN`, e.g. `wire_test_00_00_02`,
which built r2) are not read.

---

## 5. The optical codec (`helix/optical`)

One codec for every light system (`helix.optical.codec.OpticalChain`, `OpticalSpec`),
described by the system's measured single-photon response: deconvolution to photon
amplitudes, then the integer Haar transform of the raw samples with coarse sums always
kept and differences kept where the light (or the departure from it) reaches threshold.
Decoding (`unhaar`) is the exact inverse of what was stored. Method, settings, response
fitting (`helix.optical.calibrate`) and validation (`helix.optical.validate`): `docs/OPTICAL.md`.

---

## 6. Shard formats

### Coefficient shards: `helix-coeff-2` (`helix/core/coeff_io.py`)

`write_coeff_shard` / `read_coeff_event` / `audit_shard` are the schema of record. One
modality per file, flat columnar (shard-wide arrays plus an event offset index, no
per-event groups):

```
/config   attrs: format = "helix-coeff-2", n_events, dataset_name, file_index,
                 global_event_offset, readout_type = "wire",
                 wavelet, dwt_level, dwt_mode, n_ticks_raw, pad, sigma_norm,
                 basis_digest, removal_json, threshold_json, has_coords,
                 noise_json, provenance_json
          datasets: band_lengths (n_bands,) int32   padded band lengths
                    gids (G,) int32                 the plane set; row order of every per-plane table
                    n_wires (G,) int32
                    norm_sigma (G, n_bands) float32 the corpus normalisation table
/coord    band uint8, plane_gid int32, wire int32, tau int32   (M,)   absent when has_coords = false
          event_offset int64 (n_events + 1,)      event i = rows off[i]:off[i+1]
          coord_digest uint64 (n_events,)          blake2b of the event's four coordinate columns
          sigma_threshold float32 (n_events, G, n_bands)   the event's threshold sigma
          wire_valid uint8 (n_events, G, ceil(max n_wires / 8))   packed; absent = all valid
/value    float32 (M,)                             raw coefficients (quantized to value_step)
/ident    run str, source_file str, event int64 (n_events,)  [noise_seed int64 for simulation]
```

* **format** is written by every writer and required by every reader
  (`require_format` in `coeff_io`, `CoeffTPCReader`, `CoeffTPCDataset`, `audit_shard`).
  A shard without it is refused with a pointer to `scripts/migrate_format.py`.
* **basis_digest** hashes the basis (wavelet, level, mode, n_ticks_raw, pad,
  band_lengths, removal, threshold, sigma_norm) and is checked against those attributes
  on read; `band_lengths` is validated against the basis.
* **Clean values** (`*_coeff_clean_*`, `--keep-clean` only; no training reads them):
  evaluated at the noisy support, written values only (`has_coords = false`). It joins the noisy shard by `/ident` and row for row within
  an event; `coord_digest` in both shards must agree or the read raises.
  Consequence: the pair measures fidelity on what was kept, never recall of what was
  dropped.
* **Values are raw.** Normalisation `asinh(value / norm_sigma[row_of(gid), band])` is
  applied at tokenize; rows follow the POSITION of a gid in `gids`, not its value.
  `sigma_threshold` (per event, defines the kept set) and `norm_sigma` (per corpus,
  normalises) are different tables.
* **provenance_json** records the code (`version`, `git`, `git_dirty`) and, for
  simulation, the geometry and spectrum files with their sha256 and the seed formula;
  `noise_json` records how the input was noised.
* Compression is gzip with the byte shuffle; the layout, not the filter, is the contract.

### Optical shards: `helix-optical-2` (`helix/optical/shard.py`)

`/config` (format, modality = "optical", n_events), `/ident` (run, source_file, event,
subrun, time), and per system `/optical/<system>`: the codec spec, per-waveform channel,
time, length and counts, and exact integer streams (region runs, delta-coded coarse sums,
keep bits, stored differences), plus per-event channel noise. Joins its TPC shard on
(run, source_file, event). `read_optical` / `read_event` read it,
`helix.optical.tokens.shard_tokens` tokenizes it.

### Truth shards: `helix-cells-2` (`helix/data/cells.py`)

Simulation truth on the 2-wire x 16-tick cell grid, one `{name}_cells_<k>.h5` per coefficient
shard (same events, same order, `/ident` copied), written by `scripts/build_truth.py` from the
simulator's `hits` and `step` files, separately from the corpus build. Four per-event CSR
tables: `/cells` (per cell: charge, the dominant hits group's share and (y, z)),
`/cell_track` (the cells' charge split by track), `/plane_track` (per plane and track: the
charge-weighted wire / tick centroid), `/tracks` (per track: deposited energy). Every
evaluation window lies on cell edges, so the denoiser's label and the truth of every
`helix.eval` metric (floor, uprobe, denoise) come from this alone. ~197 MB per shard of 200 events.

### The corpus root

    <root>/<run>/<dataset>_{coeff,optical}_NNNN.h5   (+ coeff_clean with --keep-clean)
    <root>/<run>/<dataset>_cells_NNNN.h5   the truth, scripts/build_truth.py
    <root>/_calib/CHUNKS.txt        the plan (helix.build.plan)
    <root>/_calib/RUNS.txt          run directory names, one per line (training reads this)
    <root>/_calib/<run>.npy         per-run norm_sigma; norm_sigma_global.npy their mean

The split is computed, not stored (`holdout=` on the dataset: below).
`scripts/write_holdout.py` writes it out as `<run>/holdout.json` for inspection; nothing
reads that file.

`python -m helix.data.coeff_verify` checks a run directory (`--dataset-name`) for
cross-shard problems (pairs, gaps, duplicates, mixed code or basis), or a whole corpus
against `_calib/CHUNKS.txt` (every planned shard present with its planned events and
nothing else, optical shards holding their TPC shard's events, one global
`norm_sigma`), or compares shard files byte for byte (`compare A B ...`, provenance
attributes excepted).

---

## 7. Reading the corpus (`helix/data`)

* `CoeffTPCReader` (`coeff_reader.py`): one shard family (`modality` `coeff` or
  `coeff_clean`), standalone h5py on pimm-data's `ShardReaderBase`; shards of another
  format are recorded and refused.
* `CoeffTPCDataset` (`coeff_dataset.py`, pimm-data registry): `modalities` from `coeff`
  (always), `coeff_clean` and `cells`, joined by identity; the shard tables in
  `sample['coeff']['_meta']`; the identity-hash split (`holdout`, `split_role`:
  blake2b(run/source_file) + event, so independent of the build). Training reads
  `("coeff",)`, the denoiser `("coeff", "cells")`; `tests/test_noisy_training.py` keeps
  `coeff_clean` and `cells` out of pretraining.
* `cells.py`: the truth format and grid (above), `Q0`, and the cell-to-token join.
* `denoise.py`: `DenoiseRows`, the denoiser's label sampling on the `cells` truth.
* `transforms.py`: `AddNoise`, `Digitize` (load-time forward model, per event or per
  collated batch).
* `bins.py`: the categorical bin grid over NOISY coefficients (uniform in asinh on each
  side of zero, one bin over the sub-threshold gap); the production grid is named in
  `helix/data/data/reference_bins.json` (`coeff_bins_d1m_j000000_noisy`, from the
  campaign corpus's first run; its digests are recorded).
* `identity.py`: `corpus_runs()` reads `_calib/RUNS.txt`; `check_corpus_matches`
  refuses a checkpoint whose recorded `basis_digest` differs from the corpus being read.

---

## 8. The model (`helix/model`)

* `tokenize.CoeffTokenize`: coefficient rows -> cells of `pw x pt = 16 x 8` slots
  (`n_slot` 128) over the bands A4, D4, D3, D2, values normalised by `norm_sigma`. The
  target is the normalised input itself (training is noisy-only); it differs from the
  input only where `dead_frac` blanks it.
* `fm.FMModel` / `build_fm` (strict: an unknown key raises): encoder blocks, cross-attention
  decoder, a categorical head over the bin grid (`n_bins`), muP (`mup.py`), RoPE over time
  and wire (`layers.py`). Attention is grouped along four token orders and runs unpadded
  through flash varlen attention (`serial.py`), which needs sm80+ (A100); on CPU / fp32
  the same groups run one SDPA each (tests, CPU evaluation). `fused_qk` is the Triton
  QK-norm + RoPE kernel (`kernels.py`), with the unfused path taken off-GPU.
  Masking (`mask.py`): `random` cells by default, whole planes with probability
  `plane_frac`. `plane_id` serves as FiLM index, RoPE axis, masking group and plane
  identity at once.
* The FM has no event separation: one event per rank, and `FMTrainer` refuses any other
  per-GPU batch.
* `denoise.DenoiseModel` / `build_denoise`: the supervised denoiser, an FM run plus a task
  head. The encoder takes the FM run's architecture, tokenizer and (`pretrained`) weights,
  optionally frozen (`freeze_encoder`); `CellDecoder` cross-attends from each 2 x 16 fine cell
  to the 3 x 3 token neighbourhood per band and returns charge `log1p(q/Q0)` and a presence
  logit; `cond_charge` regresses charge on charged cells only (calorimetry) instead of on every
  queried cell (detection, maps). `param_groups` = the encoder's muP groups + the head at
  `lr_head`. Its labels are `helix.data.denoise.DenoiseRows`, a transform after
  `CoeffTokenize` on a `("coeff", "cells")` sample.
* `checkpoint.py` (`load_run`) is the one reader of a trained run, FM or denoiser: pimm's run
  directory (`resolved_config.json`, `model/model_ema.pth`, `model/last` or
  `model/model_best.pth`, `provenance.json`) -> the model (an FM with its bins), the train
  split's `PatchConfig` (a denoiser's is its FM run's), provenance. Nothing is exported.

---

## 9. pimm integration (`helix/integrations/pimm`)

helix uses pimm's training engine only (`python -m pimm.train`), not its launch layer.
A config pulls in `custom_imports = dict(imports=["helix.integrations.pimm"])`, which
registers:

| registry | names |
|---|---|
| TRANSFORMS | `CoeffTokenize`, `DenoiseRows`, `CoeffCollect` |
| DATASETS | `CoeffTPCDataset` (wraps `helix.data.CoeffTPCDataset`, runs pimm's transforms) |
| MODELS | `Coeff-FM`, `Coeff-Denoise` |
| TRAINERS | `FMTrainer`: the model's param groups (muP; the denoiser's head), the batch check, `"auto"` cadence, a `CoeffTokenize(cfg="model")` filled from the model's FM run |
| SCHEDULERS | `WSDStableLR`, `WSDCooldownLR`, `WarmupCosineLR` (the denoiser) |
| HOOKS | `HelixPathBootstrap`, `WeightEMA`, `FiniteGuard`, `CoeffFMEvaluator`, `DenoiseEvaluator` |

**Cadence is derived at run time:** a config writes `warmup="auto"`, `save_freq="auto"`, `every_n_steps="auto"`;
`FMTrainer._resolve_cadence` fills them from pimm's `iters_per_epoch * epoch` through
`helix.core.cadence` (`warmup_steps`, `eval_every`, `save_every`).

`configs/pimm/fm.py` is the whole run: corpus = `HELIX_CORPUS`'s parent with its
`_calib/RUNS.txt`, the reference bin grid, the `nz_pw16` recipe (d 768, 12 + 4 blocks, 12
heads, QK-norm, fused QK-norm + RoPE, K = 128; fused AdamW at LR 4.4e-3) + `WSDStableLR`,
batch = world size. Variants override only what
differs: `fm_cooldown.py` (warm start from `$HELIX_EXP/fm/model/last`, first 3 runs, one
epoch, `WSDCooldownLR`), `fm_smoke.py` (a
small model, 32 events, one GPU). Anything else is a `key=value` override.

`configs/pimm/denoise.py` (`_base_` fm.py) trains the denoiser through the same trainer and
launcher: `Coeff-Denoise` on the FM run `model.fm` (default `$HELIX_EXP/fm`), the M6d recipe
of docs/archive/DENOISE.md as defaults (`model.cond_charge=True` is M6dc), the FM's identity-hash
holdout, `WarmupCosineLR`, `DenoiseEvaluator` (the training loss on the val split, fixed
draws) and `model/model_best.pth` by validation loss. `data.train.n_events` trains on a fixed,
nested subset of the train split (label efficiency).

---

## 10. Sites

`helix/paths.py` resolves every external root from the environment or the site profile
`helix/sites/<site>.yaml` (`HELIX_SITE`, or detected: `/sdf` for S3DF, `$NERSC_HOST` for
NERSC). A root the site does not declare is `null` and asking for it raises. The profile
also holds the container (`runtime`, `image`, `python`, `binds`, NERSC's interactive
runtime) and scheduler facts per job kind (`train`, `corpus`).

* `scripts/helix_env.sh` (sourced) exports the roots as `HELIX_*`, the container as
  `HELIX_CONTAINER_*` and the scheduler as `HELIX_SLURM_<KIND>_*`, using a host Python
  with PyYAML (`HELIX_PYTHON`, else `$SCRATCH/helix-host-venv` from
  `scripts/make_host_env.sh`, else a system python).
* `scripts/helix_run.sh CMD...` runs a command in the site's container (apptainer,
  shifter, podman-hpc or none), rewriting a leading `python` to the in-image interpreter,
  binding both checkouts, forwarding `HELIX_*`, the site's env and `HELIX_FORWARD_ENV`,
  with `PYTHONNOUSERSITE=1` and `PYTHONPATH=pimm:helix`. podman-hpc (NERSC, interactive)
  is chosen on a TTY; `HELIX_INTERACTIVE=0` forces the batch runtime.
* `scripts/corpus.sh` and `scripts/train.sh` submit with `sbatch` from the host, with
  the site's account / partition / QOS / constraint / extra flags.

| | S3DF (`s3df.yaml`) | NERSC (`nersc.yaml`) |
|---|---|---|
| container | apptainer, `/sdf/data/neutrino/omara/images/helix-train.sif` | shifter `ghcr.io/omaralterkait/helix-nersc:jax-cuda12`; podman-hpc interactive |
| train | `mli:cider-ml`, ampere, QOS normal | `m5238_g`, constraint gpu, QOS preempt |
| corpus | `mli:default`, turing, QOS preemptable, `--requeue` | `m5238_g`, constraint gpu, QOS shared |

The container: `container/install.sh` is the one recipe, run by `container/helix-train.def`
(apptainer), `container/Dockerfile` (CI, `ghcr.io/omaralterkait/helix`) and
`container/Dockerfile.nersc` (Docker v2 manifest for shifter / podman-hpc). It installs
dependencies only, with pimm-data at the revision `pyproject.toml` pins; helix and pimm
arrive by bind mount and `PYTHONPATH`, so editing either needs no rebuild. Rebuild the
image when the pimm-data pin moves.
