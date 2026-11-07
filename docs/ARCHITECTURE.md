# Architecture

The modules, how data moves through them, and the on-disk formats. Why each piece
is shaped this way is in `docs/DECISIONS.md`; how to run it is in `docs/RUNBOOK.md`.

---

## 1. Data flow

```
  JAXTPC wire campaign         SBND extracted files            goop light files
  (HELIX_SIM_ROOT/j00X000)     (sbnd_data/extracted/<run>)     (optical only)
        |                              |                              |
  helix.build sim               helix.build sbnd                helix.build goop
  (noise model + digitize)      (TPC readout + PMT/X-ARAPUCA)
        \______________________________|______________________________/
                                       |
                     helix.build: event loop, TPCChain, OpticalChain, ShardWriter
                     scripts/corpus.sh: plan -> calibrate -> build, as Slurm arrays
                                       v
  corpus root  <run>/<dataset>_coeff_NNNN.h5        noisy coefficients (helix-coeff-2)
               <run>/<dataset>_coeff_clean_NNNN.h5  clean values, sim, --keep-clean only
               <run>/<dataset>_optical_NNNN.h5      optical streams (helix-optical-2)
               <run>/<dataset>_cells_NNNN.h5        sim truth (helix-cells-3), scripts/build_truth.py
               _calib/  CHUNKS.txt RUNS.txt norm_sigma_global.npy
                                       |
                     helix.data: CoeffTPCReader -> CoeffTPCDataset
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
| `helix.core` | no | wavelet transform, robust statistics and hysteresis, coeff shard IO, the event and its provenance |
| `helix.tpc` | no | TPC chain and its CoeffEvents, coherent gate, thresholding, noise model, readouts and plane geometry, readers (and the SBND build source) |
| `helix.optical` | no | optical codec, calibration, optical shards, tokens, the goop build source |
| `helix.model` | no (`tokenize` defers one import) | the FM and its masking, blocks, loss, tokenizer, schedule (cadence, muP), the denoiser, reading a run |
| `helix.eval` | no | evaluation: `recon`, `events` (and windows), `probe` (floor, uprobe, labels), `denoise` |
| `helix.build` | no (`SimSource` defers pimm-data's `densify`) | the build pipeline, its CLI and the simulation source |
| `helix.data` | pimm-data (`dataset` only) | corpus reader and dataset (`dataset`), verifier and identity guard (`verify`), bins, `cells` truth, denoiser labels |
| `helix.integrations.pimm` | pimm | registry adapters into pimm's trainer |

`helix.core` and `helix.tpc` never import `pimm_data`: the DSP must install with the
base dependencies (numpy, h5py, PyWavelets, scipy, PyYAML) while pimm-data requires
torch and hdf5plugin unconditionally. `tests/test_boundary.py` enforces it with
import probes in clean subprocesses, a source scan, a per-submodule sweep, and a
check that `helix.data.bins` / `helix.data.verify` stay light (`helix/data/__init__.py`
resolves the reader and dataset lazily so they do). torch is optional at `import helix`
time; the DSP modules import it when they run.

**The pimm-data pin.** The forward model (noise, digitize) moved from pimm-data to helix
(`helix.tpc.noise`); a pimm-data from before the move still registers `AddNoise` and
`Digitize`, which the image checks are absent. The pimm-data revision is named once,
in `pyproject.toml` (`[tool.uv.sources]`); `container/install.sh` reads it from there and
`tests/test_lockstep_pin.py` checks that no other build file names one.

---

## 3. The build pipeline (`helix/build.py`)

A source is a class that only reads, next to its reader: `SimSource` in `helix/build.py` (it
densifies with pimm-data, which `helix.tpc` may not import), `SBNDSource` in
`helix/tpc/io_sbnd.py`, `GoopSource` in `helix/optical/goop.py`. `python -m helix.build SOURCE`
(`SOURCES` names them; `module:Class` names any other) gives the command line documented in
`helix/build.py`'s docstring, and `helix.build.main(Source)` the same for a class. A source provides `Source(files, device,
optical=True)`, `len()`, `systems` (optical systems; empty for TPC only), `dataset` (the
shards' dataset name), `pattern` (its input files in a run directory),
`events(indices, workers)` (per event: `ident`, the TPC input as `readout` or `planes`,
optional `clean`, `optical`, `truth`), `shard_attrs()`, and for a TPC `chain_config()` /
`bind()`, for optical `optical_chain()`; optionally `file_plan(path, events_per_shard)`.

| source | `dataset` | input | TPC | optical | clean values (`--keep-clean`) |
|---|---|---|---|---|---|
| `sim` (`SimSource`) | `sim_wire` | `<task>/sensor/*_sensor_*.h5` (one run per campaign group) | planes, noised and digitized on the device (`helix.tpc.noise`, seeded per event) | none | yes |
| `sbnd` (`SBNDSource`) | `sbnd` | `*.h5` (extracted SBND) | readout (adc, pedestal, present) | `pmt`, `xarapuca` | no |
| `goop` (`GoopSource`) | `goop` | `*.h5` (goop light) | none | `goop` (+ `truth/pe_counts`) | no |

Shared after the read: per `--events-per-shard` events, shard k (`--shard-start` + 0,
1, ...) is written as `PREFIX_coeff_k.h5`, `PREFIX_optical_k.h5` and, with `--keep-clean`
when the source has noise-free planes, `PREFIX_coeff_clean_k.h5` (k as 4 digits). Nothing
trains on the clean shards; they exist to inspect the DSP. Each file is written as `.part`, audited
(`helix.core.coeff_io.audit_shard` for coeff shards) and renamed whole, so a file with its
final name is complete; `--skip-complete` makes a task whose shards all exist do nothing.
`norm_sigma` is `--norm-sigma TABLE` (one table for the corpus) or else each shard's own;
`--calibrate TABLE` writes the table of the selected events and no shards.

`plan()` (`python -m helix.build SOURCE plan OUT RUN_DIR...`) lists a corpus's tasks, one line each,
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
   (`store.step_input` 16, `store.step_target` 32; `helix.tpc.chain.quantize`).

`process_event(adc, ped, present)` takes a readout (SBND); `process_planes({gid: image})`
takes plane images (simulation). `helix.tpc.chain.coeff_event` turns the result into a
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
(`python -m helix.build sim`). Realisations come from
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

`helix.optical` compresses optical waveforms — SBND PMT and X-ARAPUCA data, and goop's simulated
PMTs — with one method. A light system is described by its measured single-photon response;
everything else is derived from it.

### Method (`helix.optical.codec`)

Per event and system:

1. **Noise and pedestal per channel, from where there is no light.** The matched-filter
   statistic (the window minus a local baseline, correlated with the zero-mean unit-norm core of
   the response — its shortest span around the peak holding 98% of its energy) marks light at
   significance `k`; the noise of the statistic, of every Haar level, and the window's clipped
   mean are measured outside it (widened by the core span) and medianed over the channel's
   windows. A window median of integer codes is off by up to half a count, and light below the
   trigger lifts a lit window's own estimate — either becomes fake or lost light.
2. **The light.** Photon amplitudes `a >= 0` minimising `|y - h*a|^2 / 2 + lambda sum(a)` (FISTA,
   then refitted without the penalty on its support), `y` the window minus its channel's
   pedestal, `lambda = k x statistic noise x |h|` (a photon enters where its matched-filter
   significance exceeds `k`). The light `h*a` covers isolated photons, pile-up and slow dense
   light alike.
3. **Stored.** The integer Haar transform of the RAW ADC samples: coarse sums always; a difference
   where the light's difference reaches `q sigma_j` (`q = 1`: keep exactly where the signal is
   worth a noise sigma — the error-optimal rule applied to the estimate; nothing is chosen by its
   own noisy value), or where the data depart from the light model by `5 sigma_j` (a wrong
   response on big pulses, artefacts that are not light; a noise coefficient passes with
   probability ~6e-7).

Decoding never uses the light estimate: it is the exact inverse Haar transform of the stored raw
coarse sums and differences; a missing difference splits its block's sum evenly.

Settings: `OpticalSpec(response, polarity, k=3, level, q=1, residual=5)`. SBND: `level` 6 (PMT 2 ns,
X-ARAPUCA 16 ns ticks); goop: 7 (1 ns). Batching groups windows within 25% padding (goop: 0.67 s
per event on a 2080 Ti).

### Responses (`helix.optical.calibrate.fit_response`)

Measured the way the codec uses them: photons from the codec's own deconvolution at significance
5; the core (±40-80 ns) by regression on isolated single photons, each against its own local
level, with equal weight for every arrival time mod 8 samples (cancels noise locked to the
sample clock); the tail by least squares over all photons with the core fixed; the tail as an
exponential model if it describes it (goop: `-0.015 exp(-t / 632 ns)`), else smoothed in log time
(SBND PMT: ringing at 16-80 ns, 20-80 standard errors, and the AC undershoot to ~600 ns).
A stack of matched-filter maxima — the earlier estimate — selects noise peaks in quiet windows
and returns the noise's autocorrelation (digitizer clock lines). Recovery of a known response
injected on noise (rms error, units of the peak): from the true response goop core 0.006, tail
0.0004, SBND PMT 0.013 / 0.004; from a 30% wider start, one pass, goop 0.017 / 0.007, PMT
0.021 / 0.004 (the start's error enters through the photons it deconvolves; passes reduce it).
On data, late light correlated with each photon (argon's slow component) trades against a longer
tail: one pass from the previous response. The X-ARAPUCA response (16 ns, short self-triggered
windows, few isolated photons) is not recovered reliably and stays the mean isolated pulse.

### Validation (`helix.optical.validate`, at tag `archive-2026-11`)

Known light — single photons, and 100 / 500-photon flashes (30% 6 ns, 70% 1.5 us) — injected on
real light-free windows (SBND) or on goop's exact noise model; error = signal dropped + noise
kept, in a fixed reference sigma_j measured on the noise before injection; the oracle keeps
exactly the true differences above sigma_j.

Previous support rule vs this codec (SBND-data-helix `archive/scripts/rule_truth2.py`, region scoring):
PMT per photon 13.0 values at 25.0 sigma^2 vs 6.4 at 11.5 (oracle 6.2 / 9.1); per flash 698 at 1310
vs 484 at 827 (oracle 495 / 645); X-ARAPUCA per photon 27.5 at 58.6 vs 9.2 at 17.3. goop
(SBND-data-helix `archive/scripts/goop_compare.py`, exact truth): whole-chunk error 3.5k sigma^2 at ~1950 values vs >= 5.7k
for coif3 at any threshold; late single photons 450 vs >= 1230.

### Files

* Optical shards (`helix.optical.shard`): one modality per file next to the TPC coefficient
  shards, joined on `(run, source_file, event)`; exact integer streams; `read_optical`.
* Tokens (`helix.optical.tokens`): 128-sample tree blocks with a stored difference + coarse
  tokens (128 slots, like a TPC token); `shard_tokens` decodes and tokenizes an event's windows
  together (`shard.read_event`; one SBND event on a CPU core: PMT ~0.3 s, X-ARAPUCA ~0.08 s).
  Not yet fed to the model.

### Running

Optical shards are written by the build pipeline (`helix/build.py`), next to the TPC shards:

* SBND: `python -m helix.build sbnd --input RUN/*.h5 --output OUT/RUN/sbnd --device cuda` —
  TPC `sbnd_coeff_NNNN.h5` + optical `sbnd_optical_NNNN.h5` (`--no-optical`: TPC only).
* goop: `python -m helix.build goop --input light_output.h5 --output OUT/RUN/goop --device cuda` —
  `goop_optical_NNNN.h5`, with goop's photon counts per PMT in `truth/pe_counts`.
* A whole SBND corpus: `scripts/corpus.sh all sbnd OUT RUN_DIR...` (`docs/RUNBOOK.md` §3).
  `corpus.sh` calibrates a TPC `norm_sigma` before building, so it does not serve an
  optical-only source (goop) as is.

Per-system overrides of the codec settings: `--optical-config` (json / yaml).

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
  A shard without it is refused (the format-1 corpora were migrated in place by
  `scripts/migrate_format.py`, at tag `archive-2026-11`).
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

### Truth shards: `helix-cells-3` (`helix/data/cells.py`)

Simulation truth on the 2-wire x 16-tick cell grid, one `{name}_cells_<k>.h5` per coefficient
shard (same events, same order, `/ident` copied), written by `scripts/build_truth.py` from the
simulator's `hits`, `step` and pysupera `labl` files, separately from the corpus build. Four per-event CSR
tables: `/cells` (per cell: charge, the dominant hits group's share and (y, z)),
`/cell_track` (the cells' charge split by track), `/plane_track` (per plane and track: the
charge-weighted wire / tick centroid), `/tracks` (per track: deposited energy and its
motif / pid class). Every evaluation window lies on cell edges, so the denoiser's label and
the truth of every `helix.eval` metric (floor, uprobe, labels, denoise) come from this alone. ~197 MB per shard of 200 events.

### The corpus root

    <root>/<run>/<dataset>_{coeff,optical}_NNNN.h5   (+ coeff_clean with --keep-clean)
    <root>/<run>/<dataset>_cells_NNNN.h5   the truth, scripts/build_truth.py
    <root>/_calib/CHUNKS.txt        the plan (helix.build.plan)
    <root>/_calib/RUNS.txt          run directory names, one per line (training reads this)
    <root>/_calib/<run>.npy         per-run norm_sigma; norm_sigma_global.npy their mean

The split is computed, not stored (`holdout=` on the dataset: below).

`python -m helix.data.verify` checks a run directory (`--dataset-name`) for
cross-shard problems (pairs, gaps, duplicates, mixed code or basis), or a whole corpus
against `_calib/CHUNKS.txt` (every planned shard present with its planned events and
nothing else, optical shards holding their TPC shard's events, one global
`norm_sigma`), or compares shard files byte for byte (`compare A B ...`, provenance
attributes excepted).

---

## 7. Reading the corpus (`helix/data`)

* `CoeffTPCReader` (`dataset.py`): one shard family (`modality` `coeff` or
  `coeff_clean`), standalone h5py on pimm-data's `ShardReaderBase`; shards of another
  format are recorded and refused.
* `CoeffTPCDataset` (`dataset.py`, pimm-data registry): `modalities` from `coeff`
  (always), `coeff_clean` and `cells`, joined by identity; the shard tables in
  `sample['coeff']['_meta']`; the identity-hash split (`holdout`, `split_role`:
  blake2b(run/source_file) + event, so independent of the build). Training reads
  `("coeff",)`, the denoiser `("coeff", "cells")`; `tests/test_boundary.py` keeps
  `coeff_clean` and `cells` out of pretraining.
* `cells.py`: the truth format and grid (above), `Q0`, and the cell-to-token join.
* `denoise.py`: `DenoiseRows`, the denoiser's label sampling on the `cells` truth.
* `bins.py`: the categorical bin grid over NOISY coefficients (uniform in asinh on each
  side of zero, one bin over the sub-threshold gap); the production grid is named in
  `helix/data/data/reference_bins.json` (`coeff_bins_d1m_j000000_noisy`, from the
  campaign corpus's first run; its digests are recorded).
* `verify.py`: the corpus verifier (`python -m helix.data.verify`, below), `corpus_runs()`
  (reads `_calib/RUNS.txt`) and `check_corpus_matches`, which
  refuses a checkpoint whose recorded `basis_digest` differs from the corpus being read.

---

## 8. The model (`helix/model`)

* `tokenize.CoeffTokenize`: coefficient rows -> cells of `pw x pt = 16 x 8` slots
  (`n_slot` 128) over the bands A4, D4, D3, D2, values normalised by `norm_sigma`. The
  target is the normalised input itself (training is noisy-only); it differs from the
  input only where `dead_frac` blanks it.
* `fm.FMModel` / `build_fm` (strict: an unknown key raises): encoder blocks, cross-attention
  decoder, a categorical head over the bin grid (`n_bins`; `loss.py`), muP (`schedule.py`), RoPE over time
  and wire (`blocks.py`). Attention is grouped along four token orders and runs unpadded
  through flash varlen attention (also `blocks.py`), which needs sm80+ (A100); on CPU / fp32
  the same groups run one SDPA each (tests, CPU evaluation). `fused_qk` is the Triton
  QK-norm + RoPE kernel (`kernels.py`), with the unfused path taken off-GPU.
  Masking (`fm.make_mask`): `random` cells by default, whole planes with probability
  `plane_frac`. `plane_id` serves as FiLM index, RoPE axis, masking group and plane
  identity at once.
* The FM has no event separation: one event per rank, and `FMTrainer` refuses any other
  per-GPU batch (the design for separation and its acceptance criterion: MULTI_EVENT_BATCHING
  at git tag `archive-2026-11`).
* `denoise.DenoiseModel` / `build_denoise`: the supervised denoiser, an FM run plus a task
  head. The encoder takes the FM run's architecture, tokenizer and (`pretrained`) weights,
  optionally frozen (`freeze_encoder`); `CellDecoder` cross-attends from each 2 x 16 fine cell
  to the 3 x 3 token neighbourhood per band and returns charge `log1p(q/Q0)` and a presence
  logit; `cond_charge` regresses charge on charged cells only (calorimetry) instead of on every
  queried cell (detection, maps). `dist` is the charge head's form: `point` (squared error,
  the default), `gauss` (+ a per-cell width, Gaussian likelihood) or `cat` (one categorical
  over no-charge + charge bins, replacing both heads; a hurdle, so `cond_charge=True`); every
  head returns `(charge, presence_logit)` first, so `predict` and the evaluation read all three
  alike. `lin_amp_scale` > 0 gives the encoder the linear-amplitude input channel
  (`FMModel`), a pretrained FM loading into it with the new columns at zero
  (`pad_input_columns`). `param_groups` = the encoder's muP groups + the head at
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
`helix.model.schedule` (`warmup_steps`, `eval_every`, `save_every`).

`configs/pimm/fm.py` is the whole run: corpus = `HELIX_CORPUS`'s parent with its
`_calib/RUNS.txt`, the reference bin grid, the `nz_pw16` recipe (d 768, 12 + 4 blocks, 12
heads, QK-norm, fused QK-norm + RoPE, K = 128; fused AdamW at LR 4.4e-3) + `WSDStableLR`,
batch = world size. Variants override only what
differs: `fm_cooldown.py` (warm start from `$HELIX_EXP/fm/model/last`, first 3 runs, one
epoch, `WSDCooldownLR`), `fm_smoke.py` (a
small model, 32 events, one GPU). Anything else is a `key=value` override.

`configs/pimm/denoise.py` (`_base_` fm.py) trains the denoiser through the same trainer and
launcher: `Coeff-Denoise` on the FM run `model.fm` (default `$HELIX_EXP/fm_cooldown`), the M6d recipe
of DENOISE (archive tag) as defaults (`model.cond_charge=True` is M6dc), the FM's identity-hash
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
* `scripts/corpus.sh` submits with `sbatch` from the host; `scripts/submit.sh` submits
  training through `pimm submit` (recipes in `configs/launch/`, submitit chains, multi-node
  under shifter with the `nccl-plugin` module). Both pass the site's account / partition /
  QOS / constraint as flags. `scripts/sweep.py` runs a study file (`configs/sweeps/`) through
  `submit.sh`, with evaluation jobs held by `--dependency=afterok`.

| | S3DF (`s3df.yaml`) | NERSC (`nersc.yaml`) |
|---|---|---|
| container | apptainer, `/sdf/data/neutrino/omara/images/helix-train.sif` | shifter `ghcr.io/omaralterkait/helix-nersc:jax-cuda12` (STALE: built with jax; rebuild and repoint, TODO); podman-hpc interactive |
| train | `mli:cider-ml`, ampere, QOS normal | `m5238_g`, constraint gpu, QOS preempt |
| corpus | `mli:default`, turing, QOS preemptable, `--requeue` | `m5238_g`, constraint gpu, QOS shared |

The container: `container/install.sh` is the one recipe, run by `container/helix-train.def`
(apptainer), `container/Dockerfile` (CI, `ghcr.io/omaralterkait/helix`) and
`container/Dockerfile.nersc` (Docker v2 manifest for shifter / podman-hpc). It installs
dependencies only, with pimm-data at the revision `pyproject.toml` pins; helix and pimm
arrive by bind mount and `PYTHONPATH`, so editing either needs no rebuild. Rebuild the
image when the pimm-data pin moves.
