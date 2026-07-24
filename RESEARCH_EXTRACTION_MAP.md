# Research → {helix, pimm-data, pimm}: the unification map

Companion to `CONSOLIDATION_PLAN.md` (which is DSP-only). This maps the WHOLE
`research/coeff_foundation_model` tree — a monolith that today does DSP + data +
model + training + eval with hardcoded paths and duplicated logic — into clean
homes, so every concern has ONE source of truth and `research/` shrinks to
experiment definitions + the record.

## 0. The layering (corrected 2026-07-24, rev 2)

**helix PROVIDES the model + representation** (DSP, tokenizer, model architecture,
forward, everything to *define & run* the FM). **pimm TRAINS and EVALS it** (the
loop, optimizer, schedule, DDP, checkpointing, probes). pimm-data is the shared
DATA layer. The goal of this map: define helix's model interface and enumerate
**what pimm needs to add** to train/eval it (§5).

```
  pimm-data    DATA layer (shared): readers, datasets (JAXTPC/LUCiD/Optical),
   ▲    ▲      noise injection, coeff-cache + identity, labels (edepsim),
   │    │      geometry, collate, NormalizationTable.   (may import helix for DSP)
   │    └───────────────────────────────────────┐
  helix                                        pimm  (particle-imaging-models)
  PROVIDES the model + representation:          TRAINS + EVALS + baselines:
   core/     wavelet DSP, CoeffSet               engines/  trainers incl. FMTrainer
   tpc/ optical/  removal, pipeline              eval/     probes, harness, deconv
   tokenize/ coeff rows → tokens                 models/   point-cloud zoo (baselines)
   model/    the FM MAE (nn.Module, forward,     configs/, launch/
             loss, param_groups, encode, mask)   └── imports helix (model) + pimm-data
   NO train/ NO eval/  (those are pimm's job)
   ▲
   │  imports pimm-data (data)
```

Dependencies: **pimm → helix (model) + pimm-data (data); helix → pimm-data (data)**;
helix has NO dependency on pimm. The rule: **model + representation code → helix;
training + eval → pimm; data → pimm-data; research keeps configs + record.**

## 1. Per-concern placement (the whole tree)

### C1. Coefficient extraction pipeline — `star_tpc.py`, `measure_coeffs.py`
The per-event chain: extract sensor → densify → inject noise → digitize → DWT →
gate → threshold → rows + σ-table. Split by concern:
| piece | → home | note |
|---|---|---|
| DWT / smart gate / threshold | **helix.tpc** (`process_plane`, `coherent_gate`, `threshold_bands`) | the DSP; §CONSOLIDATION_PLAN |
| densify / AddNoise / digitize | **pimm-data** (`dense_ops` — already there) | + the colored-spectrum fix |
| `load_geom` | **pimm-data** (`geometry.load_plane_registry`) | DELETE `measure_coeffs.load_geom` (dup) |
| `build_batch` (hand-rolled collate) | **pimm-data** (`collate`) | DELETE the hand copy |
| σ-table (2-cal-event normalization) | **pimm-data** `NormalizationTable` (versioned sidecar) | dataset-level artifact, NOT DSP, NOT in CoeffSet |
| `rows_to_struct` (rows→struct) | **helix.tokenize** | model-side, in helix |
| the ORCHESTRATION (compose the above) | **the corpus builder** (research, thin) | imports helix + pimm-data; no algorithm of its own |

Result: `star_tpc`/`measure_coeffs` become ~empty — a thin builder that loops
pimm-data grids → helix CoeffSet → pimm-data writer. (FREEZE until training done.)

### C2. Tokenizer — `vit_tpc.py` (`assemble_tpc_band`), `star_tpc.rows_to_struct`, `star_model` (SIGMA/losses/DEV)
| piece | → home |
|---|---|
| patchify (PW×PT, N_SLOT), tree parent/ancestor, cell/slot grids, asinh norm | **helix/tokenize/** |
| `FM_PW`/`FM_PT`/`FM_CELLT` env-global mutation | KILL → constructor/config args (helix) |
| `LENS_T`/pad-4336 (hardcoded ×3) | DERIVE once from helix `wavedec` band_lengths |
| `SIGMA=2.6`, `losses`, `DEV` (`star_model`) | helix (SIGMA→helix norm/NormalizationTable; losses→helix.model; DEV→helix.train) |
The tokenizer lives in **helix** — it is the FM's front half (coeff rows → tokens),
inseparable from the model it feeds.

### C3. Coeff cache — `fm/cache.py`, `fm/cache_ext.py`, `fm/charge_cache.py`, `fm/data.py`
| piece | → home | note |
|---|---|---|
| on-disk coeff dataset (read) | **pimm-data** `CoeffTPCDataset` + `CoeffRowReader` | sharded HDF5, raw values, versioned σ sidecar, run-qualified identity |
| shard writer | **pimm-data** `CoeffShardWriter` | pins the format (round-trip test) |
| cache builder (fill it) | **corpus builder** (research/pimm) | composes pimm-data grids + helix DSP + writer; replaces cache/cache_ext |
| `fm/data.py` "THROWAWAY shim" | DELETE → `CoeffTPCDataset` + tokenizer | its `batch_size=None` one-event contract preserved |
| positional identity + corrupt-skip defects | FIXED by pimm-data identity (runs= landed) | |
| charge cache | pimm-data (co-located `val_charge`) | kills the dual-directory pairing |

### C4. FM model — `model.py`, `model_serial.py` (production `SerialFMModel`), `vit_model.py`, `perceiver_*`
→ **helix/model/**. helix provides the model: the `nn.Module`, its `forward`, the
loss, `param_groups` (muP), `encode`/`encode_layers` (for probes), and the mask
generator. Variants (`vit_model`, `perceiver`) too. helix defines the model; pimm
runs it (§5). `losses`/`SIGMA`/`DEV` (`star_model`) → helix.model / helix norm.

### C5. Training — `fm/train.py`, `fm/mae_ddp.py`, `fm/slurm/*`  →  **pimm**
The training LOOP is pimm's: `pimm/engines` FMTrainer builds the helix model and
runs optimizer / muP param-group / warmup-cosine-floor schedule / DDP /
checkpoint / logging over the coeff-token DataLoader (`batch_size=None`). Drop
`_rank` from the noise seed. helix supplies the model + its param_groups + loss;
pimm supplies the loop. launch/slurm → research/pimm run config. See §5 for the
exact list of what pimm must add.

### C6. Labels & eval — `fm/pb_labels.py`, `pb_probe.py`, `probe_3d_*`, `eval_harness.py`, `deconv_*`
| piece | → home | note |
|---|---|---|
| label build (`hits.group_to_track ⨝ edepsim` PDG/KE) | **pimm-data** edepsim reader + label decoration | the MISSING loader; JAXTPC `make_labl.py` is the seed — DATA |
| probe harness (frozen `model.encode` → ridge/MLP, controls, CV) | **pimm/eval** (hooks) | calls helix `model.encode`; `probe_3d_rigor` pattern |
| deconv fine-tune program | **pimm/eval** (downstream) | fine-tunes the helix model; the value-prop result |
| eval_harness (mask-gen curve, NLL, var-explained) | **pimm/eval** (`FMReconEvaluator`) | |
EVAL is pimm's job; it consumes helix's `model.encode`/forward. The SPINE/PoLAr-MAE
baselines are pimm's point-cloud models.

### C7. Optical — `doraemon_optical.py`, `onfly_optical.py`, optical tokenizer studies
| piece | → home |
|---|---|
| optical DSP (chunk sparsify, quant) | **helix.optical** (already there) |
| optical data loader | **pimm-data** `OpticalDataset` (already there; re-validate new nested layout) |
| optical tokenizer (hybrid, D-16) | **helix/tokenize/** (the optical branch of the FM tokenizer) |
| optical FM fusion (cross-modal MAE) | **helix/model/** (multi-modal trunk) |
| `doraemon_optical` hardcoded path/loader | DELETE → `OpticalDataset` |

### C8. Design record — `DECISIONS.md`, `EXECUTION_PLAN.md`, `FULL_ARCHITECTURE.md`, `model_scoping.md`, `reviews/`, `litrev/`, `fm/*.md`
STAYS in `research/` — the intellectual record. Consolidate: one `research/README.md`
map + a live `fm/RESULTS.md` (the strongest results — deconv ladder, 3D probes —
have no written home today; write them). DECISIONS.md gets a dated addendum, not edits.

### C9. Frozen studies — `coherent_coeffs/`, `wire_denoise/`
STAY (historical). Their PRODUCTIONIZED outputs are extracted (smart gate→helix;
learned CNN grpnet = unproductionized, keep weights + note as a future U-lever).
STATUS banners; no move.

### C10. This session — `r2_qualification/`
STAYS as evidence. The faithful-injector fixtures + parity tests → **helix/tests**
as the packaged gate's qualification suite.

## 2. Unification / dedup — one source of truth for each

| duplicated today | → single home |
|---|---|
| smart gate ×3 (`measure_coeffs`, `smart.py`, `gpu_pipeline`) | helix.tpc.coherent_gate |
| threshold: `prod_threshold` + `sparsify` inline | helix.core.threshold_bands |
| geometry: `measure_coeffs.load_geom` + pimm-data | pimm-data.geometry |
| collate: `build_batch` + pimm-data | pimm-data.collate |
| band-lengths / pad-4336 ×3 hardcoded | helix wavedec-derived band_lengths |
| σ constants (SIGMA, kgate, ksig, GS) scattered | helix config defaults + pimm-data NormalizationTable |
| noise defaults: AddNoise literals vs DEFAULT_* | pimm-data DEFAULT_* constants |
| two pimm-data checkouts (submodule vs standalone) | ONE canonical + symlink; research sys.path → it |
| `.pylibs` sys.path hacks ×~15 | pip-installed env (or one `_paths.py` helper) |
| fm/data.py throwaway shim | pimm-data CoeffTPCDataset |

## 2b. The new helix package shape (model + representation, NOT training)

```
helix/
  core/       wavelet DSP, backend, threshold_bands, CoeffSet, provenance
  tpc/        wire removal (gate/multipass) + pipeline + io
  optical/    PMT chunk pipeline
  tokenize/   coeff rows → tokens (patchify, tree, asinh)   [from vit_tpc, rows_to_struct]
  model/      the FM MAE: nn.Module + forward + loss +      [from fm/model*, vit_model,
              param_groups + encode + mask generator         star_model losses/SIGMA]
  (NO train/  — pimm owns the loop)
  (NO eval/   — pimm owns probes/harness)
```
helix imports **pimm-data** for data (geometry, and the coeff-token dataset type it
tokenizes). No dependency on pimm. Extra: `helix[torch]` for tokenize+model.

## 5. THE INTERFACE — what pimm needs to train & eval the helix model

This is the deliverable. helix EXPOSES a stable model API; pimm ADDS the loop/eval.

### 5a. What helix.model must expose (the contract pimm consumes)
```python
helix.build_fm(config) -> nn.Module                 # construct the FM (arch from config)
model.forward(batch) -> dict(loss=…, **aux)         # masked-recon loss computed inside
model.param_groups(lr, wd) -> list[dict]            # muP buckets (hidden lr/m + wd·m; nodecay)
model.encode(batch) -> Tensor                       # frozen features (for probes)
model.encode_layers(batch) -> list[Tensor]          # per-layer (probe sweep)
helix.tokenize(coeff_rows, cfg) -> token_dict       # coeff rows -> the model's input dict
helix.make_mask(batch, ratio, mode) -> mask         # MAE masking (or model owns it)
# token_dict fields: inp/occ/valid/target/cell/slot/band_id/plane_id/t_phys/wire_pos/…
# checkpoint: state_dict keys stable (subclass current SerialFMModel) so existing ckpts load.
```

### 5b. What pimm must ADD (the training + eval integration — the "what's needed")
1. **FMTrainer** (`pimm/engines`, register in TRAINERS): builds `helix.build_fm`,
   runs the loop, calls `model.forward → loss`. No new model code — pimm calls helix.
2. **`batch_size=None` DataLoader** over pimm-data's coeff-token dataset (one event =
   one token set; ~215 ms CPU assembly in workers via `helix.tokenize`). pimm's
   `build_train_loader` currently hardcodes `batch_size` + role-collate → needs the
   `batch_size=None` variant.
3. **muP optimizer support**: `param_dicts="model"` branch in `pimm/utils/optimizer`
   so it takes `model.param_groups()` (pimm's keyword-matching can't express muP).
4. **Warmup-cosine-floor scheduler** in `pimm/utils/scheduler` (10%-floor cosine).
5. **BatchTransformLoader** + drop `_rank` from the noise seed (the dense-tail runner;
   needed for the on-the-fly training path, and it fixes the never-run
   `dataset.batch_transform` gap). DDP via pimm's wrapper, `set_epoch` stamping.
6. **Checkpoint compat**: pimm CheckpointSaver/Loader ↔ helix model state_dict
   (key-for-key by subclassing; a `tools/convert_fm_ckpt.py` for the existing
   `ckpt_dscale600b4_*`).
7. **Eval hooks** (`pimm/eval`): `FMReconEvaluator` (mask-gen curve, NLL,
   var-explained via `model.encode`/forward) + the frozen-probe harness
   (`model.encode` → ridge/MLP → metric, with random-init/raw/geo controls) +
   the deconv fine-tune. Ported from `fm/pb_probe`/`probe_3d_*`/`eval_harness`.
8. **Data contract**: pimm-data provides the coeff-token dataset (`CoeffTPCDataset`
   yielding coeff rows; `helix.tokenize` applied in `__getitem__` or a transform),
   labels for probes (edepsim join), and holdout/identity.

Open interface question: does `helix.tokenize` run in the DataLoader worker (CPU,
current ~215 ms/event) or inside `model.forward` (GPU)? Affects where the seam sits
between pimm-data (rows) and helix (tokens→model). Decide with the perf envelope.

## 3. The end state — research becomes thin

`research/coeff_foundation_model/` after extraction:
- `configs/` — experiment definitions composing helix (FM: DSP+tokenizer+model+train)
  + pimm-data (data).
- `build_corpus.py` — the one builder (loops pimm-data grids → helix CoeffSet →
  pimm-data writer + σ sidecar). Replaces star_tpc/measure_coeffs/cache/cache_ext.
- `docs/` — the design record (DECISIONS, RESULTS, plans).
- `coherent_coeffs/`, `wire_denoise/`, `r2_qualification/` — frozen studies + evidence.
- launch/slurm run configs.
No load-bearing DSP, data, model, tokenizer, training, or eval code remains — it
all moved into helix (FM code) or pimm-data (data).

## 6. Sequencing (respecting the freeze)

1. **helix DSP** (this worktree): CoeffSet+io, public wavedec/threshold_bands,
   coherent_gate, process_plane, CLI. (CONSOLIDATION_PLAN §5.)
2. **pimm-data data**: colored-spectrum fix; `NormalizationTable`; `CoeffShardWriter`
   + `CoeffTPCDataset` + reader; edepsim label reader. Kill `load_geom`/`build_batch` dups.
3. **corpus builder** (research, thin): compose #1+#2 → new coeff format; validate
   vs old cache. Replaces cache/cache_ext/star_tpc extraction.
4. **helix tokenize/model**: promote the tokenizer + FM model into helix (from fm/model*,
   vit_tpc, vit_model, star_model). Expose the §5a API. FREEZE-sensitive (live training
   imports these) — worktree, merge at boundary.
5. **pimm training + eval** (§5b): FMTrainer, batch_size=None loader,
   BatchTransformLoader, muP optimizer, scheduler, checkpoint compat, eval hooks.
   pimm consumes helix.build_fm + pimm-data dataset. Validate A/B vs the standalone loop.
6. **research slim-down**: delete extracted code (thin re-exports first); write
   RESULTS.md + research/README map; STATUS banners.

helix defines/runs the model; pimm trains/evals it; pimm-data feeds both. Nothing
moves out of a live-training import path except at the freeze boundary.
