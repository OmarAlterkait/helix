# Research → {helix, pimm-data, pimm}: the unification map

Companion to `CONSOLIDATION_PLAN.md` (which is DSP-only). This maps the WHOLE
`research/coeff_foundation_model` tree — a monolith that today does DSP + data +
model + training + eval with hardcoded paths and duplicated logic — into clean
homes, so every concern has ONE source of truth and `research/` shrinks to
experiment definitions + the record.

## 0. The layering (corrected 2026-07-24, rev 3 — grounded on pimm-private v0.5.1)

**helix PROVIDES the model + representation** (DSP, the *stateless* coeff-assembly
function, and the FM `nn.Module` — everything to *define & run* the FM). **pimm
TRAINS and EVALS it** (the loop, optimizer, schedule, DDP, checkpointing, probes)
**and hosts its FM-exclusive transforms**. pimm-data is the shared DATA layer and
the **migration target for dataset loading**. The goal of this map: define helix's
model interface and enumerate **what pimm needs to add** to train/eval it (§5).

The word "tokenizer" hid two separable things — split them:
- **assemble** (stateless numpy: patchify / asinh / occupancy / valid-mask / coords,
  **zero `nn.Parameter`**) — the analog of pimm-data's `GridSample`. Its *logic*
  lives in **helix** (`helix.model.tokenize.assemble`, a pure function — "helix owns the
  tokenizer"); the *transform wrapper* that runs it in the DataLoader worker is an
  **FM-exclusive transform in pimm**, registered into pimm's registry, via `helix.integrations.pimm` + `custom_imports`
  (the `HierarchicalMaskGenerator` precedent). pimm-data never imports helix at
  read time → no dependency cycle.
- **embed** (learnable `patch_embed` / `band_emb` / RoPE / FiLM / `mask_tok`) — a
  model op, in **helix.model.forward**.

```
  pimm-data    DATA layer (shared): readers, datasets (JAXTPC/LUCiD/Optical/Coeff —
   ▲    ▲      yield neutral dicts / coeff ROWS, never tokens), noise injection,
   │    │      coeff-cache + identity, labels (edepsim), geometry, NormalizationTable.
   │    │      MIGRATION TARGET: dataset loading (DataLoader build + collate) moves
   │    │      here over time.  (imports helix only at corpus-BUILD time; torch-free at read)
   │    └───────────────────────────────────────┐
  helix                                        pimm  (base: pimm-private v0.5.1)
  PROVIDES model + representation:              TRAINS + EVALS + baselines:
   core/     wavelet DSP, CoeffSet               engines/  trainers incl. FMTrainer
   tpc/ optical/  removal, pipeline              eval/     probes, harness, deconv
   tokenize/ assemble() — pure numpy fn,         transforms/  CoeffTokenize (FM-exclusive
             NO Dataset/transform machinery                   wrapper → calls helix.assemble,
   model/    the FM MAE (nn.Module, forward,                  registers into pimm-data registry)
             loss, param_groups, encode, mask)   datasets/ loading (build DataLoader + collate)
   NO train/ NO eval/ NO Dataset/loader/collate  models/   point-cloud zoo (baselines)
   ▲                                             └── imports helix (model+assemble) + pimm-data
   │  imports pimm-data (BUILD time only)
```

Dependencies: **pimm → helix (model + assemble) + pimm-data (data); helix →
pimm-data (BUILD time only)**; helix has NO dependency on pimm; **pimm-data never
imports helix at read time** (no cycle). The rule: **learnable / `nn.Module` +
stateless representation *logic* → helix; loading + generic data → pimm-data;
training + eval + FM-exclusive transforms → pimm; research keeps configs + record.**

**Now → target (per the migration directive):** in pimm-private today, dataset
*loading* lives in pimm — the trainer builds the DataLoader (`engines/train.py`)
and `collate_fn` is local (`datasets/utils.py:18`), NOT in pimm-data. The plan is
to **move loading (DataLoader construction + generic collate) into pimm-data**,
while **FM-exclusive transforms stay in pimm**. This map places new code at its
*target* home and flags the transitional pimm-side spots.

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
| `rows_to_struct` (rows→struct assembly) | **helix.model.tokenize.assemble** (pure fn) + **pimm** `CoeffTokenize` wrapper | stateless; logic in helix, wrapper in pimm (§C2) |
| the ORCHESTRATION (compose the above) | **the corpus builder** (research, thin) | imports helix + pimm-data; no algorithm of its own |

Result: `star_tpc`/`measure_coeffs` become ~empty — a thin builder that loops
pimm-data grids → helix CoeffSet → pimm-data writer. (FREEZE until training done.)

### C2. Tokenizer — `vit_tpc.py` (`assemble_tpc_band`), `star_tpc.rows_to_struct`, `star_model` (SIGMA/losses/DEV)
"Tokenizer" = **assemble** (stateless, DATA) + **embed** (learnable, MODEL). Split it:
| piece | → home | why |
|---|---|---|
| **assemble**: patchify (PW×PT, N_SLOT), tree parent/ancestor, cell/slot grids, asinh norm, occupancy/valid masks, dead-wire aug, `cell_t`/`wire_pos` coords | **helix.model.tokenize.assemble** (pure numpy fn) + **pimm `CoeffTokenize`** transform that calls it | zero `nn.Parameter` → data-shaping; analog of pimm-data `GridSample`. Logic in helix ("helix owns the tokenizer"); the worker-run transform is FM-exclusive → pimm, registered into PIMM's registry by `helix.integrations.pimm` (pulled in via a config's `custom_imports`), so pimm needs no change. **helix ⇏ Dataset/transform machinery; pimm-data ⇏ helix import at read time (no cycle).** |
| **embed**: `patch_embed`/`band_emb`/`plane_emb`/FiLM/RoPE-angles/`mask_tok` | **helix.model.forward** | learnable → model, per pimm's own `self.tokenizer`-in-`forward` precedent (voltmae spconv, PTv3 serialize) |
| `FM_PW`/`FM_PT`/`FM_CELLT` env-global mutation | KILL → explicit `PatchConfig` args | verified as the ONLY coupling to remove; passed to `CoeffTokenize` ctor + read by helix model for `patch_embed` dim |
| `LENS_T`/pad-4336 (hardcoded ×3) | DERIVE once from helix `wavedec` band_lengths | stamped into the coeff-shard schema at build time → read-time transform needs no helix import |
| `SIGMA=2.6`, `losses`, `DEV` (`star_model`) | `PatchConfig`/helix norm (SIGMA); helix.model (losses); pimm (DEV) | |
**`PatchConfig`** (PW/PT/N_SLOT/SIGMA + wavedec band_lengths) is a single source of
truth **owned by `helix/config.py`**: the experiment config feeds it to pimm's
`CoeffTokenize` (explicit args) and helix's model reads it for `patch_embed` input
dim — both agree by construction.

### C3. Coeff cache — `fm/cache.py`, `fm/cache_ext.py`, `fm/charge_cache.py`, `fm/data.py`
| piece | → home | note |
|---|---|---|
| on-disk coeff dataset (read) | **pimm-data** `CoeffTPCDataset` + `CoeffRowReader` (yields coeff ROWS, not tokens) | sharded HDF5, raw values, versioned σ sidecar, run-qualified identity |
| shard writer | **pimm-data** `CoeffShardWriter` | pins the format (round-trip test) |
| cache builder (fill it) | **corpus builder** (research/pimm) | composes pimm-data grids + helix DSP + writer; replaces cache/cache_ext |
| `fm/data.py` "THROWAWAY shim" | DELETE → `CoeffTPCDataset` + `CoeffTokenize` | **drop `batch_size=None`**: use pimm's offset-packed collate (variable token count packs via `offset`, like points) — pimm-private has NO `batch_size=None` path (§5b.2) |
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
checkpoint / logging over the coeff-token DataLoader (**offset-packed collate**, not
`batch_size=None` — §5b.2). Drop `_rank` from the noise seed. helix supplies the
model + its param_groups + loss; pimm supplies the loop. The DataLoader
construction is a pimm-side spot *now* but is part of the loading that **migrates
to pimm-data** later. launch/slurm → research/pimm run config. See §5.

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
| optical tokenizer (hybrid, D-16) | **helix.model.tokenize.assemble** (optical branch, pure fn) + pimm `CoeffTokenize` optical variant |
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
| collate: `build_batch` (hand-rolled) + pimm `datasets/utils.collate_fn` | pimm `collate_fn` now → **migrates to pimm-data** (loading); FM uses its offset-packed form |
| band-lengths / pad-4336 ×3 hardcoded | helix wavedec-derived band_lengths |
| σ constants (SIGMA, kgate, ksig, GS) scattered | helix config defaults + pimm-data NormalizationTable |
| noise defaults: AddNoise literals vs DEFAULT_* | pimm-data DEFAULT_* constants |
| two pimm-data checkouts (submodule vs standalone) | ONE canonical + symlink; research sys.path → it |
| `.pylibs` sys.path hacks ×~15 | pip-installed env (or one `_paths.py` helper) |
| fm/data.py throwaway shim | pimm-data CoeffTPCDataset |

## 2b. The new helix package shape (model + representation, NOT training/loading)

```
helix/
  core/       wavelet DSP, backend, threshold_bands, CoeffSet, provenance
  tpc/        wire removal (gate/multipass) + pipeline + io
  optical/    PMT chunk pipeline
  config/     PatchConfig (PW/PT/N_SLOT/SIGMA) + wavedec band_lengths  [SOURCE OF TRUTH]
  tokenize/   assemble(rows, PatchConfig) -> arrays — PURE NUMPY fn    [from vit_tpc,
              (patchify/asinh/occupancy/coords); NO torch, NO Dataset   rows_to_struct]
  model/      the FM MAE: nn.Module + forward + loss +      [from fm/model*, vit_model,
              param_groups + encode + mask; learnable embed  star_model losses/SIGMA]
  (NO train/ NO eval/  — pimm owns the loop + probes)
  (NO Dataset / DataLoader / collate / transform-registry  — that seam never enters helix)
```
`helix.model.tokenize.assemble` is a **pure function** (rows + PatchConfig → arrays), not a
transform — pimm's `CoeffTokenize` imports and calls it. helix imports **pimm-data**
only at corpus-BUILD time (DSP); it is **torch-free-importable at read time** and has
NO dependency on pimm. Extra: `helix[torch]` for the model only (assemble stays numpy).

## 5. THE INTERFACE — what pimm needs to train & eval the helix model

This is the deliverable. helix EXPOSES a stable model API; pimm ADDS the loop/eval.

### 5a. What helix must expose (the contract pimm consumes)
```python
# --- representation logic (stateless, numpy; called by pimm's CoeffTokenize) ---
helix.model.tokenize.assemble(coeff_rows, patch_cfg) -> arrays  # PURE fn: rows -> token arrays
helix.config.PatchConfig                                  # PW/PT/N_SLOT/SIGMA + band_lengths (SoT)
# --- the model (nn.Module) ---
helix.build_fm(config) -> nn.Module                 # construct the FM (arch from config)
model.forward(batch) -> dict(loss=…, **aux)         # flat-tensor dict in, dict-with-loss out
                                                    #   (matches pimm-private run_step: train.py:386-388)
model.param_groups(lr, wd) -> list[dict]            # muP buckets (hidden lr/m + wd·m; nodecay)
model.encode(batch) -> Tensor                       # frozen features (for probes)
model.encode_layers(batch) -> list[Tensor]          # per-layer (probe sweep)
model.make_mask(batch, ratio, mode) -> mask         # MAE masking (model-owned; per-step, GPU)
# arrays fields: inp/occ/valid/target/cell/slot/band_id/plane_id/t_phys/wire_pos/…
# batching: token rows pack via `offset` (pimm offset-concat collate) — NOT batch_size=None.
# checkpoint: state_dict keys stable (subclass current SerialFMModel) so existing ckpts load.
```
Note the split: **`assemble` is helix's but runs data-side** (pimm's `CoeffTokenize`
calls it in the worker); **the model** consumes the packed token batch. helix exposes
both, but only the model is an `nn.Module`.

### 5b. What pimm must ADD (the training + eval integration — the "what's needed")
1. **FMTrainer** (`pimm/engines`, register in TRAINERS): builds `helix.build_fm`,
   runs the loop, calls `model.forward → loss`. No new model code — pimm calls helix.
   (Contract already matches pimm-private: flat dict in, `output_dict["loss"]` out,
   `train.py:386-388`.)
2. **`CoeffTokenize` transform** (`pimm/transforms`, FM-exclusive, registered into
   pimm's registry, via `helix.integrations.pimm` + `custom_imports`): calls `helix.model.tokenize.assemble` in the DataLoader
   worker (CPU, ~215 ms/event) → token arrays. **Batching = pimm's offset-packed
   collate** (variable token count packs via `offset`, like points) — **NOT
   `batch_size=None`**: pimm-private has no such path (every loader is integer-batched
   + `collate_fn` + `drop_last=True`, `StatefulDataLoader`, `engines/train.py:498`).
   The FM path rides `collate_fn`'s offset-concat (`datasets/utils.py:18`).
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
8. **Data contract**: pimm-data provides `CoeffTPCDataset` (yields coeff ROWS, never
   tokens); pimm's `CoeffTokenize` transform (calling `helix.assemble`) turns rows →
   tokens in the pipeline; labels for probes (edepsim join) + holdout/identity.

Resolved (was: worker vs forward): **stateless assemble runs data-side in the worker**
(pimm `CoeffTokenize` → `helix.assemble`); only the **learnable embed** runs in
`model.forward`. Grounded in pimm's own convention (learnable tokenizers are
model-side *because* they're learnable — voltmae spconv, PTv3 serialize; the stateless
analog is `GridSample`, a data transform) + the no-cycle constraint (pimm-data must
not import helix at read time). Items 1-2 (DataLoader build + collate) sit pimm-side
*now* but are part of the **loading that migrates to pimm-data** per the directive.

## 3. The end state — research becomes thin

`research/coeff_foundation_model/` after extraction:
- `configs/` — experiment definitions composing helix (DSP + assemble + model) +
  pimm-data (data + loading) + pimm (loop + FM-exclusive transforms + eval).
- `build_corpus.py` — the one builder (loops pimm-data grids → helix CoeffSet →
  pimm-data writer + σ sidecar). Replaces star_tpc/measure_coeffs/cache/cache_ext.
- `docs/` — the design record (DECISIONS, RESULTS, plans).
- `coherent_coeffs/`, `wire_denoise/`, `r2_qualification/` — frozen studies + evidence.
- launch/slurm run configs.
No load-bearing DSP, data, model, assemble, training, or eval code remains — it
all moved into helix (DSP + assemble + model), pimm-data (data + loading target),
or pimm (loop + CoeffTokenize + eval).

## 6. Sequencing (respecting the freeze)

1. **helix DSP** (this worktree): CoeffSet+io, public wavedec/threshold_bands,
   coherent_gate, process_plane, CLI. (CONSOLIDATION_PLAN §5.)
2. **pimm-data data**: colored-spectrum fix; `NormalizationTable`; `CoeffShardWriter`
   + `CoeffTPCDataset` + reader; edepsim label reader. Kill `load_geom`/`build_batch` dups.
3. **corpus builder** (research, thin): compose #1+#2 → new coeff format; validate
   vs old cache. Replaces cache/cache_ext/star_tpc extraction.
4. **helix assemble/model**: promote `assemble` (pure fn, from vit_tpc/rows_to_struct)
   + `PatchConfig` + FM model (fm/model*, vit_model, star_model) into helix. Expose the
   §5a API. FREEZE-sensitive (live training imports these) — worktree, merge at boundary.
5. **pimm training + eval** (§5b): FMTrainer, `CoeffTokenize` transform (offset-packed,
   NOT batch_size=None), BatchTransformLoader, muP optimizer, scheduler, checkpoint
   compat, eval hooks. pimm consumes helix.build_fm + helix.assemble + pimm-data dataset.
   Validate A/B vs the standalone loop.
6. **research slim-down**: delete extracted code (thin re-exports first); write
   RESULTS.md + research/README map; STATUS banners.
7. **(later) loading migration**: move DataLoader construction + generic `collate_fn`
   from pimm into pimm-data; `CoeffTokenize` (FM-exclusive) stays in pimm.

helix defines the model + assemble; pimm trains/evals + hosts FM-exclusive transforms;
pimm-data feeds both (and absorbs loading over time). Nothing moves out of a
live-training import path except at the freeze boundary.
