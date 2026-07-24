# Research → {helix, pimm-data, pimm}: the unification map

Companion to `CONSOLIDATION_PLAN.md` (which is DSP-only). This maps the WHOLE
`research/coeff_foundation_model` tree — a monolith that today does DSP + data +
model + training + eval with hardcoded paths and duplicated logic — into clean
homes, so every concern has ONE source of truth and `research/` shrinks to
experiment definitions + the record.

## 0. The layering (dependency direction — strictly one-way)

```
  helix        DSP only: removal, wavelet, threshold, CoeffSet, provenance
    ▲          (no dependency on pimm-data or pimm)
    │  imports for the coeff transform
  pimm-data    data: readers, datasets, noise injection, coeff-cache + identity,
    ▲          collate, geometry, NormalizationTable   (may import helix for DSP)
    │  imports for data + DSP
  pimm         model, tokenizer, training loop, eval hooks
    ▲
    │  composes all three
  research/    experiment CONFIGS + the corpus builder + design record + frozen
               studies + qualification. Thin: no load-bearing DSP/data/model code.
```

The rule: **code flows down to its layer; research keeps only composition + record.**

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
| `rows_to_struct` (rows→struct) | **pimm** tokenizer | model-side |
| the ORCHESTRATION (compose the above) | **the corpus builder** (research/pimm, thin) | imports helix + pimm-data; no algorithm of its own |

Result: `star_tpc`/`measure_coeffs` become ~empty — a thin builder that loops
pimm-data grids → helix CoeffSet → pimm-data writer. (FREEZE until training done.)

### C2. Tokenizer — `vit_tpc.py` (`assemble_tpc_band`), `star_tpc.rows_to_struct`, `star_model` (SIGMA/losses/DEV)
| piece | → home |
|---|---|
| patchify (PW×PT, N_SLOT), tree parent/ancestor, cell/slot grids, asinh norm | **pimm/models/fm_mae/tokenize.py** |
| `FM_PW`/`FM_PT`/`FM_CELLT` env-global mutation | KILL → constructor args (pimm) |
| `LENS_T`/pad-4336 (hardcoded ×3) | DERIVE once from helix `wavedec` band_lengths |
| `SIGMA=2.6`, `losses`, `DEV` (`star_model`) | pimm (SIGMA→NormalizationTable; losses→model; DEV→trainer) |
Tokenizer stays model-side (it's a training recipe, still churning) — NOT helix, NOT pimm-data.

### C3. Coeff cache — `fm/cache.py`, `fm/cache_ext.py`, `fm/charge_cache.py`, `fm/data.py`
| piece | → home | note |
|---|---|---|
| on-disk coeff dataset (read) | **pimm-data** `CoeffTPCDataset` + `CoeffRowReader` | sharded HDF5, raw values, versioned σ sidecar, run-qualified identity |
| shard writer | **pimm-data** `CoeffShardWriter` | pins the format (round-trip test) |
| cache builder (fill it) | **corpus builder** (research/pimm) | composes pimm-data grids + helix DSP + writer; replaces cache/cache_ext |
| `fm/data.py` "THROWAWAY shim" | DELETE → `CoeffTPCDataset` + tokenizer | its `batch_size=None` one-event contract preserved |
| positional identity + corrupt-skip defects | FIXED by pimm-data identity (runs= landed) | |
| charge cache | pimm-data (co-located `val_charge`) | kills the dual-directory pairing |

### C4. FM model — `model.py`, `model_serial.py` (production `SerialFMModel`), `vit_model.py`, `perceiver_*`, `model_serial`
→ **pimm/models/fm_mae/**. Production arch + registered variants; state-dict keys
preserved (subclass) so existing checkpoints load. `vit_model`/`perceiver` = variants.

### C5. Training — `fm/train.py`, `fm/mae_ddp.py`, `fm/slurm/*`
→ **pimm/engines** (`FMTrainer`) + `BatchTransformLoader` (the dense-tail runner)
+ launch layer. The migrate-training-into-pimm track. muP param groups, warmup-
cosine-floor scheduler, `batch_size=None` loader. Drop `_rank` from the seed.

### C6. Labels & eval — `fm/pb_labels.py`, `pb_probe.py`, `probe_3d_*`, `eval_harness.py`, `deconv_*`
| piece | → home | note |
|---|---|---|
| label build (`hits.group_to_track ⨝ edepsim` PDG/KE) | **pimm-data** edepsim reader + label decoration | the MISSING loader; JAXTPC `make_labl.py` is the seed |
| probe harness (frozen encoder → ridge/MLP, controls, CV) | **pimm** eval hooks OR research eval scripts | `probe_3d_rigor` pattern; mature |
| deconv fine-tune program | **pimm** (downstream) | the value-prop result |
| eval_harness (mask-gen curve, NLL, var-explained) | pimm eval hook (`FMReconEvaluator`) | |

### C7. Optical — `doraemon_optical.py`, `onfly_optical.py`, optical tokenizer studies
| piece | → home |
|---|---|
| optical DSP (chunk sparsify, quant) | **helix.optical** (already there) |
| optical data loader | **pimm-data** `OpticalDataset` (already there; re-validate new nested layout) |
| optical tokenizer (hybrid, D-16) | **pimm/models** |
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

## 3. The end state — research becomes thin

`research/coeff_foundation_model/` after extraction:
- `configs/` — experiment definitions composing pimm(model)+pimm-data(data)+helix(DSP).
- `build_corpus.py` — the one builder (loops pimm-data grids → helix CoeffSet →
  pimm-data writer + σ sidecar). Replaces star_tpc/measure_coeffs/cache/cache_ext.
- `docs/` — the design record (DECISIONS, RESULTS, plans).
- `coherent_coeffs/`, `wire_denoise/`, `r2_qualification/` — frozen studies + evidence.
No load-bearing DSP, data, model, or tokenizer code remains — it all moved down a layer.

## 4. Sequencing (respecting the freeze; libraries bottom-up)

1. **helix DSP** (this worktree): CoeffSet+io, public wavedec/threshold_bands,
   coherent_gate, process_plane, CLI. (CONSOLIDATION_PLAN §5.) Merge at freeze boundary.
2. **pimm-data data**: colored-spectrum fix; `NormalizationTable`; `CoeffShardWriter`
   + `CoeffTPCDataset` + reader; edepsim label reader. (Independent of helix merge;
   the builder composes them.) Kill `load_geom`/`build_batch` dups.
3. **corpus builder** (research, thin): compose #1+#2, write the new coeff format
   (raw values + σ sidecar + identity). Validate vs old cache (tokenizer-port +
   builder-equivalence). Replaces cache/cache_ext/star_tpc extraction.
4. **pimm model/train/eval**: fm_mae model + tokenizer + FMTrainer + probes.
   (The migrate-into-pimm track; after the loader lands.)
5. **research slim-down**: delete extracted code (→ thin re-exports first, then
   remove once callers move); write RESULTS.md + research/README map; STATUS banners.

Each library is independently testable; research composes them. Nothing moves out
of a live-training import path except at the freeze boundary.
