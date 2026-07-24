# Research → {helix, pimm-data, pimm}: the unification map

Companion to `CONSOLIDATION_PLAN.md` (which is DSP-only). This maps the WHOLE
`research/coeff_foundation_model` tree — a monolith that today does DSP + data +
model + training + eval with hardcoded paths and duplicated logic — into clean
homes, so every concern has ONE source of truth and `research/` shrinks to
experiment definitions + the record.

## 0. The layering (corrected 2026-07-24)

**helix is the COMPLETE foundation-model library** — wavelet DSP + coefficient
representation AND the tokenizer, the FM model, training and eval. It is NOT
"DSP only." helix and pimm are then SIBLING model frameworks over one data layer:

```
  pimm-data    DATA layer (shared): readers, datasets (JAXTPC/LUCiD/Optical),
   ▲    ▲      noise injection, coeff-cache + identity, labels (edepsim),
   │    │      geometry, collate, NormalizationTable.   (may import helix for DSP)
   │    └──────────────────────────────┐
  helix                              pimm  (particle-imaging-models)
  the wavelet-coefficient FM:         the point-cloud framework:
   core/     wavelet DSP, CoeffSet      models/  PT-v3, sonata, polarmae, voltmae
   tpc/ optical/  removal, pipeline     engines/ trainers, hooks
   tokenize/ coeff rows → tokens        configs/, launch/
   model/    the FM MAE (SerialFM…)     (the SPINE/PoLAr-MAE baseline side)
   train/    the training loop
   eval/     probes, eval harness
   ▲
   │  composes helix + pimm-data
  research/  experiment CONFIGS + corpus builder + design record + frozen
             studies + qualification. Thin: no load-bearing code.
```

The rule: **the entire FM program's load-bearing code → helix; data → pimm-data;
research keeps only composition + the record.** pimm stays the separate
point-cloud framework (it consumes pimm-data too; it is the baseline/comparison
side, not where the FM lives).

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
→ **helix/model/**. Production arch + variants; the FM MAE lives in helix.
`vit_model`/`perceiver` = variants. (pimm keeps its OWN point-cloud models — PT-v3,
polarmae — for the baseline comparison; those are unrelated to the helix FM.)

### C5. Training — `fm/train.py`, `fm/mae_ddp.py`, `fm/slurm/*`
→ **helix/train/**. The FM training loop lives in helix: muP param groups,
warmup-cosine-floor scheduler, DDP, `batch_size=None` loader over the coeff-token
dataset. It CONSUMES pimm-data (the coeff dataset) but is a helix component. Drop
`_rank` from the noise seed. The launch/slurm layer stays research-side config.

### C6. Labels & eval — `fm/pb_labels.py`, `pb_probe.py`, `probe_3d_*`, `eval_harness.py`, `deconv_*`
| piece | → home | note |
|---|---|---|
| label build (`hits.group_to_track ⨝ edepsim` PDG/KE) | **pimm-data** edepsim reader + label decoration | the MISSING loader; JAXTPC `make_labl.py` is the seed — DATA, so pimm-data |
| probe harness (frozen encoder → ridge/MLP, controls, CV) | **helix/eval/** | `probe_3d_rigor` pattern; part of the FM library |
| deconv fine-tune program | **helix/eval/** (downstream head) | the value-prop result |
| eval_harness (mask-gen curve, NLL, var-explained) | **helix/eval/** | |
Note: the SPINE/PoLAr-MAE BASELINE side lives in **pimm** (its point-cloud models);
the FM's own eval is helix.

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

## 2b. The new helix package shape (full FM library)

```
helix/
  core/       wavelet DSP, backend, threshold_bands, CoeffSet, provenance
  tpc/        wire removal (gate/multipass) + pipeline + io
  optical/    PMT chunk pipeline
  tokenize/   coeff rows → tokens (patchify, tree, asinh)   [from vit_tpc, rows_to_struct]
  model/      the FM MAE (SerialFMModel + variants)         [from fm/model*, vit_model]
  train/      the FM training loop (DDP, muP, schedule)     [from fm/train, mae_ddp]
  eval/       probes + eval harness + deconv head           [from fm/pb_probe, probe_3d, eval_harness]
```
helix imports **pimm-data** for data (the coeff-token dataset, geometry, noise).
No dependency on pimm. New extras: `helix[torch]` for model/train/eval.

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

## 4. Sequencing (respecting the freeze; data layer first)

1. **helix DSP** (this worktree): CoeffSet+io, public wavedec/threshold_bands,
   coherent_gate, process_plane, CLI. (CONSOLIDATION_PLAN §5.)
2. **pimm-data data**: colored-spectrum fix; `NormalizationTable`; `CoeffShardWriter`
   + `CoeffTPCDataset` + reader; edepsim label reader. Kill `load_geom`/`build_batch` dups.
3. **corpus builder** (research, thin): compose #1+#2 → new coeff format; validate
   vs old cache. Replaces cache/cache_ext/star_tpc extraction.
4. **helix tokenize/model/train/eval**: promote the FM stack into helix (from fm/,
   vit_tpc, star_model). This is the big lift and the FREEZE-sensitive one (the live
   training imports these) — do in this worktree, merge at the freeze boundary.
5. **research slim-down**: delete extracted code (→ thin re-exports first, then
   remove once callers move); write RESULTS.md + research/README map; STATUS banners.

Each library is independently testable; research composes helix + pimm-data.
Nothing moves out of a live-training import path except at the freeze boundary.
