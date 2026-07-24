# Helix consolidation plan — the placement map

Branch `consolidation` (this worktree). Main tree stays as-is: the live training
job (`dscale600b4_20k`, babysat) imports the FREEZE_LIST files **and, transitively,
`helix/core/wavelet_ops_torch`** from the *main tree path* — so edits HERE are
safe; only the `consolidation → main` MERGE is freeze-sensitive and waits for
training to finish (see §5).

## 0. Goal & invariants

Helix becomes a clean **DSP library** exposing one front-end with three modes,
one coefficient object, and a read/write pair that round-trips:

```
                         ┌──────────── helix front-end (one code path) ─────────────┐
 raw sensor file  ─────▶ │ [removal: gate(default) | multipass | none] → DWT →       │
                         │  threshold/sparsify → CoeffSet(+provenance)               │
                         └──────────────────────────────────────────────────────────┘
   mode 1 all-at-once ───────────────┬───────────────┬──────────── → ML model
                                     ▼ mode 3 (load)  ▼ mode 2 (convert)
                          read_processed(file)     write_processed(CoeffSet) → file
                          ── INVARIANT: write∘read = identity; mode1 output ≡ mode2+3 ──
```

**Qualified removal default (this session):** smart gate, `kgate=3.0`, `ksig=3.0`,
**`npass=2`**, `group_size=64`, A-parity `sigc` (`quantile(0.5)`). See
`research/r2_qualification/REPORT.md`.

---

## 1. PLACEMENT — what goes where

Legend:  KEEP = already correctly placed · ADD = new in helix · PROMOTE = move
research→package · THIN = research file becomes a re-export · STAY = stays in
research/pimm, NOT helix · FREEZE = in the live-training import closure (worktree
edit + merge-at-boundary).

### A. `helix/core/` — detector-agnostic DSP (shared by TPC & optical)

| item | action | notes |
|---|---|---|
| `backend.py` | KEEP | lazy numpy/jax/torch dispatch. FREEZE (training imports). |
| `wavelet.py` (ThresholdSpec, SparseResult, sparsify, reconstruct) | KEEP + ADD | ADD public `wavedec`/`waverec` (alias the `_wavedec`/`_waverec`), ADD `threshold_bands()` extracted from `sparsify`. |
| `wavelet_ops_{numpy,jax,torch}.py` | KEEP + ADD | ADD `threshold_bands` per backend; refactor `sparsify` to call it. Apply A-parity `sigc`. FREEZE (`_torch` imported by training). |
| `dwt_matrix.py` | KEEP | jax matmul-DWT builder. FREEZE. |
| `provenance.py` | **ADD** | `basis_descriptor(wavelet,level,pad,removal,threshold,…)` + sha256 digest → stamped into every CoeffSet/file. |
| `filters.py` (hardcoded coif3/db1 bank, pywt→test-only) | ADD (DEFERRED) | decouples pywt from the torch path; not required for consolidation. Do only if the pywt-in-container issue bites. |

### B. `helix/tpc/` — wire pipeline (the qualified gate lands here)

| item | action | notes |
|---|---|---|
| `config.py` (DetectorConfig) | KEEP + ADD | ADD `removal='gate'` selector; `gate_kgate=3.0`, `gate_ksig=3.0`, `gate_npass=2` (per-pass list allowed); R1 fields (`mask_threshold_nsigma`/`temporal_dilation_ticks`/`n_passes`) become removal='multipass'-only. group_size stays shared. |
| `coherent.py` (R1 multipass) | KEEP + rewire | classic image-space multipass = `removal='multipass'`. Its `nuf/gs` α is validated-good (not a bug). torch stays NotImplemented (documented; gate is the torch path). |
| `coherent_ops_{numpy,jax}.py` | KEEP | R1 group primitives. |
| `coherent_gate.py` | **ADD (PROMOTE)** | THE qualified smart gate, 2-pass, k=3, A-parity, `gate_approx` explicit, NaN-guard. Ported from `research/.../measure_coeffs.smart_gate_bands` + the multipass logic from `research/r2_qualification/multipass.py`. numpy + torch backends (bands→bands). NOT in the training closure → safe to build here now. |
| `pipeline.py` (process_plane/process_event) | KEEP + REWRITE | `process_plane(image, config, removal=…) → CoeffSet`. Branch the ORDER on removal: multipass acts pre-DWT (`remove→wavedec→threshold`); **gate acts post-DWT/pre-threshold** (`wavedec→gate→threshold`, no image round-trip); none skips removal. Threshold σ computed AFTER gate. ADD optional `clean_image=` to emit paired `val_clean` (mode-1 training). |
| `io.py` | KEEP + ADD | sensor reader REVIVED (done, 28d7795). ADD `read_processed()` to pair `write_processed()`; extend the coeff schema to store everything `reconstruct`/decode needs (n_time, pad, mode, full ThresholdSpec, removal+params, basis digest, schema_version). This closes write∘read=identity. |
| `run.py` (CLI) | KEEP + ADD | `helix-tpc`: ADD `--removal gate|multipass|none`, `--to-coeffs` (mode-2 convert writing CoeffSet files), multi-file/run iteration. Fix `--backend torch` (route→gate or clear error) and `--coh-only` (currently writes nothing). |

### C. `helix/optical/` — PMT pipeline

| item | action | notes |
|---|---|---|
| `config,io,pipeline,metrics,viz` | KEEP | chunk-based, removal='none', per-chunk σ, 12-bit quant. |
| optical `write_processed`/`read_processed` | ADD (DEFERRED) | NO optical coeff file has ever been written (the 33× is a computed metric). Add only when optical mode-2/3 is actually needed; requires CoeffSet to support variable-length signals + a `quant` field. |

### D. helix root / packaging

| item | action | notes |
|---|---|---|
| `helix/{coherent,config,io,pipeline,run,wavelet,_*}.py` shims | KEEP (for now) | back-compat re-exports; `tests/conftest.py` imports through them. Retire only after migrating tests. |
| `pyproject.toml` version 0.1.0 vs `__init__` 0.2.0 | FIX | sync to 0.2.0. |
| `README.md` | REWRITE | pre-restructure; describe 3 layers + backends + 3 modes + removal family (gate default). |
| `CLAUDE.md` | UPDATE | pipeline description → removal family; "coherent has no torch backend" → gate is the torch path. |

### E. `research/` — STAYS (evidence & training; NOT promoted)

| item | disposition |
|---|---|
| `coeff_foundation_model/fm/{train,mae_ddp,model,model_serial,data,cache*}` | STAY — model/training, not DSP. FREEZE. |
| `star_tpc.py`, `measure_coeffs.py` | THIN (after freeze lifts): import `helix.tpc.coherent_gate` instead of the inline copy; keep the extraction/orchestration glue. FREEZE. |
| tokenizer (`vit_tpc.assemble_tpc_band`, `star_tpc.rows_to_struct`) | STAY (→ pimm, §G) — patch/slot layout is model-side, still churning. NOT helix. |
| `coherent_coeffs/`, `wire_denoise/` (frozen studies + RESULTS.md) | STAY — historical record; STATUS banners, no move. |
| `r2_qualification/` (this session's harnesses, REPORT, grids) | STAY — the qualifying evidence for the packaged gate. |
| one-off verify/plot scripts, prototypes (`star_model`, `vit_model`, …) | STAY, archive-tag; several are FREEZE (transitive imports — lazy-import them to shrink the freeze set). |

### F. `pimm-data` — belongs there, NOT helix

| item | disposition |
|---|---|
| noise injection (`dense_ops` torch coherent/incoherent, `noise.py`, `digitize`) | STAY in pimm-data — runs in torch-free DataLoader workers. The **colored-spectrum fix** (`series_spectrum` never passed → white corpus) lands here, separate from consolidation. |
| coeff-row cache as a first-class dataset (`CoeffTPCDataset`, `CoeffShardWriter`) | pimm-data (the earlier two-piece design) — the FM *loader*, distinct from the helix DSP consolidation. Raw values + versioned σ sidecar + stable identity. |
| geometry (`load_plane_registry`) | STAY pimm-data — helix reads it via path insert; `helix.tpc.io` now also reads `/config/num_wires` itself. |

### G. helix FM layers (tokenizer + model) — IN HELIX; training/eval → pimm

(Correction 2026-07-24 rev2: helix PROVIDES the model + representation; pimm
TRAINS + EVALS it.)
| item | → home |
|---|---|
| tokenizer (`vit_tpc.assemble`, `rows_to_struct`, patch/asinh/tree) | **helix/tokenize/** |
| FM MAE model + forward + loss + param_groups + encode + mask | **helix/model/** |
| training loop (`fm/train`, `mae_ddp`, muP, schedule, DDP) | **pimm/engines** (FMTrainer) |
| probes + eval harness + deconv | **pimm/eval** (consume helix `model.encode`) |
See `RESEARCH_EXTRACTION_MAP.md` §2b (helix shape) and §5 (the helix↔pimm interface:
what helix exposes and what pimm must add to train/eval).

### H. `pimm` (particle-imaging-models) — trains/evals the helix model + baselines

pimm owns the TRAINING LOOP and EVAL for the helix FM (FMTrainer, eval hooks — §5b)
AND keeps its own point-cloud model zoo (PT-v3, sonata, polarmae) as the
SPINE/PoLAr-MAE baseline side. pimm imports helix (the model) + pimm-data (data).

---

## 2. New modules to create (signatures)

```python
# helix/core/wavelet.py  (public transform + threshold seam)
def wavedec(x, *, wavelet="coif3", level=4, mode="periodization") -> list   # bands
def waverec(bands, *, wavelet, mode="periodization", n_time=None)
def threshold_bands(bands, th: ThresholdSpec, sigma=None) -> (bands, n_kept)

# helix/tpc/coherent_gate.py  (the qualified R2, bands→bands, backend-dispatched)
def gate_bands(bands, *, group_size=64, kgate=3.0, ksig=3.0, npass=2,
               gate_approx=True) -> bands
#   npass>1: detect signal on cleaned bands → refine mask → re-estimate → re-gate.
#   kgate may be a scalar or per-pass list. numpy + torch backends.

# helix/tpc/pipeline.py  (the front-end)
@dataclass
class CoeffSet:                       # §4
    ...
def process_plane(image, config, *, removal="gate", clean_image=None) -> CoeffSet
def process_event(planes, config, *, removal="gate") -> dict[str, CoeffSet]

# helix/tpc/io.py  (close the loop)
def write_processed(path, event_idx, coeffsets: dict, config)   # extend schema
def read_processed(path, event_idx) -> dict[str, CoeffSet]      # NEW; write∘read=id

# helix/core/provenance.py
def basis_descriptor(**kw) -> dict ;  def descriptor_digest(d) -> str
```

## 3. CoeffSet & on-disk schema (the superset that makes the 3 modes interchangeable)

`CoeffSet` must carry everything BOTH existing formats need (`write_processed`
band-COO ∪ FM rows `band/gid/wire/idx/val`) so mode-1 output == mode-3 read:

- coeffs (per-band list) + **band_lengths** + **pad** + **n_ticks_raw** + **mode**
  (pin the padded-4336 convention — what the model & caches use; else `idx=wire*Lb+τ`
  decodes wrong).
- `n_wires`, plane label / volume / gid.
- `wavelet`, `level`; full `ThresholdSpec`; removal spec (`kind`, kgate, ksig, npass,
  group_size).
- **`sigma_threshold`** (per-band, per-event — belongs in CoeffSet).
- **NOT** the cross-event σ normalization table (2-cal-event, dataset-level) → a
  separate `NormalizationTable` owned by the dataset/tokenizer, NOT CoeffSet.
- optional paired `val_clean` (mode-1 training); `val_charge` (deconv) as a sidecar.
- basis digest (provenance), `schema_version`, event identity (run/file-tag/event).
- **write∘read=identity** is the acceptance test; build it first.

## 4. Freeze constraints & merge boundary

- **Safe to build in this worktree NOW:** everything in `helix/tpc/` (coherent_gate,
  pipeline, io, config, run) — the training does NOT import helix.tpc.
- **Worktree + merge-at-boundary:** `helix/core/wavelet.py`, `wavelet_ops_torch.py`
  (training imports these). Keep changes ADDITIVE (new public wavedec/waverec,
  new threshold_bands) so behavior of `_wavedec` is unchanged — then even the merge
  is low-risk. The A-parity `sigc` change lives in `coherent_gate` (new file), not
  in the training path.
- **research THIN-outs** (`star_tpc`/`measure_coeffs` → import `helix.tpc`): FREEZE —
  do in worktree, merge only when training done (dscale600b4_20k ~ step 990k/1200k).
- **Golden gate:** `research/goldens/capture.py --check` before every merge to main.
  Two golden regimes after packaging: the k4 corpus-faithful (star_tpc, unchanged)
  and the new package gate (k3, 2-pass, A-parity) — keep both.

## 5. Ordered work plan (each step golden-checked; all in this worktree)

1. **CoeffSet + write/read + identity test** (helix/tpc/io.py + wavelet.py). The
   acceptance test for the whole architecture. — *check: read∘write==id, 3 backends.*
2. **Public `wavedec`/`waverec` + `threshold_bands`** in helix/core (additive). —
   *check: sparsify unchanged (golden); pywt parity.*
3. **`coherent_gate.py`** (promote + multipass + A-parity + gate_approx + NaN guard),
   numpy+torch, parity test vs the research reference. — *check: matches
   measure_coeffs on 1-pass; 2-pass reproduces grid numbers.*
4. **`process_plane(removal=…)`** rewrite with per-removal ordering + clean_image;
   config fields. — *check: gate path == step-3 result end-to-end; multipass unchanged.*
5. **CLI** `--removal`, `--to-coeffs`, multi-file; fix torch/coh-only. — *check: CLI
   convert→read round-trips a real shard.*
6. **Docs**: README/CLAUDE/version sync; STATUS banners on frozen research dirs.
7. **(freeze boundary)** THIN `star_tpc`/`measure_coeffs` to import `helix.tpc`;
   recapture goldens; merge `consolidation → main`.
8. **Deferred/separate tracks** (not this consolidation): optical write/read; pywt
   `filters.py`; pimm-data colored-spectrum fix + CoeffTPCDataset; pimm tokenizer/
   trainer migration.

Everything in §1's STAY/FREEZE columns is deliberately NOT moved into helix — the
library is DSP only; training, tokenization, noise injection, and the FM loader
live in research/pimm-data/pimm respectively.
