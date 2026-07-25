# CoeffEvent & the coeff corpus — object model, on-disk schema, naming

Design for the wavelet-coefficient object and its on-disk corpus, grounded in
pimm-data's existing HDF5/reader/Dataset conventions so the corpus reads as a
**natural pimm-data dataset**. Decisions locked with the user 2026-07-24:
rebuild (free design), event-scoped `CoeffEvent` in `helix/core`, provenance
**embedded in the h5** (not a sidecar), **flat-columnar shards** (not per-event
groups), **one modality per file** (noisy `coeff` + clean `coeff_clean` now, charge
deferred), corpus under **`/sdf/data/neutrino/omara/coeff_tpc/`**, optical-ready.
Everything here is designed to be pinned by a **round-trip identity test** (§7).

---

## 1. Kill "3 modes" — it's one transform + one codec

The old "mode 1/2/3" framing conflated a compute step with a read/write pair.
There are really **two operations**, and the three usage patterns are just
whether disk sits in the middle:

```
      compute (DSP)                     codec (lossless)
  raw ───────────────▶ CoeffEvent ◀───────────────▶  coeff shard (h5)
      process_event()             write / read

  A. compute-on-the-fly :  process_event → [tokenize] → model        (no disk)
  B. build corpus       :  process_event → write         (CoeffShardWriter)
  C. train from corpus  :  read → [tokenize] → model      (CoeffTPCDataset)
```

- **compute** = `process_event(raw, cfg, removal=…) → CoeffEvent` (per-plane worker
  `process_plane`). The DSP: removal → DWT → threshold. Shared by A and B.
- **codec** = a lossless `CoeffEvent ⇄ h5` pair. B and C are the two halves of the
  **same** codec — that is why naming them as separate "modes" felt incomplete.
- **tokenize** (`CoeffEvent → rows → tokens`) is a *later* stage (pimm), not a mode.

**The invariant becomes precise:** `read(write(cs)) == cs` (coeff+metadata
bit-identity) and `compute(raw)` equals that — i.e. **the corpus is provably a
cache of the compute path**, nothing more. (Old vague "write∘read=identity".)

Naming (adopted from pimm-data house style):

| concept | name | home |
|---|---|---|
| in-memory event object | `CoeffEvent` | `helix/core/coeff_event.py` |
| DSP compute | `process_event` / `process_plane` | `helix/tpc/pipeline.py` |
| pure (de)serialization | `coeff_event_to_arrays` / `arrays_to_coeff_event` | `helix/core/coeff_io.py` |
| reference shard codec + gate | `write_coeff_shard` / `read_coeff_event` | `helix/core/coeff_io.py` |
| basis/provenance | `basis_descriptor` / `descriptor_digest` | `helix/core/provenance.py` |
| production corpus writer | `write_coeff_shard` (function) | pimm-data `readers/coeff_tpc.py` |
| corpus reader (h5→flat dict) | `CoeffTPCReader(modality=…)` — one class, `modality` selects `'coeff'`/`'coeff_clean'`/(future) `'coeff_charge'` | pimm-data `readers/coeff_tpc.py` |
| corpus Dataset | `CoeffTPCDataset` (`type='CoeffTPCDataset'`, `modalities=('coeff','coeff_clean'?)`) | pimm-data `coeff.py` |

Corpus root: **`/sdf/data/neutrino/omara/coeff_tpc/<run>/`**. Flat-columnar shards,
one modality per file (§3).

(`write_processed`/`read_processed` in `helix/tpc/io.py` are **renamed+moved** to
`write_coeff_shard`/`read_coeff_event` in core — "processed" was vague and the object is
now detector-neutral. `tpc/io.py` keeps only *sensor* reading.)

---

## 2. `CoeffEvent` — the in-memory object (helix/core, detector-neutral)

Event-scoped, supersedes/absorbs `SparseResult` (already shared by tpc+optical).
**Flat columnar** — mirrors the on-disk shard and the model's flat-row consumption
(`plane_gid` a column, no per-plane sub-objects). Detector-neutral: TPC uses the
grid `(band,wire,tau)` coords; optical (later) uses an offset-packed chunk variant
(§5). As built (`helix/core/coeff_event.py`):

```python
@dataclass
class CoeffEvent:                       # one EVENT, all planes — flat sparse rows
    # --- flat coeff rows (n = total kept coeffs) ---
    band: np.ndarray            # uint8   (n,)  index into [cA, cD_L, …, cD_1]
    plane_gid: np.ndarray       # uint8   (n,)  canonical plane id (v*3 + {U,V,Y})
    wire: np.ndarray            # int32   (n,)  signal/row index within the plane
    tau: np.ndarray             # int32   (n,)  within-band coeff index
    value: np.ndarray           # float32 (n,)  RAW coeff (un-normalized)
    # --- per-plane bookkeeping (indexed by position in `gids`) ---
    gids: np.ndarray            # int32   (G,)  plane set (shard-uniform)
    n_wires: np.ndarray         # int32   (G,)  signals per plane (for reconstruct)
    sigma_threshold: np.ndarray  # float32 (G, n_bands)  per-(gid,band) threshold σ
    # --- provenance + identity ---
    basis: BasisDescriptor      # wavelet/level/mode/pad/band_lengths/removal/threshold/sigma_norm
    run: str; source_file: str; event: int
    # constructors: CoeffEvent.from_sparse_results({gid: SparseResult}, basis=…)
    #               ce.reconstruct_images(n_time) -> {gid: image}
    # targets (clean/charge) are NOT fields — separate modality files (§4b); compute
    # returns e.g. process_event(raw, clean_image=…) -> {'coeff': ce, 'coeff_clean': ce_clean}
```

`BasisDescriptor` (`helix/core/provenance.py`) carries `band_lengths` + the basis
that generates them; `.validate()` fails loudly on drift, `.digest()` is the
`/config` join key.

Key points from the audit, baked in:
- **RAW values** (decision: rebuild). Normalization (`SIGMA/σ_tab`) applied at
  **tokenize**, not stored. `value_clean`/`value_charge` also raw, treated
  identically (both optional target columns, both scaled downstream).
- **`sigma_threshold`** (per-event, defines the kept set) is *distinct* from the
  cross-event **normalization σ table** (§4, lives in `/config`). Never conflate.
- **Coordinate encoding = `(band, wire, tau)`** — self-describing, no packed `idx`.
  The model's `idx = wire*band_lengths[band] + tau` is derived at tokenize.
- **`band_lengths` is mandatory and load-bearing** — carried on the object and in
  `/config`, validated against `(n_ticks_raw, pad, wavelet, level, mode)` on read.

---

## 3. On-disk schema — flat columnar, one modality per file

Corpus root: **`/sdf/data/neutrino/omara/coeff_tpc/<run>/`** (84 T data volume, off
the near-full group volume). Shards are `{dataset}_<modality>_{NNNN}.h5`, ~N
events/shard (a builder knob).

**Two deliberate departures from the raw sensor shards, per user direction:**

1. **Flat columnar, NOT per-event groups.** A coeff shard is read *hot in the
   training loop*, so per-event HDF5 group traversal is pure overhead. Instead,
   concatenate every event's coeff rows into **shard-wide arrays** + an
   `event_offset` index (this is pimm-data's own offset idiom — optical's
   `adc`/`offsets` — applied shard-wide; it is also exactly the FM cache's proven
   flat-row shape). `plane_gid` becomes a **column**, not a group path. One
   contiguous slice per event; no `event_NNN` groups.
2. **One modality per file.** The noisy input, the clean target, and (future) the
   charge target live in **separate files/modalities**, joined by identity at read
   (`labl ↦ sensor` pattern) — never mixed in one h5.

### `coeff` modality — the noisy input (`{dataset}_coeff_{NNNN}.h5`)

```
├── /config                                  ← written ONCE per shard (shared)
│     attrs (scalars):
│       n_events, dataset_name, file_index, global_event_offset   # house identity
│       readout_type = 'wire'
│       # basis / provenance  (EMBEDDED — /config is per-file, no per-event dup)
│       wavelet='coif3', dwt_level=4, dwt_mode='periodization'
│       n_ticks_raw=4321, pad=15                     # → padded length 4336
│       removal_kind='gate', gate_kgate=3.0, gate_ksig=3.0, gate_npass=2, group_size=64
│       threshold_func='visushrink', threshold_kappa=1.0, per_band_sigma=True, threshold_approx=False
│       sigma_norm=2.6                               # SIGMA (tokenize normalization)
│       basis_digest='<sha256>'                      # provenance.descriptor_digest
│       production_version, run_id, batch_timestamp, git_*   # house provenance (NO schema_version)
│     datasets (shared tables):
│       band_lengths   (n_bands,)          int32     # PADDED lengths [271,271,542,1084,2168]
│       gids           (G,)                int32     # the plane set (row order of norm_sigma)
│       n_wires        (G,)                int32     # signals per plane, indexed by gid position
│       norm_sigma     (G, n_bands)        float32   # the NormalizationTable (frozen; row = gid POSITION)
│
├── /coord                                   ← shard-wide concatenated coords (M = Σ n_coeff)
│     band        (M,) uint8                 # 0..n_bands-1
│     plane_gid   (M,) uint8                 # v*3 + {U:0,V:1,Y:2}  (a COLUMN now)
│     wire        (M,) int32
│     tau         (M,) int32                 # within-band coeff index
│     event_offset (n_events+1,) int64       # event boundaries; slice [off[i]:off[i+1]]
│     sigma_threshold (n_events, n_gid, n_bands) float32   # per-event per-(gid,band)
│
└── /value        (M,) float32               ← RAW noisy coeff
```

Coords stored as plain int columns (blosc-zstd handles the redundancy; sorted
`(event, plane_gid, band, wire, tau)` for compression). `idx = wire*band_lengths[band]
+ tau` is derived at **tokenize**, not stored.

### `coeff_clean` modality — the clean target (`{dataset}_coeff_clean_{NNNN}.h5`)

**Separate file** (built in the same `process_event(clean_image=…)` pass, written to
its own shard). Self-describing — its own `/config` + `/coord` + `/value` — so it is
independently valid and may carry the **clean signal's own support** (the coeffs the
noisy pass missed), not just the noisy support. Joined onto `coeff` at read by
identity `(run, tag, event)`; the tokenizer aligns clean↔noisy by the `(band,
plane_gid, wire, tau)` key (gather clean at each noisy coord, 0 if absent) — robust,
positional-coupling-free. `CoeffTPCDataset(modalities=('coeff','coeff_clean'))` opts
it in; MAE-only pretraining uses just `('coeff',)`.

### `coeff_charge` modality — DEFERRED (future)

Not built now. When deconvolution work needs it, it lands as another separate
modality (`{dataset}_coeff_charge_{NNNN}.h5`), same joined-by-identity pattern
(different producer: `hits` truth via pywt DWT). Nothing in the layout blocks it.

Notes:
- **`gid` is a column** (`plane_gid`), matching the FM cache — no per-event group
  path to walk.
- **delta encoding dropped** in the flat layout — plain int columns compress well
  under blosc-zstd and avoid per-event cumsum bookkeeping. (Delta+cumsum was the
  per-event-group idiom; unnecessary here.)
- **zero bands are representable** — a band simply has no rows; `band_lengths` in
  `/config` tells the decoder every band's shape regardless.
- **noisy vs clean are separate files** — cleanly separated, each self-describing.

---

## 4. The two σ's, and who owns the NormalizationTable

- **`sigma_threshold`** — per-event, per-band, defines the VisuShrink kept set.
  Lives in `/coord` as a `(n_events, n_gid, n_bands)` table. In CoeffEvent.
- **`norm_sigma`** — the cross-event 2-cal-event normalization table, per `(gid,
  band)`, frozen for the corpus. **Embedded in `/config`** as a dataset (it's
  shared-per-shard, like `num_wires`). Written by `CoeffShardWriter`, computed once
  by the corpus builder. `SIGMA=2.6` is a `/config` attr and also pinned in helix
  `PatchConfig`. Tokenize does `arcsinh(value * (SIGMA/norm_sigma[gid,band]) / SIGMA)`
  = `arcsinh(value / norm_sigma[gid,band])` — normalization lives entirely at
  tokenize, corpus stores raw.

This is the answer to "who owns NormalizationTable": **embedded in the shard
`/config`, written by pimm-data's corpus writer, computed by the builder** — no
separate sidecar file, consistent with "embed everything in the h5."

---

## 4b. Targets — every target is a separate modality/file, joined by identity

**Decision (user): noisy and clean are NOT in the same h5.** Every target is its own
self-describing modality file, joined onto `coeff` at read by identity `(run, tag,
event)` — uniform with how `charge` (and `labl ↦ sensor`) work. This makes targets
optional, objective-specific, and **independently (re)buildable** without touching
the noisy shards.

| modality | file | producer | status |
|---|---|---|---|
| `coeff` | `{ds}_coeff_{NNNN}.h5` | noisy pass (removal→DWT→threshold) | **now** — the input |
| `coeff_clean` | `{ds}_coeff_clean_{NNNN}.h5` | clean-only pass, same `process_event(clean_image=)` | **now** — denoising target |
| `coeff_charge` | `{ds}_coeff_charge_{NNNN}.h5` | `hits` truth via pywt DWT | **deferred** (future) |

- **Alignment at read** is by the `(band, plane_gid, wire, tau)` **key**, not by
  position: the tokenizer gathers each target's value at every noisy coord (0 if
  absent). So a target may have its **own support** (e.g. clean coeffs the noisy
  pass missed) — the separate self-describing file makes that natural, and there is
  no fragile positional coupling.
- **Objective selects modalities:** MAE pretraining = `modalities=('coeff',)`;
  denoising = `('coeff','coeff_clean')`; deconvolution (future) adds `coeff_charge`.
- **Role tag** `('target', <name>)` — DEFERRED to the tokenizer stage (not yet in
  `CoeffTPCDataset`, which currently returns plain rows). Intended so it
  distinguishes
  targets from the input in `_roles` so tokenize/loss select correctly.
- **Extensible:** a new target = a new modality file. No schema change, no rewrite of
  existing shards.

## 5. Optical readiness (future) — the second-row-space path

Optical coeffs are variable-**length** chunks, not a wire×tick grid. They map onto
pimm-data's existing **second row-space / offset machinery** (the same mechanism
optical waveforms already use), so `CoeffEvent` + the codec absorb optical as a
*field-subset*, not a fork:

- `PlaneCoeffs` → a `ChannelCoeffs` variant whose coords are `(band, chunk)` with a
  per-chunk **`length`** and packed flat `value` (ΣL,) instead of the grid COO.
- On disk: store the payload as one flat concatenated array + CSR `offsets`
  (leading-0 cumsum), exactly like optical `adc`/`offsets`.
- Role: the dataset stamps `'_roles': {'value': ('instance', 'coeff_wave_offset')}`;
  `Collect(offset_keys_dict=dict(offset='chunk_id', wave_offset='value'))` emits both
  offsets; `collate` cumsums each independently; `split_event` slices the payload by
  its own span. Round-trip identity guaranteed by the existing machinery.
- `/config` for optical carries the optical basis + `quant` field instead of
  `num_wires`.

So the detector-neutral `CoeffEvent` in core carries **either** a grid-COO plane spec
(TPC) **or** an offset-packed chunk spec (optical). No optical file is written until
we need it (deferred), but nothing in the TPC schema blocks it.

---

## 6. One contract, two implementations, no import cycle

**pimm-data must not import helix at read time** (else the build-time
`helix → pimm-data` dep becomes a cycle). Resolved the same way every pimm-data
reader already works — standalone h5py:

- **helix** defines the schema-of-record: `coeff_event_to_arrays`/`arrays_to_coeff_event`
  (pure `CoeffEvent ⇄ dict-of-arrays+attrs`) + the reference single-file codec
  `write_coeff_shard`/`read_coeff_event`. This is the **acceptance gate** (§7).
- **pimm-data** implements the *same on-disk contract* independently and
  torch/helix-free: `CoeffTPCReader` (h5 → flat dotted dict, per the reader
  protocol) + `CoeffShardWriter` (build many events/shard + identity + target
  columns + `/config`). It never imports helix; it reads the documented layout with
  h5py like `jaxtpc_sensor.py` does.
- The two are kept in sync by a **cross-repo golden test**: a shard written by
  helix's `write_coeff_shard` must read identically through pimm-data's
  `CoeffTPCReader`, and vice-versa. The layout is documented in pimm-data
  `docs/SCHEMA-on-disk.md` (append a Coeff section) — the prose contract.

Dependency graph unchanged: `pimm → helix + pimm-data`; `helix → pimm-data`
(build only); **pimm-data ⊥ helix** (read time). No cycle.

---

## 7. The round-trip identity test (built first — the acceptance gate)

Two levels, both bit-identical:

1. **helix codec** (`helix/core`): for real shards and synthetic edge cases
   (empty planes, all-zero bands, single coeff, full plane):
   `read_coeff_event(write_coeff_shard(cs)) == cs` on every field (coeff arrays +
   metadata), **float32 end-to-end**; and a decode check
   `reconstruct(read(write(cs))) == reconstruct(cs)` (proves stored metadata
   suffices to decode — fails loudly if `mode`/`n_ticks`/`band_lengths` missing).
2. **pimm-data pipeline** (mirrors the optical round-trip test):
   `read_event (slice by event_offset) → get_data → Collect → collate_with_roles →
   split_event` returns the per-event arrays bit-identically; and the clean modality
   joins onto coeff by identity with correct key-alignment.
3. **cross-repo**: helix-written shard ↔ pimm-data-read (and reverse) agree.

Build order (per CONSOLIDATION_PLAN §5, revised): **schema + codec + identity test
FIRST**, before `process_plane`/gate rewiring, so the whole architecture has a
green acceptance gate from step 1.

---

## 8. Resolved choices (user, 2026-07-24)

- **O1. Shard granularity → many events/shard** (builder knob). Kills the
  200k-inode bomb.
- **O2. Modality name → new modality `'coeff'`** (+ `'coeff_clean'`, later
  `'coeff_charge'`); add entries to `VALID_MODALITIES`.
- **O3. Layout → flat columnar**, NOT per-event groups: shard-wide `/coord` +
  `/value` + `event_offset`; `plane_gid` a column. Fastest hot-loop read; matches
  the FM cache's flat-row shape and pimm-data's offset idiom.
- **O4. Targets → each a separate modality/file**, joined by identity (§4b): `clean`
  = `coeff_clean` shard **now**; `charge` = `coeff_charge` **deferred to future**.
  Noisy and clean are never in the same h5.
- **Corpus root** → `/sdf/data/neutrino/omara/coeff_tpc/<run>/`.
- **Charge** → deferred (future), alongside optical.
- **O5. Bands → KEEP ALL `level+1`** (incl. D1), unlike the old build which dropped
  D1 at the cache level. Rationale (audit-quantified): after per-band VisuShrink only
  ~0.009% of D1's noise slots survive, so D1 adds only ~1–2k kept coeffs/event
  (~1–3%) — the "D1 = 50% of padded slots" figure is a red herring (storage counts
  *kept sparse* coeffs, not slots). Keeping D1 is nearly free AND makes the corpus
  reconstructable (the old 4-band cache could not `waverec`). The model/tokenizer
  selects which bands to use.

**Audit (5+5 agents, 2026-07-24):** all confirmed defects fixed + regression-tested
(plane_gid uint8→int32; coeff glob no longer matches coeff_clean; coeff↔coeff_clean
join by identity not position; multipass padding; cal_events guard; clean-gather +
sigma + digest + writer validation). Confirmed-correct: numpy≡torch DWT, gate port,
byte-parity, freeze-safety. Intentional-and-correct: k3/2pass (qualified), A-parity
quantile, raw+norm_sigma.
