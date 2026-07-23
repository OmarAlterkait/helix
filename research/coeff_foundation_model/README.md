# coeff_foundation_model — LArTPC wavelet-coefficient foundation model

Design handoff + all the data-measurement / profiling tooling that produced its
numbers. **`DESIGN_HANDOFF.md` is the primary document** — read it first.

## Contents

| file | what |
|---|---|
| `DESIGN_HANDOFF.md` | the design handoff (architecture, data measurements, cost model, open points) |
| `cross_level_operator_analysis.md` | unified TPC/optical pre-attention design: per-level ops, tree V-cycle cross-level operator, aggregation, skip/MLP rules, test ladder; §9 = measured optical tree statistics |
| `blocks_and_assumptions.md` | the step-back audit: every block, its decisions, its assumptions (ranked by risk), and how each block uses the scale axis |
| `mechanism_tests.md` | the variant spaces + test battery: weight-structure ladder W0–W7 (what levels represent, physical-scale conditioning), cross-scale topology menu C0–C4 (many-hops analysis), tests T-A/B/C/D with decision rules; doraemon optical corpus facts (20k noise-free events). **Test battery superseded by `reviews/review_synthesis.md`** |
| `reviews/` | skeptic + simplification agent reviews of the plan (verbatim) + `review_synthesis.md` — convergent fixes (kill CKA/grad-conflict tests, one ceiling model + star design, value-level twins in bits, doraemon loader = critical path), the big catches (tree-lift envelope confound, orphan/support-hole reframing of non-Markov, dense-conv category error, contaminated noise-chunk class, bit-count over-provisioning) |
| `EXECUTION_PLAN.md` | **the operative plan** (M0–M5 + global protocol + settled user decisions) |
| `DECISIONS.md` | the decision log: D-01..D-07 with the numbers that made them; registered predictions P1–P3 |
| `doraemon_optical.py` | loader for the doraemon label_N optical set (per-interaction chunks + truth) |
| `measure_coeffs_doraemon.py` | M0 clean-support tree statistics (nominal σ=2.6) → `artifacts/optical_tree_stats_doraemon.json` + `typical_event_coeffs_doraemon.npz` |
| `info_audit.py` | M2 information audit in bits: exact activity-CMI with envelope controls, value-level copula (C)MI incl. lateral control, orphan census, group-delay probe, anchor sweep, registered bit-count predictions → `artifacts/info_audit.json` |
| `microbench_treeops.py` | M1 cost microbench: topology gathers (C0–C3), per-band dense-vs-gather, batched stems → `artifacts/microbench_treeops.json` |
| `compute_budget.py` | token budget → trunk cost → tokenizer overhead, pure arithmetic on measured anchors → `artifacts/compute_budget.json` (joint 31k tokens at 8×4+1024; ViT-L MAE step ~208 ms ≈ 17k GPU-h/30ep; tokenizer 19–28% of step) |
| `dump_coeff_dataset.py`, `star_model.py` | M3: 300-event clean-support coefficient dataset + the star-design substrate (arms = topology masks × weight ties on one point-set family) → `artifacts/star_results.jsonl` |
| `model_scoping.md` | earlier scoping: data path, pimm-data dataloader, what the model needs (§5b dataloader, §5c first coeff prototype) |
| `measure_coeffs.py` | **Items 0–5 + timing** measurement script. GPU pipeline: extract → densify → GPU coherent+intrinsic noise → coif3-L4 DWT → coeff-space smart removal (kgate=4) → per-band-σ threshold. `--scan N --kappa K`. Holds the reusable helpers (`load_geom`, `build_batch`, `event_coeffs`, `smart_gate_bands`, `prod_threshold`) the other scripts import. |
| `measure_coeffs_optical.py` | **optical twin** of measure_coeffs (CPU, production-faithful κ=1.2): per-band survival, full-depth tree lift + strict Markov test, alignment shift sweep, value stats, coarse-column occupancy → `artifacts/optical_tree_stats.json` + `artifacts/typical_event_coeffs_optical.npz`; results in `cross_level_operator_analysis.md` §9 |
| `patch_sweep.py` | single-event patch sweep + cathode continuity on the dumped npz (pure recompute, no GPU) |
| `patch_sweep_multi.py` | patch sweep + cathode continuity aggregated over N events (`--events N`); produced §3.3/3.4/3.6/3.7 |
| `coeff_prototype.py` | first prototype: materialize per-level `{gid:{band:(B,W,len)}}` coeff tensors for clean+noisy |
| `viz_2x2_coeff.py` | 2×2 panels (clean \| noisy \| coeff-space smart-removed kgate=4 \| diff), **full plane** → `figures/full_*.png` |
| `viz_2x2_jaxtpc.py` | 2×2 panels using the **sample-space** `helix.tpc.remove_coherent` (the earlier variant) → `figures/panel_*.png` |
| `profile_load.py` | GPU load-path profiler: per-stage time vs batch size (densify/coherent/incoherent/digitize/dwt+thr) |
| `artifacts/typical_event_coeffs_smart.npz` | the reference dumped event (ev33, smart-removed, kgate=4) — recompute Items 3–5 at any patch size |
| `artifacts/typical_event_coeffs.npz` | earlier dump (pre-threshold clean+noisy prototype) |
| `figures/` | the 2×2 panels (full-plane coeff-space `full_*`; sample-space `panel_*`) |

## Environment (read-only pimm container)

The pimm container lacks `pywt` / `hdf5plugin` / `pytest` and its site is read-only;
they were installed to `helix/.pylibs`. Every script wires this in its `__main__`:

```python
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")                       # pywt, hdf5plugin, pytest
sys.path.insert(0, "/sdf/group/neutrino/omara/particle-imaging-models/libs/pimm-data/src")  # pimm_data (extract only)
sys.path.insert(0, "/sdf/group/neutrino/omara/helix")                               # helix
import hdf5plugin   # MUST precede any h5py read (blosc/zstd codecs for the compressed HDF5)
```

Scripts that import `measure_coeffs` rely on running from this folder (its dir is on
`sys.path[0]`), e.g. `python research/coeff_foundation_model/patch_sweep_multi.py --events 200`.

## Data / pipeline facts

- Data: `/sdf/data/neutrino/omara/JAXTPC_Wire/test_00_00_02`, split `run_0027575766`,
  `dataset_name="sim_wire"`, modality `sensor` (19999 events; **stored sensor = clean truth**, noise added at load).
- Geometry: `pimm-data/.../data/cubic_wireplane_geometry.json` (U/V 1969 wires ped 1843, Y 1443 ped 410, n_ticks 4321).
- Coherent removal: `research/coherent_coeffs/smart.py::smart_removal` (default `kgate=4.0`); GPU twin fused in DWT space is `measure_coeffs.smart_gate_bands`.
- Coherent noise generation runs on GPU (`pimm_data/dense_ops.py::_coherent_torch`, default in `add_intrinsic_noise`); `coherent_numpy=True` recovers the bit-exact numpy oracle.

## Headline numbers (post-smart-removal, see DESIGN_HANDOFF §3)
- Token budget: 8×4 → ~25.4k/event (6 planes), event-independent (±10%); 16×8 → ~11.6k.
- Wavelet-tree lift: P(D3|D4)/P(D3) ≈ 31×, P(D2|D4)/P(D2) ≈ 29× → structured tree-coupling justified.
- Cathode continuity: direct 0.253 > mirrored 0.228 ≈ chance 0.219 → cross-volume alignment is DIRECT (same tick).
