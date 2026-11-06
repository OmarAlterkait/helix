# Handoff: the consolidated line, for continuing at NERSC (2026-11-06)

Who this is for: the session that continues the work (helix-nersc), on the data of this
branch once it is transferred. It states what exists, how it fits together, what was
assumed, and which choices are still open. Procedures are in `docs/RUNBOOK.md`, module
shapes and formats in `docs/ARCHITECTURE.md`, the standing decisions and their evidence in
`docs/DECISIONS.md`; this file does not repeat them, it points at them.

---

## 1. State at handoff

| | |
|---|---|
| branch | `consolidated`, tip 8e7b3eb + this file; origin/main (039f3ee) + 50 commits; merge-base = origin/main |
| tests | 540 passed, 9 skipped, 1 xfailed (S3DF, 2026-11-04, with pimm and data) |
| corpus | `coeff_tpc_d1m/j000000`: 1000 `helix-coeff-2` shards x 200 = 200k events + 1000 `helix-cells-3` truth shards, 480 GB |
| basis_digest | `0dd4f3a223909fd7d8150f54854111ce6e7c89f030227a20113aabf2a27c9457` (built at 1317c04, RTX 2080 Ti) |
| bin grid | `coeff_bins_d1m_j000000_noisy` (K = 128, 120 events of j000000; digests in `helix/data/data/reference_bins.json`) |
| runs | `$HELIX_EXP/fm` (stable, 115,488 steps = 3 epochs, 4 A100, LR 2.2e-3), `$HELIX_EXP/fm_cooldown` (9,386 steps) |
| eval | `eval2_last.json` in each run dir, from `helix.eval` at 8e7b3eb, `--weights last`, 4113 probe events |

The other four campaign groups (j001000-j004000) are **not built**. No clean shards exist
(`--keep-clean` was off). Nothing optical is in this corpus (§4.5).

### Results (fm_cooldown; [16, 84]% event-bootstrap intervals in the JSON)

| metric | reference (no fit) | fm (stable) | fm_cooldown |
|---|---|---|---|
| floor eff @1% FPR/view, < 0.1 MeV | amplitude 0.08 | 0.35 | 0.40 |
| floor, 0.1-0.2 MeV | 0.18 | 0.55 | 0.62 |
| floor, 0.2-0.5 MeV | 0.84 | 0.95 | 0.97 |
| floor, > 0.5 MeV | 0.99 | 0.99 | 0.99 |
| uprobe within-group R^2 (U / V / Y) | geometry 0.03 | 0.52 | 0.53 (0.56 / 0.56 / 0.49) |
| same, Fisher-mean r (checks.py, 600 events) | 0.15 | -- | 0.75 |
| labels motif, balanced accuracy | token layout 0.29 | 0.80 | 0.80 |
| labels pid, balanced accuracy | 0.24 | 0.65 | 0.65 |
| recon var_expl | -- | 0.645 | 0.693 |

pid recall per class (cooldown): electron 0.40, photon 0.48, muon 0.78, proton 0.92.
Background windows per view: U 27,420, V 27,420, Y 26,539.

**What these numbers do not show** (by helix-nersc's measurements on r1): a single seed
moves the floor by 0.03-0.04 at d768, so stable vs cooldown below that is unresolved;
validation loss rank-correlates 0.15 with probes, so recon cannot pick between models;
nothing here compares pretrained against from-scratch (§5, C8).

---

## 2. Structure

```
 doraemon_2026_1M_lartpc  (JAXTPC wire stage: sensor, step, hits; pysupera labl)
        |  scripts/build_sim.py  (helix noise model + digitize, per-event seed)
        v
 helix.build + helix.tpc.TPCChain        DWT coif3 L4 -> coherent gate -> threshold -> quantize
        |  scripts/corpus.sh plan | calibrate | build       (turing, one GPU type per corpus)
        v
 <root>/j000000/sim_wire_coeff_NNNN.h5   noisy coefficients          helix-coeff-2
 <root>/j000000/sim_wire_cells_NNNN.h5   truth on the 2 x 16 grid    helix-cells-3  (scripts/build_truth.py)
 <root>/_calib/                          CHUNKS.txt, RUNS.txt, <run>.npy, norm_sigma_global.npy
        |  helix.data.CoeffTPCDataset (identity-hash split) -> helix.model.tokenize.CoeffTokenize
        v
 pimm.train + helix.integrations.pimm    configs/pimm/{fm, fm_cooldown, fm_smoke, denoise}.py
        |  scripts/train.sh CONFIG [key=value]   (sbatch chain, resumes)
        v
 $HELIX_EXP/<name>/  (pimm run dir)  ->  helix.model.checkpoint.load_run  ->  python -m helix.eval
```

| package | lines | role | used on the result path |
|---|---|---|---|
| `helix.core` | 1.3k | wavelet, robust stats, hysteresis, coeff shard IO, provenance, cadence | yes |
| `helix.tpc` | 1.5k | TPC chain, coherent gate, threshold, noise model, readers (sim, SBND) | yes |
| `helix.build` | 0.4k | the event loop / shard writer for every source | yes |
| `helix.data` | 2.8k | reader, dataset, split, bins, identity guard, truth (`cells.py`), denoiser labels | yes |
| `helix.model` | 2.2k | tokenizer, FM, masking, loss, muP, varlen attention, denoiser, `load_run` | yes |
| `helix.integrations.pimm` | 1.4k | registry adapters, `FMTrainer`, hooks, evaluators, schedulers | yes |
| `helix.eval` | 1.2k | recon, floor, uprobe, labels, denoise; shared probe / windows / events | yes |
| `helix.optical` | 1.4k | optical codec, response calibration, optical shards, tokens | no: not fed to the model |

Boundary: `helix.core` / `helix.tpc` never import pimm-data (`tests/test_boundary.py`).
pimm-data is pinned once in `pyproject.toml`; helix registers `AddNoise` / `Digitize`, so
the two move in lockstep. pimm is used only as the training engine (`python -m pimm.train`).

**Source of every number:** the eval JSON records the code commit, run, tokenizer, weights
and seed; the training run records `provenance.json`; every shard records its code commit,
dirty flag and `basis_digest`; `helix.data.identity` refuses a checkpoint whose
`basis_digest` differs from the corpus being read.

---

## 3. The data

### 3.1 The campaign (input)

`HELIX_SIM_ROOT` = `<campaign>/data/jaxtpc_wire` (S3DF: the mirror
`/sdf/data/neutrino/omara/doraemon_2026_1M_lartpc/data/jaxtpc_wire`, whose `sensor/`,
`step/`, `hits/` are symlinks into `/sdf/data/neutrino/doraemon/...` and whose `labl/` is
real). Layout `<group>/<task>/{sensor,step,hits,labl}/sim_wire_<kind>_<task>.h5`; 5 groups
`j00X000` x 200 tasks x 1000 events. `event_id` (0..999999) is the campaign's event and
helix's; `/ident` = (run = group, source_file = sensor file name, event = event_id). The
campaign also has `edepsim`, `jaxtpc_pixel`, `supera_wire`, `supera_pixel` stages, and **no
light simulation**.

`labl` (pimm-data's `supera_labl` output, pysupera) is present for all 5 groups x 200 tasks.

### 3.2 The corpus

* Basis: coif3, level 4, periodization, D1 dropped; coherent gate on 64-wire "block"
  groups, kgate = ksig = 3, npass 2, tau 0.05, sigc global; threshold kappa 1.0, floor 2.0,
  sigma per plane; values rounded to a power of two <= sigma/16 (input), sigma/32 (target).
  Same basis as r1 but a different chain: helix-nersc's r1 digest is `8c4542b6...`, so
  **no r1 number transfers**; re-measure.
* Noise is added once, at build time, from the event's identity (blake2b seed). Training
  reads that one realisation every epoch; `AddNoise` at load time exists but `fm.py` does
  not use it.
* Training drops events above `max_event_size = 600_000` rows (fm.py); the fraction dropped
  on d1m has not been measured.
* Split: identity hash, train / val / probe = 0.95 / 0.03 / 0.02 (j000000: 4113 probe events).

### 3.3 The truth (`helix-cells-3`, `helix/data/cells.py` docstring is the spec)

Per 2-wire x 16-tick cell: `q` = the hits file's electrons (pre-response: the charge reaching
the wires, before field response and electronics), built from hits independently of the
CORPUS threshold, but **not complete**: JAXTPC f0d2ed39 writes hits after its own cuts (each
hits entry records `threshold: 25.0`; per helix-nersc, samples <= 1 e- per group x wire x tick
are dropped in box compaction and <= 25 e- at the CSR encode, on 1ea2282). That drops every
deposit group's diffusion tails, ~1.6% of in-window charge, growing with drift (complete /
hits = 1.0155-1.0161 on r1). Fine for the particle labels, centroids and track splits; for
charge targets see C13. Also per cell: its
split by track, and the dominant hits group's share and 3D (y, z). Per (plane, track) the
pixel-charge centroid. Per track: deposited energy (step dE summed, MeV), `motif` (shower,
track, michel, delta, le) and `pid` (photon, electron, muon, pion, proton, le), -1 = none,
from labl by pimm-data's walk. ~197 MB / shard. >99% of charge is labelled.

Building truth needs the campaign's hits / step / labl, which exist only at S3DF.

### 3.4 Sizes for the transfer

| what | size | needed for |
|---|---|---|
| `coeff_tpc_d1m/j000000/*_coeff_*` + `_calib/` | ~283 GB | training |
| `coeff_tpc_d1m/j000000/*_cells_*` | ~197 GB | denoiser labels, every eval metric |
| bin table `coeff_bins_d1m_j000000_noisy.pt` | small | verify at NERSC with `derive_coeff_bins.py --verify` instead of copying (RUNBOOK §5) |
| `$HELIX_EXP/fm`, `fm_cooldown` | checkpoints | baselines to compare a NERSC rerun against |
| campaign labl / hits / step | large | only to (re)build truth at NERSC |

---

## 4. Assumptions

Each is either measured (with where) or assumed (with what would break it).

### 4.1 Simulation and corpus
1. **The noise model stands in for the detector** (MicroBooNE-spectrum ENC, coherent
   waveforms per 64-wire block). Assumed; it matters only for real data, which is out of
   scope for now.
2. **The five groups are one distribution**, so a grid and a `norm_sigma` from j000000
   describe all of them. Assumed for d1m (measured for r1's runs: 0.16 bin widths,
   DECISIONS "FREEZE the bin grid"). Breaks if a group's derived grid differs by much more.
3. **One GPU type per corpus.** Measured: turing and A100 differ on 0.016% of kept
   coefficients. j000000 was built on 2080 Ti.
4. **The value step is harmless.** Measured at dc6ae8b: added variance <= 1.6e-4 sigma^2,
   input moves <= 0.013 against bins 0.08-0.11 wide.
5. **One fixed noise realisation per event is enough.** Assumed; helix-nersc's repetition
   result (d768 +0.046 loss at 310 epochs, flat for d512 to 77) says 3 epochs are far from
   the problem.

### 4.2 Truth and labels
6. **hits = the charge a perfect detector would see** (pre-response), apart from JAXTPC's
   per-group 25 e- cut (§3.3), which removes ~1.6% of the charge as diffusion tails. The
   label of every metric here; helix-nersc's denoiser moved to the uncut (complete) truth.
7. **The 2 x 16 grid is finer than any patch under test** (a pw16 token spans 8 cells in
   wire), and every window edge lies on it, so models with different tokenizers score on the same
   cells.
8. **A track lies in one fragment**; the ~1 in 10^6 groups labl lists as overrides are not
   read.
9. **A cell's class is its dominant track's.** Mixed cells take the majority.
10. **A particle's energy is its whole track's step energy**, not the window's charge.
    Measured: Spearman 0.94-0.96 per view against window charge, ~20k e/MeV.

### 4.3 Evaluation (definitions in DECISIONS "The evaluation's definitions")
11. **iso windows** (32 w x 128 t): >= 90% of the particle's plane charge inside, < 5% of
    the window from anything else, origin offset at random (+-6 w, +-32 t). **bg windows**:
    kept coefficients, no truth within +-16 w / +-128 t, at most 10 per plane per event.
12. **1% FPR per view** (U, V, Y; both volumes pooled per view). Measured: a pooled
    threshold ran Y at 2.4-3.2%.
13. **References are not fitted probes**: floor vs max |coeff| / sigma in the cell, uprobe
    vs geometry columns, labels vs token layout. Measured: random-init and Gaussian-noise
    features under the same MLP scored below these (the probe's dilution, not a
    representation), so they are not usable references.
14. **The probe is an MLP** on standardised features (per channel, fit-row statistics),
    early-stopped on a by-event validation split; ridge lost (floor 0.29 vs 0.35 below 0.1
    MeV, Y uprobe R^2 0.01 vs 0.48).
15. **Fit on the first 2/3 of the probe events, score the rest**, at most 1M fit rows per
    probe; bootstrap over events as draw counts.
16. **uprobe reports within-group R^2**, not Fisher-mean r (r ignores scale: geometry alone
    has r 0.15, R^2 0.03). helix-nersc's 0.83-0.87 is Fisher r: compare it with our 0.75.
17. **Labels use class-balanced CE and balanced accuracy** (michel is 0.4% of cells).

### 4.4 Training
18. **One event per rank**; batch = GPU count; the FM has no event separation.
19. **LR = 4.4e-3 sqrt(B/16)** (2.2e-3 at our B = 4), measured at 29.2k / 4k steps on r1;
    the optimum falls with length, so 115k steps at 2.2e-3 is probably above the optimum.
    Not measured on d1m.
20. **The cooldown annealed on a file-order subset.** `fm_cooldown` ran with
    `data.train.max_len=46200` (a single-run corpus makes `COOLDOWN_RUNS = 3` a whole epoch,
    a third of the stable phase), i.e. on the first ~46k events of j000000 in file order, which
    DECISIONS warns against. It is ~8% of the stable phase.
21. **Eval read `last` weights** for both runs, not EMA.

### 4.5 Optical
22. **No optical labels exist for this corpus.** The campaign has no light stage; optical
    data exist only for SBND (real) and goop (simulated PMTs, `truth/pe_counts`). The codec
    is validated (docs/OPTICAL.md); tokens exist but no model reads them.

---

## 5. Choices to make

Numbered for reference; each with the options, the recommendation from this side, and
what it costs. Several are also in `TODO.md`.

**C1. The base.** This branch is the base (it already contains origin/main to 039f3ee).
helix-nersc's 10 commits merge on top: take 494a8f6, 3dfc02e (truth label), 8cfd2cf
(linear-amplitude input), e68fe7e (charge-scale augmentation; its volume-aware drift
layouts not as a default), 23bd7f8 (FM into an encoder with added input columns),
2a8533e + 975218f (record, read together). Skip 42f6188 (charge-weight power: biases faint
cells; does not import), 5f70a66 / e0eb082 (NERSC launchers). The Gaussian-NLL head is
uncommitted and waits on M15. **Port:** the denoiser read `helix.probe.resolution`
(`cell_inputs`, `cell_neighbors`, `set_cell`); here those live in `helix.data.cells` /
`helix.data.denoise` and `helix.eval`. 3dfc02e's label and `helix-cells-3`'s `q` should be
checked to be the same quantity (both: complete hits charge on 2 x 16) before one replaces
the other.

**C2. How much corpus.** One group (200k) or all five (1M, ~176 more turing GPU-hours and
~0.9 TB). Recommendation: calibrate the recipe on j000000 first (C5); build more only when
a model size or length is chosen that the repetition results say needs it.

**C3. `norm_sigma` and bins for more groups.** `corpus.sh calibrate` writes a table per
run and **overwrites** `norm_sigma_global.npy` with their mean, and `plan` overwrites
`CHUNKS.txt` / `RUNS.txt`. Running `corpus.sh all` over all five groups therefore gives
the new shards a different table from j000000's, and `coeff_verify` (one global table)
fails the corpus. Options: (a) build the new groups with j000000's existing table
(`--norm-sigma _calib/norm_sigma_global.npy`, plan appended, no recalibration); (b)
recalibrate over all five and rebuild j000000. Recommendation (a), with the bin grid
frozen (DECISIONS), after checking per-group tables deviate little (the calibrate job
prints the run-to-run deviation). Either way `corpus.sh` needs a small change to plan an
added run without rewriting the existing plan.

**C4. Where to build.** One GPU type per corpus, and j000000 is turing-built. NERSC has
A100s only, so groups built there would mix GPU types. Options: build the remaining groups
at S3DF on turing and transfer; or rebuild everything at NERSC (an A100 corpus, new
digest, new grid, new truth). Truth needs the campaign (S3DF only) either way, unless
hits / step / labl are transferred too.

**C5. The training recipe on d1m** (cheap calibration before any long run, judged by
probes, not loss): a seed pair at d768 (the noise floor); LR 1.1e-3, 1.55e-3, 2.2e-3 at
~30k steps with a 10% cooldown, at the batch NERSC will use; optionally d1024 at equal
compute (width beat depth; d1536 gave a more seed-stable floor on noisy targets). Batch:
B = 4 per node; multi-node is deferred (TODO), so B = 16 needs that first. The critical
batch is constant in cells, not events: recompute it for d1m's event sizes.

**C6. The cooldown mechanism.** With a single-run corpus, `COOLDOWN_RUNS` cannot set the
length, and `max_len` takes a file-order prefix (assumption 20). Options: a hashed event
fraction for the cooldown subset; or a cooldown length in steps over the whole train split.
Either replaces both `COOLDOWN_RUNS` and the `max_len` override.

**C7. The value target.** The top bins saturate for bright charge (helix-nersc: 79% of
>= 256k e Y cells have a slot past the last finite edge; not yet measured on d1m's grid),
so pretraining cannot reward bright-charge precision; frozen features add ~15% over raw
inputs there. Options: extend the grid to the observed maximum; a continuous distributional
target; both. A pretraining change, judged by frozen probes with the amplitude metric (C8).
The +-1e18 open edges go when the format next changes.

**C8. Evaluation additions.** Agreed with helix-nersc, none implemented yet:
* a threshold-free companion to the floor (AUC, or efficiency at 5% FPR): seed pairs agree
  to ~0.005 on AUC against ~0.04 at 1% FPR;
* an amplitude metric: median and hw68 of log(pred / truth) per view and per truth-charge
  bin (1-4k, 4-16k, 16-64k, 64-256k, >= 256k e), for the denoiser and a frozen charge
  probe; it is what exposed the ~3% bright floor;
* two seeds per arm;
* the pretrained-vs-scratch label-efficiency curve (64 / 256 / 1k / 4k / 16k labelled
  events) for the denoiser (`data.train.n_events`, `model.pretrained=False` already exist)
  and for the particle labels;
* the corpus `basis_digest` in the eval JSON;
* optional: the bg-window score tail (99th percentile) as a diagnostic;
* localisation only if token geometry changes (C10); it was dropped because averaged over
  undetected windows it measured noise centroids, so a revived one must condition on
  detection.
Open: which uprobe statistic is primary (R^2 here, Fisher r there); reporting both costs
nothing.

**C9. Denoiser defaults.** Adopt linear-amplitude input and charge-scale augmentation
(settled by helix-nersc). NLL head after M15. `cond_charge` (M6d vs M6dc) stays a switch:
the two serve detection and calorimetry. Encoder default: `model.fm` defaults to
`$HELIX_EXP/fm` (stable), not the cooldown (TODO).

**C10. Switches to settle.** `dec_frac`, `n_masks`: delete (helix-nersc: dec_frac hurt and
was seed-unstable on the v7 probe; n_masks never scored). `rope_split` / `wire_rope`: keep
until the RoPE re-test with a full-width single-axis path (TODO). Token geometry 32 x 16:
untested; only with localisation back.

**C11. Optical.** A joint TPC + light model needs a simulated light source paired with
the TPC; the campaign has none. Options: goop for optical-only work, SBND data for real
light, or a light stage added to a campaign. Until then optical stays outside the model.
If and when it enters, `plane_id`'s four roles (FiLM index, RoPE axis, masking group,
plane identity) must separate, and charge-light correlation is physics: test couplings with
controlled perturbations, not augmentation.

**C12. NERSC setup** (RUNBOOK §11): rebuild the image without jax and repoint
`container.image`; `make_host_env.sh` (login python is 3.6); stage to `$SCRATCH`, durable
copy in `$CFS/m5238/users/oalter/coeff_tpc_d1m`; verify the bin grid there; run the suite
with `HELIX_REQUIRE_PIMM=1 HELIX_REQUIRE_DATA=1`. Evaluation fits in a `shared` GPU job (32
cores, 56 GB): the floor holds features in host memory, ~25 GB at `FIT_ROWS` = 1M.

**C13. The charge label: hits or complete truth.** cells-3 `q` carries the production's
25 e- cut (§3.3). helix-nersc's complete truth (`scripts/build_complete_corpus_truth.py` on
their branch: every step deposit through JAXTPC's own drift, attenuation and diffusion, no
cuts; `--validate` re-applies the cut and must reproduce hits) gave the denoiser +0.035-0.05
detection, removed the bright bias (median 0.972 -> 0.995) and presence ECE 0.0084 -> 0.0031,
with presence defined at q > 25 e- (`--support-min-q 25`: without it 37% of charged cells
hold <= 25 e-). Options: (a) an uncut `q` column in cells-3, presence at 25 e-; (b) a
separate complete-truth family. Either is built at S3DF (needs step + hits and a JAXTPC
checkout; `/sdf/group/neutrino/omara/JAXTPC` now has f0d2ed39 fetched) and is validated
against the hits-based `q` first (`--validate`: the cut rebuild must reproduce hits; r1
reference median relative L2 0.0038 per event-plane, thresholded / hits 1.00005). Checked
against helix-nersc's 1ea2282: of the four JAXTPC functions the script imports,
`compute_wire_distances`, `calculate_max_diffusion_sigmas` and `prepare_deposit_with_diffusion`
do the same maths; `create_sim_config` differs in `num_s` (diffusion kernel levels 16 -> 32)
and a pixel-only origin fix; `config/cubic_wireplane_config.yaml` is unchanged (the campaign
is the same cubic detector, so the script's constants -- 1.6 mm/us, 10 ms lifetime, 0.5 us
tick -- hold). The script reads hits through the removed `helix.probe.truth`, so it needs
porting to `helix.tpc.io` / `helix.data.cells` first. Cost: ~6,000 s per shard on 2 CPU
workers, ~1,700 worker-hours for j000000. Recommendation (a): one truth format, the
particle labels unchanged. Whether the floor / uprobe / labels windows should use the uncut
charge too is part of the choice (window selection uses charge fractions, so the change is
small but not zero).

---

## 6. Pitfalls that cost time here

* A job must not run from a tree being edited: shards record `code.git` / dirty and
  `coeff_verify` rejects mixed commits. Here every job ran from a worktree frozen at its
  commit (`.worktrees/<purpose>-<hash>`).
* S3DF CPU partitions are preemptable for every account (truth builds were preempted three
  times on milano); `ampere` CPU-only jobs run at normal QOS.
* `FIT_ROWS`, `CAP_SIGNAL`, `N_ISO`, `N_BG_PLANE` set the eval's memory; the old per-group
  loops were quadratic in events. Keep every bootstrap statistic a weighted function of
  per-event counts.
* `exp/helix/report/` (S3DF) holds the studies behind the eval definitions (`checks.py`,
  `checks_analyze.py`); they import helix internals as of 8e7b3eb's parent and are records,
  not re-runnable against the current tree.
