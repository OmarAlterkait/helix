# Decisions

Standing decisions live here; the measurements behind the older ones are in SCIENCE.md and
the other retired documents and study records cited by name below (PERFORMANCE, DENOISE,
ENCODER_POOLING, NEXT, ...), which are at git tag `archive-2026-11` under `docs/archive/`
(`git show archive-2026-11:docs/archive/SCIENCE.md`). The shape of the code they apply to is
in `docs/ARCHITECTURE.md`. The newest round is first (the 2026-11 handoff, the 2026-10 consolidation, then
§0); the sections after them are older and name the corpora and scripts of their time
(r1, the retired launchers); where a later entry supersedes one, it says so. Two documents referenced by older entries --
`RESEARCH_EXTRACTION_MAP.md` (the research-tree extraction map) and
`CONSOLIDATION_PLAN.md` -- described work that is now finished and were retired
on 2026-09-12; they remain in git history and in
`$HELIX_ARCHIVE/retirement-backups/superseded-docs_2026-09-12/`.

Standing decisions for the coefficient FM, each with the evidence behind it and
what would overturn it. The commit log has the narrative; this is the state.

A decision without a measurement next to it is a preference, and several entries
here started as preferences that the measurement then contradicted. Those are
marked **RETRACTED** rather than deleted — knowing which way a question was
answered wrongly is worth as much as the answer.

---

## State at the 2026-11 handoff (2026-11-06)

What existed when the work moved to NERSC (helix-nersc); the choices it left open are in
`TODO.md`.

| | |
|---|---|
| corpus | `coeff_tpc_d1m/j000000`: 1000 `helix-coeff-2` shards x 200 = 200k events + 1000 `helix-cells-3` truth shards, 480 GB (~283 GB coefficients + `_calib/`, ~197 GB truth); no clean shards, nothing optical |
| basis_digest | `0dd4f3a223909fd7d8150f54854111ce6e7c89f030227a20113aabf2a27c9457` (built at 1317c04, RTX 2080 Ti) |
| bin grid | `coeff_bins_d1m_j000000_noisy` (K = 128, 120 events of j000000; digests in `helix/data/data/reference_bins.json`) |
| runs | `$HELIX_EXP/fm` (stable, 115,488 steps = 3 epochs, 4 A100, LR 2.2e-3), `$HELIX_EXP/fm_cooldown` (9,386 steps) |
| eval | `eval2_last.json` in each run dir, `helix.eval` at 8e7b3eb, `--weights last`, 4113 probe events |

| metric | reference (no fit) | fm (stable) | fm_cooldown |
|---|---|---|---|
| floor eff @1% FPR/view, < 0.1 MeV | amplitude 0.08 | 0.35 | 0.40 |
| floor, 0.1-0.2 MeV | 0.18 | 0.55 | 0.62 |
| floor, 0.2-0.5 MeV | 0.84 | 0.95 | 0.97 |
| floor, > 0.5 MeV | 0.99 | 0.99 | 0.99 |
| uprobe within-group R^2 (U / V / Y) | geometry 0.03 | 0.52 | 0.53 (0.56 / 0.56 / 0.49) |
| labels motif, balanced accuracy | token layout 0.29 | 0.80 | 0.80 |
| labels pid, balanced accuracy | 0.24 | 0.65 | 0.65 |
| recon var_expl | -- | 0.645 | 0.693 |

pid recall per class (cooldown): electron 0.40, photon 0.48, muon 0.78, proton 0.92. What
these do not show (helix-nersc, on r1): one seed moves the floor by 0.03-0.04 at d768, so
stable vs cooldown below that is unresolved; validation loss rank-correlates 0.15 with the
probes; nothing compares pretrained against from-scratch. fm's column read raw `last`
weights of a WSD stable phase, so it probably understates fm (score it with `--weights ema`).

**The data, as built.** `/ident` is (run = campaign group, source_file = the sensor file,
event = the campaign's `event_id`); the noise is added once, at build, from the event's
identity (blake2b seed), and training reads that one realisation every epoch. Basis: coif3,
level 4, periodization, D1 dropped; the coherent gate on 64-wire blocks (kgate = ksig = 3,
npass 2, tau 0.05, global sigc); threshold kappa 1.0, floor 2.0, sigma per plane; values
rounded to a power of two <= sigma/16 (input), sigma/32 (target). Same basis as r1 but a
different chain (r1's digest is `8c4542b6...`), so no r1 number transfers. Training drops
events above `max_event_size = 600_000` rows (the fraction on d1m is unmeasured); the split is
0.95 / 0.03 / 0.02 by identity hash (4113 probe events in j000000).

**The truth is hits truth, not complete.** cells-3 `q` is the hits file's electrons
(pre-response), and JAXTPC f0d2ed39 writes hits after its own cuts (`threshold: 25.0`:
samples <= 1 e- per group x wire x tick dropped in box compaction, <= 25 e- at the CSR
encode), which drops each deposit group's diffusion tails, ~1.6% of in-window charge,
growing with drift (complete / hits = 1.0155-1.0161 on r1). Fine for the particle labels,
centroids and track splits; for charge targets see TODO C13. A track lies in one fragment
(labl's ~1 in 10^6 override groups are not read); a cell's class is its dominant track's; a
particle's energy is its whole track's step energy (Spearman 0.94-0.96 per view against window
charge, ~20k e/MeV).

**Measured then, standing now.** One GPU type per corpus (turing and A100 differ on 0.016% of
kept coefficients). The value step is harmless (added variance <= 1.6e-4 sigma^2 at dc6ae8b,
input moves <= 0.013 against bins 0.08-0.11 wide). `dec_frac` and `n_masks` were deleted
(model section below). The evaluation's definitions are the entry "The evaluation's
definitions" below.

**Operational, learned in that round.** A job must not run from a tree being edited (shards
record `code.git` and the dirty flag): every job ran from a worktree frozen at its commit. S3DF
CPU partitions are preemptable for every account (truth builds were preempted three times on
milano); `ampere` CPU-only jobs run at normal QOS. `exp/helix/report/` (S3DF) holds the
studies behind the evaluation's definitions (`checks.py`, `checks_analyze.py`); they import
helix internals as of 8e7b3eb's parent and are records, not re-runnable.

---

## The simulation source is the 1M-event campaign (2026-10-28)

**The simulation corpus is built from `doraemon_2026_1M_lartpc`, and only from it.** The
reader (`helix.tpc.io`) reads JAXTPC's format 2 (`entry_NNNNNN`, `/dprod/events`) in the
campaign layout `<group>/<task>/{sensor,step,hits}/`; format 1 (`event_NNN`,
`wire_test_00_00_02`) is not read. *Why:* one source, no second identity rule or path
layout kept for a production nothing new is built from. *Cost:* r2 (built from wire_test)
can be read, trained on and evaluated, but neither it nor its cells truth can be rebuilt
or re-migrated by this code; SBND-data-helix `jobs/regress.sh`'s simulation reference must
be re-made from a campaign task.

**The event is the campaign's.** `/ident` is (run = the campaign group `j00X000`,
source_file = the task's sensor file name, event = the campaign-unique `event_id`); the
noise seed and the identity-hash split are functions of it, so they are the same whatever
helix's sharding. A corpus run is a group (5 runs of 200,000 events), a build task a task
(1000 events), and shard k holds campaign events `200k .. 200k+199` (generator job k): a
shard's name says which events it holds. Within an unnumbered source file the reader's
index follows the events (`[hash(run, file) | event]`), so dataset order is file order.

**Step positions are `uint16 * pos_step_mm + pos_origin`** (JAXTPC `save_event_step` /
`load_event_step`), not `volume_ranges`. *Why:* the old decode put every r2 cells
`dom_y`/`dom_z` in [-2160, -1281] mm, compressed 0.22x and offset per event; the
along-wire probe's label was wrong on r2. Same `helix-cells-2` format (columns unchanged);
the label text says which decode a file used.

**No physics setting changes.** The campaign's detector and production configs are those
wire_test ran (same `cubic_wireplane_config.yaml`, `production_cubic_wireplane_doraemon_300micro`
thresholds, 4321 ticks, 1969/1969/1443 wires, pedestals 1843/410, 182 e-/ADC, 12 bits,
noise off) and helix's noise model and geometry are unchanged; the input differs in the
generator (MPV/MPR particle bomb) and JAXTPC f0d2ed3 (diffusion kernel `num_s` 16 -> 32,
`group_sizes` uint16).

**`helix.data.verify` judges a gap against the plan**, not against a sensor-file name
(`--source-root` is gone): a gap the plan never wrote is fine, a planned shard absent is
an error.

## The 2026-10 consolidation: noisy-only, one code path (2026-10-27 .. 28)

origin/main's denoising research was merged into this line and then consolidated. The
studies behind these entries are records only, all on the r1 corpus: DENOISE,
ENCODER_POOLING, SCIENCE §§10-11 and PERFORMANCE §11 (at the archive tag).

**Training is noisy-only.** The tokenizer's target is its own normalised input (`tgt`
differs from `inp` only where `dead_frac` blanks the input); no training config, dataset
default or training module reads `coeff_clean` (`tests/test_boundary.py` scans
for it). *Why:* real detector data has no noise-free counterpart, so a recipe that needs
the simulated clean target cannot be the recipe for data. It costs floor on simulation
(SCIENCE §11: 0.45-0.61 at 1% FPR against 0.64-0.65 with clean targets; width recovers
part of it); that is the price. Clean shards are a build option (`--keep-clean`, off by
default), kept only to inspect the DSP. *Would overturn it:* nothing on the simulation
side; a real-data target would.

**The bin grid bins noisy coefficients, with one gap bin.** Uniform in asinh on each side
of zero, with one bin spanning the sub-threshold gap; no target parameter, no clean
mode, no legacy-table defaults. *Why:* noisy coefficients exist only beyond the
~3.35-3.74 sigma threshold, so a single uniform grid leaves ~30-40% of its bins
unreachable (SCIENCE §11). The reference is `coeff_bins_d1m_j000000_noisy`
(`helix/data/data/reference_bins.json`), derived 2026-10-29 on milano from 120 events of
`coeff_tpc_d1m/j000000`, K = 128, with no empty bin in any band (it replaced r2's
`coeff_bins_r2_run0027575715_noisy` when the corpus moved to the campaign).

**One model, on varlen attention.** One `FMModel` (registered `Coeff-FM`) whose only
attention path is the unpadded grouped (varlen) attention, with its CPU / fp32 per-group
SDPA reference kept for tests and CPU evaluation; one categorical head; `build_fm` raises
on an unknown key. *Why:* every option was a branch only one study exercised, and varlen +
`fused_qk` was 6% faster per step and 0.005 better in cooled loss than the padded groups,
which attended duplicated tokens (PERFORMANCE §11). *Consequence:* flash varlen attention
needs sm80+, so training and evaluation run on `ampere` (A100) at S3DF, not `turing`
(which still builds the corpora). Removed, with their verdicts:

* the full-attention and padded paths, the serial / varlen switches: superseded by varlen
  (PERFORMANCE §11);
* the MSE / NLL heads (`nll`, `loss_fused`, `alpha` / `beta` / `varb`, the tokenizer's row
  `target`) and `no_bce`: the categorical head is calibrated (PIT mean 0.512, std 0.289:
  SCIENCE's header, NEXT §1) and no config used them;
* `vis_frac` (visible tokens decoded): needs clean targets, and gave no floor change even
  with them (SCIENCE §11);
* encoder pooling (`band_pool`, `pool_skip`, `pool_sub`, `mask_cell`, location masks):
  pooling costs the floor (0.09-0.13 at 1% FPR against plain pw8), and at pw16 the
  location mask unit costs it (0.569 against 0.62-0.65) (ENCODER_POOLING §6);
* block and `plane_any` masks, `plane_mode`: `plane_frac`, whole-plane masks mixed into
  random ones per step, is the plane masking in use (SCIENCE §2);
* `d_dec` (costs map_r: ENCODER_POOLING §6b), `n_sink`, `pad_mask`, `cond` / `adaln`,
  `dec_mode`, FiLM subsets, `n_wirefeat`, `fm_names`, the centroid time coordinate and
  `sigma_norm`: unused by any config, or superseded;
* `dec_frac` (partial decoding) and `n_masks` (several masks per step), 2026-11: `dec_frac`
  hurt the floor on the pooled encoder (ENCODER_POOLING §6) and was seed-unstable on
  helix-nersc's v7 probe; `n_masks` was never scored. A run that recorded either still
  reads (`helix.model.checkpoint.RETIRED_TRAIN_OPTS`);
* `PatchConfig.cell_t`, which accepted one value (`helix.model.tokenize.RETIRED_KEYS`).

Kept as switches until measured: `rope_split` (default False, as in every config) and
`wire_rope`, for the RoPE re-test (TODO); `dead_frac`.

**The bf16-weights package is removed:** `bf16_params`, `fp32_stream`, `FlatAdamW`, the
fp32 all-reduce hook and WeightEMA's FlatAdamW-master branch; with them `ddp_bf16_grads`,
`bucket_by_size` and its sampler, the OneCycleLR `max_lr` expansion and
`helix.integrations.pimm._compat`. The trainer is plain fused AdamW on fp32 weights under
bf16 autocast. *Why:* bf16 weights were 11% faster but 0.010-0.015 worse in cooled loss
(PERFORMANCE §11; `fp32_stream` was the arm meant to explain it); bucketing has nothing
to sort at one event per rank; `_compat` patched pimm's legacy single-file checkpoint
loader, and helix's split checkpoints restore RNG state through DCP instead.

**The recipe is `nz_pw16`** (`configs/pimm/fm.py`): d768, 12 + 4 blocks, 12 heads (head
dim 64), QK-norm, fused QK-norm + RoPE, K = 128, AdamW, the WSD stable phase plus a
separate cooldown (`fm_cooldown.py`). *Why:* it is the recipe the noisy-only results were
measured with (SCIENCE §11). The peak LR is `4.4e-3 * sqrt(GPUs / 16)`: 4.4e-3 was tuned at
16 events per step, the optimum scales as sqrt(batch) (SCIENCE §5: 2.2e-3 was measured best
at 4 events per step, the one clean arm of that sweep), and muP transfers it across width
only. *Caveat:* the optimum also falls with run length (SCIENCE §9); a production-length
run should confirm it (TODO). Supersedes the m113 architecture (d512) named in §0.

**Truth is the `cells` modality, built separately from the corpus.**
`scripts/build_truth.py --corpus <run dir>` writes `{name}_cells_<k>.h5` beside each
coefficient shard (same events, same order, identity copied; format `helix-cells-3`,
`helix/data/cells.py`): per 2-wire x 16-tick cell its charge, its split by track and its
dominant hits group's share and position; per (plane, track) the charge centroid; per
track the deposited energy and its motif / pid class (pysupera's `labl`, written by
pimm-data's `supera_labl` beside the task, walked as pimm-data walks `segment_motif` /
`segment_pid`). `CoeffTPCDataset(modalities=("coeff", "cells"))` joins it by
identity; pretraining refuses any config or module that names it. *Why:* labels change
more often than the corpus, and a corpus rebuild is the expensive step; every evaluation
window and padding box lies on a cell edge, so the cell grid carries everything the
denoiser label and the evaluation need (on run 0027575715 shard 0 it reproduces the
pixel-level windows exactly). Size: ~197 MB per shard. It replaces the per-script truth
dumps (`dump_probe_truth`, `dump_resolution_truth`, `build_denoise_truth`, ...).

**One evaluation command.** `python -m helix.eval RUN --metrics recon,floor,uprobe,labels,denoise
--out FILE` scores a run from its pimm directory: `recon` (the training evaluator's
metric code, under the random and the region mask), `floor` (the detection floor on frozen
features against the raw inputs and the fit-free coefficient amplitude), `uprobe` (the
along-wire probe against the raw inputs and the geometry columns), `labels` (each signal
cell's particle class against the raw inputs and the token layout), `denoise` (a denoiser on
the floor's windows, and its charge on every charged cell). *Why:* sixteen scripts
(`eval_resolution`, `eval_checkpoint`, `eval_denoise`, `run_probe`, ...) each read the model
and the truth their own way. `Q0` is defined once (`helix.data.cells.Q0`).

**The evaluation's definitions, and why they scale.** Shared by every metric (`helix.eval.Context`):
one encoder, one fit/score split, and per event one truth draw (`Event.truth`).
*Definitions* (each changed numbers when set, 2026-11; the measurements behind the last
four: `exp/helix/report/checks.py`, 600 events of fm / fm_cooldown):
* a cell no token covers is SCORED as predicted 0, never dropped -- dropping them made the
  scored cells depend on the tokenizer;
* the false-positive rate is per VIEW: each view's 99th percentile of its own background
  windows, sampled per plane -- one pooled threshold, set by the views with most background
  (U and V: 5.3k / 5.9k windows against Y's 0.8k), ran Y at 2.4-3.2% and raised the faint
  efficiency 0.35 -> 0.41;
* every probe arm has a reference that is not a probe of another encoder: the floor's is the
  fit-free coefficient amplitude, uprobe's the geometry columns, labels' the token layout
  (and every probe has the raw-input arm beside them, below).
  The randomly initialised encoder was the control until measured: under the same probe it
  scored BELOW the amplitude at every energy up to 1 MeV (0.06 vs 0.08 below 0.1 MeV, 0.52
  vs 0.83 at 0.2-0.5) and below geometry along the wire (within-group R^2 -0.02 vs 0.03) --
  Gaussian-noise features did the same, so it bounds the probe's fitting (thousands of empty
  columns drown a weak signal), not the representation;
* the probe stays an MLP: ridge on the same features lost the faint floor (0.29 vs 0.35
  below 0.1 MeV) and the Y view along the wire (R^2 0.01 vs 0.48);
* uprobe reports the pooled within-(event, plane) R^2, not a Fisher-z mean of
  correlations -- a correlation ignores scale (geometry alone: r 0.15, R^2 0.03), and the
  per-group row threshold it needed is gone;
* labels (motif, pid per signal cell) are scored by balanced accuracy and fitted with a
  class-weighted loss: plain argmax never predicted a Michel (0.4% of cells) or a delta;
* the bootstrap resamples events as draw counts: an event drawn twice weighs twice
  everywhere;
* dropped: map rows as a scored set (`map_r`, presence / faint AUC: their mix of signal,
  near-signal and noise rows was a choice, not a population), localisation (averaged over
  windows the probe never detected, it measured noise centroids), pair separation, RankMe,
  energy bins above 0.5 MeV (every arm is at 0.97-0.99 there), and the random-init arm.
*Added after an audit of this package (2026-11-06; each confirmed in the code, every earlier
number still under its own key, the new ones beside it):*
* a NEAR background beside the far one. The far (`bg`) windows must hold no truth within
  +-16 w / +-128 t, while an iso window caps foreign charge only INSIDE itself (5%), so a
  prediction that hazes charge around tracks gained iso efficiency at no false-positive cost.
  `near` windows (`helix.data.denoise.near_windows`, the denoiser's own training windows:
  empty with a one-cell margin, charge inside the padding; kept only with a kept coefficient
  inside, as every `bg` window has; drawn after iso and bg, which stay as they were) give
  `floor_eff1pct_near_<E>` (the threshold set on them) and `near_fpr_at_far` (the far
  threshold's rate on them; 1% if the prediction cannot tell them apart). The far efficiency is
  unchanged and still the floor; measured on fm_cooldown (`last`, the 142 scored events of the
  NERSC j000000 sample; far numbers identical to the previous code on the same events): the
  trained arm's 0.44 / 0.64 below 0.1 / at 0.1-0.2 MeV fall to 0.04 / 0.15 with the threshold
  on the near windows, and the far threshold fires on 19% of them (raw arm 1.8%, amplitude
  3.3%). One run and 142 events: that the probe predicts a haze around activity is the reading
  this suggests, not yet shown cell by cell;
* the threshold-free companion `auc_<E>`: P(iso above a far window of its own view). Seed
  pairs agree to ~0.005 on it against ~0.04 on the efficiency at 1%. Every window number is
  also reported per view (`_U`, `_V`, `_Y`), since a pooled one hides a view;
* a `raw` arm in floor, uprobe and labels: the same probe (rows, MLP, standardisation,
  selection) on the covering tokens' raw tokenizer inputs (`inp` and `occ`, 2 x n_slot per
  band) instead of encoder features, and `d_over_raw` = trained - raw, bootstrapped paired.
  The existing references bound the probe, not pretraining: labels' token layout carries no
  charge at all (0.29 / 0.24 balanced accuracy against chance 0.20 / 0.17). The predecessor
  (`scripts/eval_resolution.py`, origin/main) had this arm. On fm_cooldown (as above) trained -
  raw is +0.38 / +0.53 floor efficiency below 0.1 / at 0.1-0.2 MeV, +0.53 uprobe R^2 and
  +0.30 / +0.21 motif / pid balanced accuracy (raw 0.50 / 0.43). Raw scores BELOW geometry along
  the wire (R^2 -0.01 vs 0.02): its 1024 columns drown the geometry, the known weakness of the
  probe;
* every probe arm is fitted `--probe-seeds` (default 2) times on the same rows (seeds
  `seed + k`), reported as `seed_mean` / `seed_half_range` ((max - min) / 2) beside the
  interval, which comes from the first fit: the event bootstrap resamples events under ONE
  fitted probe and leaves the fit's own variance out. On fm_cooldown the trained arm's
  half-range is 0.003-0.011 on the efficiency, <= 0.002 on the AUC and R^2;
* per-event weighting beside the pooled numbers (`_ev`): the fit takes at most
  `FIT_ROWS // n_train` rows per event but scoring pooled every row, so dense events set the
  score. Floor: each event's iso windows of a bin weigh 1 in all; uprobe: each (event, plane)
  group's sums divided by its rows (groups weigh equally; a mean of per-group R^2 would be set
  by the groups with least variance); labels: each event's rows weigh 1 in all in every
  per-class recall. Recon stays token-pooled on purpose (its own docstring: the per-event mean
  is ~0.09 nats optimistic);
* recon is also scored under the predecessor's REGION mask (`region`: 75% of (plane, 32 wire,
  128 tick) regions hidden whole, chosen by crc32 of the region key, the same in every event),
  beside the model's random mask: the predecessor's 0.31-0.39 var_expl against this package's
  ~0.69 is the gap between interpolating from neighbouring tokens and reconstructing what no
  neighbour shows; fm_cooldown (`last`): 0.693 random, 0.297 region;
* the denoise metric also scores CHARGE, which detection does not: on every truth-charged
  covered cell, per view, the median and hw68 of log(pred / true) per truth bin (1-4k ...
  >=256k e-), the median true / pred per predicted-charge bin (linearity), the per-event plane
  total sum pred / sum true (median over events; on charged cells, so haze is the near
  window's to price), and the presence probability's ECE (15 bins), Brier and log-loss against
  q > 25 e- (the training's presence label, `--presence-q`) on the covered cells of the charged
  set and of the windows -- every covered cell of an event is millions of cells. One noise
  realisation, the corpus's own, as on real data;
* the JSON records what it scored: `corpus.basis_digest` (from the evaluated shards, not the
  training run's record), `truth.q_cut` (the cells shards' `hits_threshold`, 25 e- on d1m:
  cells-3 `q` is hits truth, cut per sample) and `weights` per metric -- the audit found the
  floor read `last` while the denoiser's encoder starts from EMA, which the JSON did not say.
*Scaling:* probes fit on at most `FIT_ROWS` = 1M rows and score each later event as it
arrives (holding predictions only); every bootstrap statistic is a weighted function of
per-event counts, so a resample costs a few bincounts, not a rebuild of ~8M rows. The
4113-event probe split had needed >240 GB and, in uprobe scoring, a per-group loop that was
quadratic in events.

**The denoiser trains on pimm, like the FM.** `configs/pimm/denoise.py` (`_base_` fm.py),
launched by `scripts/submit.sh denoise -- model.fm=<FM run>`: `Coeff-Denoise` = the encoder of
the FM run (its architecture, its tokenizer and, with `pretrained=True`, its EMA weights;
`pretrained=False` is the from-scratch baseline) plus `CellDecoder` with a presence
logit. The default is M6d (`cond_charge=False`: detection and maps); `model.cond_charge=True`
is M6dc (calorimetry). The split is fm.py's identity-hash holdout; the best checkpoint by
validation loss is kept. *Why:* M6d is better or tied on detection, gated-map detection
and localisation, M6dc better on energy (DENOISE, "Dense rerun ... M6d / M6dc"); one
trainer, launcher and checkpoint format for both models instead of
`scripts/train_denoise.py`'s own DDP loop and its `holdout.json` split.

**The denoiser's encoder default stays `$HELIX_EXP/fm_cooldown`.** *Why:* the SBND session
measured the stable phase `fm` with its EMA weights against `fm_cooldown` on every probe and
found them equal within 0.011, so either serves; the config, README, RUNBOOK and ARCHITECTURE
all name `fm_cooldown`, and changing the default would change the recipe of record for nothing.

**Presence is q > 25 e-, everywhere.** `DenoiseRows(support_min_q=25)` (its default, and
`configs/pimm/denoise.py`'s `_SUPPORT_MIN_Q`), the evaluation's `--presence-q`. *Why:* cells-3
`q` is hits truth cut at 25 e- per sample by the simulation (d1m), so 25 is the truth's own
detection threshold. The transform's default was 0; on d1m the two select the same cells (every
surviving sample is above 25 e-, so every nonzero 2 x 16 cell is), so no d1m label changes: the
threshold is now stated where it is used instead of inherited from the truth.

**One noise realisation per event; no charge-scaling augmentation, no fresh noise.** *Why:* real
data has one realisation and no clean counterpart, and the augmentation needs clean shards
(gain on the SIGNAL), which this corpus does not have and a detector never will. The study
branch's `gain_sigma_*` / `clean_root` / `draw_gain` (origin/denoise-m14) are not ported.
*Cost, measured on r1:* the whole-event gain response (charge out / in when one event's charge
is scaled) was 0.87 without the augmentation against 0.97 with it (M10lin vs M11).

**The linear-amplitude input channel is an option, default off until d1m confirms it.**
`model.lin_amp_scale=S` (`FMModel(lin_amp_scale)`) adds `sinh(inp) / S`, the coefficient in
noise units uncompressed, as `n_slot` more embedding columns; a pretrained FM without them
loads with the new columns at zero (`helix.model.denoise.pad_input_columns`), so the denoiser
starts exactly at the FM. 0 is bit-identical to the model without it. *Why:* the FM reads
`asinh(v / sigma)`, which compresses bright amplitudes, and that was the one encoder cause of
the flat high-charge floor found (`baselines/uboone_sp/causes/results/m7_tests_results.md` on
origin/denoise-m14). On r1, M7dc's full 40k-step recipe with the channel from step 0 (M10lin,
S = 30) cut the per-cell hw68 against M7dc by 8 / 18 / 27% on Y (16-64k / 64-256k / >= 256k
e-), 6 / 14 / 21% on U, 1 / 6 / 18% on V, growing with charge; added only for a 6k-step
fine-tune (M9lin) it gained less. *Default 0* because r1 is retired and no r1 number transfers
to d1m (§"The data, as built"): set 30 once a d1m run reproduces the gain (TODO). The study
branch's physical-drift-coordinate option from the same commit is not ported.

**The charge head's form is a switch, `model.dist`: `point` by default until one full-length
comparison picks `gauss` or `cat` (TODO).** The presence head is a calibrated probability, but
the conditional charge was a point value trained by squared error -- a Gaussian of the SAME
width for every cell, which left a flat ~3% relative error at high charge. Two candidates
(`helix.model.denoise`):
* `gauss`: a log-width per cell beside the mean, trained by the Gaussian likelihood with
  beta = 0 (on the charged rows with `cond_charge`, on every row without). *Measured on r1*
  (M14: 6k-step fine-tunes of M10lin, paired against the point control on 48 events): hw68
  0.71 / 0.61 / 0.59 of the point head's at 16-64k / 64-256k / >= 256k e- on Y (U 0.80 / 0.71 /
  0.71, V 0.83 / 0.74 / 0.67), ~1% worse at 1-4k; calibrated from 4k e- up (|z| < 1 for
  0.67-0.70, < 2 for 0.94-0.96, median width = error hw68 within ~5%), under-covered at 1-4k
  (0.50-0.62: heavier tails than a Gaussian). beta-NLL 0.5 was worse at every bin above 4k (Y
  64-256k 0.73 against 0.61), so beta is 0 and not a config key. It followed a whole-event
  charge scaling less than the point head (gain response 0.76 vs 0.82), which no augmentation
  corrects here (above). Its width layer starts at zero, so a point head's weights start it at
  the point head; unify has no fine-tune-from-a-denoiser path, so the study's explicit warm
  start (`init_dist_from_point`) is not ported.
* `cat`: ONE categorical replacing both heads -- bin 0 = no charge (q <= `support_min_q`,
  25 e-), K = 256 bins uniform in y = log1p(q / Q0) from y(25 e-) to 7.5 (3.7M e-), HL-Gauss
  targets of width 0.75 bin (Imani & White 2018; Farebrother et al. 2024) renormalised over
  the charged bins, a hard bin-0 target for empty rows; presence = 1 - p0, the charge the mean
  of y given charge (the median is also exposed). Its cross-entropy is exactly the presence
  BCE plus the charged rows' conditional cross-entropy, so it is a hurdle -- `cond_charge` --
  by construction; `cond_charge=False` is refused rather than silently ignored. *Not measured
  yet.* r1's binned head (M14hl, 1024 bins beside a separate presence head, 18-43% WORSE than
  the point head when bright) was a 6k-step fine-tune warm-started from the point head, whose
  widths stayed at their 0.05 initial value; cat trains from the FM init over a full run, so
  that result does not decide it.
`hlgauss` (the study's binned conditional head) is not ported. Each head's validation `loss` is
its own objective (point mse + bce, gauss nll + bce, cat ce), so `model_best` compares
checkpoints of one head; all three also report `mse` and `bce`.
`python -m helix.eval ... --metrics denoise` scores a distributional head's calibration as
`cells.dist`: per truth bin and view, the coverage of |z| < 1 / 2 (gauss) or of the PIT's 68 /
95% intervals (cat), and the median width against the error's hw68.

**Checkpoints load directly.** `helix.model.checkpoint.load_run` reads pimm's run directory
(`resolved_config.json`, the WeightEMA sidecar, `model/last` or `model/model_best.pth`,
`provenance.json`); nothing is exported. *Why:* the artifact layer (`artifact.py`,
`export_artifact.py`, format detection, migrations, backfills) carried what the run
directory already records. There is no compatibility with clean-target checkpoints or
exported artifacts (m113 included): they trained on another target and grid, so their
numbers are not comparable with today's anyway. Supersedes the artifact entries below.

---

## 0. One of each (2026-10-24 .. 26)

**One build entry point.** `helix/build.py` holds the event loop, the TPC and optical chains,
the shard writer, calibration and the task plan; a source is one reader class (sim, sbnd,
goop; `python -m helix.build SOURCE`; 2026-11: moved from `scripts/build_{sim,sbnd,goop}.py`
next to their readers). *Why:* three builders
(`build_coeff_corpus.py`, `helix.tpc.sbnd_run`, `helix.optical.goop_run`) each carried
their own copy of everything after the read. A new source is now a new class and
nothing else. Shards are written `.part` and renamed, so `--skip-complete` lets a
resubmitted task skip finished work; `scripts/corpus.sh` replaced
`submit_coeff_corpus.sh` and `calibrate_norm_sigma.sh`. Checked output-neutral: rebuilt
simulation and SBND shards were byte-identical to r2 and v1.

**One noise model, torch.** `helix.tpc.noise` builds the simulation corpus and is the
load-time `AddNoise` / `Digitize`. *Why:* the numpy and torch implementations
(`dense_ops.py`, the `coherent_numpy` switch) had to be kept in step by hand. The numpy
port survives only as the test reference `tests/_noise_reference.py`. Output-neutral.

**One coherent gate, on the readout's channel groups.** `helix.tpc.coherent_gate` takes
group labels from the readout: SBND front-end boards across planes, the simulation's
64-wire noise blocks per plane. `coherent.groups` names the grouping (null: none);
`group_size` lives only in the readout. *Why:* the contiguous-block gate and the labeled
gate were two implementations of one operation, and the block gate silently ignored the
`sigc_scope: group` its config recorded (simulation now records `global`, what it always
did). Not output-neutral for simulation: the trailing partial block of a 1969 / 1443-wire
plane is reduced over a padded group (ulp-level float32 changes; run_0027575715 shard 0:
23 of 200 events differ, 2.4e-5 of the kept set). SBND coefficients byte-identical.

**One corpus verifier.** `python -m helix.data.verify` checks a run, or a whole
corpus against its plan (`_calib/CHUNKS.txt`), and `compare`s shards byte for byte.
*Why:* SBND-data-helix's `verify_corpora.py`, `readback.py` and `compare_shards.py`
duplicated it outside the repo that owns the format.

**A format stamp, and no legacy read paths.** Every coeff shard carries `/config`
`format = "helix-coeff-2"`, and every reader requires it. The fallbacks for shards
without `/ident`, without `coord_digest`, and the flat JAXTPC sensor schema are removed.
*Why:* each fallback was a branch that only old data exercised, and a reader that
accepts anything cannot say what it read. Corpora built before the stamp are migrated in
place (`scripts/migrate_format.py`, now at tag `archive-2026-11`: metadata only, old attributes
backed up); r2 and SBND
v1, the production corpora, were migrated so and verify against their plans; pre-tau `coeff_tpc` and r1 are retired and unreadable.

**Configs state intent; step counts are derived at run time.** `configs/pimm/fm.py` is
the whole run (corpus = `HELIX_CORPUS`'s parent and its `_calib/RUNS.txt`, the m113
architecture, the WSD stable phase); `fm_cooldown`, `fm_smoke` override only
what differs, and anything else is `key=value`. Warmup and eval / save cadence are
`"auto"`, filled by `FMTrainer` from pimm's `iters_per_epoch * epoch` through
`helix.model.schedule`. *Why:* nine configs carried `N_TRAIN_EVENTS` tables and step
counts copied between them; a fix applied to one was shadowed by another. The run's
length now follows from the corpus, the split and the GPU count, which is what it
depends on. `HELIX_LEGACY_CORPUS` and `resolve_cooldown_subsets.py` went with them.
*Superseded in part* (2026-10 consolidation): the architecture is `nz_pw16`, not m113's.

**An sbatch chain instead of `pimm submit`.** `scripts/train.sh CONFIG [key=value ...]`
submits `LINKS` jobs chained `--dependency=afterany`; each runs `python -m pimm.train`
in the site container and resumes from the latest complete checkpoint. *Why:* `pimm
submit` could not run from S3DF (its preflight imports the training stack on the host,
and its job script finds configs only inside pimm's own `configs/`). helix needs pimm's
training engine, not its launch layer, so `configs/launch/`, the launcher venv
(`make_launcher_env.sh`) and the S3DF-only `submit_coeff_fm_train.sh` /
`chain_coeff_fm_train.sh` went; the host needs only PyYAML (`scripts/make_host_env.sh`).
`afterany` continues whether a preempted job is requeued (NERSC) or cancelled (S3DF).
Checked: a 3-link `fm_smoke` chain whose first link timed out resumed with loss and LR
continuous.
*Superseded* (2026-11-06): **training launches through `pimm submit` again**, by
`scripts/submit.sh CONFIG [--recipe R] [flags] [-- key=value]` with recipes in
`configs/launch/` and studies in `configs/sweeps/` (`scripts/sweep.py`). *Why:* `train.sh` was
single-node (`--nodes=1`), and every recipe result was measured at 16 GPUs; pimm's launcher
already runs multi-node under shifter with the `nccl-plugin` module (4.9 GiB/s across nodes
against podman-hpc's 1.23), and the user wants pimm's launcher structure. The two blockers
above are fixed in pimm-private's `helix-launch` branch: `train.sh` and `submit` accept an
external config by absolute path (the run goes to `$HELIX_EXP/<name>`, no config group), and
preflight reads the config with `WORLD_SIZE` set to the launch's ranks, where it had refused
every run above 4 GPUs (it imports no torch since `import_custom_modules=False`, so the host
needs pimm's launch dependencies and helix's base install, `scripts/make_host_env.sh`). The
launcher no longer injects `hooks.CheckpointSaverIteration.backend=dcp` (deprecated, and
`None` already resolves to the same format), which named a hook helix does not have and so
killed every multi-node helix job. `train.sh`'s guard -- the same run name continues only
the same config and overrides -- moved into `submit.sh`. Checked at NERSC: `fm_smoke` on 2
nodes x 4 GPUs on the d1m sample (8 ranks, steps, validation, DCP checkpoint, EMA, clean
exit). Not yet checked: a requeue across links under submitit, and S3DF (QOS normal, so no
preemption; links end on walltime).

**The production bin grid is r2's** (`coeff_bins_r2_run0027575715`, derived from r2's
first run with r1's parameters: 120 events, K = 128). *Why:* r2's inner edges sit a
median 0.35 / 0.46 / 0.64 / 0.98 local bin widths (A4 / D4 / D3 / D2, max 2.0) from r1's,
against 0.16 between r1 runs, so by the yardstick of the freeze decision below the r1
grid does not describe r2. Models trained on r1's grid are not value-comparable with
models trained on this one.
*Superseded* (2026-10 consolidation): that grid binned clean coefficients. The reference
is now the noisy grid `coeff_bins_r2_run0027575715_noisy`, same corpus and parameters, not
yet derived.

**Sites are profiles; the target is NERSC.** Paths, container and scheduler live in
`helix/sites/<site>.yaml` and reach shell scripts through `scripts/helix_env.sh` and
`scripts/helix_run.sh`, so moving to NERSC changes a profile, not the scripts.

---

## Corpus

**Ship R1 (`gate_tau=0.05`), not k30.** R1 wins on denoising (SSE 0.714,
off-signal 80k→67k), support cleanliness (band-0 zero-clean 4.81%→1.90%), and
ties or wins cross-plane. Both corpora live at
`/sdf/data/neutrino/omara/coeff_tpc{,_r1}`.

**Build all 8 runs; the corpus is 790 shards / 344 GB / ~158k events.** Seven
runs have 100 source files, `run_0027670361` has 90 upstream — complete at 90, not
a build failure. Identity split: **train 150,239 / val 4,641 / probe 3,111**.

**FREEZE the bin grid. Do not re-derive on the pooled corpus.** Bins derived on
run 2 alone vs run 1 alone, both at 480 events so both converged, differ by
**0.129% of a band's span = 0.16 bin widths**, which is *smaller* than the
sampling spread of the 120-event table actually shipped (0.159%). Run-to-run over
sampling = **0.81×**. `cent_ratio` agrees to 0.38% median.

Re-deriving would move the grid less than the noise already in it, while costing
value-CE comparability with every earlier run and invalidating every trained
K=128 head. Ship `coeff_bins_r1_tau05_run0027575715_v2.pt`.

*Since:* the grid was re-derived at the second site, because the corpus there is
not byte-identical to the one v2 fingerprints (same run, same `basis_digest`,
different arrays; cause not established). The DECISION above is unchanged --
bins are not re-derived when the corpus GROWS -- but the shipped name is no
longer a literal. It is declared in `helix/data/data/reference_bins.json` and
read by `helix.data.bins.reference_table()`. A number produced against v3 is not
comparable with one against v2; see that file's `_rerecord_note`.

*Would overturn it:* a run whose bins differ from run 1's by materially more than
the 0.16-bin-width yardstick — i.e. evidence the runs are not one distribution.

*Superseded for r2* (§0): r2 is a different corpus, not a grown one, and its grid
was derived anew.

**The frozen grid costs 1.75% worst case, not 11.5%.** Applying run 1's grid to
run 2's data reclassifies 7.8% of coefficients carrying 11.5% of |charge| — but
reclassification is *not* error: adjacent bins of a K=128 partition have
near-equal centroids. Measured charge error is **1.75%** worst case, ~0.4% by the
centroid comparison. Do not quote 7.8/11.5% as an error.

**RETRACTED — hold out a whole RUN for validation.** Proposed as "a materially
stronger generalisation test"; the measurement above kills it. The runs are the
same distribution, so a held-out run tests nothing extra, and it would couple the
split to run composition. The existing identity hash is better and was verified,
not assumed: run 1's counts are 19,034 / 577 / 388 with one run built and
**identical** with all eight. Growing the corpus reassigns nothing.

---

## Model, metrics, artifacts

**Centroids are never optional.** `set_bins` derives any table it is not given,
so no centroid buffer is ever NaN. They used to be optional, so consumers asked
"are these present?" — three files asked, one forgot, and `var_expl` /
`charge_closure` / `charge_bias` were absent from **every** eval log ever
produced. Centroid args are keyword-only.

`checkpoint.apply_bins` is the caller on the RESUME path (and was in the retired
`scripts/smoke_train_fm.py`), not the sole caller — several test files call
`model.set_bins(edges)` directly. That distinction matters: the guarantee here comes from `set_bins`
DERIVING any table it is not given, not from a single choke point upstream of it.
A reader who believed the choke point existed could add a caller and expect the
invariant to hold for free.

**`cent_lin` and `bin_cent_measured` are deleted — both were tables nothing
read.** `cent_lin` also carried a real bias (planes pooled across 22%-different
sigmas). `bin_cent_measured` was added *by this work* two commits after removing
`cent_lin`, committing the same sin. Old checkpoints carrying either still load
(`_STALE_BUFFERS`). *Superseded* (2026-10 consolidation): `_STALE_BUFFERS` is gone with
every other compatibility path; such checkpoints are not loaded.

**Validation scores the whole val set.** Every rank evaluates its shard and the
sums all-reduce. Previously rank 0 evaluated alone with a `DistributedSampler`
loader, i.e. **1/world_size** of the set reported as the number — the
`batches=145` in the 4-GPU logs. Costs nothing: the other ranks were blocked at a
barrier anyway.

**`model_best` OFF for the stable phase, ON for the cooldown.** A flat WSD
schedule has no best step — raw val loss is a plateau plus noise whose max is the
luckiest eval, and it would select *raw* weights while everything downstream
reads the EMA. An annealing schedule has a real minimum and its raw weights *are*
the annealed model. Confirmed: the completed 1-run stable phase recorded
`best_metric_value: -inf` after 118,950 steps; the cooldown writes monotonically
improving `Best validation` lines.

**The cooldown is a separate short run, never a tail on the stable phase.** WSD's
stable phase commits to no horizon, so only the cooldown config knows
`total_steps`, which `1 − √p` needs. Warm-start with `weight=`, **never**
`resume=` — resume restores the step counter and would put the run past `p = 1`,
at the floor from step one, which looks fine (loss falls) and anneals nothing.

*Verified live:* `Lr(423)/Lr(1)` observed 0.8349 vs 0.8348 predicted for `1 − √p`
— 0.01%.

**Cooldown length uses a RUN subset, not `max_len`.** `epoch` is an integer and
one epoch over all 8 runs is 33% of the stable phase. Three runs = 14,264 steps =
12.7%, inside the usual 10–20% band. `max_len` would take the first N events of a
run-ordered index and anneal almost entirely on run 1. Sound only because the
runs are one distribution — see above.

---

## Structure

**`helix/integrations/pimm` is a package, not a 1089-line module.** Split by
concern: `_compat` (import-time pimm patches, must import first), `data`,
`model`, `trainer`, `hooks` (what a run *writes*), `eval` (what a run
*measures*). The module path is unchanged, so `imports=["helix.integrations.pimm"]`
and every symbol import still work. *Superseded in part* (2026-10 consolidation):
`_compat` is removed.

**Eval stays in helix**, against the extraction map's "EVAL is pimm's
job". That line was written when eval meant downstream probes and baselines,
which do belong there. This is the *training* metric and it reaches into helix
internals no framework should know about — `core.raw_heads`, the bin buffers,
`losses_cat`. Moving it inverts the one-way dependency the integration exists to
preserve.

**No top-level shims.** The nine re-export modules are gone; import from
`helix.core.*` / `helix.tpc.*`. (`helix.tpc.wavelet`, once the `(image,
DetectorConfig)` adapter, no longer exists; the TPC chain calls `helix.core.wavelet`
directly.)

**Tests exercise shipped code, not copies.** `bucketize_bins` exists because the
tests previously pinned hand-copies of logic that lived elsewhere.

---

## Known-complex, deliberately unchanged

**The ±1e18 edge sentinels are gratuitous** — every consumer slices them off
(`edges[b, 1:-1]`), and `_close_open_edges` exists only to undo them. Not
removed: `edges` is `(n_band, K+1)` in every checkpoint on disk including m113,
the frozen goldens and the trained run. Churning a persisted format for tidiness
would break all three. Delete when the format next changes for a real reason.
*Since* (2026-10 consolidation): m113 and the clean-target runs are no longer loaded,
so the reason that remains is every noisy-only checkpoint written since.

**Deriving centroids from edges** is unreachable for anything current (the v2
sidecar measures both; live checkpoints carry both). Kept only for pre-fix blobs
like m113, which has no measured `cent_ratio` and never will. (The research
bundle has since retired; this is the remaining reason to keep the fallback.)
*Superseded* (2026-10 consolidation): m113 is no longer loadable; the derivation
remains as `set_bins`' behaviour for a table it is not given.

---

## Operational

Details and incantations live in `docs/RUNBOOK.md`; `TODO.md` is what is NOT
done. There is one launcher, `scripts/submit.sh` through `pimm submit` (§0). The decisions are:

- **Corpus builds on `turing` only.** The DSP is architecture-sensitive: turing
  and A100 disagree on 0.016% of surviving coefficients and are not
  bit-comparable. Run 1 was built entirely on turing.
- **Training on `ampere`.** K=128 at full event size makes the logits ~2.4 GB for
  one event before the backward doubles it, and the varlen attention needs sm80+
  (evaluation too).
- **Never edit the helix tree mid-build.** `helix.data.verify` fails a corpus whose
  shards carry different `code.git`; it caught exactly this and 7 shards were
  rebuilt.
- **`helix.data.verify` judges a gap against the plan** (`_calib/CHUNKS.txt`): an upstream gap
  is not planned, so not a failure; a planned shard absent is.
- **Arrays of at most 100 tasks** — `MaxArraySize` is 100 here, so the old
  `#SBATCH --array=0-799` was unsubmittable; `corpus.sh build` splits the plan into
  arrays of `ARRAY_MAX` (100).
- **Chained links**, each resuming iff a complete checkpoint exists. Training
  runs at QOS `normal` (`helix/sites/s3df.yaml`): preemptable is priority 1
  against normal's 10000, so a long job waits rather than runs, and S3DF
  preemption cancels rather than requeues. Resume needs `weight=` **and** `resume=`, which
  pimm's `scripts/train.sh` passes from its latest-checkpoint resolver on a requeued link.
