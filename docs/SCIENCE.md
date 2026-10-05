# Science record

> **The posterior-width diagnosis is now measured.** Over 5.4M held-out slots
> the categorical head's PIT has mean 0.512 and std 0.289, against the
> 0.5 / 0.2887 of a uniform PIT — the forecast is calibrated. The read-back is a
> posterior mean, which is MMSE-optimal by construction, so `var_expl` = 0.708
> is at its Bayes limit and **no loss change can raise it**. Reconstruction is
> finished as an optimisation target; the remaining headroom is in the
> representation, which is what the probe measures. See
> `docs/REVIEW_FIELD.md` §9b, and note the probe's resolving power (~0.013)
> recorded in the same section before designing a comparison against it.


What was measured, what it means, and what is still open. Numbers here come from
runs on this cluster; where a claim was later refuted, the refutation is recorded
rather than the claim quietly removed.

---

## 1. The coherent gate and `tau`

`tau` is the occupancy tolerance in the coherent-noise gate. A wire group of 64
is judged to hold signal if enough of its wires are occupied; `tau` is the
fraction below which a group is treated as noise-only and its coherent component
subtracted:

    refuse = refuse & (occ1 > tau)

`tau = 0.05` is 3 wires of 64.

Measured over **1,200 (event, plane) pairs**:

| | without `tau` | with `tau = 0.05` |
|---|---|---|
| stripe residual | baseline | **5.32x lower** |
| off-signal pixels > 5 ADC | baseline | **2.61x fewer** |

This is the ONLY difference between the two corpus generations, and it is why
they must never be mixed:

| corpus | gate | `basis_digest` |
|---|---|---|
| `coeff_tpc` | pre-tau | `7f954a84...` |
| `coeff_tpc_r1` | `tau = 0.05` | `8c4542b6...` |

They share run names and agree on wavelet, bands, gids, `sigma_norm` and noise
model. Only the surviving coefficients differ. Each is internally consistent, so
a reader pointed at the wrong one is perfectly satisfied and the failure is a
plausible NUMBER rather than an error. `helix/data/identity.py` exists for this.

---

## 2. Plane masking

### The question

The foundation model is trained by masked autoencoding over wavelet
coefficients. Random masking teaches local inpainting. Masking whole planes
should force something harder: inferring a plane's signal from the OTHER planes'
views of the same charge, which is the structure a 3D reconstruction needs.

### Two modes, and why the distinction matters

* **`plane`** — n_planes masked **per VOLUME**, so every volume is punctured.
* **`plane_any`** — n_planes drawn from the whole event, so a volume can survive
  intact. This is the historical behaviour.

With `VIEWS_PER_VOLUME = 3`, measured on the real tokenizer:

| mode | n | cells masked | always-intact structure |
|---|---|---|---|
| `plane_any` | 1 | 16.5% | one volume, always |
| `plane` | 1 | 32.9% | (2,2) |
| `plane` | 2 | 66.1% | (1,1) |

### A hypothesis that was refuted

I claimed the intact volume in `plane_any` gave the model a shortcut: it could
interpolate from a complete view rather than infer across planes, and that this
was why `plane_any` scored poorly.

**It is wrong.** Measured at n=1, `plane` and `plane_any` score the SAME
(0.5061 vs 0.5095) despite `plane` masking twice as many cells. If the intact
volume were the shortcut, removing it should have changed the score. It did not.
The rationale is kept as a quotation in `helix/model/mask.py` rather than deleted,
because the reasoning is a plausible trap worth seeing.

### The result

`plane25` run (112,677 steps) against the 8-run baseline, both cooled, same
evaluator:

| task | coolbase | coolplane25 | delta |
|---|---|---|---|
| random 0.75 | 0.7030 | 0.6948 | **-0.008** |
| `plane` n=1 | 0.5061 | **0.6665** | **+0.160** |
| `plane_any` n=1 | 0.5095 | 0.5742 | +0.065 |

3D probe, same pair:

| probe | coolbase | coolplane25 |
|---|---|---|
| `[mlp] trained` | 0.5326 | **0.8513** |
| `[tri] cross` | 0.7433 | **0.8734** |
| `[tri] solo` | 0.5326 | 0.8513 |
| `xwire` (control) | 0.6789 | 0.6789 |

**Read it as:** plane masking buys a large gain on the cross-plane task and on
the 3D probe, for a cost on random masking that is within noise of zero. The
`xwire` control is identical across checkpoints, as it must be — it is
checkpoint-independent, and a difference there would have invalidated the
comparison rather than supported it.

`plane_frac` selects the mode via the `plane_mode` training option.

---

## 3. Data scale

8x the data buys **representation, not reconstruction**:

* probe **+0.038**, with 8x the standard error
* variance explained **-0.007**
* controls identical

The 8 R1 runs are one distribution (measured), which is what makes pooling them
legitimate.

---

## 4. Settled calls

**R1 is not degraded — ship it.** The L12 probe that suggested otherwise is a
head-geometry detector, and RankMe reads backwards in this setting.

**The 3D probe is sound.** It rises 0.50 -> 0.76 with training at scale. The
instability seen early was starved training and starved evaluation, not a defect
in the metric.

**Charge closure was broken and is fixed** (2026-08-24). Closure is now
`sum E[|X|]`, `charge_bias` became `charge_resid`, and a regression test guards
it. Numbers computed before that date under the old closure are not comparable.

**`m113`'s operating point was never stored in its checkpoint** — `rope_split=0`,
`cell_t=canonical` (helix `grid_center`), `mask=0.75`, `plane_frac`/EMA. Every
evaluation of it before that was discovered used the wrong config. That is the
origin of the eval-artifact format: architecture, operating point, bins and
provenance beside the weights, in one directory that stands alone. m113 now
lives at `archive/fm_m113_artifact` — weights digest
`7d795cc3ab49f90a79f98927647028b5`, trained on the pre-tau corpus
(`basis_digest 7f954a84…`), raw `model` weights rather than the EMA shadow.

---

## 5. Learning rate and batch size

> **READ THIS FIRST (2026-09-20).** Every arm below at B >= 8 ran with
> `num_worker_per_gpu = 0`, and pimm derives the per-rank seed as
> `cfg.seed + rank * num_worker_per_gpu` -- so every rank got the SAME seed and
> drew the SAME mask. Sixteen events masked identically is not sixteen
> independent draws, and it costs **0.18-0.24 val at every step** (measured: the
> same B=16 config re-run after the fix is uniformly that much better). The B=4
> arms are clean (`per_gpu = 1`); the B=8 and B=16 arms are NOT. Treat the
> sqrt(B) claim and the matched-step table as unestablished until re-run. The
> fix is in `configs/pimm/coeff_fm_train.py` (`WORKERS_PER_GPU * _WORLD`).
>
> What SURVIVES: the B=4 learning-rate curve, and everything in the S(B)
> subsection at the end, which was measured after the fix.

**The shipped learning rate is well below optimal, at every batch size tested.**
`configs/pimm/coeff_fm_train.py` carries `lr=1.1e-3`, inherited from the m113
lineage where it was tuned at batch 4 and never re-tuned. Measured 2026-09-19 on
NERSC A100s, 16,000 events per arm, warmup pinned to a fixed 4% of steps so the
schedule shape is identical across arms:

| lr | B=4 | B=8 | B=16 |
|---|---|---|---|
| 5.5e-4  | 3.6543 | 3.8183 | 3.9399 |
| 1.1e-3 *(shipped)* | 3.5610 | 3.7555 | 3.9017 |
| **2.2e-3** | **3.4715** | 3.7419 | 3.8839 |
| **4.4e-3** | 3.4793 | **3.7370** | **3.8702** |
| 8.8e-3  | — | — | 3.9757 |
| 1.76e-2 | — | — | 4.0612 |
| 3.52e-2 | — | — | 4.1039 |

(final `val`; lower is better)

**Noise floor = 0.0026 val.** Two runs identical but for the seed gave 3.5652 and
3.5626 — so anything above ~0.003 is signal. The shipped rate costs **0.094 val
at B=4**, 36x that floor. This is not a large-batch problem; it applies to the
four-GPU configuration as it stands.

**LR\* scales as sqrt(B).** 2.2e-3 at B=4 to 4.4e-3 at B=16 is 2x over a 4x
batch range. Extrapolating, B=128 wants **~1.2e-2**. The B=16 curve is resolved
over 64x and has a clean interior minimum, degrading gracefully above rather than
diverging — so the usable range is wide, but the optimum is distinct and the
penalty is asymmetric: 8.8e-3 is worse than 5.5e-4. **Err low.**

**At fixed DATA, fewer steps costs real progress**, and learning rate does not
buy it back. At each batch's own best lr: 3.4715 (B=4, 4000 steps) ->
3.7370 (B=8, 2000) -> 3.8702 (B=16, 1000). Monotone, and ~100x the noise floor
per halving.

That last row is why "3 epochs" is the wrong way to size a large-batch run. At
B=128 three epochs is 3,521 steps against the production run's 112,679. Whether
the larger batch earns that back by making more progress PER STEP is a separate
measurement — fixed steps, data varying — and it is the one that decides whether
128 GPUs is cheap or merely fast.

### How many GPUs a batch can usefully use: S(B), measured clean

Steps to reach `val = 3.25`, every arm on the same code path, eval cadence and
post-fix seeding:

| B | steps | model `S_min(1 + B_crit/B)` |
|---|---|---|
| 4 | >6,000 (budget ran out; model says ~7,500) | — |
| 8 | **4,750** | 4,679 |
| 16 | **3,250** | 3,250 |
| 32 | **2,250** | 2,536 |
| 128 | **2,000** | 2,000 |

**B_crit = 12.5, S_min = 1,821 steps**, mean error 4% over a 16x range in batch.
The fit uses only B=16 and B=128; **B=8 is held out and predicted to 1.5%**,
which is the reason to believe the two parameters mean something rather than
merely interpolating four points.

The curve is everywhere sub-linear, which the earlier contaminated numbers were
not -- they implied a 16->32 speedup FASTER than perfect scaling, which this
model cannot produce and which should have been read as a broken measurement
rather than a finding.

What it means for sizing a run **at this horizon**: past B ~ 12 the step count
flattens toward S_min, so no batch beats ~1,800 steps to this loss. B=128 uses 8x
the compute of B=16 to save 1.6x the steps -- **5x worse per GPU-hour**. Large
batch buys wall clock, not efficiency.

The qualifier is load-bearing. `B_crit` GROWS as training proceeds (Section 5's
gradient-noise measurement puts it at ~step^0.8, and the two methods agree at
~12 for this horizon), so a 10^5-10^6 step run on a 1-10M event corpus will
support a far larger batch than 12. Do not carry 12 forward as a constant.


## 6. Throughput: what multi-node training actually costs

**The gap to linear scaling is event-size variance, not the network.** Measured
2026-09-20 on Perlmutter A100s, 400 steps per rung, one event per rank:

| GPUs | workers/GPU | s/step | data wait | events/s | vs linear |
|---|---|---|---|---|---|
| 1 | 4 | 0.148 | 0.006 | 6.8 | 100% |
| 4 (1 node) | 1 | 0.166 | 0.006 | 24.1 | 89% |
| 4 (4 nodes) | 1 | 0.162 | 0.006 | 24.7 | 91% |
| 8 | **0** | 0.296 | 0.108 | 27.0 | 50% |
| 16 | **0** | 0.311 | 0.110 | 51.4 | 48% |
| 32 | 4 | 0.229 | 0.006 | 139.7 | 65% |
| 64 | 4 | 0.236 | 0.006 | 271.2 | 63% |
| 128 | 1 / 4 / 16 | 0.241 / 0.244 / 0.241 | 0.006 | ~527 | 61% |

Three things are ruled out by that table and are worth stating because each was
the first guess at some point:

* **Not the interconnect.** `4x1` and `1x4` run the SAME global batch with the
  same work per rank and differ only in whether the all-reduce crosses the
  fabric or stays on NVLink. It costs **-2.4%** -- the fabric version is
  marginally faster. (This is after the libnl fix; before it, NCCL fell back to
  TCP and the same all-reduce took 11.8x longer.)
* **Not the filesystem.** At 128 GPUs, 1 / 4 / 16 workers per GPU -- 128 to 2,048
  concurrent Lustre readers -- give 0.241 / 0.244 / 0.241 s with the data wait
  pinned at 0.006 s.
* **Not the dataloader, any more.** The 50% and 48% rows are a config bug:
  `num_worker` is a GLOBAL count that pimm divides by world size, so the shipped
  literal 4 became ZERO workers per GPU above eight ranks and every batch was
  read inline. See `configs/pimm/coeff_fm_train.py`.

**What it is.** The FM takes one event per rank and DDP makes a step cost the
SLOWEST rank. Events are not the same size: over 1,600 corpus events the
coefficient count runs **39,462 to 759,710, a 2.77x spread**. So a step costs the
MAXIMUM of N draws from that distribution, not the mean, and the penalty grows
with N.

That is not an analogy, it is the model. Fitting step time against event size on
ONE GPU -- where no rank waits for another -- gives

    step = 91 ms + 0.228 us per 1,000 coefficients        R^2 = 0.88

and feeding those two constants plus the corpus size distribution into
`a + b * E[max of N]` predicts every other rung with **nothing fitted to the
multi-node data**:

| GPUs | predicted | measured | error |
|---|---|---|---|
| 1 | 152.7 ms | 148.0 | +3.2% |
| 4 | 176.6 ms | 166.0 | +6.4% |
| 32 | 204.4 ms | 229.0 | -10.7% |
| 64 | 212.7 ms | 236.0 | -9.9% |
| 128 | 220.2 ms | 244.0 | -9.8% |

The residual is systematic -- the model under-predicts by a near-constant ~10%
from 32 GPUs up -- and that residual is the genuine all-reduce and sync cost a
single-GPU fit cannot contain. So: **~90% of the multi-node penalty is event-size
stragglers and ~10% is communication.**

**The obvious fix does not work, and that is the result.** If a step costs the
largest event in it, bucketing events by size so every rank gets comparable work
should remove `E[max]` from the expression and leave `a + b * mean` = 153 ms at
any node count -- 1.6x at 128 GPUs. It was implemented
(`helix/integrations/pimm/sampler.py`, `bucket_by_size`) and measured at 16 GPUs
against an otherwise identical run:

| | median s/step | final val | var_expl |
|---|---|---|---|
| bucketing off | 0.198 | **3.4023** | **0.4436** |
| bucketing on | 0.214 | 3.5189 | 0.3667 |

**8% slower AND 0.117 worse val**, the latter consistent at every matched step
from 1,000 on and ~45x the 0.0026 seed-to-seed noise floor. Both arms ran 3,000
steps at B=16 with the same seed and rate; only the sampler differed.

The ordering itself is correct -- unit tests pin within-step max/mean below 1.15
against ~1.5 for a random batch -- so the batches really are homogeneous, the
step times simply do not follow, and the convergence cost that homogeneity was
always going to risk is real rather than hypothetical. Whatever cancels the
straggler gain (a per-node resource contended by all four GPUs when every rank
is large at once is the obvious candidate) was not worth chasing: a few percent
of step time is not where the returns are.

So the predictive model in the table above stands as a DESCRIPTION -- it forecasts
five node counts from two single-GPU constants -- but not as a lever. The 61%
scaling efficiency at 128 GPUs is what the machine gives, and a few percent
either way is not worth chasing.

`bucket_by_size` stays **0**. The code is kept because it is tested and costs
nothing switched off, and because the same machinery would be needed if event
sizes ever became far more skewed than 2.77x.

**A caution learned the hard way.** The first A/B measured nothing. The flag was
set, the trainer logged "length bucketing ON", and the sampler then fell back to
pimm's ordinary order because the REGISTERED dataset wrapper
(`helix/integrations/pimm/data.py`) did not forward `event_sizes()` -- the third
time that wrapper's re-declaration has hidden something the inner dataset
gained, and the first time it degraded instead of raising. Both arms were
identical runs and the 8% between them was noise. Check that a run did what its
config said before reading a number off it.

---

## 7. Open

**Charge R2 is unrun.** It is the metric that would separate "the model
represents charge" from "the model represents where charge is".

**Bins are per-corpus-generation and this is a trap.** The edges shipped with
`m113` were derived from an old cache built with a WHITE noise model; the current
corpus is colored, coherent + incoherent. Bins are training-set statistics.
Re-derive them per corpus (`scripts/derive_coeff_bins.py`), never inherit them.

---

## 8. What the retired cache was, and why it could not be reused

540 GB of per-event `.npz` (199,990 events, built Jun 13) was deleted on
2026-09-11. It is recorded here because "we threw away 540 GB of training data"
deserves a reason that survives.

It was superseded by `coeff_tpc_r1` on four independent counts:

1. **MORE events, and this is the one reason that does NOT hold.** The cache
   held 199,990 events against the corpus's 157,991. An earlier version of this
   document claimed the opposite, from a count that summed each shard's
   `n_events` attribute — but `coeff_tpc_r1` writes TWO coeff shards per source
   file (100 source files -> 200 shards per run), so that sum double-counts.
   Counting distinct `(source_file, event)` pairs gives 19,999 per run and
   157,991 across the 8 runs, which is exactly what
   `configs/pimm/coeff_fm_train_8run.py` records as its split
   (150,239 + 4,641 + 3,111). The cache was the larger set of events. It was
   still unusable, for the three reasons below, but "it was smaller" was not one
   of them and should not have been offered as one.
2. **Wrong noise model** — white, where the corpus is colored with coherent and
   incoherent components. helix's own `scripts/derive_coeff_bins.py` says so:
   "the edges shipped with m113 were derived from the old cache, which used a
   different (white) noise model."
3. **Pre-tau** — it predates BOTH sharded corpora (§1).
4. **Unstamped** — no `basis_digest`, no run or event id, no config. Nothing
   could prove what a model trained on it had seen, and `identity.py` could only
   warn.

The one thing it uniquely held was `val_clean`, the paired noise-free target that
r1 shards do not store. That is regenerable at current physics —
`build_coeff_corpus.py` emits it by design — and the cache's copy was paired to
the wrong noise anyway.

Full evidence, and file lists of all 220,333 deleted files, are in
`$HELIX_ARCHIVE/retirement-backups/caches-retired_2026-09-11/`.

---

## 9. Scaling results after the port (2026-09-24 to 10-04)

All on the 8-run corpus (150,239 training events), muP, WSD schedule, scored at a
fixed 0.75 mask. "Cooled" = a 10% `(1-sqrt p)` cooldown; laws are fit on cooled
loss only (the cooldown gain grows with stable length: +0.04 after 4.5k steps,
+0.17 after 47.5k at d512, and rankings at equal steps survive it).

**Recipe now in use:** QK-norm (`model.qk_norm=True`), FiniteGuard, LR 4.4e-3
with warmup `500*sqrt(d/128)` steps, mask 0.75, `plane_frac` 0.1, 4-block decoder.
A d768 long run went NaN at 19.4k steps (qkv-bias growth 7 -> 18); QK-norm fixed
the stability (cooled 2.712 / probe 0.862 vs 2.716-2.724 / 0.83-0.85 without).

**Learning rate.** One LR transfers across width once warmup scales with
sqrt(d) (d1024 at 4.4e-3 with a 1,414-step warmup: 2.939). The optimum falls with
run length: d512 B=16 over 44.8k steps prefers 2.2e-3 (2.764) to 4.4e-3 (2.778);
B=64 cooled prefers 6.2e-3 (2.717) to 8.8e-3 (2.745). The stability edge sits
4-8x above the optimum; sqrt(B) scaling to B=512 lands inside it.

**Critical batch** (cooled, examples to reach a target): B_crit ~33 events at
val 2.90, ~40 at 2.85, ~59 at 2.80 (~2M cells). Constant in CELLS, not events:
cropping events to 1/2 and 1/4 raised B_simple by 2.2x and 5.8x. Large batch
buys wall clock, not compute.

**Width at equal compute ("tier 1",** ~9,500 s of training on 16 nodes, B=64,
cooled in-run): d768 2.620 (47k steps), d1024 2.579 (33.2k), d1536 2.579 (17.7k),
d2048 2.616 (11.5k). Width beats depth (d768x12 2.962 vs d512x24 2.993 at equal
cost). The cooled fit on 12 points, E 2.47, size exponent 0.35, data ~1.0,
underpredicts long runs (2.71 predicted for a d768 run that reached 2.585).
Best cooled loss so far: d1024, LR 6.2e-3, two 16-node slots: 2.519 (var_expl 0.787).

**Repetition** (cooled on each run's own events): d512 loss flat to 77 epochs
(2.785 at 5 and 19 epochs), +0.011 at 310; d768 +0.046 at 310. The probe is noisy
(0.70-0.87 across seeds). The 150k-event corpus supports ~10M presentations.

**Seed noise.** Early pairs: 0.002-0.008 in loss, ~0.03 in probe. The QK-norm d768
pair (29.2k steps, `ab_ref` vs `res_pw16_s1`) differs by 0.023 (2.718 vs 2.741);
treat single-seed d768 differences below ~0.02 as unresolved.

**Design screens** (d512 and d768): defaults kept. `plane_frac=0` collapses the
along-wire probe (0.51-0.56); an 8-block decoder gives the best loss and a worse
probe (0.68-0.77); mask 0.85 probes best (0.866, within noise). Loss does not rank
representations: rank correlation with the probe is 0.15 over 17 models, and the
probe itself saturates at ~0.83-0.87 for good d768 models.

---

## 10. Patch size and resolution

Validation loss cannot compare tokenizers (a bigger masked token is a harder
target), and the along-wire probe scores per patch. `helix.probe.resolution`
scores frozen features on a fixed 2-wire x 16-tick grid instead
(`scripts/dump_resolution_truth.py`, then `scripts/eval_resolution.py`): charge
map, faint-charge detection, floor efficiency at a 1% false-positive rate (and
floor AUC), localisation, close-pair separation, each with an event-bootstrap
interval. Cells no token covers are predicted 0: the probe never sees an all-zero
input, and its extrapolated constant used to set window maxima -- the first
version of this table carried that bug, and it reversed the pw32 conclusion.

Equal-step runs, d768 QK-norm recipe, 29.2k steps, one token change each
(variants in `configs/pimm/variants/`; tokens per event vs today's 32,480):

| metric | pw8 | pw16 s0 | pw16 s1 | pw32 | pt16 | noD2 |
|---|---|---|---|---|---|---|
| tokens vs today | 1.29 | 1.00 | 1.00 | 0.74 | 0.74 | 0.75 |
| charge map r | 0.967 | 0.938 | 0.939 | 0.891 | 0.888 | 0.926 |
| floor eff @1% FPR, <0.1 MeV | 0.22 | 0.18 | 0.26 | 0.24 | 0.10 | 0.05 |
| floor eff @1% FPR, 0.1-0.2 MeV | 0.54 | 0.39 | 0.48 | 0.35 | 0.13 | 0.16 |
| floor eff @1% FPR, 0.2-0.5 MeV | 0.97 | 0.93 | 0.94 | 0.89 | 0.58 | 0.81 |
| floor AUC, 0.1-0.2 MeV | 0.93 | 0.92 | 0.92 | 0.92 | 0.89 | 0.84 |
| localisation >1 MeV, wires / ticks | 0.35 / 2.5 | 0.54 / 3.1 | 0.58 / 2.8 | 0.73 / 3.2 | 0.83 / 4.6 | 0.60 / 3.0 |

At equal cost (0.74-0.75x tokens) the three reductions are not equal. Coarser
TIME (pt16) and dropping D2 both lose the floor (efficiency at 0.1-0.2 MeV 0.13
and 0.16 against 0.39-0.48); D2 appears to help reject noise although it rarely
carries a faint deposit alone (floor AUC 0.92 -> 0.84), and pt16 also loses time
localisation (3.1 -> 4.6 ticks). Coarser WIRE (pw32) keeps most of the floor
(0.35 at 0.1-0.2, AUC unchanged) and costs wire localisation (0.54 -> 0.73) and
map fidelity. Finer wire (pw8, 1.29x tokens) localises ~35% better and is at
least as sensitive.

How big a difference is real: the two pw16 seeds agree to 0.003 on map/AUC
metrics and ~0.03 wire on localisation; floor efficiency spreads ~0.08 between
them (and ~0.05 when the truth windows are re-drawn), so floor differences below
~0.1 are not resolved. Close-pair separation spread up to 0.12 between seeds and
is not used for decisions.
