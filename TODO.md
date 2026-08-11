# Deferred work

Things consciously left undone, with enough context to pick up cold. Ordered by
what blocks what, not by size.

Larger deferred designs get their own file: see `MULTI_EVENT_BATCHING.md`.

---

## 1. `losses_cat` memory — an efficiency win, NOT a blocker

**Status:** deferred. Not blocking anything; cap `--n-cells` on a small card.

Correction to an earlier reading of this: full-event training with the
categorical head **was already done**. `fm/data.py::get_cached` accepts a `cap`
argument and never applies it —

```python
return _to_fm(B)     # full token count (~25-37k, fits)
```

— so m113 trained on whole events on the research hardware. Nothing here blocks
that.

What it is: `helix/model/loss.py::losses_cat` finds the true bin per slot with

```python
binid = (tgt.unsqueeze(-1) >= ec[:, None, 1:-1]).sum(-1).clamp(0, K - 1)
```

The comparison is bool, but `.sum(-1)` accumulates in **int64**, so the
intermediate materialises at 8 bytes/element:

```
30976 x 128 x 127 = 503,545,856 elements  x 8 B = 3.75 GiB
```

That is a transient spike on top of the backward activations. It has headroom on
an A100/H100 and tips over an 11 GB card, which is why
`scripts/smoke_train_fm.py` has `--n-cells`.

Two independent improvements, either of which helps:

* `.sum(-1, dtype=torch.int16)` keeps the shape but drops the intermediate ~8x
  (3.75 GiB -> ~470 MB). One-word change, same algorithm.
* `torch.bucketize(tgt, ec[b])` removes the intermediate entirely, O(N log K).

Both should be bit-identical (all three are "count edges below the value"), but
`losses_cat` is extracted-verbatim research code, so either wants a test
asserting identical `binid` and identical loss on real data before it lands.

Worth doing for the headroom — it raises the model/event size that fits on a
given card — not because anything is currently blocked.

## 2. `FMTrainer` — only when a real run is wanted

**Status:** deliberately not built. `scripts/smoke_train_fm.py` covers the
"does the loop work" case without any pimm dependency.

The point of using pimm at all is to stop maintaining a bespoke training loop:
pimm brings DDP, checkpoint/resume, W&B + structured logging, the hook lifecycle,
slurm launch, and — the real prize — the pretrain eval suite
(`MAEEvaluator`, `EventProbeSuiteEvaluator`, linear probes).

When it is built it is two method overrides:

* `build_optimizer` → `AdamW(model.param_groups(lr, weight_decay=wd), lr=lr,
  betas=(0.9, 0.95))`. pimm's own `build_optimizer` groups parameters by
  substring matching on names, which cannot express muP.
* `build_scheduler` → preserve the per-group LR ratios. `fm/mae_ddp.py` does
  `ratio = [pg["lr"]/lr ...]` then `pg["lr"] = lr_at(step) * ratio[i]`. A
  scheduler that sets one LR for all groups **silently discards muP**. pimm
  expresses this as `OneCycleLR`'s per-group `max_lr` list.

Home: `helix/integrations/pimm.py`, with the other adapters — it encodes model
knowledge (use the model's own param groups) rather than training policy, and it
keeps pimm untouched.

## 3. Re-derive the categorical bins — marginal

**Status:** optional. The existing bins still fit.

`tier1_bins.pt` was derived from the old white-noise cache. Measured against the
corpus: overflow 0.01-0.20% (design target ~0.1%) and the corpus fills 92-104% of
the grid's range. Re-deriving with the retained `tier1_setup_bins.py` would
recover ~6-8% of range in bands 0-2 and pull band 3's 0.20% overflow back to
~0.1%. Worth doing when convenient, not a correctness issue.

Note the bins are UNIFORM in `arcsinh(clean/sigma)` space (= log-spaced in raw
charge, constant relative precision), NOT quantile-matched — so skewed bin
occupancy is by design and is not evidence of staleness.

## 4. Mirror the jax forward ops

`helix/tpc/{noise,dense_ops,geometry}.py` mirror pimm-data for the numpy and
torch paths, pinned by `tests/test_forward_mirror.py`. The **jax** path
(`noise_jax`, `dense_ops_jax`) is not mirrored, so `build_coeff_corpus.py
--backend jax` still imports pimm-data. torch is the production backend, so this
is the last loose end rather than a blocker.

## 5. Reusing pimm's MAEEvaluator needs two changes

The coeff FM IS a masked autoencoder, so `pimm/engines/hooks/eval/pretrain/mae.py`
is the natural eval hook. It does not fit as-is, in two different ways — both
pinned by `tests/test_pimm_step_contract.py` so neither surprises us on the first
eval step:

* **Hard failure.** It calls `model(input_dict, return_pred=return_viz)`.
  `FMModel.forward(batch, tok_mask=None)` does not accept that kwarg, so eval
  raises `TypeError` immediately. Fix by accepting (and honouring) `return_pred`,
  or by writing a coeff-specific evaluator.
* **Silent degradation.** It reads `coord_loss`, `feat_loss` and
  `mask_ratio_actual` via `.get(..., 0.0)`. We emit `bce`, `val` and
  `masked_frac`, so those metrics would log as 0.0 rather than fail — the worse
  of the two failure modes.

Its point-cloud visualisation path (`viz_visible_coord`) has no coeff analogue
either. A thin `CoeffFMEvaluator` reusing its averaging + `neg_val_loss`
checkpoint-selection logic is probably less work than bending it.

## 6. Verify the pimm registry wiring

`helix/integrations/pimm.py` registers three names, and
`tests/test_integration_pimm.py` checks them — but both tests **skip** here,
because this environment has pimm checked out without its dependencies
(`pyarrow`, `addict`). Nothing has exercised `build_dataset` / `build_model`
through a real config. Needs an environment with pimm's deps installed.

## 7. The retirement bundle — gated on FMTrainer parity

13 modules, ~3,700 lines, all reachable from `fm/mae_ddp.py`:

```
fm/     mae_ddp  model  model_serial  train  data
top     star_tpc  measure_coeffs  vit_tpc  star_model  vit_model
        baseline_tpc  doraemon_optical  onfly_optical
```

They form one connected graph — removing any member breaks the A/B reference —
so they retire together, in the commit that shows the new trainer reproduces the
old one. `research/goldens/capture.py` depends on this bundle too (it calls
`star_tpc.prep_tpc_rows`), so the DSP golden retires with it.

Caveat discovered late: the A/B can only verify **loop mechanics**. `mae_ddp`'s
recorded numbers are on the white-noise distribution, so it cannot serve as a
physics reference against a corpus built with the measured spectrum.

## 8. Housekeeping

* `tests/test_coeff_dataset.py` in pimm-data pins the cross-repo codec golden to
  the hardcoded path `/sdf/group/neutrino/omara/helix-consolidate`. As
  `extraction` diverges, it silently compares against the wrong tree. Point it at
  the installed helix.
* Version split: `pyproject.toml` says `0.1.0`, `helix/__init__.py` says `0.2.0`.
* The 10 back-compat flat shims (`helix/_backend.py`, `helix/io.py`, …) are
  imported by 6 scripts and 4 tests, all internal. helix has no external
  consumer, so they are dead weight — delete and repoint the importers.
* Branches are held, not merged: helix `extraction`, pimm-data `coeff-corpus`.
