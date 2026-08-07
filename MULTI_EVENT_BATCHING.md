# Multi-event batching for the coefficient FM

**Status: deferred, deliberately.** The FM processes ONE event per forward pass.
This document records why, what the hazard is, and exactly what the work would be
if we return to it — so the decision can be re-opened on evidence rather than
re-derived from scratch.

Written 2026-08-06, against `helix.model` at the point where the tokenizer gained
its inverse.

---

## 1. TL;DR

* One event is already ~31–40k tokens. **The GPU is saturated ~8× below that**, so
  batching several events into one forward buys essentially **no throughput**.
* The model has **no event separation at all**. Concatenating two events lets
  event A's tokens attend to event B's — silent cross-event leakage, the same bug
  class pimm fixed for Panda in `9491b0b`.
* pimm-data **already has every piece** needed on the data side (role-driven
  collate, offsets, second row-spaces, `node_bases`). Nothing to add there.
* The work is entirely helix-side and is **~50–70 lines**, but it buys
  *correctness*, not speed. That is why it is deferred rather than dropped.

## 2. The measurement that decided it

`SerialFMModel` (m113 config: d=512, 12 enc, 4 dec, 8 heads, 59.2M params,
gp=1024, gd=2048), bfloat16 autocast, RTX 2080 Ti 11 GB, `raw_heads` forward:

| n_cells | forward (ms) | µs/cell |
|--------:|-------------:|--------:|
|   4,000 |         87.9 |   21.98 |
|   8,000 |        169.3 |   21.17 |
|  16,000 |        326.2 |   20.39 |
|  32,000 |        645.9 |   20.19 |

An **8× increase in tokens buys 8% better per-cell cost**. The curve is flat by
4,000 cells. Real events (from the tokenizer golden) are 30,976 and 39,689 cells
— 8–10× past the saturation point, and one event nearly fills 11 GB.

So the usual reason to batch — filling the device — does not apply. This is also
why the research trained the way it did; `fm/mae_ddp.py` line 3:

> Each rank processes 1 event/step; gradients all-reduced => global batch = world events

That is the right answer for this data shape, not a limitation worked around.

**Caveat.** Measured on a 2080 Ti. A larger card raises the saturation point and
the memory ceiling. The conclusion is robust to that (the curve is flat an order
of magnitude below one event) but if training moves to A100/H100 and someone
wants several events resident, **re-measure before acting on this document**.

## 3. The current contract

* One event per sample; `batch_size=1` per rank; scale via DDP world size.
* At B=1 pimm's `collate_fn` is a passthrough (`torch.cat` of a one-element
  list), so no index needs rebasing and both loss paths work.
* Do **not** `Collect` `n_cells`: collate turns the int into `tensor([30976])`
  while `make_mask` does `n = B["n_cells"]`. `FMModel.forward` derives it from
  `plane_id.shape[0]`, which is also correct for a concatenated batch.

## 4. The hazard, stated plainly

`helix/model/serial.py` contains zero references to offsets, batches or event
boundaries. Attention runs over whatever tokens it is given:

* `_sched` sorts **globally** (`argsort(plane*1e9 + t)`), so tokens from
  different events interleave and share attention groups.
* `uniform_attn` / `grouped_cross` pad to a multiple of `g` over the **whole**
  input, not per event.

Setting `batch_size=2` therefore does not fail — it silently trains a model whose
tokens attend across unrelated events. Nothing in the loss, the metrics or the
logs would show it.

**Until this is implemented, `batch_size` must stay 1.** The cheapest interim
protection is a ~10-line guard: accept an optional `offset` and raise if it
implies more than one event. That converts a silent correctness bug into a loud
failure without doing any of the real work below.

## 5. What pimm-data already provides (no changes needed)

`pimm_data.collate.collate_with_roles` is role-driven, and the roles already
cover this exactly:

| role | behaviour |
|---|---|
| `point` / `raw` | concat along dim 0 |
| `<part>_offset` | per-event counts → cumulative |
| `('instance', offset_key)` | rows live in a **second row-space** counted by that offset |
| `('edge', 'self' \| (src,dst))` | concat **with index shifting** |
| `label` | concat + distinct renumber |

plus `_roles.node_bases(offset)` ("the amount to add to event *i*'s within-event
indices so concatenated index arrays stay globally valid") and
`batch_ops.split_event` to undo the concat.

Coeff tokens map onto it with nothing new:

```
part `tok`; tok_offset counts CELLS
    tok_{inp,occ,valid,tgt,band_id,plane_id,t_phys,wire_pos,wirefeat,cell_wb,cell_tb}
        -> point (concat)

second row-space; tok_active_offset counts ACTIVE SLOTS
    tok_{cell,slot,target}
        -> ('instance', 'tok_active_offset')
```

This is the same shape as the optical case already in production
(`('instance','sensor_wave_offset')` for packed waveform samples). Per that
convention `cell` stays per-event-local and the consumer adds
`node_bases(tok_offset)`.

**Wiring note.** pimm's `Trainer` uses *pimm's* `collate_fn`, not pimm-data's
role-driven one. `build_train_loader` is a `Trainer` method, so an `FMTrainer`
subclass overrides it to pass `collate_with_roles`.

## 6. What helix would need

Four changes, all in `helix/model/`:

1. **`forward` / `raw_heads` accept an optional `offset`** (cumulative cell
   counts, no leading zero — pimm-data's convention). Absent ⇒ one event ⇒
   today's behaviour exactly.

2. **Event-major ordering in `_sched`.** Vectorised, no Python loop over events:

   ```python
   o = torch.argsort(key)                                   # within-event key order
   o = o[torch.argsort(batch_idx[o], stable=True)]          # then event-major
   ```

   `batch_idx` comes from `offset_to_batch(offset)` (helix needs its own
   3-line copy; it must not import pimm-data).

3. **Per-event padding in `uniform_attn` / `grouped_cross`.** They already pad to
   a multiple of `g` (`npad = ((T + g - 1) // g) * g`); make that per event so no
   attention group ever spans two events. **At B=1 this computes the same `npad`
   as today**, so the single-event path stays bit-identical — check it against
   `tests/goldens_fm.json`.

4. **Per-event masking + index rebasing.** `make_mask` must draw per event
   (`mask_mode='plane'` selects planes, and across a batch that must be per
   event, not global). The sparse loss must rebase `cell` by
   `node_bases(offset)`; the dense `losses_fused` path needs nothing.

### Why padding, not masking

A dense block-diagonal attention mask is the obvious alternative and is much
worse here. Grouped attention costs `nb·g² = T·g`; a mask covering it needs
`T·g` booleans ≈ **36M per event**. Per-event padding costs at most `g-1` extra
cells, averaging `g/2` ≈ 512 — about **1.6%** at the measured 20 µs/cell.

Padding is ~4 orders of magnitude cheaper than masking for this geometry. Flash-
attn varlen (`cu_seqlens`) would remove even that 1.6%, but adds a dependency to
buy back 1.6% of a thing that already buys no throughput.

## 7. Acceptance criterion

**Bit-exact batch invariance.** For events `e1`, `e2`:

```
forward(concat(e1, e2), offset=[n1, n1+n2])   sliced per event
    ==   forward(e1)  and  forward(e2)        run separately
```

Exact equality, not tolerance — the goldens already pin CPU/single-thread
determinism (`tools/capture_fm_golden.py`), so any cross-event leakage shows up
as a bit difference rather than a statistical wobble. Also re-run
`tools/capture_fm_golden.py --check` to confirm the B=1 path did not move.

## 8. When to revisit

Reopen this if any of these becomes true:

* **Training hardware changes** and several events fit comfortably — re-measure
  the µs/cell curve first; the whole argument rests on it being flat.
* **Events get much smaller** (a coarser tokenizer, fewer bands, a patch-size
  change). If a typical event drops near the ~4k-cell saturation point, batching
  starts to matter.
* **Eval wants multi-event passes** — e.g. a probe that needs many events'
  features in one forward. This is the most likely trigger, and note it needs
  only the *inference* path to be event-aware, which is strictly less work
  (no mask, no loss rebasing).
* **Someone needs `batch_size>1` for any reason.** In that case implement
  section 6 first; do not raise the config value against the current model.
