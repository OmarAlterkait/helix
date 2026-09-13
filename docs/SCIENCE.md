# Science record

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
`cell_t=canonical`, `mask=0.75`, `plane_frac`/EMA. Every evaluation of it before
that was discovered used the wrong config. This is why
`tools/convert_fm_ckpt.py` writes a self-contained checkpoint: architecture,
weights, bins and provenance in one file.

---

## 5. Open

**Charge R2 is unrun.** It is the metric that would separate "the model
represents charge" from "the model represents where charge is".

**The deferred dense-chain tests.** 32 tests for sparse -> densify -> noise ->
digitize, parked in `tests/_deferred/` because the chain now spans the
helix/pimm-data boundary and needs fixtures from both trees.

**Bins are per-corpus-generation and this is a trap.** The edges shipped with
`m113` were derived from an old cache built with a WHITE noise model; the current
corpus is colored, coherent + incoherent. Bins are training-set statistics.
Re-derive them per corpus (`scripts/derive_coeff_bins.py`), never inherit them.

---

## 6. What the retired cache was, and why it could not be reused

540 GB of per-event `.npz` (199,990 events, built Jun 13) was deleted on
2026-09-11. It is recorded here because "we threw away 540 GB of training data"
deserves a reason that survives.

It was superseded by `coeff_tpc_r1` on four independent counts:

1. **Smaller, not extra** — 199,990 events against the corpus's 315,982.
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
