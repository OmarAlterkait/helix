# The three open items

Everything else this round produced is either landed (`docs/PERFORMANCE.md`) or
retired with a number against it (`docs/REVIEW_FIELD.md` §7, §9b). These three
are what is left, and each is stated the same way: **what is measured, what is
proposed, and what has to be tested before anyone believes the proposal.**

The discipline matters here. Six claims in this round were written down and then
falsified by their own follow-up measurement — including two of mine. Nothing
below should be acted on from the reasoning alone.

---

## 1. CRPS, per-slot NLL and PIT in `CoeffFMEvaluator`

**Measured.** The categorical head is calibrated: PIT mean 0.512, std 0.289 over
5.4M held-out slots, against the 0.5 / 0.2887 of a uniform PIT. The read-back is
a posterior mean and a posterior mean is MMSE-optimal, so **`var_expl` = 0.708 is
at its Bayes limit and no loss change can raise it.** Separately, the probe's
per-group r has std 0.344 over 897 groups (SEM 0.0115), so it cannot adjudicate
an effect below ~0.013 on this corpus.

**The consequence.** There is currently no instrument in the codebase that can
measure an objective-side change. `var_expl` is pinned; the probe is too coarse.

**Proposed.** Add three numbers beside `var_expl`:

| | what it scores | units | note |
|---|---|---|---|
| per-slot NLL | likelihood | nats | = the training CE on held-out data; **not** comparable across bin counts |
| CRPS | the whole predictive distribution | asinh(coeff/sigma) | reduces to \|pred − truth\| for a point forecast; IS comparable across bin counts |
| PIT mean/std | calibration | — | uniform ⇒ 0.5 / 0.2887 |

**To test before trusting it.** Reproduce the three numbers above from the
evaluator on the same held-out split (CE 2.941, CRPS 0.6717, PIT 0.512/0.289) —
if the evaluator disagrees with `tools/profile/i6_metrics.py` (on the
`perf/fast-path` branch), one of them is
wrong. Then confirm the metric MOVES on a change `var_expl` cannot see; a metric
that never separates two models is not yet earning its place.

**Cost.** No training. One evaluator change.

**Watch for.** The open outer bins are stored as **±1e18**, not `±inf`, despite
`helix/data/bins.py` saying "extended to +-inf". A CRPS integral that trusts
`isfinite` returns ~1e14. That cost one run here.

---

## 2. The RoPE bandwidth fix

**Measured.** `helix/model/layers.py` `rope_tables` — when an axis is disabled,
`apply_rope` leaves that half of every head **unrotated** rather than
reallocating it to the live axis. `tests/test_serial.py` asserts this
behaviour, so it is a fact about the code, not a reading of it.

**The consequence.** The result the production setting rests on — "removing wire
RoPE freezes reconstruction, var_expl ~2%" — **cannot distinguish "wire RoPE is
essential" from "half the positional dimension vanished".** `rope_split=False`
is therefore supported by a confounded negative, not by evidence.

**Proposed.** A single-axis path: `rope_angles(pos, 2*hd)` to get `hd/2`
frequencies, applied across the full head width, so a disabled axis costs
nothing rather than half the bandwidth. ~5 lines.

**To test before concluding anything about wire RoPE.** Re-run `wire_rope=0` and
`rope_split=True` against the corrected full-width baseline. Three outcomes,
three different conclusions:

* the collapse persists ⇒ wire RoPE is genuinely load-bearing, and
  `rope_split=False` is vindicated for the first time;
* the collapse disappears ⇒ the original result measured bandwidth, and
  time-only RoPE on the cross-plane layers becomes a live option — which is what
  the multi-view literature favours, since the wire axis is not metric across
  planes;
* something between ⇒ report it as between. Do not round.

**Cost.** ~5 lines, then short runs. Judge on the probe AND CRPS — the probe
alone cannot resolve a small effect.

---

## 3. The tokenizer's cell geometry

**Measured.** ~32,000 tokens per event against PoLAr-MAE's ~400 on the same
detector family. 6.87% slot occupancy. The binding constraint on training is the
batch: **4 independent events per step**, pushing 2.5x PoLAr-MAE's tokens through
32x fewer independent samples. Cell count, not coefficient count, is what the
current code costs (t = 7.0 vs t = 0.4 in a two-variable fit).

**Cheaper than it looks.** CLAUDE.md is explicit that cell geometry is a config
change plus a retrain, **not** a 344 GB corpus rebuild — the corpus stores
coefficients and `pw`/`pt` are applied at tokenize time. The bin table is over
coefficient values, so it survives. Only `n_slot` changes, so `val_head`
reshapes and there is no warm start.

**Proposed, as the cheapest probe of the question.** `pw x pt = 32 x 16`
(`n_slot = 512`): 4x fewer tokens at unchanged occupancy, which **isolates token
count from density**. Batch 16 at the same memory.

**To test before believing it helps.** Equal-wall-clock, not equal-step: the
whole point is that the smaller token count buys batch. Score the probe and
CRPS. And measure, do not assume:

* occupancy actually unchanged (it is 6.87% at 16x8; confirm at 32x16);
* the token count actually drops ~4x on real events, not on the tiling model;
* whether the probe's patch tuple still resolves what it needs at coarser cells
  — a 32x16 cell is a coarser label, and PoLAr-MAE's own reported failure is
  fine structure (Michel F1 0.440) attributed to fixed-resolution tokenization.
  That is the risk this change carries, and it is the one to look for first.

**Cost.** One config change, one full retrain, no corpus work.

---

## Not on this list, and why

`docs/REVIEW_FIELD.md` §7 lists ten things a reviewer will "correct" and be
wrong about. §9b retires, with a number each: the `ceil` padding partition,
plane-pure blocks, the Reformer/boundary cluster, varlen-for-correctness
(0.016%), bin count (oracle read-back 0.999917), and register tokens (carriers
reconstruct *better* than matched-occupancy peers — the opposite of the
signature). Hyperparameters, muP and decoder width are deliberately out of scope
here.
