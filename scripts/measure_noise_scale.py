"""Gradient noise scale: the batch size past which more parallelism stops paying.

WHAT THIS MEASURES AND WHY IT IS NOT A SWEEP
--------------------------------------------
A batch-size sweep answers "which B was best on THIS corpus at THIS horizon".
It is expensive -- one full training run per point -- and it does not
extrapolate, which matters here because the corpus this repository trains on
today (157,991 events) is one to two orders of magnitude smaller than the one it
is meant for, and the number of epochs is meant to grow with it.

The gradient noise scale answers the underlying question directly, from a SINGLE
checkpoint, in minutes on ONE GPU. For a batch of size B the expected squared
gradient norm is

    E|G_B|^2  =  |G|^2  +  S / B          S = tr(Sigma), the per-example variance

which is a straight line in 1/B. Its slope over its intercept

    B_simple  =  S / |G|^2

is the batch size at which gradient noise and gradient signal are equal. Below
it, doubling B roughly halves the steps to a target; above it, doubling B buys
almost nothing and the extra hardware is spent on variance reduction that the
optimiser cannot use. It is the same quantity that appears in

    S(B) = S_min * (1 + B_crit / B)

as B_crit, so this script and a matched-step sweep measure the SAME thing by
different routes -- which is the point: the sweep calibrates, this extrapolates.

The estimator is McCandlish et al. (2018), "An Empirical Model of Large-Batch
Training", and the one property that makes it cheap is that a batch of B events
costs the same forward/backward work whether it is run as one batch or as B
single-event passes accumulated. helix's FM takes exactly one event per rank
(``MULTI_EVENT_BATCHING.md``), so the accumulated form is not even a
reformulation -- it is what training already does.

WHAT IT DOES NOT MEASURE
------------------------
* **Adam.** The relation above is derived for SGD. With a preconditioner the
  relevant noise lives in the UPDATE, not the raw gradient, so the number here
  is an approximation. Treat its TREND across checkpoints as the result, and
  calibrate its absolute level against a matched-step sweep (docs/SCIENCE.md
  Section 5 has one at B=4, 8, 16).
* **A target loss.** B_simple grows as training proceeds -- that is the whole
  reason to measure it at several checkpoints rather than one. A single number
  from a single checkpoint says nothing about where a long run ends up.

The masking is part of the noise, deliberately. Two forward passes on the SAME
event give different gradients because the mask is redrawn, and that variance is
real variance in the training signal: it is what a larger batch averages over.
Fixing the mask would report a noise scale for a procedure nobody runs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _powers_of_two(n):
    """1, 2, 4, ... up to and including n."""
    out, b = [], 1
    while b <= n:
        out.append(b)
        b *= 2
    return out


def _fit(points):
    """Least squares of y = c0 + c1 * x over (x=1/B, y=E|G_B|^2).

    Unweighted on purpose. The natural weighting would be by the number of
    independent samples per B, which under the nested scheme below is largest at
    B=1 -- and B=1 is precisely the point most contaminated by the Adam caveat
    in the docstring. An unweighted fit lets the large-B points, which are the
    ones a large-batch decision rests on, carry their share.
    """
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    det = n * sxx - sx * sx
    if abs(det) < 1e-30:
        raise SystemExit("degenerate fit: need at least two distinct batch sizes")
    c1 = (n * sxy - sx * sy) / det
    c0 = (sy - c1 * sx) / n
    return c0, c1


def main(argv=None):
    # pimm's own DictAction, not a hand-rolled `k, _, v = s.partition("=")`.
    # The difference is not cosmetic: a hand-rolled split leaves every value a
    # STRING, so `data.train.split=['run_x']` becomes the eleven-character
    # string "['run_x']", which merges into the config, survives until
    # `cfg.dump()` re-emits it as `split='['run_x']'`, and dies in yapf with a
    # SyntaxError forty frames from the cause. DictAction evaluates brackets,
    # ints, floats and bools the way `pimm train --options` does, which is what
    # "as pimm's train CLI takes them" below is promising.
    #
    # Imported here rather than at module scope because this file lives in
    # scripts/ and must stay importable without pimm on the path.
    from pimm.utils.config import DictAction

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="a helix pimm training config; the same one the "
                         "checkpoint was trained with")
    ap.add_argument("--options", nargs="+", action=DictAction, default={},
                    help="key=value overrides, as pimm's train CLI takes them. "
                         "`weight=<path>` is how a checkpoint is named.")
    ap.add_argument("--max-batch", type=int, default=128,
                    help="largest B to estimate; also the events per repeat")
    ap.add_argument("--repeats", type=int, default=16,
                    help="independent streams. Each contributes one sample at "
                         "the largest B and 2^k at B = max_batch / 2^k.")
    ap.add_argument("--out", default=None, help="append one JSON row here")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)

    import torch
    from pimm.distributed.distributed import move_batch_to_device
    from pimm.engines.defaults import default_config_parser, default_setup
    from pimm.engines.train import TRAINERS
    from pimm.utils.events import EventStorage

    cfg = default_config_parser(a.config, a.options)
    cfg = default_setup(cfg)

    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    # before_train is what loads the weights. Without it this measures a
    # randomly initialised model, whose noise scale is a real number about
    # nothing -- so refuse rather than report it.
    with EventStorage() as trainer.storage:
        trainer.before_train()

    src = getattr(cfg, "weight", None) or (cfg.model or {}).get("checkpoint")
    if not src:
        raise SystemExit(
            "no `weight` and no `model.checkpoint`. The noise scale of an "
            "untrained model is not the noise scale of your training run, and "
            "it changes by orders of magnitude over the first thousand steps.")

    model = trainer.model
    device = trainer.parallel_context.device
    params = [p for p in model.parameters() if p.requires_grad]
    n_param = sum(p.numel() for p in params)

    batches = _powers_of_two(a.max_batch)
    # sums[B] collects samples of |G_B|^2 across repeats and across the
    # non-overlapping windows within a repeat.
    sums = {b: [] for b in batches}

    loader = trainer.train_loader
    it = iter(loader)
    n_events = 0

    print(f"model: {n_param:,} trainable parameters on {device}")
    print(f"plan : {a.repeats} repeats x {a.max_batch} events "
          f"= {a.repeats * a.max_batch:,} forward/backward passes")

    model.train()
    for rep in range(a.repeats):
        # Non-overlapping windows: within one repeat, the prefix of length B is
        # reset every B events, so the samples at a given B are independent of
        # each other. Nesting ACROSS B (the B=2 windows sit inside the B=4
        # windows) is unavoidable and harmless -- the fit needs the means, and
        # the means are unbiased either way.
        window = {b: ([torch.zeros_like(p) for p in params], 0) for b in batches}
        for i in range(a.max_batch):
            try:
                raw = next(it)
            except StopIteration:
                it = iter(loader)
                raw = next(it)
            batch = move_batch_to_device(raw, device)

            model.zero_grad(set_to_none=False)
            out = model(batch)
            out["loss"].backward()
            n_events += 1

            for b in batches:
                buf, cnt = window[b]
                for t, p in zip(buf, params):
                    if p.grad is not None:
                        t.add_(p.grad)
                cnt += 1
                if cnt == b:
                    # Reduce on the GPU and cross to the host ONCE. The obvious
                    # spelling, `sum(float(t.pow(2).sum()) for t in buf)`, puts
                    # a float() -- a device synchronisation -- inside the loop,
                    # so closing one window costs 228 stalls, one per parameter
                    # tensor. That made the measurement several times slower
                    # than the training step whose gradients it is reading.
                    # Same arithmetic, same result; only the syncs change.
                    sq_t = torch.zeros((), device=device, dtype=torch.float64)
                    for t in buf:
                        sq_t += t.double().pow(2).sum()
                    sums[b].append(float(sq_t) / (b * b))
                    for t in buf:
                        t.zero_()
                    cnt = 0
                window[b] = (buf, cnt)

        done = (rep + 1) * a.max_batch
        print(f"  repeat {rep + 1}/{a.repeats}  ({done:,} events)", flush=True)

    rows = []
    for b in batches:
        vals = sums[b]
        mean = sum(vals) / len(vals)
        var = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
               if len(vals) > 1 else float("nan"))
        sem = math.sqrt(var / len(vals)) if len(vals) > 1 else float("nan")
        rows.append(dict(B=b, n=len(vals), mean_sq=mean, sem=sem))

    c0, c1 = _fit([(1.0 / r["B"], r["mean_sq"]) for r in rows])
    g_sq, s_tr = c0, c1
    b_simple = s_tr / g_sq if g_sq > 0 else float("inf")

    print()
    print(f"{'B':>5} {'n':>5} {'E|G_B|^2':>14} {'sem':>12}")
    for r in rows:
        print(f"{r['B']:5d} {r['n']:5d} {r['mean_sq']:14.6e} {r['sem']:12.3e}")
    print()
    print(f"|G|^2      = {g_sq:.6e}   (intercept: the signal)")
    print(f"tr(Sigma)  = {s_tr:.6e}   (slope: the per-example noise)")
    if g_sq <= 0:
        print("B_simple   = INFINITE -- the fitted intercept is not positive, "
              "which means every B measured is still far below the noise scale. "
              "Re-run with a larger --max-batch.")
    else:
        print(f"B_simple   = {b_simple:,.1f} events")
        print()
        print("read it as: at this point in training, a batch of "
              f"{b_simple:,.0f} events is where gradient noise equals gradient "
              "signal. Well below it, doubling the batch nearly halves the "
              "steps to a target; well above it, it does not.")

    row = dict(tag=a.tag or os.path.basename(str(src)), weight=str(src),
               config=a.config, n_param=n_param, events=n_events,
               max_batch=a.max_batch, repeats=a.repeats,
               curve=rows, g_sq=g_sq, tr_sigma=s_tr, b_simple=b_simple)
    if a.out:
        with open(a.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"\nappended to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
