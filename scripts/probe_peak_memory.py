"""Peak GPU memory against event size — the number that decides a large run.

WHY THIS EXISTS
---------------
The FM takes one event per rank, and events are not the same size: over the
corpus the coefficient count runs 39,462 to 759,710, a 2.77x spread. Memory
scales with that, and the categorical head is the dominant term -- the logits are
``(n_cells, n_slot, K)``, which the config notes is ~2.4 GB at a full event
before ``cross_entropy`` upcasts to fp32 and holds a log-softmax buffer.

The failure this guards against is specific to SCALE and cannot be seen at small
rank counts. Each step draws one event per rank, so the LARGEST event in a step
is the p_{1-1/N} quantile of the size distribution:

    N = 16 ranks  -> the ~94th percentile
    N = 128       -> the ~99.2nd
    N = 512       -> the ~99.8th, essentially every step

If the out-of-memory threshold sits between those quantiles, a job runs fine at
128 GPUs and dies at 512 -- and because a single rank OOMing leaves every other
rank blocked in a collective, it does not die quickly. It hangs until the NCCL
watchdog fires, burning the whole allocation.

Run this before requesting many nodes. It is one GPU and a few minutes, and it
answers whether ``data.train.max_len`` needs to cap the tail.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _batch_size_of(obj):
    """Total tensor elements in a batch, as a stand-in for event size.

    Deliberately not a named key. By the time a batch reaches the model it has
    been through the tokenizer, so the raw ``coeff.value`` column is gone and the
    field that carries "how big is this event" depends on the transform chain.
    Total element count is monotone in event size, needs no knowledge of the
    schema, and survives the tokenizer being changed -- which is the point,
    since this script exists to be run before a big job, not to track a schema.
    """
    import torch as _t
    if isinstance(obj, _t.Tensor):
        return obj.numel()
    if isinstance(obj, dict):
        return sum(_batch_size_of(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_batch_size_of(v) for v in obj)
    return 0


def main(argv=None):
    from pimm.utils.config import DictAction

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--options", nargs="+", action=DictAction, default={})
    ap.add_argument("--steps", type=int, default=200,
                    help="forward/backward passes to sample the size distribution")
    a = ap.parse_args(argv)

    import torch
    from pimm.distributed.distributed import move_batch_to_device
    from pimm.engines.defaults import default_config_parser, default_setup
    from pimm.engines.train import TRAINERS
    from pimm.utils.events import EventStorage

    cfg = default_setup(default_config_parser(a.config, a.options))
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    with EventStorage() as trainer.storage:
        trainer.before_train()

    model = trainer.model
    device = trainer.parallel_context.device
    total = torch.cuda.get_device_properties(device).total_memory / 2 ** 30
    print(f"{torch.cuda.get_device_name(device)}, {total:.0f} GiB total")

    rows = []
    it = iter(trainer.train_loader)
    model.train()
    for i in range(a.steps):
        try:
            raw = next(it)
        except StopIteration:
            break
        batch = move_batch_to_device(raw, device)
        # Per-step reset: we want the peak of THIS event, not a running maximum,
        # so that the relationship to event size is readable.
        torch.cuda.reset_peak_memory_stats(device)
        model.zero_grad(set_to_none=True)
        out = model(batch)
        out["loss"].backward()
        torch.cuda.synchronize(device)
        rows.append((_batch_size_of(batch),
                     torch.cuda.max_memory_allocated(device) / 2 ** 30))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{a.steps} steps", flush=True)

    rows.sort()
    print(f"\n{'batch elems':>13} {'peak GiB':>9} {'GiB per 100k':>13}")
    for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0):
        n, m = rows[min(len(rows) - 1, int(q * (len(rows) - 1)))]
        print(f"{n:13,} {m:9.2f} {m / (n / 1e5):13.3f}")

    # Extrapolate to the tail the sampler will actually hand a large run.
    big = max(rows, key=lambda r: r[0])
    slope = big[1] / big[0]
    print(f"\nlinear at {slope * 1e6:.3f} GiB per 1M batch elements")
    for label, n in (("2.77x the largest sampled", int(big[0] * 2.77 / 1.0)),):
        print(f"  {label:12s} {n:9,} -> {slope * n:5.2f} GiB of {total:.0f} "
              f"({'FITS' if slope * n < total * 0.9 else 'AT RISK'})")
    print("\nIf the corpus max is at risk, cap data.train.max_len at a percentile "
          "of event_sizes() rather than discovering it at 512 ranks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
