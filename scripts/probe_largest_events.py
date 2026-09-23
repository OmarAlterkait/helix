"""Peak memory on the corpus's actual largest events — measured, not extrapolated.

``probe_peak_memory.py`` samples the distribution and fits a line. That answers
"how does memory grow with event size" but not "does the worst event fit", and
extrapolating the fit is how a 2.77x max/mean ratio gets applied to an already-
large sampled event and produces a number 40% too high.

This takes the five largest events in the corpus by ``event_sizes()`` and runs a
real forward/backward on each. One GPU, a couple of minutes, no extrapolation.

It matters because of how batches are drawn: one event per rank, so the largest
event in a step is the p_{1-1/N} quantile. At 512 ranks that is the ~99.8th
percentile essentially every step, and a single rank running out of memory leaves
the other 511 blocked in a collective until the watchdog fires.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    from pimm.utils.config import DictAction

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--options", nargs="+", action=DictAction, default={})
    ap.add_argument("--top", type=int, default=5, help="how many of the largest to try")
    a = ap.parse_args(argv)

    import numpy as np
    import torch
    from pimm.distributed.distributed import move_batch_to_device
    from pimm.engines.defaults import default_config_parser, default_setup
    from pimm.engines.train import TRAINERS
    from pimm.utils.events import EventStorage

    cfg = default_setup(default_config_parser(a.config, a.options))
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    with EventStorage() as trainer.storage:
        trainer.before_train()

    loader = trainer.train_loader
    ds = loader.dataset
    device = trainer.parallel_context.device
    total = torch.cuda.get_device_properties(device).total_memory / 2 ** 30
    sizes = np.asarray(ds.event_sizes(), dtype=np.int64)
    print(f"card {total:.1f} GiB | {len(sizes):,} events | "
          f"median {int(np.median(sizes)):,} max {int(sizes.max()):,} coefficients")

    collate = getattr(loader, "collate_fn", None)
    trainer.model.train()
    worst = 0.0
    for rank, idx in enumerate(np.argsort(sizes)[::-1][: a.top], start=1):
        sample = ds[int(idx)]
        batch = move_batch_to_device(collate([sample]) if collate else sample, device)
        torch.cuda.reset_peak_memory_stats(device)
        trainer.model.zero_grad(set_to_none=True)
        try:
            out = trainer.model(batch)
            out["loss"].backward()
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
            worst = max(worst, peak)
            print(f"  #{rank} {int(sizes[idx]):9,} coeff -> peak {peak:6.2f} GiB "
                  f"({peak / total * 100:3.0f}% of card)")
        except torch.cuda.OutOfMemoryError:
            print(f"  #{rank} {int(sizes[idx]):9,} coeff -> OUT OF MEMORY")
            torch.cuda.empty_cache()
    if worst:
        print(f"\nworst measured {worst:.2f} GiB of {total:.1f} "
              f"({worst / total * 100:.0f}%). Headroom matters: `empty_cache=False` "
              f"means the allocator does not return fragments, so a run sits above "
              f"its own peak.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
