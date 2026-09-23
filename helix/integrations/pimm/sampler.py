"""Length bucketing: make every rank in a step do comparable work.

THE MEASUREMENT THIS EXISTS FOR
-------------------------------
The FM takes exactly one event per rank, and DDP makes a step cost the SLOWEST
rank. Events are not the same size -- over 1,200 corpus events the coefficient
count runs 39,462 to 759,710, a 2.75x spread -- so a step costs the maximum of N
draws from that distribution rather than the mean, and the penalty grows with N.

Measured on the scaling ladder, step time against the single-GPU mean of 0.153 s:

      GPUs   step     ratio    E[max of N sizes]/mean
         4   0.166    1.08x    1.38
        32   0.229    1.49x    1.85
        64   0.236    1.54x    1.98
       128   0.244    1.59x    2.11

Fitting those gives ~56% of a step scaling with event size and the rest fixed.
That accounts for the ENTIRE gap to linear scaling: it is not the interconnect
(moving the same all-reduce from NVLink to the fabric costs -2.4%), not the
filesystem (1 vs 16 dataloader workers at 2,048 concurrent readers are
indistinguishable), and not the dataloader.

So the 61% scaling efficiency at 128 GPUs is a SOFTWARE property, worth ~1.59x
throughput (525 -> 835 events/s) to fix, and the win grows with node count
rather than shrinking.

HOW
---
``StatefulRandomSampler`` hands rank r the slice ``indices[r::num_replicas]``,
so the N events of one step are a CONTIGUOUS WINDOW of the shuffled order. Making
a step homogeneous therefore needs no change to how ranks read -- only a
reordering of that one permutation, which is what this does:

    shuffle -> cut into megabatches of `mega` steps -> sort each megabatch by
    event size -> the consecutive blocks of `num_replicas` are now homogeneous
    -> shuffle the ORDER of those blocks -> stride by rank

WHY THE MEGABATCH, AND WHY SHUFFLE THE BLOCKS
---------------------------------------------
Both guard the statistics rather than the speed. Sorting globally would make
step k's batch a deterministic function of k and walk the model from smallest
events to largest, which is a curriculum nobody chose. Sorting only within a
megabatch keeps each batch a random draw from a `mega`-step-wide window, and
shuffling the block order removes any correlation between size and training
step. What remains -- batches that are internally size-homogeneous -- is a real
change to the sampling distribution, which is why this is OFF by default and why
it ships with an A/B against loss, not only against throughput. A throughput win
that costs convergence is not a win.
"""
from __future__ import annotations

import numpy as np


def bucketed_sampler_class(base, mega=32, log=None):
    """Return a subclass of pimm's sampler `base` that buckets by event size.

    Built as a subclass of whatever pimm currently uses rather than a rewrite,
    so the epoch/position/checkpoint semantics -- which resume depends on --
    stay exactly pimm's. Only the ORDER changes.
    """

    class LengthBucketedSampler(base):
        def _build_order(self):
            sizes = _sizes_of(self.data_source)
            if sizes is None:
                # WARNING, not info. Falling back is the right behaviour -- a
                # sampler is a bad place to fail a run -- but the caller asked
                # for bucketing by setting a config flag, and quietly not doing
                # it produced an A/B that compared two identical runs. A run
                # that is not doing what its config says should say so loudly.
                if log:
                    log.warning(
                        "bucket_by_size is set but %s exposes no event_sizes(); "
                        "training with pimm's ORDINARY order. Bucketing is NOT "
                        "active.", type(self.data_source).__name__)
                return super()._build_order()

            length = len(self.data_source)
            if length == 0 or self.num_replicas <= 1:
                return super()._build_order()

            # Same permutation, same seeding, same padding as the base class, so
            # that with bucketing off this is bit-identical to pimm's order.
            rng = np.random.default_rng(self.seed + self.epoch)
            idx = (rng.permutation(length) if self.shuffle
                   else np.arange(length, dtype=np.int64))

            if not self.drop_last:
                pad = self.total_size - len(idx)
                if pad > 0:
                    reps = int(np.ceil(pad / len(idx)))
                    idx = np.concatenate([idx, np.tile(idx, reps)[:pad]])
            else:
                idx = idx[: self.total_size]

            R = self.num_replicas
            step_blocks = []
            window = R * max(1, int(mega))
            for start in range(0, len(idx), window):
                chunk = idx[start:start + window]
                # A trailing chunk shorter than one step cannot form a block; it
                # is carried as-is so no event is dropped.
                order = np.argsort(sizes[chunk], kind="stable")
                chunk = chunk[order]
                for b in range(0, len(chunk), R):
                    step_blocks.append(chunk[b:b + R])

            rng.shuffle(step_blocks)
            out = np.concatenate(step_blocks) if step_blocks else idx
            return out[self.rank: self.total_size: R].tolist()

    LengthBucketedSampler.__name__ = "LengthBucketedSampler"
    return LengthBucketedSampler


def _sizes_of(dataset):
    """Per-event sizes aligned with `dataset`'s index, or None."""
    fn = getattr(dataset, "event_sizes", None)
    if fn is None:
        return None
    try:
        s = np.asarray(fn(), dtype=np.int64)
    except Exception:
        # A sampler is not the place to fail a run: without sizes the correct
        # behaviour is pimm's ordinary order, which is what the caller gets.
        return None
    return s if s.ndim == 1 and s.size else None
