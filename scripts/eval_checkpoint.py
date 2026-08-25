"""Load a checkpoint and score it once, without taking an optimizer step.

``CoeffFMEvaluator`` hooks ``after_step`` and ``after_epoch`` only, so every
trainer-driven route to a number runs at least one optimizer step first and
reports weights that have already moved. For a comparison between two frozen
checkpoints that is not a rounding detail — it is the difference between scoring
the artifact and scoring something near it.

This drives pimm's own lifecycle up to the point where the weights are loaded and
then calls the evaluator's own ``eval()``. Nothing about the metric is
reimplemented here: ``eval()`` is the shipped code path, the same one training
logs come from, so a number from this script and a number from a training log are
produced by the same lines.

    python scripts/eval_checkpoint.py \
        --config configs/pimm/coeff_fm_eval_probe.py \
        --options weight=/path/run/model/model_ema.pth save_path=/path/out
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--options", nargs="*", default=[],
                    help="key=value overrides, as pimm's train CLI takes them")
    ap.add_argument("--tag", default=None,
                    help="label recorded in the emitted row (default: basename of weight)")
    ap.add_argument("--out", default=None, help="append one JSON row here")
    a = ap.parse_args(argv)

    from pimm.engines.defaults import default_config_parser, default_setup
    from pimm.engines.train import TRAINERS
    from pimm.utils.events import EventStorage

    opts = {}
    for kv in a.options:
        k, _, v = kv.partition("=")
        opts[k] = v
    cfg = default_config_parser(a.config, opts)
    cfg = default_setup(cfg)

    # Weights arrive by ONE of two routes: cfg.weight (a pimm checkpoint, loaded
    # by CheckpointLoader) or model.checkpoint (a converted blob carrying its own
    # architecture and bins, restored by build_coeff_fm). Requiring cfg.weight
    # alone rejected the second, which is how the m113 reference is packaged.
    src = getattr(cfg, "weight", None) or (cfg.model or {}).get("checkpoint")
    if not src:
        raise SystemExit(
            "no `weight` and no `model.checkpoint`. This script scores a "
            "CHECKPOINT; without one it would silently report a randomly "
            "initialised model, which looks like a valid row.")

    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))

    # before_train is what actually loads the weights (CheckpointLoader) and
    # gives every other hook its trainer reference. The training LOOP is
    # deliberately never entered.
    with EventStorage() as trainer.storage:
        trainer.before_train()
        ev = next((h for h in trainer.hooks
                   if type(h).__name__ == "CoeffFMEvaluator"), None)
        if ev is None:
            raise SystemExit(
                "no CoeffFMEvaluator in the hook list — this config cannot "
                "produce the metrics this script exists to read")
        metrics = ev.eval()
        if metrics is None:
            raise SystemExit(
                "the evaluator returned None. On a non-zero rank that means "
                "'this rank is not authoritative' and is expected — run this "
                "on one rank. On rank 0 it means the evaluator produced nothing.")
        if not metrics:
            raise SystemExit(
                "the evaluator returned an empty result: it was authoritative "
                "and had nothing to measure. Check that data.val.split_role "
                "names a split this corpus actually has.")
        metrics = {k: float(v) for k, v in metrics.items()}

    tag = a.tag or os.path.basename(os.path.dirname(os.path.dirname(str(src))))
    row = dict(tag=tag, weight=str(src), config=a.config,
               # Recorded because it CHANGES THE NUMBER: the head decodes through
               # this partition, so two rows with different bins are not the same
               # measurement even on identical events.
               bins=str((cfg.model or {}).get("bins")),
               split_role=str(getattr(cfg.data.val, "split_role", "?")),
               data_root=str(getattr(cfg.data.val, "data_root", "?")),
               metrics=metrics)
    print(json.dumps(row, indent=1, sort_keys=True))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
