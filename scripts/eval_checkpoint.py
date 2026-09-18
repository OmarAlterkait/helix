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


def _check_corpus(cfg, weight, *, strict):
    """Compare the corpus about to be read against the one the run recorded.

    The record lives in ``<run>/provenance.json``, written by HelixPathBootstrap
    and appended once per chain link; the LAST entry is the live one. Runs that
    predate the stamp carry none, and are reported as unchecked rather than
    refused -- failing them would make the guard unadoptable on every existing
    checkpoint.
    """
    import json
    from helix.data.identity import corpus_identity, check_corpus_matches

    d = (cfg.data or {}).get("val", {}) or (cfg.data or {}).get("train", {})
    root = d.get("data_root")
    if not root:
        return
    split = d.get("split")
    if isinstance(split, (list, tuple)):
        split = split[0] if split else None
    actual = corpus_identity(root, dataset_name=d.get("dataset_name", "sim_wire"),
                             split=split)

    # WHERE the record lives depends on which route the weight arrived by, and
    # this used to assume the pimm one for both.
    #
    #   cfg.weight        -> <run>/model/<file>.pth, so dirname(dirname(...)) is
    #                        the run directory. Correct.
    #   model.checkpoint  -> an eval ARTIFACT DIRECTORY, so the same arithmetic
    #                        walks TWO levels ABOVE it. For the shipped m113
    #                        artifact that is /sdf/data/neutrino/omara, which has
    #                        no provenance.json -- so the guard printed "corpus
    #                        identity NOT RECORDED", a false reassurance, and
    #                        returned. 100% inert on that route, which is the
    #                        DEFAULT m113 eval. Meanwhile the artifact carries
    #                        its corpus in artifact.json, in the very directory
    #                        the guard was handed.
    recorded = None
    if os.path.isdir(str(weight)):
        try:
            from helix.model.artifact import inspect as _inspect
            recorded = (_inspect(str(weight)).provenance or {}).get("corpus")
        except Exception:
            recorded = None            # not an artifact; fall through to the path below
    run_dir = os.path.dirname(os.path.dirname(str(weight)))
    pj = os.path.join(run_dir, "provenance.json")
    if recorded is None and os.path.exists(pj):
        try:
            with open(pj) as fh:
                blob = json.load(fh)
            entries = blob if isinstance(blob, list) else [blob]
            for e in reversed(entries):             # last link wins
                if isinstance(e, dict) and e.get("corpus"):
                    recorded = e["corpus"]
                    break
        except Exception:
            recorded = None
    try:
        print(f"[corpus] {check_corpus_matches(recorded, actual, where=weight)}",
              flush=True)
    except ValueError as e:
        if strict:
            raise SystemExit(f"{e}\n\nPass --allow-corpus-mismatch if this is "
                             f"deliberate (e.g. a cross-corpus study).")
        print(f"[corpus] WARNING (--allow-corpus-mismatch): {e}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--options", nargs="*", default=[],
                    help="key=value overrides, as pimm's train CLI takes them")
    ap.add_argument("--tag", default=None,
                    help="label recorded in the emitted row (default: basename of weight)")
    ap.add_argument("--out", default=None, help="append one JSON row here")
    ap.add_argument("--allow-corpus-mismatch", action="store_true",
                    help="score a checkpoint against a corpus it did NOT train "
                         "on. Refused by default because the result is a "
                         "plausible wrong number rather than an error.")
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

    # Does this checkpoint belong to this corpus?
    #
    # Nothing checked. Each corpus is internally consistent, so a reader pointed
    # at the WRONG one is satisfied and the only validation here was that the
    # split name exists -- which every corpus satisfies. Two corpora differing
    # only in the coherent-removal gate (tau) read identically and score
    # differently, so the failure mode was a plausible number, not a crash.
    _check_corpus(cfg, src, strict=not a.allow_corpus_mismatch)

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
