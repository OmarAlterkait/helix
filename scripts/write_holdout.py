#!/usr/bin/env python
"""Write ``<corpus>/holdout.json`` — the split, materialised.

WHY THIS EXISTS. configs/pimm/coeff_fm_train.py says the split is "Resolved once
and written to holdout.json beside the corpus", and the reason it gives is right:
"reproducible is not the same as auditable, and a probe holdout that exists only
as code cannot be inspected, diffed or cited."

Nothing wrote it. `CoeffTPCDataset.holdout_manifest` computes it, and
dump_probe_truth.py and run_probe.py READ the file, but no code put it there. The
production corpus has one; a freshly built corpus never does, so the probe cannot
run on any corpus built from scratch. Found by building one.

The split is keyed on blake2b(run/source_file) + event -- the SIMULATION event's
identity -- so it is independent of basis, noise model, shard size and shard
order. A corpus rebuilt with a different gate or wavelet gets the SAME split, and
this file must therefore reproduce production's byte for byte for the same runs
and fractions. That is a check, not a coincidence: use --compare.

    python scripts/write_holdout.py --corpus <run dir> [--compare <other holdout.json>]
"""
from __future__ import annotations

import argparse
import json
import os


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True, help="corpus RUN dir")
    ap.add_argument("--dataset-name", default="sim_wire")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", type=float, default=0.95)
    ap.add_argument("--val", type=float, default=0.03)
    ap.add_argument("--probe", type=float, default=0.02)
    ap.add_argument("--compare", default=None,
                    help="an existing holdout.json to check against (same runs/fractions "
                         "must give an IDENTICAL split -- the identity hash does not "
                         "depend on the corpus's contents)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    from helix.data.coeff_dataset import CoeffTPCDataset

    fractions = dict(train=a.train, val=a.val, probe=a.probe)
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise SystemExit(f"fractions must sum to 1, got {fractions} = {sum(fractions.values())}")
    holdout = dict(seed=a.seed, fractions=fractions)

    corpus = os.path.abspath(a.corpus)
    out = {
        "corpus": corpus,
        "seed": a.seed,
        "fractions": fractions,
        "note": ("Resolved from blake2b(run/source_file) + event. Reproducible from "
                 "seed and fractions; written so the split can be inspected, diffed "
                 "and cited without running code. train is the complement of "
                 "val+probe."),
    }
    counts = {}
    for role in ("val", "probe"):
        ds = CoeffTPCDataset(data_root=os.path.dirname(corpus),
                             split=os.path.basename(corpus),
                             dataset_name=a.dataset_name,
                             holdout=holdout, split_role=role)
        # _ident_at yields {run, source_file, event}; normalise event to int so
        # the JSON matches production's (numpy ints serialise but compare badly).
        out[role] = [dict(run=str(d["run"]), source_file=str(d["source_file"]),
                          event=int(d["event"]))
                     for d in ds.holdout_manifest()]
        counts[role] = len(out[role])
    # train is the complement and is NOT enumerated: it is 95% of the corpus and
    # listing it would make the file 20x larger for no auditable gain.
    ds = CoeffTPCDataset(data_root=os.path.dirname(corpus),
                         split=os.path.basename(corpus),
                         dataset_name=a.dataset_name,
                         holdout=holdout, split_role="train")
    counts["train"] = len(ds.holdout_manifest())
    out["counts"] = counts

    print(f"  {counts}")

    if a.compare:
        ref = json.load(open(a.compare))
        for role in ("val", "probe"):
            mine = [(d["run"], d["source_file"], d["event"]) for d in out[role]]
            theirs = [(d["run"], d["source_file"], d["event"]) for d in ref.get(role, [])]
            same = mine == theirs
            print(f"  {role}: {len(mine)} vs {len(theirs)}  IDENTICAL={same}")
            if not same:
                raise SystemExit(
                    f"{role} differs from {a.compare}. The identity split must NOT "
                    f"depend on corpus contents — investigate before trusting either.")

    path = os.path.join(corpus, "holdout.json")
    if a.dry_run:
        print(f"  --dry-run: would write {path}")
        return 0
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
