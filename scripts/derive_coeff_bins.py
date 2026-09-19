#!/usr/bin/env python
"""Derive categorical-head bin edges from a coeff corpus.

The algorithm and the reasoning behind it live in ``helix.data.bins``; this is
the command line over it. The bins are TRAINING-SET STATISTICS, so they must come
from the corpus a model will be trained on — the edges shipped with m113 were
derived from the old cache, which used a different (white) noise model.

Three modes::

    # derive a fresh grid
    python scripts/derive_coeff_bins.py --corpus <run-dir> --out bins.pt

    # reproduce an existing grid, using the parameters it records
    python scripts/derive_coeff_bins.py --like bins.pt --corpus <run-dir> --out new.pt

    # reproduce it and assert the result matches, writing nothing
    python scripts/derive_coeff_bins.py --verify bins.pt --corpus <run-dir>

``--verify`` is the handover check: it proves a receiving site can regenerate the
grid from its own copy of the corpus rather than carry the file. It exits
non-zero if the tables differ.
"""

from __future__ import annotations

import argparse

from helix.data import bins as binlib


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", help="corpus dir (<root>/<run>/); with --like or "
                                     "--verify, defaults to the recorded path")
    ap.add_argument("--out", help="output .pt (omit only with --verify)")
    ap.add_argument("--like", help="reuse the derivation parameters this .pt records")
    ap.add_argument("--verify", help="rederive and compare against this .pt; writes nothing")
    ap.add_argument("--dataset-name", default=binlib.DEFAULTS["dataset_name"])
    ap.add_argument("--events", type=int, default=binlib.DEFAULTS["events"],
                    help="events to pool (research used 120)")
    ap.add_argument("--K", type=int, default=binlib.DEFAULTS["K"], help="bins per band")
    ap.add_argument("--n-bands", type=int, default=binlib.DEFAULTS["n_bands"],
                    help="bands the tokenizer keeps (D1 and beyond are dropped)")
    ap.add_argument("--lo-pct", type=float, default=binlib.DEFAULTS["lo_pct"])
    ap.add_argument("--hi-pct", type=float, default=binlib.DEFAULTS["hi_pct"])
    # The basis check below exists to catch the pre-tau/r1 confusion, which is a
    # real hazard: those generations differ in which coefficients survive the
    # gate, so a grid from one does not describe the other. But a SYNTHETIC
    # corpus is a legitimately different generation and the caller knows it --
    # docs/RUNBOOK.md 0b builds one from pimm_data.testing and derives a grid
    # over it, and without an escape hatch that documented path cannot run at
    # all. It is opt-in and it says what it did.
    ap.add_argument("--allow-foreign-basis", action="store_true",
                    help="derive from a corpus whose basis_digest is not the "
                         "packaged reference generation (synthetic corpora, or a "
                         "deliberately different DSP)")
    a = ap.parse_args(argv)

    if a.like and a.verify:
        ap.error("--like and --verify both name a reference; pass one")
    reference = a.like or a.verify
    if not reference and not a.corpus:
        ap.error("--corpus is required unless --like or --verify supplies one")
    if not a.verify and not a.out:
        ap.error("--out is required unless --verify")

    # Input-side check first: one shard header, a second, and it names the cause.
    # It cannot tell WHICH run this is -- see helix.data.bins.check_corpus -- so
    # it is a cheap filter in front of the content digest, not a replacement.
    target = a.corpus
    if target:
        c = binlib.check_corpus(target, dataset_name=a.dataset_name)
        if c["status"] == "basis-mismatch" and not a.allow_foreign_basis:
            print(f"FATAL: {target}\n"
                  f"  basis_digest {c['actual']}\n"
                  f"  expected     {c['expected']}  ({c.get('generation')})\n"
                  "  A different DSP produced this corpus -- most likely the pre-tau\n"
                  "  generation rather than r1. They differ in which coefficients\n"
                  "  survive the coherent gate, so a grid from one does not describe\n"
                  "  the other. Pass --corpus for the generation you mean, or\n"
                  "  --allow-foreign-basis if you MEANT a different generation.")
            return 2
        if c["status"] == "basis-mismatch":
            print(f"note: --allow-foreign-basis: deriving from basis_digest "
                  f"{c['actual'][:12]}…, which is NOT the packaged reference "
                  f"generation ({c.get('generation')}). The resulting grid describes "
                  f"THIS corpus and is not comparable with the released models.")
        if c["status"] == "run-name-differs":
            print(f"note: corpus directory is {c['actual']!r}, the reference grid was "
                  f"derived from {c['expected']!r}. The name is only a convention, so "
                  f"this is not conclusive either way -- the digest check below is.")
        elif c["status"] == "unreadable":
            print(f"FATAL: no shard found under {target}")
            return 2

    if reference:
        ref = binlib.load(reference)
        p = binlib.params_of(ref)
        corpus = a.corpus or p["corpus"]
        print(f"reference {reference}: events={p['events']} K={p['K']} "
              f"n_bands={p['n_bands']} corpus={p['corpus']}")
        if a.corpus and a.corpus != p["corpus"]:
            print(f"  deriving from {corpus} instead — the recorded path is where "
                  f"it lived when it was derived")
        table = binlib.rederive(ref, corpus=corpus, report=print)
    else:
        table = binlib.derive(a.corpus, dataset_name=a.dataset_name,
                              events=a.events, K=a.K, n_bands=a.n_bands,
                              lo_pct=a.lo_pct, hi_pct=a.hi_pct, report=print)

    if a.verify:
        r = binlib.compare(ref, table)
        for k, v in r["arrays"].items():
            print(f"  {k:12s} identical={v['identical']}  "
                  f"max|diff|={v['max_abs_diff']:.3e}")
        for k, (x, y) in r["param_mismatches"].items():
            print(f"  {k}: reference {x!r} != derived {y!r}")
        if r["identical"]:
            print("VERIFIED: rederived bit-identically — this grid need not be copied")
            return 0
        print("MISMATCH: the rederived grid differs from the reference.\n"
              "  A grid is the objective a model was trained against, so this is a\n"
              "  DIFFERENT grid, not a rounding difference. Check that --corpus names\n"
              "  the SAME run the reference records, with an identical shard set:\n"
              "  'first N events' is defined by dataset order.")
        return 1

    # Self-check against the packaged fingerprint. A site that arrived without a
    # copy of the original grid has nothing to --verify against, and deriving
    # from the wrong run of a multi-run corpus yields a grid that is entirely
    # self-consistent and simply not the one the released models were trained
    # on. Nothing else would notice, so this is not opt-in.
    r = binlib.check_reference(table)
    ref = r.get("reference") or {}
    if r["status"] == "match":
        print(f"matches the packaged reference grid ({ref.get('name')}) — this is "
              f"the grid the released models were trained against")
    elif r["status"] == "mismatch":
        print(f"WARNING: does NOT match the packaged reference grid "
              f"({ref.get('name')}).")
        rc = ref.get("corpus") or {}
        print(f"  The reference was derived from corpus run "
              f"{rc.get('run')}, generation {rc.get('generation')}.")
        print("  Same parameters, different numbers means a different corpus run or a\n"
              "  different shard set. The grid is still usable — it is simply not the\n"
              "  one existing checkpoints were trained against, so numbers from a model\n"
              "  trained on it are not comparable to the published ones.")
    elif r["status"] == "params":
        pm = ", ".join(f"{k}: reference {x!r} != derived {y!r}"
                       for k, (x, y) in r["param_mismatches"].items())
        print(f"note: derived with different parameters from the packaged reference "
              f"({pm}), so they are not comparable. Deliberate if you meant to change "
              f"the grid.")
    else:
        print("note: no packaged reference fingerprint installed, so this grid could "
              "not be checked against the released one.")

    binlib.save(table, a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
