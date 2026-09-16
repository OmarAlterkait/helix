#!/usr/bin/env python3
"""Promote a ``pimm export`` into a helix EVAL ARTIFACT.

An export directory is portable but not self-describing. Three things it
structurally cannot record, each of which changes the number:

  WHICH WEIGHT SET.  pimm's ``_sanitize_config`` nulls ``weight`` on every
    export, and the exported file is always named ``model.safetensors`` /
    ``model.bin`` whatever it was exported from. So every real export reads
    "unknown", every probe row carries ``weights_are_ema=None``, and an EMA arm
    and a raw arm can be compared in silence. On a flat-LR WSD run the raw
    weights sit at full LR noise for the whole stable phase — which is the
    entire reason the EMA exists — so that comparison is two noisy draws, not
    two models.

  WHICH CORPUS.  ``basis_digest`` pins the DSP basis, and a model trained on
    ``coeff_tpc`` (7f954a84…) scored against ``coeff_tpc_r1`` (8c4542b6…) is
    reading coefficients from a different gate. Nothing in an export names it.

  WHICH CODE.  The helix commit that defined the tokenizer the weights saw.

``--weights`` is REQUIRED and unguessable on purpose: you are asserting what you
exported, and the assertion is what makes every downstream row attributable.
This is the only place a human has to know, and it is one flag at the moment the
knowledge is actually in hand.

Usage:
    pimm export --run-dir <save_path> model_ema.pth /tmp/exp
    python scripts/export_artifact.py /tmp/exp --weights ema \\
        --corpus /sdf/data/neutrino/omara/coeff_tpc_r1 -o <artifact_dir>

The result is readable by everything that reads a checkpoint here — run_probe,
feats_rank, the encode recipe — because there is one reader
(:mod:`helix.model.artifact`) and this is one of its formats.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir", help="a `pimm export` directory, or a converted blob")
    ap.add_argument("-o", "--out", required=True, help="artifact directory to write")
    ap.add_argument("--weights", required=True, choices=("ema", "raw"),
                    help="WHICH weight set you exported. Required: pimm cannot "
                         "record it and nothing downstream can recover it.")
    ap.add_argument("--corpus", default=None,
                    help="the corpus these weights were trained on; its "
                         "basis_digest is recorded so a mismatched eval is "
                         "detectable rather than merely wrong")
    ap.add_argument("--cell-t", default=None, choices=("grid_center", "centroid"),
                    help="fill in a tokenizer cell_t the export did not record")
    ap.add_argument("--dataset-name", default="sim_wire")
    ap.add_argument("--note", default=None, help="free text carried into provenance")
    ap.add_argument("--force", action="store_true", help="overwrite an existing artifact")
    a = ap.parse_args(argv)

    from dataclasses import replace

    from helix.core.coeff_io import _code_version
    from helix.model.artifact import load, save, weights_digest

    if os.path.exists(os.path.join(a.out, "artifact.json")) and not a.force:
        raise SystemExit(f"{a.out} already holds an artifact; pass --force to replace it")

    art = load(a.export_dir)
    op = art.op

    # The operating point must be COMPLETE. An artifact whose cell_t is unknown
    # is exactly the artifact this file exists to stop being written: the probe
    # would fall back to a default, and 94.06% of cells would carry a time
    # coordinate the model never saw (mean |delta| 19.5 ticks).
    if op.cell_t is None:
        if a.cell_t is None:
            raise SystemExit(
                f"{a.export_dir} records no tokenizer cell_t, and --cell-t was "
                f"not given. Refusing to write an artifact that cannot say which "
                f"time coordinate its weights were trained on -- that is the one "
                f"failure this format exists to prevent. helix configs train "
                f"grid_center; read it out of the run's config.py to be sure.")
        op = replace(op, cell_t=a.cell_t)
    elif a.cell_t is not None and a.cell_t != op.cell_t:
        raise SystemExit(
            f"{a.export_dir} records cell_t={op.cell_t!r} but --cell-t "
            f"{a.cell_t!r} was given. One of the two is wrong; picking either "
            f"silently is how a model gets scored on a coordinate it never saw.")

    missing = [k for k in ("pw", "pt", "n_bands") if getattr(op, k) is None]
    if missing:
        raise SystemExit(
            f"{a.export_dir}: the operating point is incomplete ({', '.join(missing)} "
            f"not recorded). Re-export with the run's resolved config, which "
            f"carries the CoeffTokenize transform.")

    corpus = None
    if a.corpus:
        from helix.data.identity import corpus_identity
        corpus = corpus_identity(a.corpus, dataset_name=a.dataset_name)
        if corpus is None:
            raise SystemExit(f"{a.corpus}: no {a.dataset_name}_coeff_*.h5 shard found")
        corpus = dict(corpus, root=os.path.abspath(a.corpus))

    prov = dict(source=os.path.abspath(a.export_dir), source_format=art.fmt,
                source_weights=art.weights_source, corpus=corpus,
                helix=_code_version(), step=art.provenance.get("step"),
                weights_digest=weights_digest(art.state_dict),
                exported_at=int(time.time()), note=a.note)
    save(a.out, state_dict=art.state_dict, arch=art.arch, op=op,
         weights=a.weights, provenance=prov)

    print(json.dumps(dict(out=os.path.abspath(a.out), weights=a.weights,
                          cell_t=op.cell_t, pw=op.pw, pt=op.pt,
                          n_bands=op.n_bands,
                          basis_digest=(corpus or {}).get("basis_digest", ""),
                          helix=prov["helix"].get("git", "unknown"),
                          weights_digest=prov["weights_digest"]),
                     indent=2))
    if corpus is None:
        print("\nNOTE: no --corpus given, so this artifact does not name the "
              "basis_digest its weights were trained on. A later eval against a "
              "different corpus will not be detectable.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
