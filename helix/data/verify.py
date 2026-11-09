"""What a corpus is, and whether it is whole: its identity (the guard a checkpoint is checked
against) and the corpus-level integrity check. numpy and h5py only (no pimm-data), so a launcher
host can read a corpus's run list and identity (``tests/test_boundary.py``).

Usage::

    python -m helix.data.verify <corpus_root>                  every run, against _calib/CHUNKS.txt
    python -m helix.data.verify <run_dir> --dataset-name sim_wire [--expect 1000]
    python -m helix.data.verify compare A B [A2 B2 ...]        same content, byte for byte

IDENTITY. Each corpus is internally consistent (``CoeffTPCReader`` refuses shards that disagree
with each other), so a reader pointed at the WRONG corpus is perfectly satisfied and computes a
plausible number against data the model never trained on. ``coeff_tpc`` and ``coeff_tpc_r1``
share a run name and differ only in the coherent-removal gate. ``basis_digest`` hashes the whole
DSP recipe (wavelet, level, mode, ticks, pad, band lengths, removal, threshold, sigma_norm), so
it is the one field a run records and is checked against.

INTEGRITY. ``helix.core.coeff_io.audit_shard`` checks one file as it is written; what goes wrong
in a build spans files -- a failed task leaves a shard missing, overlapping ranges duplicate
events, a rebuilt shard comes from another commit or another norm table. :func:`verify` checks a
corpus against its plan (``_calib/CHUNKS.txt``, :func:`helix.build.plan`), or one run directory
on its own.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter

import h5py
import numpy as np


# ---- identity --------------------------------------------------------------------------------

def corpus_identity(data_root, dataset_name="sim_wire", split=None):
    """``{'basis_digest', 'removal_json', 'sigma_norm', 'shard'}`` of the first shard, or ``None``
    when there is none ("not a corpus" and "a corpus that disagrees" are different failures)."""
    roots = [os.path.join(data_root, split)] if split else [data_root]
    for r in roots:
        for pat in (f"{dataset_name}_coeff_*.h5", f"*/{dataset_name}_coeff_*.h5"):
            hits = sorted(glob.glob(os.path.join(r, pat)))
            if hits:
                with h5py.File(hits[0], "r") as f:
                    cfg = f["config"]
                    return {
                        "basis_digest": str(cfg.attrs.get("basis_digest", "")),
                        "removal_json": str(cfg.attrs.get("removal_json", "")),
                        "sigma_norm": float(cfg.attrs.get("sigma_norm", 0.0)),
                        "shard": os.path.basename(hits[0]),
                    }
    return None


def check_corpus_matches(recorded, actual, *, where=""):
    """Refuse when a run's recorded corpus identity disagrees with the one open; returns a note.

    A MISSING record passes with a note (checkpoints trained before the stamp have none); a
    PRESENT record that disagrees always raises -- the case where a number would otherwise come
    out plausible and wrong."""
    if actual is None:
        raise ValueError(f"{where}: no coeff shard found — this is not a corpus")
    if not recorded or not recorded.get("basis_digest"):
        return ("corpus identity NOT RECORDED by this run (predates the stamp); "
                f"reading basis_digest={actual['basis_digest'][:12]}… unchecked")
    a, b = str(recorded["basis_digest"]), str(actual["basis_digest"])
    if a != b:
        raise ValueError(
            f"{where}: corpus mismatch — this run trained on basis_digest={a}\n"
            f"  but the corpus being read is basis_digest={b}\n"
            f"  recorded removal: {recorded.get('removal_json')}\n"
            f"  actual   removal: {actual['removal_json']}\n"
            "  These are different DSP, so the coefficients differ. Point at the "
            "corpus this run trained on, or evaluate a checkpoint trained on this one.")
    return f"corpus identity OK (basis_digest={b[:12]}…)"


def resolve_corpus(recorded_root, *, override=None, identity=None, dataset_name="sim_wire", split=None):
    """``(path, note)``: the corpus a run recorded, found at THIS site, and its identity check.

    A run records its corpus as the absolute path it trained at, so a run trained at S3DF names a
    path NERSC does not have. In order: ``override`` (the caller's ``--corpus``); the recorded
    path, if it exists; this site's copy under the same name -- ``HELIX_CORPUS_ROOT/<name>``, or
    ``HELIX_CORPUS``'s parent when that is ``<name>`` (the tail rule :func:`read_plan` applies to
    a moved plan). Whatever is found is then held to ``identity``, the run's recorded
    :func:`corpus_identity`, by :func:`check_corpus_matches`: a different ``basis_digest`` is
    refused, a run that recorded none is let through with "unchecked" in the note. ``split``
    (one run directory, or the run's list of them: the first is read) locates a shard."""
    from helix import paths

    recorded_root = str(recorded_root)
    name = os.path.basename(os.path.normpath(recorded_root))
    if override:
        path = str(override)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"--corpus {path}: not a directory")
    elif os.path.isdir(recorded_root):
        path = recorded_root
    else:
        tried = []
        try:
            root, _ = paths.resolve("HELIX_CORPUS_ROOT")
            one, _ = paths.resolve("HELIX_CORPUS")
        except paths.SiteError:
            root = one = None
        if root is not None:
            tried.append(root / name)
        if one is not None and one.parent.name == name:
            tried.append(one.parent)
        path = next((str(p) for p in tried if p.is_dir()), None)
        if path is None:
            raise FileNotFoundError(
                f"the run's corpus {recorded_root} is not at this site (looked for {name!r} at "
                f"{', '.join(map(str, tried)) or 'no configured corpus root'}); pass its location (--corpus)")
    sp = split[0] if isinstance(split, (list, tuple)) and split else (split or None)
    note = check_corpus_matches(identity, corpus_identity(path, dataset_name, sp), where=path)
    return path, note


def corpus_runs(corpus_root=None) -> list[str]:
    """The run list, in written order, from the corpus's own ``_calib/RUNS.txt`` (written by
    ``scripts/corpus.sh plan``). ``corpus_root`` is the generation directory, defaulting to
    ``HELIX_CORPUS``'s parent. The training configs read it rather than retyping the runs."""
    if corpus_root is None:
        from helix.paths import root
        corpus_root = root("HELIX_CORPUS").parent
    manifest = os.path.join(str(corpus_root), "_calib", "RUNS.txt")
    if not os.path.isfile(manifest):
        raise FileNotFoundError(
            f"no run manifest at {manifest}. _calib/RUNS.txt is written by "
            f"`scripts/corpus.sh plan` (docs/RUNBOOK.md §3); a corpus "
            f"without it is incomplete, not merely undocumented.")
    with open(manifest, "r", encoding="utf-8") as fh:
        runs = fh.read().split()
    if not runs:
        raise ValueError(f"{manifest} is empty")
    return runs


#: What every shard of one corpus shares. ``basis_digest`` covers the basis, the band lengths, the
#: removal gate, the threshold and sigma_norm; ``removal_json`` is named on its own so a message
#: can show the gate; ``norm_sigma`` must be FROZEN across a corpus (row i normalises any event in
#: any shard). ``CoeffTPCReader`` refuses a disagreement, :func:`verify` reports it.
SIGNATURE = ("basis_digest", "removal_json", "gids", "n_wires", "norm_sigma")


def shard_signature(cfg):
    """``{name: value}`` of :data:`SIGNATURE` for an open shard's ``/config`` (lists and str, so
    ``==`` compares them; ``None`` where absent)."""
    out = {}
    for k in SIGNATURE:
        v = cfg[k][()] if k in cfg else cfg.attrs.get(k)
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        out[k] = v.tolist() if isinstance(v, np.ndarray) else (None if v is None else str(v))
    return out


# ---- integrity -------------------------------------------------------------------------------

def read_plan(root):
    """[(prefix, first, count, events_per_shard, shard_start, inputs)] of ``root/_calib/CHUNKS.txt``.

    The plan records each shard prefix as the absolute path it was built at
    (``<built root>/<run>/<dataset>``). A corpus moved since -- staged to $SCRATCH, transferred
    to another site -- keeps the plan but not the path, so a prefix whose run directory is
    absent resolves under ``root`` by its last two components (run, dataset)."""
    out = []
    for line in open(os.path.join(root, "_calib", "CHUNKS.txt")):
        if line.strip():
            p, first, count, eps, k, *inputs = line.split()
            if not os.path.isdir(os.path.dirname(p)):
                p = os.path.join(root, *p.rstrip("/").split("/")[-2:])
            out.append((p, int(first), int(count), int(eps), int(k), inputs))
    return out


def _ident(f):
    s = lambda x: x.decode() if isinstance(x, bytes) else str(x)       # noqa: E731
    g = f["ident"]
    return [(s(r), s(sf), int(e)) for r, sf, e in zip(g["run"][:], g["source_file"][:], g["event"][:])]


def _index(path):
    return int(path.rsplit("_", 1)[1][:-3])


def verify(root, dataset_name=None, *, expect_events=None):
    """Problems of a corpus (empty: whole); entries starting ``NOTE:`` do not fail it.

    ``root`` with ``_calib/CHUNKS.txt`` (no ``dataset_name``): every planned shard is present with
    its planned number of events and nothing else is, every optical shard holds its TPC shard's
    events, and every coeff shard carries ``_calib/norm_sigma_global.npy``. A run directory with
    ``dataset_name``: its ``{dataset_name}_coeff_*.h5`` (every gap reported; ``expect_events`` the
    total). Then, over all coeff shards: one :func:`shard_signature`, one helix commit and a clean
    tree, no event identity twice. Pure file inspection: a partial corpus can be checked too."""
    probs, planned = [], {}
    if dataset_name is None:
        for prefix, _, count, eps, k, _ in read_plan(root):
            for j in range(-(-count // eps)):
                planned[(prefix, k + j)] = min(eps, count - j * eps)
        g = os.path.join(root, "_calib", "norm_sigma_global.npy")
        norm = np.load(g) if os.path.exists(g) else None
        for prefix in sorted({p for p, _ in planned}):
            found = {_index(q) for kind in ("coeff", "optical") for q in glob.glob(f"{prefix}_{kind}_[0-9]*.h5")}
            probs += [f"{prefix}: shard {k} is not in the plan" for k in sorted(found - {k for p, k in planned
                                                                                       if p == prefix})]
        coeff, optical = [], {}
        for (prefix, k), n in sorted(planned.items()):
            have = {kind: q for kind in ("coeff", "optical")
                    if os.path.exists(q := f"{prefix}_{kind}_{k:04d}.h5")}
            if not have:
                probs.append(f"{prefix}: shard {k} missing")
            if "coeff" in have:
                coeff.append((have["coeff"], n))
            if "optical" in have:
                try:
                    with h5py.File(have["optical"], "r") as f:
                        if int(f["config"].attrs["n_events"]) != n:
                            probs.append(f"{have['optical']}: {int(f['config'].attrs['n_events'])} events, "
                                         f"planned {n}")
                        optical[have.get("coeff")] = (have["optical"], _ident(f))
                except Exception as e:                              # unreadable or truncated
                    probs.append(f"{have['optical']}: cannot read ({type(e).__name__}: {e})")
    else:
        norm, optical = None, {}
        paths = sorted(glob.glob(os.path.join(root, f"{dataset_name}_coeff_[0-9]*.h5")))
        if not paths:
            return [f"no {dataset_name}_coeff_*.h5 shards under {root}"]
        have = [_index(p) for p in paths]
        gap = sorted(set(range(have[0], have[-1] + 1)) - set(have))
        if gap:
            probs.append(f"file_index gap — shards {gap} absent (a failed job leaves exactly this; the "
                         f"corpus's plan says whether it was meant to exist: verify the corpus root)")
        coeff = [(p, None) for p in paths]

    ref = None
    commits, dirty, no_commit, inputs, ident, total = Counter(), [], [], set(), Counter(), 0
    for p, want in coeff:
        try:
            with h5py.File(p, "r") as f:
                cfg = f["config"]
                sig, n = shard_signature(cfg), int(cfg.attrs["n_events"])
                prov = json.loads(str(cfg.attrs.get("provenance_json", "{}")))
                noise = str(cfg.attrs.get("noise_json", ""))
                if norm is not None and not np.array_equal(cfg["norm_sigma"][:], norm):
                    probs.append(f"{p}: norm_sigma is not _calib/norm_sigma_global.npy")
                ids = _ident(f)
                ident.update(ids)
            if p in optical and optical[p][1] != ids:
                probs.append(f"{optical[p][0]}: events differ from its TPC shard")
        except Exception as e:                                      # unreadable or truncated
            probs.append(f"{p}: cannot read ({type(e).__name__}: {e})")
            continue
        total += n
        if want is not None and n != want:
            probs.append(f"{p}: {n} events, planned {want}")
        if ref is None:
            ref, ref_p = sig, p
        for k in SIGNATURE:
            if sig[k] != ref[k]:
                probs.append(f"shards disagree on {k}: {os.path.basename(ref_p)} vs {os.path.basename(p)}"
                             + ("  (norm_sigma must be FROZEN across a corpus)" if k == "norm_sigma" else ""))
        code = prov.get("code") or {}
        if code.get("git"):
            commits[code["git"]] += 1
            if code.get("git_dirty"):
                dirty.append(os.path.basename(p))
        else:
            no_commit.append(os.path.basename(p))
        # the DSP's inputs, which basis_digest does not cover: geometry, noise spectrum, noise model
        inputs.add(json.dumps([prov.get("geom_sha256"), prov.get("spectrum_sha256"), noise]))

    if coeff and not commits:
        probs.append(f"NO shard records the helix commit that built it ({len(coeff)} shards): "
                     f"a mixed-tree build is undetectable")
    if len(commits) > 1:
        probs.append(f"shards were built by DIFFERENT helix commits: {dict(commits)}")
    if dirty:
        probs.append(f"corpus was built from a DIRTY working tree ({len(dirty)} shards, e.g. {dirty[0]}) "
                     f"— the recorded commit does not describe the code that ran")
    if commits and no_commit:
        probs.append(f"NOTE: {len(no_commit)} shards record no commit (git timed out under a wide job "
                     f"array?), e.g. {no_commit[0]}")
    if len(inputs) > 1:
        probs.append(f"shards were built against DIFFERENT DSP inputs ([geometry sha256, spectrum "
                     f"sha256, noise]): {sorted(inputs)}")
    dups = [k for k, c in ident.items() if c > 1]
    if dups:
        probs.append(f"{len(dups)} event identities appear more than once (overlapping ranges "
                     f"duplicate training data invisibly); first few: {dups[:5]}")
    if expect_events is not None and total != expect_events:
        probs.append(f"event count {total} != expected {expect_events}")
    return probs


# ---- compare ---------------------------------------------------------------------------------

SKIP_ATTRS = {"provenance_json", "created", "created_utc", "elapsed_s", "host"}


def compare(pa, pb):
    """Differences between two shard files (empty: the same content): every dataset
    byte-identical (variable-length strings by value), every attribute equal except the
    build's provenance (commit, host, time)."""
    def items(f):
        out = {}
        f.visititems(lambda k, v: out.__setitem__(k, v))
        return out

    def attrs(o):
        return {k: v for k, v in o.attrs.items() if k not in SKIP_ATTRS}

    def same(x, y):
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            return np.asarray(x).shape == np.asarray(y).shape and np.asarray(x).tobytes() == np.asarray(y).tobytes()
        return x == y

    diffs = []
    with h5py.File(pa, "r") as a, h5py.File(pb, "r") as b:
        ia, ib = items(a), items(b)
        diffs += [f"{k}: only in {'A' if k in ia else 'B'}" for k in sorted(set(ia) ^ set(ib))]
        for k in sorted(set(ia) & set(ib)) + ["/"]:
            oa, ob = (a, b) if k == "/" else (ia[k], ib[k])
            if isinstance(oa, h5py.Dataset):
                da, db = oa[()], ob[()]
                if da.dtype.kind == "O":
                    eq = oa.dtype == ob.dtype and da.shape == db.shape and list(da.ravel()) == list(db.ravel())
                else:
                    eq = oa.dtype == ob.dtype and da.tobytes() == db.tobytes()
                if not eq:
                    diffs.append(f"{k}: data differ")
            ta, tb = attrs(oa), attrs(ob)
            diffs += [f"{k}@{n}: attribute differs" for n in sorted(set(ta) | set(tb))
                      if n not in ta or n not in tb or not same(ta[n], tb[n])]
    return diffs


def _report(probs):
    notes = [p for p in probs if p.startswith("NOTE:")]
    fatal = [p for p in probs if not p.startswith("NOTE:")]
    for n in notes:
        print(f"  ~ {n}")
    if fatal:
        print(f"CORPUS FAILED ({len(fatal)} problem(s)):")
        for p in fatal:
            print(f"  - {p}")
        return 1
    print("corpus OK")
    return 0


def main(argv=None):
    import argparse
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "compare":
        bad = 0
        for pa, pb in zip(argv[1::2], argv[2::2]):
            d = compare(pa, pb)
            print(f"{'IDENTICAL' if not d else 'DIFFERENT'}  {pa}  vs  {pb}" + "".join(f"\n    {x}" for x in d[:20]))
            bad += bool(d)
        return 1 if bad else 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_root", help="a corpus root with _calib/CHUNKS.txt, or a run dir (with --dataset-name)")
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument("--expect", type=int, default=None, help="expected total event count (a run dir)")
    ap.add_argument("--modalities", nargs="*", choices=["coeff"], help=argparse.SUPPRESS)   # coeff is the only one
    a = ap.parse_args(argv)
    if a.dataset_name is None and not os.path.exists(os.path.join(a.data_root, "_calib", "CHUNKS.txt")):
        ap.error("--dataset-name is required unless DATA_ROOT has _calib/CHUNKS.txt")
    return _report(verify(a.data_root, a.dataset_name, expect_events=a.expect))


if __name__ == "__main__":
    raise SystemExit(main())
