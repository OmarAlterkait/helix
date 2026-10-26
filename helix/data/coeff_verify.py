"""Corpus-level integrity check for a coeff corpus.

``helix.core.coeff_io.audit_shard`` is thorough but strictly PER FILE — it runs
inside the builder after every shard write and cannot see anything that spans
shards. Every failure the build actually produced in practice is cross-file:

- a job that died after writing its noisy shard, leaving no ``coeff_clean`` pair
  (observed: an OOM-killed 1000-event build left a complete, valid, readable
  1.36 GB noisy shard and nothing noticed);
- a shard missing from the middle of a run (a failed array task);
- overlapping ``--event-start`` ranges duplicating events across shards;
- shards that disagree on the frozen ``norm_sigma`` / basis / plane set.

None of these corrupt a file. Each one silently changes how much data the corpus
contains, which no per-file audit and no training run will ever report.

A whole corpus built by ``scripts/corpus.sh`` is also checked against its plan
(``_calib/CHUNKS.txt``, :func:`helix.build.plan`): every planned shard exists with its planned
number of events and nothing else is there, every optical shard holds its TPC shard's events,
and every shard carries the corpus's one ``_calib/norm_sigma_global.npy``.

Usage::

    python -m helix.data.coeff_verify <run_dir> --dataset-name sim_wire [--expect 1000]
    python -m helix.data.coeff_verify <corpus_root>          every run, against _calib/CHUNKS.txt
    python -m helix.data.coeff_verify compare A B [A2 B2 ...]   same content, byte for byte
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter

import numpy as np
import h5py


def verify_corpus(data_root, dataset_name, *, split="", expect_events=None,
                  modalities=("coeff", "coeff_clean"), source_root=None):
    """Check a built corpus for cross-shard problems. Returns a list of problem
    strings (empty == clean). Does not open the dataset; pure file inspection, so
    it works on a partially built corpus too."""
    probs = []
    base = os.path.join(data_root, split) if split else data_root

    def shards(mod):
        pat = os.path.join(base, f"{dataset_name}_{mod}_[0-9]*.h5")
        return sorted(glob.glob(pat))

    noisy = shards("coeff")
    if not noisy:
        return [f"no {dataset_name}_coeff_*.h5 shards under {base}"]

    def idx_of(p):
        return int(os.path.basename(p).rsplit("_", 1)[-1][:-3])

    # --- 1. pair completeness -------------------------------------------------
    if "coeff_clean" in modalities:
        n_idx = {idx_of(p) for p in noisy}
        c_idx = {idx_of(p) for p in shards("coeff_clean")}
        if n_idx - c_idx:
            probs.append(f"coeff shards with no coeff_clean pair: {sorted(n_idx - c_idx)}")
        if c_idx - n_idx:
            probs.append(f"coeff_clean shards with no coeff pair: {sorted(c_idx - n_idx)}")

    # --- 2. file_index contiguity --------------------------------------------
    # A gap usually means a build job died. It can also mean the SOURCE file was
    # never there: run_0027670361 upstream has 90 files, not 100, so shards
    # 51-56 and 94-97 can never exist and the corpus is complete without them.
    # Those two cases look identical from inside the corpus, and the difference
    # is one directory away — so `source_root` is asked for the evidence rather
    # than a flag being offered to silence the check. With it, a gap whose source
    # is ABSENT is fine and a gap whose source EXISTS is still an error, which is
    # a strictly stronger check than the unconditional one it replaces.
    have = sorted(idx_of(p) for p in noisy)
    missing = sorted(set(range(have[0], have[-1] + 1)) - set(have))
    if missing and source_root:
        src_dir = os.path.join(source_root, split) if split else source_root
        recoverable = [i for i in missing if os.path.exists(
            os.path.join(src_dir, f"{dataset_name}_sensor_{i:04d}.h5"))]
        upstream = [i for i in missing if i not in recoverable]
        if upstream:
            print(f"  note: shards {upstream} have no source under {src_dir} — "
                  f"absent upstream, not a failed build")
        missing = recoverable
    if missing:
        probs.append(f"file_index gap — shards {missing} absent between "
                     f"{have[0]} and {have[-1]} (a failed job leaves exactly this"
                     + ("; pass --source-root to distinguish an upstream gap)"
                        if not source_root else ", and the source file IS present)"))

    # --- 3. cross-shard agreement + identity uniqueness ----------------------
    ref = None
    ident = Counter()
    code = []
    dsp = []
    total = 0
    for p in noisy:
        try:
            with h5py.File(p, "r") as f:
                cfg = f["config"]
                cur = dict(
                    band_lengths=cfg["band_lengths"][:].tolist(),
                    gids=cfg["gids"][:].tolist(),
                    n_wires=cfg["n_wires"][:].tolist(),
                    norm_sigma=(cfg["norm_sigma"][:].tolist()
                                if "norm_sigma" in cfg else None),
                    basis_digest=str(cfg.attrs.get("basis_digest", "")),
                    noise_json=str(cfg.attrs.get("noise_json", "")),
                )
                # Which helix built this shard. Compared separately from the
                # fields above because absence is LEGACY, not disagreement:
                # shards predating the field simply have none, and flagging
                # every one of them would bury the case this exists to catch —
                # two shards of one corpus built by different code.
                _prov = json.loads(str(cfg.attrs.get("provenance_json", "{}")))
                code.append((os.path.basename(p), _prov.get("code")))
                # The DSP's INPUTS: which geometry file and which noise spectrum
                # produced this shard. Both hashes are already written by
                # the sim build and were never compared — so a corpus
                # assembled from shards built against different geometry, or
                # against a different (or white) noise spectrum, passed clean.
                # Handled like `code`, not like the fields above: absence is
                # LEGACY, so only shards that carry them are compared.
                dsp.append((os.path.basename(p),
                            {k: _prov.get(k) for k in
                             ("geom", "geom_sha256", "spectrum", "spectrum_sha256")}
                            if any(k in _prov for k in
                                   ("geom_sha256", "spectrum_sha256")) else None))
                n = int(cfg.attrs["n_events"])
                total += n
                if "ident" in f:
                    runs = f["ident"]["run"][:]
                    srcs = f["ident"]["source_file"][:]
                    evs = f["ident"]["event"][:]
                    for r, s, e in zip(runs, srcs, evs):
                        r = r.decode() if isinstance(r, bytes) else str(r)
                        s = s.decode() if isinstance(s, bytes) else str(s)
                        ident[(r, s, int(e))] += 1
                else:
                    probs.append(f"{os.path.basename(p)}: no /ident group")
        except Exception as e:                       # unreadable/truncated
            probs.append(f"{os.path.basename(p)}: cannot read ({type(e).__name__}: {e})")
            continue
        if ref is None:
            ref, ref_p = cur, p
        else:
            for k in cur:
                if cur[k] != ref[k]:
                    probs.append(
                        f"shards disagree on {k}: {os.path.basename(ref_p)} vs "
                        f"{os.path.basename(p)}"
                        + ("  (norm_sigma must be FROZEN across a corpus — build "
                           "with --norm-sigma)" if k == "norm_sigma" else ""))

    # --- 3b. code version ----------------------------------------------------
    known = [(f, c) for f, c in code if c]
    if not known and code:
        # A corpus where NOTHING records its builder must fail, not pass. The
        # guard below only compares shards that carry `code`, so a corpus with
        # zero of them skipped the check entirely and reported clean — measured
        # on run_0027575715: 0 of 100 shards carry it, verify_corpus said OK,
        # and one surviving build log shows 12 shards were rebuilt from a
        # different worktree. "Which code produced this data" is the one
        # question a provenance check exists to answer.
        probs.append(
            f"NO shard records provenance['code'] ({len(code)} shards): the "
            f"corpus cannot say which helix built it, so a mixed-tree build is "
            f"undetectable. Rebuild with a current builder, or record the "
            f"provenance alongside the shards.")
    if known:
        # Compare FIELD BY FIELD over the keys a pair of records actually share,
        # not by serialising the whole dict. A record that omits `git_dirty`
        # because `git status` timed out is INCOMPLETE, not DIFFERENT: it does
        # not contradict a record that has it. Comparing json.dumps() treated
        # {"git": X, "git_dirty": false} and {"git": X} as two versions and
        # failed a corpus built entirely by commit X -- which is a false refusal,
        # and the worst kind, because the fix looks like "rebuild 344 GB".
        conflicts = {}
        for key in {k for _, c in known for k in c}:
            vals = {json.dumps(c[key], sort_keys=True) for _, c in known if key in c}
            if len(vals) > 1:
                conflicts[key] = sorted(vals)
        if conflicts:
            probs.append(
                "shards were built by DIFFERENT helix versions: "
                + "; ".join(f"{k}={' vs '.join(v)}" for k, v in sorted(conflicts.items()))
                + "  (basis_digest and the geom/spectrum hashes pin the DSP's "
                  "inputs, not its code, so this is the only field that shows it)")
        elif len({frozenset(c) for _, c in known}) > 1:
            # Same values everywhere, but some shards recorded fewer fields.
            # Not a refusal -- nothing disagrees -- but say so, because it means
            # provenance capture failed on some tasks and the NEXT build might
            # lose a field that would have mattered.
            missing = sorted({k for _, c in known for k in c}
                             - set.intersection(*[set(c) for _, c in known]))
            probs.append(
                f"NOTE: provenance is INCOMPLETE on some shards (missing: {missing}). "
                f"Every field that IS recorded agrees, so this is not a mixed "
                f"build -- helix.core.coeff_io._code_version gives up on a git "
                f"timeout, which happens under a wide job array on a shared "
                f"filesystem.")
        elif any(c.get("git_dirty") for _, c in known):
            probs.append(
                f"corpus was built from a DIRTY working tree "
                f"({known[0][1].get('git')}) — the recorded commit does not "
                f"describe the code that ran")
        if len(known) != len(code):
            probs.append(
                f"{len(code) - len(known)} of {len(code)} shards predate the "
                f"provenance 'code' field (legacy); the rest record "
                f"{known[0][1].get('git', '?')}")

    # --- 3c. DSP inputs (geometry + noise spectrum) ---------------------------
    known_dsp = [(f, d) for f, d in dsp if d]
    if known_dsp:
        seen = {json.dumps(d, sort_keys=True) for _, d in known_dsp}
        if len(seen) > 1:
            probs.append(
                "shards were built against DIFFERENT DSP inputs (geometry or "
                "noise spectrum): " + "; ".join(sorted(seen))
                + "  (a corpus mixing geometries or noise models is not one "
                  "corpus; the coefficients are not comparable)")
        if len(known_dsp) != len(dsp):
            probs.append(
                f"{len(dsp) - len(known_dsp)} of {len(dsp)} shards predate the "
                f"geom/spectrum provenance hashes (legacy); the rest record "
                f"geom={known_dsp[0][1].get('geom')} "
                f"spectrum={known_dsp[0][1].get('spectrum')}")

    # --- 4. duplicate events -------------------------------------------------
    dups = [k for k, c in ident.items() if c > 1]
    if dups:
        probs.append(
            f"{len(dups)} event identities appear more than once (overlapping "
            f"--event-start ranges duplicate training data invisibly); first few: "
            f"{dups[:5]}")

    # --- 5. expected count ---------------------------------------------------
    if expect_events is not None and total != expect_events:
        probs.append(f"event count {total} != expected {expect_events}")

    return probs


# ------------------------------------------------------------------------------------- plan
def read_plan(root):
    """[(prefix, first, count, events_per_shard, shard_start, inputs)] of ``root/_calib/CHUNKS.txt``."""
    out = []
    for line in open(os.path.join(root, "_calib", "CHUNKS.txt")):
        if line.strip():
            p, first, count, eps, k, *inputs = line.split()
            out.append((p, int(first), int(count), int(eps), int(k), inputs))
    return out


def _ident(f):
    s = lambda x: x.decode() if isinstance(x, bytes) else str(x)       # noqa: E731
    g = f["ident"]
    return [(s(r), s(sf), int(e)) for r, sf, e in zip(g["run"][:], g["source_file"][:], g["event"][:])]


def verify_plan(root):
    """A corpus against its plan: problems (empty == complete). Then every run dir with
    :func:`verify_corpus`."""
    probs, want = [], {}
    for prefix, _, count, eps, k, _ in read_plan(root):
        for j in range(-(-count // eps)):
            want.setdefault(prefix, {})[k + j] = min(eps, count - j * eps)
    g = os.path.join(root, "_calib", "norm_sigma_global.npy")
    norm = np.load(g) if os.path.exists(g) else None
    for prefix, shards in sorted(want.items()):
        found = {int(p.rsplit("_", 1)[1][:-3]) for kind in ("coeff", "optical")
                 for p in glob.glob(f"{prefix}_{kind}_[0-9]*.h5")}
        probs += [f"{prefix}: shard {k} is not in the plan" for k in sorted(found - set(shards))]
        for k, n in sorted(shards.items()):
            paths = {kind: f"{prefix}_{kind}_{k:04d}.h5" for kind in ("coeff", "optical")}
            paths = {kind: q for kind, q in paths.items() if os.path.exists(q)}
            if not paths:
                probs.append(f"{prefix}: shard {k} missing")
                continue
            ids = {}
            for kind, q in paths.items():
                try:
                    with h5py.File(q, "r") as f:
                        if int(f["config"].attrs["n_events"]) != n:
                            probs.append(f"{q}: {int(f['config'].attrs['n_events'])} events, planned {n}")
                        ids[kind] = _ident(f)
                        if kind == "coeff" and norm is not None and not np.array_equal(
                                f["config/norm_sigma"][:], norm):
                            probs.append(f"{q}: norm_sigma is not _calib/norm_sigma_global.npy")
                except Exception as e:                          # unreadable or truncated
                    probs.append(f"{q}: cannot read ({type(e).__name__}: {e})")
            if len(ids) == 2 and ids["optical"] != ids["coeff"]:
                probs.append(f"{paths['optical']}: events differ from its TPC shard")
    for prefix in sorted({p for p in want if glob.glob(f"{p}_coeff_[0-9]*.h5")}):
        d, name = os.path.split(prefix)
        probs += [f"{os.path.basename(d)}: {p}" for p in verify_corpus(
            d, name, modalities=("coeff", "coeff_clean") if glob.glob(f"{prefix}_coeff_clean_*.h5")
            else ("coeff",))]
    return probs


# ------------------------------------------------------------------------------------- compare
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
    # "NOTE:" entries are observations that do not contradict anything -- they
    # are printed, but they do not fail the corpus. Everything else does.
    notes = [p for p in probs if "NOTE:" in p]
    fatal = [p for p in probs if "NOTE:" not in p]
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
    ap.add_argument("data_root", help="a run dir (with --dataset-name), or a corpus root with _calib/CHUNKS.txt")
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument("--split", default="")
    ap.add_argument("--expect", type=int, default=None,
                    help="expected total event count across the corpus")
    ap.add_argument("--modalities", nargs="*", default=["coeff", "coeff_clean"])
    ap.add_argument("--source-root", default=None,
                    help="the SENSOR tree the corpus was built from. Given, a "
                         "file_index gap is judged against it: absent upstream "
                         "is fine, present-but-unbuilt is still an error.")
    a = ap.parse_args(argv)
    if a.dataset_name is None:
        if not os.path.exists(os.path.join(a.data_root, "_calib", "CHUNKS.txt")):
            ap.error("--dataset-name is required unless DATA_ROOT has _calib/CHUNKS.txt")
        return _report(verify_plan(a.data_root))
    return _report(verify_corpus(a.data_root, a.dataset_name, split=a.split,
                                 expect_events=a.expect, modalities=tuple(a.modalities),
                                 source_root=a.source_root))


if __name__ == "__main__":
    raise SystemExit(main())
