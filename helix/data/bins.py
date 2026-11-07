"""Derive the categorical-head bin grid from a coeff corpus.

The FM's value head predicts a coefficient's magnitude as one of K bins rather
than regressing it. Those bins are **training-set statistics**: they are derived
from the corpus a model trains on, they are not recoverable from a trained
checkpoint, and a model is only meaningful against the grid it was trained with.

This module exists so the grid does not have to be *carried*. The derivation is
deterministic — it pools the FIRST ``events`` events in dataset order, with no
sampling, seed or RNG anywhere — so a receiving site with the corpus can
reproduce a bin table bit-identically instead of treating an 8 KB file as a
single point of failure. :func:`rederive` does exactly that from a table's own
recorded parameters, and :func:`compare` checks the result.

The algorithm must stay unchanged: altering it silently invalidates every checkpoint trained
against an existing grid. The command line (:func:`main`) has three modes::

    # derive a fresh grid
    python -m helix.data.bins --corpus <run-dir> --out bins.pt

    # reproduce an existing grid, using the parameters it records
    python -m helix.data.bins --like bins.pt --corpus <run-dir> --out new.pt

    # reproduce it and assert the result matches, writing nothing
    python -m helix.data.bins --verify bins.pt --corpus <run-dir>

``--verify`` is the handover check: it proves a receiving site can regenerate the
grid from its own copy of the corpus rather than carry the file. It exits
non-zero if the tables differ.

Algorithm:

* work in ``tgt = arcsinh(coeff / sigma)`` of the NOISY coefficients — the space
  the head predicts in (training is noisy-only: real data has no clean counterpart)
* K bins UNIFORM in that space over the band's robust range ``[lo_pct, hi_pct]``,
  with ONE bin spanning the sub-threshold gap ``(-a, a)``: noisy coefficients exist
  only beyond the significance threshold, so a single uniform grid would put ~30-40%
  of its bins where no value can fall. Uniform-in-asinh is log-spaced in raw charge,
  i.e. constant RELATIVE precision — NOT quantile bins, so non-uniform occupancy is
  expected and correct
* outer two bins extended to +-inf so the tails cannot fall off the grid

``sigma`` is the per-``(plane, band)`` ``norm_sigma`` table, so the edges are in
exactly the space ``CoeffTokenize`` produces.

Produces ``edges`` (n_bands, K+1), ``cent_asinh`` (n_bands, K) — the point
estimate in model space — and ``cent_ratio`` (n_bands, K) = E[coeff/sigma | bin],
the charge read-back centroid.

It does not produce ``cent_lin`` (E[raw ADC | bin]). That table pooled planes
whose ``norm_sigma`` differ by 22%, nothing consumed it, and ``set_bins`` now
rejects unknown centroid tables rather than let one bind to the wrong slot. An
older sidecar that still carries the key applies fine — ``apply_bins`` ignores it.

``torch`` is imported inside :func:`save` and :func:`load` rather than at module
scope: the derivation itself is pure numpy, and helix keeps torch optional
wherever it can.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


def reference_path() -> Path:
    """The packaged fingerprint of the grid the released models were trained on."""
    return Path(__file__).resolve().parent / "data" / "reference_bins.json"


def reference() -> dict[str, Any] | None:
    """Load the packaged reference fingerprint, or None if it is not installed."""
    p = reference_path()
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def reference_name() -> str:
    """The canonical NAME of the production grid, e.g. ``coeff_bins_r2_run0027575715_noisy``.

    The ``.pt`` on disk is ``<name>.pt`` under ``HELIX_ARCHIVE``, so this is what a
    config builds that filename from. The name is written once, in
    ``reference_bins.json``: a literal retyped elsewhere goes stale when the grid is
    re-derived, and then names a file that does not exist or, worse, one that holds
    a grid derived from a different corpus.
    """
    ref = reference()
    if not ref or not ref.get("name"):
        raise SystemExit(
            "reference_bins.json is missing or declares no `name`. It is the "
            "definition of the production grid; without it the bin table cannot "
            "be located by name. A wheel built without package-data has no copy."
        )
    return str(ref["name"])


def reference_table(archive: Any = None) -> Path:
    """Where the production ``.pt`` lives: ``<archive>/<reference_name()>.pt``.

    ``archive`` defaults to ``helix.paths.root("HELIX_ARCHIVE")``. Imported
    lazily so this module stays usable with no site configured, which is the
    case in unit tests that only exercise the digests.
    """
    if archive is None:
        from helix.paths import root as _root
        archive = _root("HELIX_ARCHIVE")
    return Path(archive) / f"{reference_name()}.pt"


#: Fallback only. The authority is ``data/reference_bins.json`` — see DEFAULTS
#: below, which reads from it. These values exist so the module still imports
#: with its package data stripped, not as a second place to edit them.
_FALLBACK_DEFAULTS = dict(dataset_name="sim_wire", events=120, K=128, n_bands=4,
                          lo_pct=0.05, hi_pct=99.95)


def _defaults() -> dict[str, Any]:
    """Derivation parameters, from the packaged spec.

    The parameters are configuration, not a property of this code, so they are
    declared once in ``data/reference_bins.json`` and read here. The CLI takes
    its argparse defaults from this, which is what keeps "what the production
    grid is" from existing in two places that can disagree.
    """
    ref = reference()
    if not ref:
        return dict(_FALLBACK_DEFAULTS)
    p = dict(_FALLBACK_DEFAULTS)
    p.update({k: v for k, v in (ref.get("params") or {}).items() if k in p})
    return p


#: The derivation parameters, resolved once at import from the packaged spec.
DEFAULTS = _defaults()

#: Keys a derived table always carries. ``compare`` checks the arrays; the
#: scalars are provenance, and are what makes :func:`rederive` possible.
ARRAY_KEYS = ("edges", "cent_asinh", "cent_ratio")
PARAM_KEYS = ("K", "n_bands", "events", "corpus")

#: The numpy CPU features the grid is derived WITHOUT. Where the CPU has AVX-512 (S3DF's turing
#: Xeons), numpy runs arcsinh, log, mean and percentile through AVX-512 kernels that round
#: differently in the last bits, so the same corpus gives a grid up to ~20 float32 ulp away
#: (6.5e-5 at most, in cent_ratio) and different digests: that was the S3DF/NERSC "v2/v3"
#: disagreement, with both corpora byte-identical. The reference is the grid WITHOUT them -- AMD
#: Milan (S3DF milano, NERSC), or turing with this set -- so set it before numpy is imported;
#: :func:`main` does that itself (it re-executes itself with it).
CANONICAL_NUMPY_ENV = {"NPY_DISABLE_CPU_FEATURES":
                       "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"}


def avx512_active() -> bool:
    """Whether this process's numpy dispatches to AVX-512 kernels (see CANONICAL_NUMPY_ENV)."""
    import os
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as feats
    except ImportError:                                    # numpy < 2
        from numpy.core._multiarray_umath import __cpu_features__ as feats
    off = set(os.environ.get("NPY_DISABLE_CPU_FEATURES", "").replace(",", " ").split())
    return bool(feats.get("AVX512F")) and "AVX512F" not in off


def derive(corpus, *, dataset_name=DEFAULTS["dataset_name"],
           events=DEFAULTS["events"], K=DEFAULTS["K"],
           n_bands=DEFAULTS["n_bands"], lo_pct=DEFAULTS["lo_pct"],
           hi_pct=DEFAULTS["hi_pct"],
           report: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Pool ``events`` events from ``corpus`` and return the bin table.

    ``corpus`` is a run directory (``<root>/<run>/``), not a corpus root — the
    grid is derived from one run's statistics and the run it came from is
    recorded in the result.

    Returns a plain dict of numpy arrays plus the parameters used. Pass it to
    :func:`save` to write the ``.pt`` the trainer reads. ``report`` receives
    one progress line per band; pass ``print`` for the CLI's behaviour.
    """
    from helix.data.dataset import CoeffTPCDataset
    from helix.model.tokenize import sigma_for_rows

    say = report or (lambda _msg: None)

    ds = CoeffTPCDataset(data_root=corpus, dataset_name=dataset_name,
                         modalities=("coeff",), transform=None)
    n = min(events, len(ds))
    say(f"pooling {n} of {len(ds)} events from {corpus}")

    vals: dict[Any, list] = {b: [] for b in range(n_bands)}
    for i in range(n):
        s = ds.get_data(i)
        c = s["coeff"]
        meta = c["_meta"]
        band = np.asarray(c["band"], np.int64)
        gid = np.asarray(c["plane_gid"], np.int64)
        value = np.asarray(c["value"], np.float32).reshape(-1)
        keep = band < n_bands
        band, gid, value = band[keep], gid[keep], value[keep]
        sig = np.maximum(sigma_for_rows(gid, band, meta["gids"], meta["norm_sigma"]), 1e-6)
        t = np.arcsinh(value / sig).astype(np.float64)
        # value/sigma == sinh(t), the DIMENSIONLESS coefficient. This is what a
        # read-back needs. The alternative — E[raw ADC | bin] — pools planes whose
        # norm_sigma differ by 22%, biasing a Y-plane coefficient 13% low and a
        # U/V one 6% high. That cancels on a random mask and does NOT cancel on a
        # plane mask (measured 0.857 vs 1.021 cross-plane): invisible on the
        # metric people look at, wrong on the one that matters.
        ratio = (value / sig).astype(np.float64)
        for b in range(n_bands):
            sel = band == b
            vals[b].append(t[sel])
            vals.setdefault(("raw", b), []).append(value[sel])
            vals.setdefault(("ratio", b), []).append(ratio[sel])

    edges = np.zeros((n_bands, K + 1), np.float32)
    cent_a = np.zeros((n_bands, K), np.float32)
    cent_r = np.zeros((n_bands, K), np.float32)   # E[coeff/sigma | bin]
    for b in range(n_bands):
        t = np.concatenate(vals[b])
        v = np.concatenate(vals[("raw", b)])
        r = np.concatenate(vals[("ratio", b)])
        lo, hi = np.percentile(t, lo_pct), np.percentile(t, hi_pct)
        # Same bin width on both sides of the gap (-a, a); one bin spans it.
        a = float(np.percentile(np.abs(t), lo_pct))
        w = ((-a - lo) + (hi - a)) / (K - 1)
        kn = int(round((-a - lo) / w))
        e = np.concatenate([np.linspace(lo, -a, kn + 1), np.linspace(a, hi, K - kn)])
        idx = np.clip(np.digitize(t, e[1:-1]), 0, K - 1)
        for k in range(K):
            m = idx == k
            if m.any():
                cent_a[b, k] = t[m].mean()
                # E[sinh t | bin], MEASURED. Not sinh(E[t | bin]) — the head is
                # categorical, and a posterior spread over many bins makes
                # sinh(mean) a 31%-low estimate of the charge (Jensen). The
                # reference reads back as sum_k p_k * centroid_k and never
                # applies sinh to a mean.
                cent_r[b, k] = r[m].mean()
            else:                                  # empty bin: fall back to its centre
                cent_a[b, k] = 0.5 * (e[k] + e[k + 1])
                cent_r[b, k] = float(np.sinh(cent_a[b, k]))
        empty = int((np.bincount(idx, minlength=K) == 0).sum())
        e[0], e[-1] = -1e18, 1e18                  # tails cannot fall off the grid
        edges[b] = e
        say(f"  band {b}: n={len(t):>10,}  tgt[{lo:+.2f},{hi:+.2f}]  "
            f"empty bins={empty}  |coeff|max={np.abs(v).max():.0f}")

    return dict(edges=edges, cent_asinh=cent_a, cent_ratio=cent_r,
                K=K, n_bands=n_bands, corpus=str(corpus), events=n)


def save(table: Mapping[str, Any], path) -> None:
    """Write a derived table as the ``.pt`` the trainer's ``bins=`` reads."""
    import torch

    out = dict(table)
    for k in ARRAY_KEYS:
        out[k] = torch.as_tensor(np.asarray(out[k]))
    torch.save(out, path)


def load(path) -> dict[str, Any]:
    """Read a bin table ``.pt`` back as numpy, for comparison or rederivation."""
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=False)
    out = dict(raw)
    for k in ARRAY_KEYS:
        if k in out:
            out[k] = out[k].numpy() if hasattr(out[k], "numpy") else np.asarray(out[k])
    return out


def params_of(table: Mapping[str, Any]) -> dict[str, Any]:
    """The derivation parameters a table records, filled in from DEFAULTS.

    A table stores ``corpus``, ``events``, ``K`` and ``n_bands`` but not the
    percentiles, which have never been anything but the defaults. Anything
    missing falls back rather than raising, so an older sidecar still rederives.
    """
    p = dict(DEFAULTS)
    for k in PARAM_KEYS:
        if k in table:
            p[k] = table[k]
    return p


def rederive(reference, *, corpus=None,
             report: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Re-derive a table using the parameters the reference itself records.

    ``reference`` is a path to a ``.pt`` or an already-loaded table. Pass
    ``corpus`` to point at the same run in a new location — the recorded path is
    where it lived when it was derived, which is not where it lives now.

    This is the handover path: the bin grid does not need to be copied, because
    the corpus plus this function reproduces it exactly.
    """
    table = load(reference) if isinstance(reference, (str, bytes)) or hasattr(reference, "__fspath__") else reference
    p = params_of(table)
    return derive(corpus if corpus is not None else p["corpus"],
                  dataset_name=p["dataset_name"], events=p["events"],
                  K=p["K"], n_bands=p["n_bands"],
                  lo_pct=p["lo_pct"], hi_pct=p["hi_pct"], report=report)


def fingerprint(table: Mapping[str, Any]) -> dict[str, str]:
    """``sha256`` of each array, over its float32 bytes.

    Digests rather than the arrays themselves, because the arrays are the thing
    the corpus regenerates. What a clone needs shipped is not the grid — it is
    the means to tell whether the grid it just derived is the right one.
    """
    out = {}
    for k in ARRAY_KEYS:
        a = np.ascontiguousarray(np.asarray(table[k], dtype=np.float32))
        out[k] = "sha256:" + hashlib.sha256(a.tobytes()).hexdigest()
    return out


def check_corpus(corpus, *, dataset_name=None) -> dict[str, Any]:
    """Check a corpus against the spec BEFORE deriving from it.

    Reads one shard header, so it costs a second rather than the minutes a
    derivation takes, and it fails with the cause named instead of a digest
    mismatch after the fact.

    What it can decide, and what it cannot, is worth being precise about:

    * ``basis_digest`` **is** checkable, and catches the confusion that actually
      matters -- the pre-tau ``coeff_tpc`` generation against ``coeff_tpc_r1``.
      Those differ in which coefficients survive the coherent gate, so a grid
      derived from one is wrong for the other.
    * *Which run* is **not** checkable. ``basis_digest`` hashes the DSP recipe,
      so all eight r1 runs share it, and a shard's ``/config`` carries only
      ``band_lengths``, ``gids``, ``n_wires`` and ``norm_sigma`` -- no
      source-run identifier. The directory name is the only signal, and a copy
      can be renamed. Detecting a wrong RUN is therefore left to the output
      digest in :func:`check_reference`, which is the only mechanism that can.

    ``status`` is ``ok``, ``basis-mismatch``, ``run-name-differs`` (a warning,
    not an error -- the path is a convention), ``unreadable``, or
    ``no-reference``.
    """
    ref = reference()
    if ref is None:
        return dict(status="no-reference")

    import os

    from helix.data.verify import corpus_identity

    want = ref.get("corpus") or {}
    ident = corpus_identity(corpus, dataset_name or DEFAULTS["dataset_name"])
    if ident is None:
        return dict(status="unreadable", corpus=str(corpus))

    got = ident.get("basis_digest")
    if want.get("basis_digest") and got != want["basis_digest"]:
        return dict(status="basis-mismatch", expected=want["basis_digest"],
                    actual=got, generation=want.get("generation"))

    name = os.path.basename(str(corpus).rstrip("/"))
    if want.get("run") and name != want["run"]:
        return dict(status="run-name-differs", expected=want["run"], actual=name)
    return dict(status="ok", basis_digest=got)


def check_reference(table: Mapping[str, Any]) -> dict[str, Any]:
    """Check a derived table against the packaged fingerprint.

    This is the check available to a site that has **no copy of the original
    grid** — the common case after a handover. Deriving from the wrong run of a
    multi-run corpus produces a grid that is perfectly self-consistent and
    simply not the one the released models were trained against; without a
    reference to compare to, nothing else would notice.

    ``status`` is one of:

    ``match``        the derived grid IS the released grid.
    ``mismatch``     same parameters, different numbers — wrong run, or a
                     different shard set.
    ``params``       derived with different K/n_bands/events, so the digests are
                     not comparable. Deliberate if you meant to change the grid.
    ``not-recorded`` the reference names the grid but its digests are not yet recorded
                     (derived for the first time: record these).
    ``no-reference`` the fingerprint is not installed (an editable checkout
                     missing package data, or a stripped wheel).
    """
    ref = reference()
    if ref is None:
        return dict(status="no-reference", reference=None)

    p = ref.get("params", {})
    differing = {k: (p.get(k), table.get(k)) for k in ("K", "n_bands", "events")
                 if k in p and p[k] != table.get(k)}
    if differing:
        return dict(status="params", reference=ref, param_mismatches=differing)

    got = fingerprint(table)
    if not ref.get("digests"):
        return dict(status="not-recorded", reference=ref, digests=got)
    bad = {k: (ref["digests"][k], got[k]) for k in ARRAY_KEYS
           if ref["digests"].get(k) != got[k]}
    out = dict(status="mismatch" if bad else "match", reference=ref,
               digest_mismatches=bad, digests=got)
    if bad and avx512_active():
        out["hint"] = ("derived in a process whose numpy uses AVX-512 kernels: the digests are not "
                       "comparable; derive with bins.CANONICAL_NUMPY_ENV set")
    return out


def compare(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two bin tables. Returns per-array equality and max abs difference.

    ``identical`` is True only if every array matches exactly. The grid defines
    the model's objective, so "close" is not the question being asked — a
    near-miss means a different grid, not a rounding difference.
    """
    arrays = {}
    for k in ARRAY_KEYS:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        same = x.shape == y.shape and bool(np.array_equal(x, y))
        diff = float(np.abs(x - y).max()) if x.shape == y.shape else float("inf")
        arrays[k] = dict(identical=same, max_abs_diff=diff)
    params = {k: (a.get(k), b.get(k)) for k in ("K", "n_bands", "events")
              if a.get(k) != b.get(k)}
    return dict(identical=all(v["identical"] for v in arrays.values()) and not params,
                arrays=arrays, param_mismatches=params)


# ---- the command line ------------------------------------------------------------------------

def main(argv=None):
    """The command line (the module docstring's three modes)."""
    import argparse
    import os
    import sys

    if avx512_active() and argv is None:      # the grid is defined without numpy's AVX-512 kernels
        os.execve(sys.executable, [sys.executable, "-m", "helix.data.bins"] + sys.argv[1:],
                  dict(os.environ, **CANONICAL_NUMPY_ENV))
    ap = argparse.ArgumentParser(prog="python -m helix.data.bins",
                                 description="Derive categorical-head bin edges from a coeff corpus.")
    ap.add_argument("--corpus", help="corpus dir (<root>/<run>/); with --like or "
                                     "--verify, defaults to the recorded path")
    ap.add_argument("--out", help="output .pt (omit only with --verify)")
    ap.add_argument("--like", help="reuse the derivation parameters this .pt records")
    ap.add_argument("--verify", help="rederive and compare against this .pt; writes nothing")
    ap.add_argument("--dataset-name", default=DEFAULTS["dataset_name"])
    ap.add_argument("--events", type=int, default=DEFAULTS["events"],
                    help="events to pool (the reference grid: 120)")
    ap.add_argument("--K", type=int, default=DEFAULTS["K"], help="bins per band")
    ap.add_argument("--n-bands", type=int, default=DEFAULTS["n_bands"],
                    help="bands the tokenizer keeps (D1 and beyond are dropped)")
    ap.add_argument("--lo-pct", type=float, default=DEFAULTS["lo_pct"])
    ap.add_argument("--hi-pct", type=float, default=DEFAULTS["hi_pct"])
    # The basis check below exists to catch the pre-tau/r1 confusion, which is a
    # real hazard: those generations differ in which coefficients survive the
    # gate, so a grid from one does not describe the other. But a SYNTHETIC
    # corpus is a legitimately different generation and the caller knows it --
    # the CI workflow (.github/workflows/docker.yml) builds one from
    # pimm_data.testing and derives a grid over it, which needs this escape hatch.
    # It is opt-in and it says what it did.
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
        c = check_corpus(target, dataset_name=a.dataset_name)
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
        ref = load(reference)
        p = params_of(ref)
        corpus = a.corpus or p["corpus"]
        print(f"reference {reference}: events={p['events']} K={p['K']} "
              f"n_bands={p['n_bands']} corpus={p['corpus']}")
        if a.corpus and a.corpus != p["corpus"]:
            print(f"  deriving from {corpus} instead — the recorded path is where "
                  f"it lived when it was derived")
        table = rederive(ref, corpus=corpus, report=print)
    else:
        table = derive(a.corpus, dataset_name=a.dataset_name,
                              events=a.events, K=a.K, n_bands=a.n_bands,
                              lo_pct=a.lo_pct, hi_pct=a.hi_pct, report=print)

    if a.verify:
        r = compare(ref, table)
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
    r = check_reference(table)
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
    elif r["status"] == "not-recorded":
        print(f"the packaged reference ({ref.get('name')}) records no digests yet; if this is "
              f"its first derivation, record these in helix/data/data/reference_bins.json:")
        print(json.dumps(r["digests"], indent=2))
    else:
        print("note: no packaged reference fingerprint installed, so this grid could "
              "not be checked against the released one.")

    save(table, a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
