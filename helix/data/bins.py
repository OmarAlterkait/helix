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

The algorithm is unchanged from ``scripts/derive_coeff_bins.py``, which is now a
thin CLI over this module. It must stay unchanged: altering it silently
invalidates every checkpoint trained against an existing grid.

Algorithm, matching research ``fm/tier1_setup_bins.py``:

* work in ``tgt = arcsinh(clean / sigma)`` — the space the head predicts in
* K bins UNIFORM in that space over the band's robust range
  ``[lo_pct, hi_pct]``. Uniform-in-asinh is log-spaced in raw charge, i.e.
  constant RELATIVE precision, which is the physically motivated choice — NOT
  quantile bins, so non-uniform occupancy is expected and correct
* outer two bins extended to +-inf so the tails cannot fall off the grid

The one deliberate difference from the research script: it read ``val_clean``
from the npz cache and divided by a global ``SIGMA=2.6``, because that cache
stored values already scaled by ``SIGMA/sigma_tab``. The corpus stores RAW
coefficients, so we divide by the per-``(plane, band)`` ``norm_sigma`` table
directly. The two are the same quantity — SIGMA cancels — and doing it this way
means the edges are in exactly the space ``CoeffTokenize`` produces.

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
    from helix.data.coeff_dataset import CoeffTPCDataset
    from helix.model.tokenize import sigma_for_rows

    say = report or (lambda _msg: None)

    ds = CoeffTPCDataset(data_root=corpus, dataset_name=dataset_name,
                         modalities=("coeff", "coeff_clean"), transform=None)
    n = min(events, len(ds))
    say(f"pooling {n} of {len(ds)} events from {corpus}")

    vals: dict[Any, list] = {b: [] for b in range(n_bands)}
    for i in range(n):
        s = ds.get_data(i)
        c, cc = s["coeff"], s["coeff_clean"]
        meta = c["_meta"]
        band = np.asarray(c["band"], np.int64)
        gid = np.asarray(c["plane_gid"], np.int64)
        clean = np.asarray(cc["value"], np.float32).reshape(-1)
        keep = band < n_bands
        band, gid, clean = band[keep], gid[keep], clean[keep]
        sig = np.maximum(sigma_for_rows(gid, band, meta["gids"], meta["norm_sigma"]), 1e-6)
        t = np.arcsinh(clean / sig).astype(np.float64)
        # clean/sigma == sinh(t), the DIMENSIONLESS coefficient. This is what a
        # read-back needs. The alternative — E[raw ADC | bin] — pools planes whose
        # norm_sigma differ by 22%, biasing a Y-plane coefficient 13% low and a
        # U/V one 6% high. That cancels on a random mask and does NOT cancel on a
        # plane mask (measured 0.857 vs 1.021 cross-plane): invisible on the
        # metric people look at, wrong on the one that matters.
        ratio = (clean / sig).astype(np.float64)
        for b in range(n_bands):
            sel = band == b
            vals[b].append(t[sel])
            vals.setdefault(("raw", b), []).append(clean[sel])
            vals.setdefault(("ratio", b), []).append(ratio[sel])

    edges = np.zeros((n_bands, K + 1), np.float32)
    cent_a = np.zeros((n_bands, K), np.float32)
    cent_r = np.zeros((n_bands, K), np.float32)   # E[coeff/sigma | bin]
    for b in range(n_bands):
        t = np.concatenate(vals[b])
        v = np.concatenate(vals[("raw", b)])
        r = np.concatenate(vals[("ratio", b)])
        lo, hi = np.percentile(t, lo_pct), np.percentile(t, hi_pct)
        e = np.linspace(lo, hi, K + 1)
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

    from helix.data.identity import corpus_identity

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
    bad = {k: (ref["digests"][k], got[k]) for k in ARRAY_KEYS
           if ref["digests"].get(k) != got[k]}
    return dict(status="mismatch" if bad else "match", reference=ref,
                digest_mismatches=bad, digests=got)


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
