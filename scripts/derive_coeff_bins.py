#!/usr/bin/env python
"""Derive categorical-head bin edges from a coeff corpus.

The FM's value head can predict a coefficient's magnitude as one of K bins
instead of regressing it. The bins are TRAINING-SET STATISTICS, so they must come
from the corpus a model will be trained on — the edges shipped with m113 were
derived from the old cache, which used a different (white) noise model.

Algorithm, matching research ``fm/tier1_setup_bins.py``:

  * work in ``tgt = arcsinh(clean / sigma)`` — the space the head predicts in
  * K bins UNIFORM in that space over the band's robust range
    ``[p0.05, p99.95]``. Uniform-in-asinh is log-spaced in raw charge, i.e.
    constant RELATIVE precision, which is the physically motivated choice —
    NOT quantile bins, so non-uniform occupancy is expected and correct
  * outer two bins extended to +-inf so the tails cannot fall off the grid

The one deliberate difference from the research script: it read
``val_clean`` from the npz cache and divided by a global ``SIGMA=2.6``, because
that cache stored values already scaled by ``SIGMA/sigma_tab``. The corpus stores
RAW coefficients, so we divide by the per-``(plane, band)`` ``norm_sigma`` table
directly. The two are the same quantity — SIGMA cancels — and doing it this way
means the edges are in exactly the space ``CoeffTokenize`` produces.

Writes ``edges`` (n_band, K+1), ``cent_asinh`` (n_band, K) — the point estimate
in model space — and ``cent_ratio`` (n_band, K) = E[coeff/sigma | bin], the
charge read-back centroid.

It no longer writes ``cent_lin`` (E[raw ADC | bin]). That table pooled planes
whose ``norm_sigma`` differ by 22%, nothing consumed it, and ``set_bins`` now
rejects unknown centroid tables rather than let one bind to the wrong slot. An
older sidecar that still carries the key applies fine — ``apply_bins`` ignores it.

Usage::

    python scripts/derive_coeff_bins.py --corpus <dir> --out bins.pt [--events 120]
"""

from __future__ import annotations

import argparse

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True, help="corpus dir (<root>/<run>/)")
    ap.add_argument("--dataset-name", default="sim_wire")
    ap.add_argument("--out", required=True, help="output .pt")
    ap.add_argument("--events", type=int, default=120,
                    help="events to pool (research used 120)")
    ap.add_argument("--K", type=int, default=128, help="bins per band")
    ap.add_argument("--n-bands", type=int, default=4,
                    help="bands the tokenizer keeps (D1 and beyond are dropped)")
    ap.add_argument("--lo-pct", type=float, default=0.05)
    ap.add_argument("--hi-pct", type=float, default=99.95)
    a = ap.parse_args(argv)

    import torch
    from pimm_data import CoeffTPCDataset
    from helix.model.tokenize import sigma_for_rows

    ds = CoeffTPCDataset(data_root=a.corpus, dataset_name=a.dataset_name,
                         modalities=("coeff", "coeff_clean"), transform=None)
    n = min(a.events, len(ds))
    print(f"pooling {n} of {len(ds)} events from {a.corpus}")

    vals = {b: [] for b in range(a.n_bands)}
    for i in range(n):
        s = ds.get_data(i)
        c, cc = s["coeff"], s["coeff_clean"]
        meta = c["_meta"]
        band = np.asarray(c["band"], np.int64)
        gid = np.asarray(c["plane_gid"], np.int64)
        clean = np.asarray(cc["value"], np.float32).reshape(-1)
        keep = band < a.n_bands
        band, gid, clean = band[keep], gid[keep], clean[keep]
        sig = np.maximum(sigma_for_rows(gid, band, meta["gids"], meta["norm_sigma"]), 1e-6)
        t = np.arcsinh(clean / sig).astype(np.float64)
        # clean/sigma == sinh(t), the DIMENSIONLESS coefficient. This is what a
        # read-back needs. The alternative — E[raw ADC | bin] — pools planes whose
        # norm_sigma differ by 22%, biasing a Y-plane coefficient 13% low and a
        # U/V one 6% high. That cancels on a random mask and does NOT cancel on a
        # plane mask (measured 0.857 vs 1.021 cross-plane): invisible on the
        # metric people look at, wrong on the one that matters. It was written
        # here for four commits and read nowhere; it is not written now.
        ratio = (clean / sig).astype(np.float64)
        for b in range(a.n_bands):
            sel = band == b
            vals[b].append(t[sel])
            vals.setdefault(("raw", b), []).append(clean[sel])
            vals.setdefault(("ratio", b), []).append(ratio[sel])

    K = a.K
    edges = np.zeros((a.n_bands, K + 1), np.float32)
    cent_a = np.zeros((a.n_bands, K), np.float32)
    cent_r = np.zeros((a.n_bands, K), np.float32)   # E[coeff/sigma | bin]
    for b in range(a.n_bands):
        t = np.concatenate(vals[b])
        v = np.concatenate(vals[("raw", b)])
        r = np.concatenate(vals[("ratio", b)])
        lo, hi = np.percentile(t, a.lo_pct), np.percentile(t, a.hi_pct)
        e = np.linspace(lo, hi, K + 1)
        idx = np.clip(np.digitize(t, e[1:-1]), 0, K - 1)
        for k in range(K):
            m = idx == k
            if m.any():
                cent_a[b, k] = t[m].mean()
                # E[sinh t | bin], MEASURED. Not sinh(E[t | bin]) — the head is
                # categorical, and a posterior spread over many bins makes
                # sinh(mean) a 31%-low estimate of the charge (Jensen). The
                # reference (fm/train.py:125, fm/e7_cat_eval.py:47,
                # fm/viz_cat.py:25) reads back as sum_k p_k * centroid_k and
                # never applies sinh to a mean.
                cent_r[b, k] = r[m].mean()
            else:                                  # empty bin: fall back to its centre
                cent_a[b, k] = 0.5 * (e[k] + e[k + 1])
                cent_r[b, k] = float(np.sinh(cent_a[b, k]))   # tier1_setup_bins.py:41
        empty = int((np.bincount(idx, minlength=K) == 0).sum())
        e[0], e[-1] = -1e18, 1e18                  # tails cannot fall off the grid
        edges[b] = e
        print(f"  band {b}: n={len(t):>10,}  tgt[{lo:+.2f},{hi:+.2f}]  "
              f"empty bins={empty}  |coeff|max={np.abs(v).max():.0f}")

    torch.save(dict(edges=torch.tensor(edges), cent_asinh=torch.tensor(cent_a),
                    cent_ratio=torch.tensor(cent_r), K=K, n_bands=a.n_bands,
                    corpus=a.corpus, events=n), a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
