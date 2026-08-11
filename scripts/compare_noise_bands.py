#!/usr/bin/env python
"""Measure what the incoherent noise model does to each wavelet band.

The corpus is built with the MEASURED (MicroBooNE) incoherent spectrum; the old
research cache defaulted to WHITE. Every downstream difference — the per-band
normalisation table, which coefficients survive thresholding, the categorical bin
grid, and m113 being out-of-distribution — is claimed to follow from that one
change. This measures the claim instead of inferring it from the spectrum shape.

Runs the SAME events through the real builder twice, once per noise model, and
reports per band:

  sigma        the normalisation table (MAD of the noisy coefficients). This is
               the quantity everything else is divided by.
  n_coeff      how many coefficients survive the gate + threshold. Noise sets the
               threshold, so the surviving SUPPORT changes too — not just the
               scale.
  tgt range    the robust span of arcsinh(clean/sigma), which is exactly what
               sets the categorical bin grid.

It also prints the prediction from the spectrum alone: relative noise POWER per
band, colored vs white, normalised to equal total power. If the mechanism is what
we think, sigma_colored/sigma_white should track sqrt(that).

Band <-> frequency at 2 MHz sampling (Nyquist 1 MHz), levels (4,4,3,2):
  band 0 = A4   0-62.5 kHz     band 2 = D3  125-250 kHz
  band 1 = D4   62.5-125 kHz   band 3 = D2  250-500 kHz   (D1 500k-1M is dropped)

Usage (needs a GPU for the torch backend)::

    python scripts/compare_noise_bands.py --shard <sensor.h5> --events 6
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDER = os.path.join(HERE, "scripts", "build_coeff_corpus.py")
SPECTRUM = "/sdf/group/neutrino/omara/JAXTPC/config/noise_spectrum.npz"
NYQ = 1.0e6
BANDS = [("0 A4", 0, NYQ / 16), ("1 D4", NYQ / 16, NYQ / 8),
         ("2 D3", NYQ / 8, NYQ / 4), ("3 D2", NYQ / 4, NYQ / 2),
         ("(D1)", NYQ / 2, NYQ)]


def spectrum_prediction():
    """Relative noise power per band, colored vs white, at equal total power."""
    d = np.load(SPECTRUM)
    f, a = d["spectrum_freqs_hz"], d["spectrum_shape"].astype(float)
    p = a ** 2
    p = p / p.sum()
    out = []
    for name, lo, hi in BANDS:
        white = (hi - lo) / NYQ                      # power is proportional to bandwidth
        col = p[(f >= lo) & (f < hi)].sum()
        out.append((name, white, col, np.sqrt(col / white) if white else np.nan))
    return out


def run_build(shard, out_dir, events, white, calibrate_to=None):
    cmd = [sys.executable, BUILDER, "--shard", shard, "--out", out_dir,
           "--dataset-name", "sim_wire", "--run", "run_0027575715",
           "--file-index", "0", "--event-start", "0", "--events", str(events),
           "--mode", "serial", "--backend", "torch"]
    if white:
        cmd.append("--white")
    if calibrate_to:
        cmd += ["--calibrate", "--save-norm-sigma", calibrate_to]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit(f"builder failed ({'white' if white else 'colored'})")
    return r.stdout


def shard_stats(path, n_band=4):
    """Per-band coefficient count and robust tgt range from a built shard."""
    import h5py
    from helix.model.tokenize import sigma_for_rows
    with h5py.File(path, "r") as f:
        cfg = f["config"]
        gids, ns = cfg["gids"][:], cfg["norm_sigma"][:]
        band = f["coord"]["band"][:].astype(np.int64)
        gid = f["coord"]["plane_gid"][:].astype(np.int64)
        val = f["value"][:]
    sig = np.maximum(sigma_for_rows(gid, band, gids, ns), 1e-6)
    t = np.arcsinh(val / sig)
    return {b: (int((band == b).sum()),
                float(np.percentile(t[band == b], 0.05)),
                float(np.percentile(t[band == b], 99.95)))
            for b in range(n_band)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", required=True)
    ap.add_argument("--events", type=int, default=6)
    ap.add_argument("--work", default="/lscratch/omara/bandcmp")
    a = ap.parse_args(argv)
    os.makedirs(a.work, exist_ok=True)
    sys.path.insert(0, HERE)

    print("=" * 78)
    print("PREDICTION from the spectrum alone (equal total power)")
    print("=" * 78)
    print("%-6s %12s %12s %14s" % ("band", "white frac", "colored frac", "sqrt(col/white)"))
    pred = {}
    for name, w, c, r in spectrum_prediction():
        print("%-6s %12.3f %12.3f %14s" % (name, w, c, "%.2f" % r if np.isfinite(r) else "-"))
        pred[name[0]] = r

    print()
    print("=" * 78)
    print("MEASURED: the same %d events built both ways" % a.events)
    print("=" * 78)
    sig = {}
    for mode in ("colored", "white"):
        p = os.path.join(a.work, "sigma_%s.npy" % mode)
        run_build(a.shard, a.work, a.events, white=(mode == "white"), calibrate_to=p)
        sig[mode] = np.load(p)                      # (n_gid, n_band)
    print("norm_sigma, averaged over planes (MAD of the NOISY coefficients):")
    print("%-6s %10s %10s %10s %10s" % ("band", "colored", "white", "col/white", "predicted"))
    for b in range(4):
        c, w = float(sig["colored"][:, b].mean()), float(sig["white"][:, b].mean())
        print("%-6s %10.4f %10.4f %10.2f %10.2f"
              % (BANDS[b][0], c, w, c / w, pred[str(b)]))

    print()
    print("support + tgt range from actual shards:")
    stats = {}
    for mode in ("colored", "white"):
        d = os.path.join(a.work, mode)
        os.makedirs(d, exist_ok=True)
        run_build(a.shard, d, min(a.events, 3), white=(mode == "white"))
        stats[mode] = shard_stats(os.path.join(d, "sim_wire_coeff_0000.h5"))
    print("%-6s %12s %12s %9s   %18s %18s"
          % ("band", "n colored", "n white", "ratio", "tgt range colored", "tgt range white"))
    for b in range(4):
        nc, cl, ch = stats["colored"][b]
        nw, wl, wh = stats["white"][b]
        print("%-6s %12,d %12,d %9.3f   [%+6.2f,%+6.2f]  [%+6.2f,%+6.2f]"
              .replace(",d", "d") % (BANDS[b][0], nc, nw, nc / max(nw, 1), cl, ch, wl, wh))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
