"""Wavelet-packet best-basis (Coifman-Wickerhauser) vs fixed DWT.

Question: at the same noise-relative (universal) threshold, does an adaptive
wavelet-packet best basis represent the optical signal with FEWER surviving
coefficients than the standard DWT? Best basis chosen to minimize kept-coeff
count (the metric we care about). CPU/pywt, subset of events.

Usage: python scripts/optical/wpt_compare.py [n_events] [maxlevel]
"""
import sys, math
sys.path.insert(0, "/sdf/group/neutrino/omara/helix")
import numpy as np
import pywt
from helix.optical import config_from_file, list_events
from helix.optical.io import read_event_chunks, pad_batch

PATH = "/sdf/home/y/youngsam/sw/dune/sim/goop/data/light_output.h5"
WL = "coif3"


def kept(data, t):
    return int((np.abs(data) >= t).sum())


def best_basis_kept(node, maxlevel, t):
    """Return (min kept-coeff count over admissible bases below node, n_leaves)."""
    if node.level >= maxlevel:
        return kept(node.data, t), 1
    ka, na = best_basis_kept(node["a"], maxlevel, t)
    kd, nd = best_basis_kept(node["d"], maxlevel, t)
    here = kept(node.data, t)
    if here <= ka + kd:            # cheaper to stop splitting here
        return here, 1
    return ka + kd, na + nd


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    maxlevel = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    cfg0 = config_from_file(PATH)
    events = list_events(PATH)[:n]
    tot_dwt = tot_wpt = tot_samp = 0
    nchunks = 0
    for ev in events:
        ec = read_event_chunks(PATH, ev, cfg0)
        for ch in ec.chunks:
            x = ch.astype(np.float64)
            N = len(x)
            # noise sigma from finest DWT detail
            cd = pywt.wavedec(x, WL, level=1, mode="periodization")[-1]
            sigma = np.median(np.abs(cd)) / 0.6745
            t = sigma * math.sqrt(2 * math.log(N))
            # DWT kept (fixed basis = split only approx)
            co = pywt.wavedec(x, WL, level=maxlevel, mode="periodization")
            kdwt = sum(kept(c, t) for c in co)
            # WPT best-basis kept
            wp = pywt.WaveletPacket(x, WL, mode="periodization", maxlevel=maxlevel)
            kwpt, nleaf = best_basis_kept(wp, maxlevel, t)
            tot_dwt += kdwt; tot_wpt += kwpt; tot_samp += N; nchunks += 1
        print(f"{ev}: dwt_kept={tot_dwt:,} wpt_kept={tot_wpt:,} ratio={tot_wpt/max(tot_dwt,1):.3f}", flush=True)
    print(f"\n=== {nchunks} chunks, {tot_samp:,} samples, maxlevel={maxlevel} ===")
    print(f"DWT  total kept: {tot_dwt:,}   (compression {tot_samp/max(tot_dwt,1):.2f}x by count)")
    print(f"WPT  total kept: {tot_wpt:,}   (compression {tot_samp/max(tot_wpt,1):.2f}x by count)")
    print(f"WPT/DWT kept ratio = {tot_wpt/max(tot_dwt,1):.3f}  "
          f"({'WPT sparser' if tot_wpt<tot_dwt else 'DWT sparser/equal'})")


if __name__ == "__main__":
    main()
