"""Estimate the EZW/SPIHT (zerotree) coding gain over independent indexing.

The classic wavelet codec codes the significance map using parent->child
correlation across scales (a significant coarse coeff predicts significant fine
coeffs at the same time). We quantify the achievable gain as:
   marginal significance-map bits   = Sum_band L_b * H(p_b)
   context (zerotree) bits          = root marginal + Sum_child H(sig_child | sig_parent)
gain = marginal / context.  (Value bits are unchanged; this is the position side.)

Usage: python scripts/optical/zerotree_estimate.py [n_events]
"""
import sys, math
sys.path.insert(0, "/sdf/group/neutrino/omara/helix")
import numpy as np
import pywt
from helix.optical import config_from_file, list_events
from helix.optical.io import read_event_chunks

PATH = "/sdf/home/y/youngsam/sw/dune/sim/goop/data/light_output.h5"
WL, LEVEL = "coif3", 10


def Hb(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    cfg0 = config_from_file(PATH)
    events = list_events(PATH)[:n]
    marg_tot = ctx_tot = 0.0
    for ev in events:
        ec = read_event_chunks(PATH, ev, cfg0)
        for x in ec.chunks:
            x = x.astype(np.float64); N = len(x)
            cd1 = pywt.wavedec(x, WL, level=1, mode="periodization")[-1]
            sig = np.median(np.abs(cd1)) / 0.6745
            t = sig * math.sqrt(2 * math.log(N))
            co = pywt.wavedec(x, WL, level=LEVEL, mode="periodization")  # [cA, cD_L..cD_1]
            S = [(np.abs(c) >= t).astype(np.int8) for c in co]           # significance maps
            # marginal map cost
            marg = sum(len(s) * Hb(s.mean()) for s in S if len(s) > 1)
            # context cost: detail bands fine->coarse, parent at next-coarser band, child idx//2
            ctx = len(S[1]) * Hb(S[1].mean())   # coarsest detail band: marginal (root)
            for j in range(2, len(S)):           # finer detail bands have a parent one level up
                child = S[j]; parent = S[j - 1]
                par_of_child = parent[np.minimum(np.arange(len(child)) // 2, len(parent) - 1)]
                for pv in (0, 1):
                    m = par_of_child == pv
                    if m.sum() > 0:
                        ctx += m.sum() * Hb(child[m].mean())
            marg_tot += marg; ctx_tot += ctx
        print(f"{ev}: marginal_pos={marg_tot:,.0f}b  context_pos={ctx_tot:,.0f}b  "
              f"gain={marg_tot/max(ctx_tot,1):.3f}x", flush=True)
    print(f"\n=== zerotree/context significance-map coding gain ===")
    print(f"marginal (independent): {marg_tot:,.0f} bits")
    print(f"context  (zerotree):    {ctx_tot:,.0f} bits")
    print(f"position-coding gain = {marg_tot/max(ctx_tot,1):.2f}x  "
          f"(applies to the index/position part of the rate only)")


if __name__ == "__main__":
    main()
