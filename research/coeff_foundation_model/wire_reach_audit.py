#!/usr/bin/env python
"""TPC WIRE-direction reach audit (data-only) — the axis the original reach
audit never measured. Per band, on the dumped typical event:

  activity: I(c; W1)            wire-neighbors +-1 (same band, same time)
            I(c; W1 | T1)       ... beyond time-neighbor context
            I(c; W2 | W1)       incremental at wire distance 2
            I(c; W4 | W1, W2)   ... distance 4
  value:    r(c, W1), r(c, W2 | W1)   Gaussian-copula partials, co-active

Run:  python wire_reach_audit.py
"""
import sys, os, json

sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
import numpy as np

from info_audit import load_tpc, mi_from_table, TPC_BANDS, ART
from reach_audit import partial_r


def main():
    units = load_tpc(os.path.join(ART, "typical_event_coeffs_smart.npz"))
    out = {}
    print(f"{'band':>5} | {'I(c;W1)':>8} {'I(c;W1|T1)':>10} {'I(c;W2|W1)':>10} "
          f"{'I(c;W4|W12)':>11} | {'r(c,W1)':>8} {'r(c,W2|W1)':>10} {'n':>8}")
    for bi in range(1, 5):
        Ta = np.zeros(32, np.int64)    # c, W1u, W1d, T1l, T1r
        Tb = np.zeros(32, np.int64)    # c, W1u, W1d, W2u, W2d
        Tc = np.zeros(128, np.int64)   # c, W1u,W1d, W2u,W2d, W4u,W4d
        vp, vt = [], []
        for u in units:
            m, v = u[bi]
            nw, n = m.shape
            if nw < 9:
                continue
            s = slice(4, nw - 4)
            c = m[s, 1:-1]
            W = {k: (m[4 - k:nw - 4 - k, 1:-1], m[4 + k:nw - 4 + k, 1:-1])
                 for k in (1, 2, 4)}
            T1 = (m[s, 0:-2], m[s, 2:])
            def code(bits):
                z = np.zeros(c.shape, np.int64)
                for b in bits:
                    z = (z << 1) | b.astype(np.int64)
                return z
            Ta += np.bincount(code([c, *W[1], *T1]).ravel(), minlength=32)
            Tb += np.bincount(code([c, *W[1], *W[2]]).ravel(), minlength=32)
            Tc += np.bincount(code([c, *W[1], *W[2], *W[4]]).ravel(), minlength=128)
            # values along wire (upper neighbor chain)
            tri = c & W[1][0] & W[2][0]
            if tri.any():
                vv = v[s, 1:-1]
                vp.append(np.stack([np.abs(vv[tri]),
                                    np.abs(v[4 - 1:nw - 4 - 1, 1:-1][tri]),
                                    np.abs(v[4 - 2:nw - 4 - 2, 1:-1][tri])], 1))
        T5a = Ta.reshape((2,) * 5)
        T5b = Tb.reshape((2,) * 5)
        T7 = Tc.reshape((2,) * 7)
        i_w1 = mi_from_table(T5a, (0,), (1, 2))
        i_w1_t1 = mi_from_table(T5a, (0,), (1, 2), (3, 4))
        i_w2_w1 = mi_from_table(T5b, (0,), (3, 4), (1, 2))
        i_w4 = mi_from_table(T7, (0,), (5, 6), (1, 2, 3, 4))
        r1 = r2 = float("nan")
        nv = 0
        if vp:
            V = np.concatenate(vp)
            nv = len(V)
            if len(V) > 300000:
                V = V[np.random.default_rng(0).choice(len(V), 300000, replace=False)]
            if len(V) >= 100:
                r1 = partial_r(V[:, 0], V[:, 1], [])
                r2 = partial_r(V[:, 0], V[:, 2], [V[:, 1]])
        print(f"{TPC_BANDS[bi]:>5} | {i_w1:>8.5f} {i_w1_t1:>10.5f} {i_w2_w1:>10.5f} "
              f"{i_w4:>11.5f} | {r1:>8.3f} {r2:>10.3f} {nv:>8}")
        out[TPC_BANDS[bi]] = dict(I_W1=i_w1, I_W1_given_T1=i_w1_t1,
                                  I_W2_given_W1=i_w2_w1, I_W4_given_W12=i_w4,
                                  r_W1=r1, r_W2_given_W1=r2, n=int(nv))
    with open(os.path.join(ART, "wire_reach_audit.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("-> artifacts/wire_reach_audit.json")


if __name__ == "__main__":
    main()
