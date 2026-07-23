#!/usr/bin/env python
"""Within-band REACH audit (Phase 2, data-only) — how far must the within-band
operator look? The M2 audit only used +-1 neighbors; this measures the
incremental information of distances 2 and 4 given closer context, activity
and value level, per band, on the dumped typical events.

Activity (exact, contingency):  I(c; L1R1)
                                I(c; L2R2 | L1R1)
                                I(c; L4R4 | L1R1, L2R2)
Value (Gaussian copula, left-chain co-active):  r(c,L1), r(c,L2|L1),
                                                r(c,L4|L1,L2)

Run:  python reach_audit.py
"""
import sys, os, json

sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
import numpy as np

from info_audit import (load_tpc, load_opt, mi_from_table, normal_scores,
                        TPC_BANDS, OPT_BANDS, ART)


def partial_r(y, x, Z):
    """corr(y, x | columns of Z) on normal scores."""
    ys, xs = normal_scores(y), normal_scores(x)
    Zs = np.stack([normal_scores(z) for z in Z], 1) if Z else None
    if Zs is not None:
        A = np.concatenate([Zs, np.ones((len(ys), 1))], 1)
        ys = ys - A @ np.linalg.lstsq(A, ys, rcond=None)[0]
        xs = xs - A @ np.linalg.lstsq(A, xs, rcond=None)[0]
    d = np.sqrt((ys ** 2).sum() * (xs ** 2).sum())
    return float((ys * xs).sum() / max(d, 1e-12))


def audit(units, band_range, band_names, name, out):
    print(f"\n===== {name}: within-band reach =====")
    print(f"{'band':>5} | {'I(c;L1R1)':>10} {'I(c;2|1)':>9} {'I(c;4|1,2)':>10} | "
          f"{'r(c,L1)':>8} {'r(c,L2|1)':>9} {'r(c,L4|1,2)':>11} {'n_val':>8}")
    rows = []
    for bi in band_range:
        T = np.zeros(128, np.int64)
        vals = []                                  # (c, L1, L2, L4) co-active
        pairs1 = []
        for u in units:
            m, v = u[bi]
            n = m.shape[-1]
            if n < 9:
                continue
            s = slice(4, n - 4)
            c = m[..., s]
            bits = [c]
            for k in (1, 2, 4):
                bits.append(m[..., 4 - k:n - 4 - k])   # L_k
                bits.append(m[..., 4 + k:n - 4 + k])   # R_k
            code = np.zeros(c.shape, np.int64)
            for b in bits:
                code = (code << 1) | b.astype(np.int64)
            T += np.bincount(code.ravel(), minlength=128)
            La = m[..., 3:n - 5] & m[..., 2:n - 6] & m[..., 0:n - 8] & c
            if La.any():
                vals.append(np.stack([np.abs(v[..., s][La]),
                                      np.abs(v[..., 3:n - 5][La]),
                                      np.abs(v[..., 2:n - 6][La]),
                                      np.abs(v[..., 0:n - 8][La])], 1))
            P1 = c & m[..., 3:n - 5]
            if P1.any():
                pairs1.append(np.stack([np.abs(v[..., s][P1]),
                                        np.abs(v[..., 3:n - 5][P1])], 1))
        T7 = T.reshape((2,) * 7)         # axes: c, L1, R1, L2, R2, L4, R4
        i1 = mi_from_table(T7, (0,), (1, 2))
        i2 = mi_from_table(T7, (0,), (3, 4), (1, 2))
        i4 = mi_from_table(T7, (0,), (5, 6), (1, 2, 3, 4))
        r1 = r2 = r4 = float("nan")
        if pairs1:
            P = np.concatenate(pairs1)
            if len(P) > 300000:
                P = P[np.random.default_rng(0).choice(len(P), 300000, replace=False)]
            r1 = partial_r(P[:, 0], P[:, 1], [])
        nv = 0
        if vals:
            V = np.concatenate(vals)
            nv = len(V)
            if len(V) > 300000:
                V = V[np.random.default_rng(0).choice(len(V), 300000, replace=False)]
            if len(V) >= 100:
                r2 = partial_r(V[:, 0], V[:, 2], [V[:, 1]])
                r4 = partial_r(V[:, 0], V[:, 3], [V[:, 1], V[:, 2]])
        print(f"{band_names[bi]:>5} | {i1:>10.5f} {i2:>9.5f} {i4:>10.5f} | "
              f"{r1:>8.3f} {r2:>9.3f} {r4:>11.3f} {nv:>8}")
        rows.append(dict(band=band_names[bi], I_c_1=i1, I_c_2_given_1=i2,
                         I_c_4_given_12=i4, r1=r1, r2_given_1=r2,
                         r4_given_12=r4, n_quad_coactive=int(nv)))
    out[name] = rows


def main():
    out = {}
    p = os.path.join(ART, "typical_event_coeffs_smart.npz")
    if os.path.exists(p):
        audit(load_tpc(p), range(1, 5), TPC_BANDS, "tpc_smart", out)
    for tag, fn in (("optical_lightout", "typical_event_coeffs_optical.npz"),
                    ("optical_doraemon", "typical_event_coeffs_doraemon.npz")):
        p = os.path.join(ART, fn)
        if os.path.exists(p):
            audit(load_opt(p), range(1, 11), OPT_BANDS, tag, out)
    jp = os.path.join(ART, "reach_audit.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\n-> {jp}")


if __name__ == "__main__":
    main()
