"""Token-budget patch sweep + cathode-continuity, on the smart-removed typical event.

Pure recomputation on research/artifacts/typical_event_coeffs_smart.npz (ev33).
D1 (band_id 4) dropped. Coarse grid = n_wires x 271 (A4/D4 resolution).

Exact band->coarse-time mapping (inverse of the footprint rule): a patch covering
coarse [c0:c0+Pc] covers A4/D4 [c0:c0+Pc], D3 [2c0:2c0+2Pc], D2 [4c0:4c0+4Pc]; so a
coeff at band time index t sits in coarse index t (A4/D4), t//2 (D3), t//4 (D2).
patch = (wire//P_wire, coarse//P_ctime); partial edge patches kept by construction.
"""
import numpy as np

NPZ = "/sdf/group/neutrino/omara/helix/research/coeff_foundation_model/artifacts/typical_event_coeffs_smart.npz"
NWIRES = {0: 1969, 1: 1969, 2: 1443, 3: 1969, 4: 1969, 5: 1443}
PNAME = {0: "vol0_U", 1: "vol0_V", 2: "vol0_Y", 3: "vol1_U", 4: "vol1_V", 5: "vol1_Y"}
CMAX = 270                                   # coarse index range 0..270 (271 steps)
SIZES = [(8, 4), (8, 8), (16, 4), (16, 8)]


def coarse_index(band, t):
    """band time index -> coarse (/16) index. A4/D4=t, D3=t//2, D2=t//4."""
    c = np.where(band <= 1, t, np.where(band == 2, t // 2, t // 4))
    return c.astype(np.int64)


def plane_cells(d, gid, Pw, Pc, mirror=False):
    """Return (wp, tp, is_coarse) per active coeff of plane gid (D1 already dropped)."""
    m = d["plane_gid"] == gid
    band, wire, t = d["band_id"][m], d["wire"][m].astype(np.int64), d["time_index"][m]
    c = coarse_index(band, t)
    if mirror:
        c = CMAX - c
    return wire // Pw, c // Pc, (band <= 1)


def occupancy_and_counts(d, gid, Pw, Pc):
    """Per-plane: n_occupied, fine(D3+D2)/coarse(A4+D4) counts per occupied token."""
    wp, tp, is_coarse = plane_cells(d, gid, Pw, Pc)
    cell = wp * 100000 + tp
    uniq, inv = np.unique(cell, return_inverse=True)
    coarse_per = np.bincount(inv, weights=is_coarse.astype(np.float64), minlength=len(uniq))
    fine_per = np.bincount(inv, weights=(~is_coarse).astype(np.float64), minlength=len(uniq))
    return len(uniq), fine_per, coarse_per


def main():
    d = dict(np.load(NPZ, allow_pickle=True))
    keep = d["band_id"] != 4                                   # drop D1
    for k in ("plane_gid", "band_id", "wire", "time_index", "value"):
        d[k] = d[k][keep]
    print(f"event {int(d['event'][0])}  active coeffs (no D1): {len(d['band_id']):,}\n")

    hdr = (f"{'patch':>7} " + " ".join(f"{PNAME[g]:>9}" for g in range(6))
           + f" {'6sum':>8} {'vol0':>7} {'vol1':>7}"
           + f" {'fan_mean':>8} {'fan_p95':>7} {'fan_max':>7}"
           + f" {'crs_mean':>8} {'crs_p95':>7} {'crs_max':>7}")
    print(hdr)
    for Pw, Pc in SIZES:
        per_plane, fine_pool, coarse_pool = [], [], []
        for g in range(6):
            n, fine, coarse = occupancy_and_counts(d, g, Pw, Pc)
            per_plane.append(n); fine_pool.append(fine); coarse_pool.append(coarse)
        fine = np.concatenate(fine_pool); coarse = np.concatenate(coarse_pool)
        s6 = sum(per_plane); v0 = sum(per_plane[:3]); v1 = sum(per_plane[3:])
        row = (f"{Pw}x{Pc:<4} " + " ".join(f"{n:>9,}" for n in per_plane)
               + f" {s6:>8,} {v0:>7,} {v1:>7,}"
               + f" {fine.mean():>8.2f} {np.percentile(fine,95):>7.0f} {int(fine.max()):>7}"
               + f" {coarse.mean():>8.2f} {np.percentile(coarse,95):>7.0f} {int(coarse.max()):>7}")
        print(row)

    # closest per-volume sum to ~7500
    print()
    best = None
    for Pw, Pc in SIZES:
        pp = [occupancy_and_counts(d, g, Pw, Pc)[0] for g in range(6)]
        for vol, sub in (("vol0", pp[:3]), ("vol1", pp[3:])):
            s = sum(sub)
            if best is None or abs(s - 7500) < abs(best[2] - 7500):
                best = (f"{Pw}x{Pc}", vol, s)
    print(f"NOTE: per-volume sum closest to ~7500 -> patch {best[0]} ({best[1]} = {best[2]:,}; "
          f"6-plane ~= {2*best[2]:,})")

    # ── cathode continuity: collection Y planes (gid 2 vol0, gid 5 vol1), patch (8,4) ──
    Pw, Pc = 8, 4
    def cellset(gid, mirror):
        wp, tp, _ = plane_cells(d, gid, Pw, Pc, mirror=mirror)
        return set(zip(wp.tolist(), tp.tolist()))
    O0 = cellset(2, False)
    O1_direct = cellset(5, False)
    O1_mirror = cellset(5, True)
    fdir = len(O0 & O1_direct) / max(len(O0), 1)
    fmir = len(O0 & O1_mirror) / max(len(O0), 1)
    print("\nCATHODE CONTINUITY (collection Y, patch 8x4):")
    print(f"  vol0_Y occupied tokens = {len(O0):,}")
    print(f"  shared-occupancy fraction  direct  (same ctime)        = {fdir:.4f}")
    print(f"  shared-occupancy fraction  mirrored (vol1 c -> 270-c)   = {fmir:.4f}")
    print(f"  NOTE: {'MIRRORED' if fmir > fdir else 'DIRECT'} alignment shows more shared "
          f"occupancy ({max(fdir,fmir):.3f} vs {min(fdir,fmir):.3f}).")


if __name__ == "__main__":
    main()
