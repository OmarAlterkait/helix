"""Patch sweep + cathode continuity aggregated over many events.

Same definitions as research/patch_sweep.py, but recomputed on N events through the
GPU production pipeline (densify -> GPU coherent+intrinsic noise -> torch DWT ->
coeff-space smart removal kgate=4 -> per-band-sigma hard threshold), D1 dropped.

Reports, per patch size: per-plane token means; 6-plane / per-volume token
distribution (mean, p5, p50, p95, max over events); fan-out (D3+D2) and coarse-load
(A4+D4) per occupied token pooled over ALL tokens of ALL events (mean, p95, max).
Cathode continuity (collection Y, 8x4): direct vs mirrored shared-occupancy fraction,
mean over events, vs chance baseline.

    python research/patch_sweep_multi.py --events 150
"""
from __future__ import annotations
import argparse, sys, time
import numpy as np

for p in ("/sdf/group/neutrino/omara/helix/.pylibs",
          "/sdf/group/neutrino/omara/particle-imaging-models/libs/pimm-data/src",
          "/sdf/group/neutrino/omara/helix"):
    if p not in sys.path:
        sys.path.insert(0, p)
import hdf5plugin  # noqa
import torch
import measure_coeffs as M   # same folder

CMAX = 270
SIZES = [(8, 4), (8, 8), (16, 4), (16, 8)]
PNAME = ["vol0_U", "vol0_V", "vol0_Y", "vol1_U", "vol1_V", "vol1_Y"]


def plane_arrays(bands):
    """Thresholded bands [A4,D4,D3,D2] (D1 dropped) -> (wire, coarse, is_coarse) arrays."""
    wires, coarse, isc = [], [], []
    for bi, b in enumerate(bands[:4]):                 # 0=A4,1=D4,2=D3,3=D2
        nz = torch.nonzero(b, as_tuple=False)
        if nz.numel() == 0:
            continue
        w = nz[:, 0].cpu().numpy().astype(np.int64)
        t = nz[:, 1].cpu().numpy().astype(np.int64)
        c = t if bi <= 1 else (t // 2 if bi == 2 else t // 4)
        wires.append(w); coarse.append(c)
        isc.append(np.full(w.shape[0], bi <= 1))
    if not wires:
        return (np.zeros(0, np.int64),) * 2 + (np.zeros(0, bool),)
    return np.concatenate(wires), np.concatenate(coarse), np.concatenate(isc)


def occ_counts(w, c, isc, Pw, Pc, mirror=False):
    """-> (n_occupied, fine_per_token, coarse_per_token)."""
    if w.size == 0:
        return 0, np.zeros(0), np.zeros(0)
    cc = (CMAX - c) if mirror else c
    cell = (w // Pw) * 100000 + (cc // Pc)
    uniq, inv = np.unique(cell, return_inverse=True)
    coarse_per = np.bincount(inv, weights=isc.astype(np.float64), minlength=len(uniq))
    fine_per = np.bincount(inv, weights=(~isc).astype(np.float64), minlength=len(uniq))
    return len(uniq), fine_per, coarse_per


def cellset(w, c, isc, Pw, Pc, mirror=False):   # isc unused (matches plane_arrays 3-tuple)
    if w.size == 0:
        return set()
    cc = (CMAX - c) if mirror else c
    return set(zip((w // Pw).tolist(), (cc // Pc).tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=150)
    args = ap.parse_args()

    from pimm_data import JAXTPCDataset
    from pimm_data.detector_transforms import Densify, AddNoise, Digitize
    from helix.core import backend
    backend.set_backend("torch")
    ops = backend.ops("helix.core.wavelet_ops")
    geom, nts = M.load_geom()
    dwt = {"ops": ops, "pad": (-nts) % (2 ** M.LEVEL)}
    stages = [Densify(geom), AddNoise(geom=geom, coherent=True, incoherent=True),
              Digitize(geom=geom)]
    ds = JAXTPCDataset(data_root=M.DATA_ROOT, split=M.SPLIT,
                       modalities=("sensor",), dataset_name=M.DATASET_NAME)
    gids = sorted(geom.keys())

    # accumulators
    tok = {s: [[] for _ in range(6)] for s in SIZES}          # per-size per-plane token counts/event
    fine_pool = {s: [] for s in SIZES}
    coarse_pool = {s: [] for s in SIZES}
    cath = {"direct": [], "mirror": [], "chance": []}

    t0 = time.perf_counter()
    for ev in range(args.events):
        cf = M.event_coeffs(ds, ev, geom, dwt, stages, 1.0)
        pa = [plane_arrays(cf[g]) for g in gids]               # per-plane (w,c,isc)
        for s in SIZES:
            Pw, Pc = s
            for pi in range(6):
                n, fine, coarse = occ_counts(*pa[pi], Pw, Pc)
                tok[s][pi].append(n)
                fine_pool[s].append(fine); coarse_pool[s].append(coarse)
        # cathode: Y planes gid2 (idx2) vol0, gid5 (idx5) vol1, patch 8x4
        O0 = cellset(*pa[2], 8, 4)
        O1d = cellset(*pa[5], 8, 4, mirror=False)
        O1m = cellset(*pa[5], 8, 4, mirror=True)
        if O0:
            cath["direct"].append(len(O0 & O1d) / len(O0))
            cath["mirror"].append(len(O0 & O1m) / len(O0))
            ny = geom[gids[5]]["n_wires"]
            ncells = ((ny + 7) // 8) * ((CMAX + 1 + 3) // 4)
            cath["chance"].append(len(O1d) / ncells)
        if ev % 50 == 0:
            print(f"  ev {ev} ({time.perf_counter()-t0:.0f}s)")
    print(f"processed {args.events} events in {time.perf_counter()-t0:.0f}s\n")

    def pct(a, q): return np.percentile(np.asarray(a), q)
    print(f"=== per-plane MEAN occupied tokens over {args.events} events ===")
    print(f"{'patch':>7} " + " ".join(f"{n:>9}" for n in PNAME) + f" {'6sum':>8} {'vol0':>7} {'vol1':>7}")
    for s in SIZES:
        means = [np.mean(tok[s][pi]) for pi in range(6)]
        s6 = sum(means); v0 = sum(means[:3]); v1 = sum(means[3:])
        print(f"{s[0]}x{s[1]:<4} " + " ".join(f"{m:>9.0f}" for m in means)
              + f" {s6:>8.0f} {v0:>7.0f} {v1:>7.0f}")

    print(f"\n=== token-budget spread over events (6-plane sum; per-volume sum) ===")
    print(f"{'patch':>7} {'6sum p5':>8} {'6sum p50':>9} {'6sum p95':>9} {'6sum max':>9} "
          f"{'vol p50':>8} {'vol p95':>8} {'vol max':>8}")
    for s in SIZES:
        per_ev_6 = np.sum([tok[s][pi] for pi in range(6)], axis=0)
        vol = np.concatenate([np.sum([tok[s][pi] for pi in range(3)], axis=0),
                              np.sum([tok[s][pi] for pi in range(3, 6)], axis=0)])
        print(f"{s[0]}x{s[1]:<4} {pct(per_ev_6,5):>8.0f} {pct(per_ev_6,50):>9.0f} "
              f"{pct(per_ev_6,95):>9.0f} {per_ev_6.max():>9.0f} "
              f"{pct(vol,50):>8.0f} {pct(vol,95):>8.0f} {vol.max():>8.0f}")

    print(f"\n=== fan-out (D3+D2) & coarse-load (A4+D4) per occupied token, pooled all events ===")
    print(f"{'patch':>7} {'fan_mean':>8} {'fan_p95':>7} {'fan_max':>7} {'crs_mean':>8} {'crs_p95':>7} {'crs_max':>7}")
    for s in SIZES:
        fine = np.concatenate(fine_pool[s]); coarse = np.concatenate(coarse_pool[s])
        print(f"{s[0]}x{s[1]:<4} {fine.mean():>8.2f} {pct(fine,95):>7.0f} {int(fine.max()):>7} "
              f"{coarse.mean():>8.2f} {pct(coarse,95):>7.0f} {int(coarse.max()):>7}")

    # closest per-volume mean to 7500
    best = None
    for s in SIZES:
        v = np.mean([np.mean(tok[s][pi]) for pi in range(3)]) + 0  # vol0 mean
        for vol_mean in (sum(np.mean(tok[s][pi]) for pi in range(3)),
                         sum(np.mean(tok[s][pi]) for pi in range(3, 6))):
            if best is None or abs(vol_mean - 7500) < abs(best[1] - 7500):
                best = (f"{s[0]}x{s[1]}", vol_mean)
    print(f"\nNOTE: per-volume MEAN closest to ~7500 -> patch {best[0]} ({best[1]:.0f}; 6-plane ~{2*best[1]:.0f})")

    d = np.array(cath["direct"]); m = np.array(cath["mirror"]); ch = np.array(cath["chance"])
    print(f"\n=== CATHODE CONTINUITY (collection Y, 8x4, {len(d)} events) ===")
    print(f"  direct  (same ctime)      mean={d.mean():.4f}  std={d.std():.4f}")
    print(f"  mirrored (c->270-c)       mean={m.mean():.4f}  std={m.std():.4f}")
    print(f"  chance baseline           mean={ch.mean():.4f}")
    print(f"  NOTE: {'MIRRORED' if m.mean()>d.mean() else 'DIRECT'} alignment higher "
          f"({max(d.mean(),m.mean()):.3f} vs {min(d.mean(),m.mean()):.3f}; chance {ch.mean():.3f}).")


if __name__ == "__main__":
    main()
