"""Stage 1 of the group-level probe suite: LABEL DUMP + VERIFICATION.

For each probe event (cache ev_NNNNN, held-out), build per-pixel group-truth labels:
  D  log charge-weighted 3D RMS dispersion of contributing groups (+ D>5mm flag)
  F  top-1 group charge fraction
  B1 leading-group 3D centroid (x,y,z mm) + charge-weighted mean deposit (theta,phi)
  X  group tables for cross-plane same-gid ranking (center time, member span)
plus the pixel->token map (4 bands, group-delay + per-plane offset corrected) and
presence bits against the cache token set.

BUG-HUNTING built in (the point of this stage):
  - alignment discrimination: token-coverage of hit pixels for the MATCHED cache event
    vs a MISMATCHED one (matched ~0.99 charge-weighted, mismatched ~0.1-0.3)
  - distribution checks vs the red-team's measured numbers (D pctl, F pctl, coverage,
    candidate counts, censoring floor, group footprint)
"""
import sys, os, json, argparse
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, "/sdf/group/neutrino/omara/particle-imaging-models/libs/pimm-data/src")
import hdf5plugin  # noqa
import h5py
import numpy as np

CACHE = "/sdf/group/neutrino/omara/helix/research/coeff_foundation_model/artifacts/fm_cache_tpc"
DROOT = "/sdf/data/neutrino/doraemon/wire_test_00_00_02"
OUT = "/sdf/group/neutrino/omara/helix/research/coeff_foundation_model/artifacts/probe_labels"
# cache filename -> (run, out_start); from cache_ext logs (red-team verified ev_00000<->run_..766/sh0/ev0)
RUNS = [(0, "run_0027575766"), (20000, "run_0027575767"), (40000, "run_0027575768"),
        (60000, "run_0027575769"), (80000, "run_0027587649"), (100000, "run_0027587651"),
        (120000, "run_0027587652"), (140000, "run_0027587653"), (160000, "run_0027587654"),
        (180000, "run_0027651460")]
LEV_T = np.array([4, 4, 3, 2]); DEC = 1 << LEV_T                 # band decimation 16,16,8,4
LENS_T = np.array([271, 271, 542, 1084])
DELTA_T = np.array([-2.38, .62, .75, .50])
PLANES = ["U", "V", "Y"]
TOFF = {"U": -17.4, "V": 2.6, "Y": 5.5}                          # per-plane (sensor - hits) ticks
QTOT_MIN = float(os.environ.get("QTOT_MIN", 250.0))                                                 # censoring confidence cut


def run_of(ev):
    for start, run in reversed(RUNS):
        if ev >= start:
            return run, ev - start
    raise ValueError(ev)


class ShardIndex:
    """NOMINAL event locator (shard = idx//200, key = event_{idx%200:03d}).
    Valid for runs with no missing events (run_0027575767: avail=20000). A full
    key scan of ~100 Lustre shards stalls for tens of minutes (verified) — so we
    locate nominally and rely on the per-event COVERAGE check to catch any
    misalignment (mismatch reads ~0.1-0.3 vs matched ~0.99)."""

    def __init__(self, modality, run, prefix):
        d = os.path.join(DROOT, modality, run)
        self.pat = os.path.join(d, f"sim_wire_{prefix}_%04d.h5")

    def locate(self, idx):
        fp = self.pat % (idx // 200)
        key = f"event_{idx % 200:03d}"
        with h5py.File(fp, "r") as f:                            # cheap existence check
            if key not in f:
                raise KeyError(f"{key} not in {fp} (run has gaps -> nominal indexing invalid)")
        return fp, key


def decode_hits(fp, evkey):
    """Per (volume, plane): flat per-sample arrays wire, time, gid, charge (CSR decode,
    mirrors pimm jaxtpc_hits); plus bridges deposit_to_group / group list per volume."""
    out = {}; bridges = {}
    with h5py.File(fp, "r") as f:
        g = f[evkey]
        for vk in sorted(g.keys()):
            if not vk.startswith("volume_"):
                continue
            v = int(vk.split("_")[1]); vg = g[vk]
            if "deposit_to_group" in vg:
                bridges[v] = dict(deposit_to_group=vg["deposit_to_group"][:].astype(np.int64))
            for pl in PLANES:
                if pl not in vg:
                    continue
                p = vg[pl]
                cw = p["center_wires"][:].astype(np.int64); ct = p["center_times"][:].astype(np.int64)
                dw = p["delta_wires"][:].astype(np.int64); dt = p["delta_times"][:].astype(np.int64)
                qs = p["charges_u16"][:].astype(np.float64); gid = p["group_ids"][:].astype(np.int64)
                sz = p["group_sizes"][:].astype(np.int64); pk = p["peak_charges"][:].astype(np.float64)
                off = np.concatenate([[0], np.cumsum(sz)])
                rep = np.repeat(np.arange(len(sz)), sz)          # member -> group row
                wire = cw[rep] + dw; time = ct[rep] + dt
                q = qs / 65535.0 * pk[rep]                       # dequantize vs group peak
                out[(v, pl)] = dict(wire=wire, time=time, gid=gid[rep], q=q,
                                    grp_gid=gid, grp_ct=ct, grp_sz=sz)
    return out, bridges


def load_step(fp, evkey):
    """Per volume: deposit positions (mm, dequantized), charge, theta, phi."""
    out = {}
    with h5py.File(fp, "r") as f:
        vr = f["config"]["volume_ranges"][:]                     # (2,3,2) mm
        g = f[evkey]
        for vk in sorted(g.keys()):
            if not vk.startswith("volume_"):
                continue
            v = int(vk.split("_")[1]); vg = g[vk]
            u = vg["positions"][:].astype(np.float64)
            lo, hi = vr[v, :, 0], vr[v, :, 1]
            pos = lo + u / 65535.0 * (hi - lo)
            out[v] = dict(pos=pos, q=vg["charge"][:].astype(np.float64),
                          theta=vg["theta"][:].astype(np.float64), phi=vg["phi"][:].astype(np.float64))
    return out


def group_centroids(step_v, d2g):
    """gid -> charge-weighted centroid (3,), mean theta/phi; d2g is deposit_to_group (contiguous runs)."""
    n = d2g.shape[0]
    assert n == step_v["pos"].shape[0], f"deposit count mismatch {n} vs {step_v['pos'].shape[0]}"
    order = np.argsort(d2g, kind="stable")
    dg = d2g[order]; pos = step_v["pos"][order]; q = step_v["q"][order]
    th = step_v["theta"][order]; ph = step_v["phi"][order]
    uniq, starts = np.unique(dg, return_index=True)
    ends = np.concatenate([starts[1:], [n]])
    cen = {}; ang = {}
    for gi, s, e in zip(uniq, starts, ends):
        w = q[s:e]; W = max(w.sum(), 1e-12)
        cen[int(gi)] = (pos[s:e] * w[:, None]).sum(0) / W
        ang[int(gi)] = ((th[s:e] * w).sum() / W, (ph[s:e] * w).sum() / W)
    return cen, ang


def token_sets(cache_npz):
    """Cache token key set {(gid, band, wb, tb)} from the sparse coeff rows."""
    d = np.load(cache_npz)
    band = d["band"].astype(np.int64); gid = d["gid"].astype(np.int64)
    wire = d["wire"].astype(np.int64); idx = d["idx"].astype(np.int64)
    tau = idx % LENS_T[band]
    key = (gid << 40) | (band << 36) | ((wire // 16) << 18) | (tau // 8)
    return set(np.unique(key).tolist())


def pix_token_keys(gid_plane, wire, t, toff):
    """For raw pixels: the 4 per-band token keys (group-delay + plane-offset corrected)."""
    t_sig = t + toff
    keys = []
    for b in range(4):
        tau = np.round(t_sig / DEC[b] - DELTA_T[b]).astype(np.int64)
        tau = np.clip(tau, 0, LENS_T[b] - 1)
        keys.append((np.int64(gid_plane) << 40) | (np.int64(b) << 36) | ((wire // 16) << 18) | (tau // 8))
    return keys                                                   # list of 4 arrays


def coverage(tok, keys, q):
    """Per-band (plain, charge-weighted) coverage of hit pixels by the cache token set."""
    cov = []
    for b in range(4):
        inset = np.fromiter((k in tok for k in keys[b].tolist()), bool, len(keys[b]))
        cov.append((inset.mean(), (q * inset).sum() / max(q.sum(), 1e-9)))
    return cov


def process_event(ev, idx_h, idx_s, tok_mismatch=None):
    run, src = run_of(ev)
    fp_h, k_h = idx_h.locate(src); fp_s, k_s = idx_s.locate(src)
    hits, bridges = decode_hits(fp_h, k_h)
    step = load_step(fp_s, k_s)
    tok = token_sets(os.path.join(CACHE, f"ev_{ev:05d}.npz"))
    rep = dict(ev=ev, run=run, src=src)

    pix = dict(gidp=[], wire=[], t=[], qtot=[], F=[], D=[], b1=[], theta=[], phi=[], pres=[], dom=[], lgid=[])
    grp_tab = dict(vol=[], plane=[], gid=[], tmean=[])
    covm = np.zeros((4, 2)); covx = np.zeros((4, 2)); nv = 0
    for (v, pl), h in hits.items():
        gid_plane = v * 3 + PLANES.index(pl)
        cen, ang = group_centroids(step[v], bridges[v]["deposit_to_group"])
        # per-pixel aggregation
        pk = h["wire"] * 100000 + h["time"]
        order = np.argsort(pk, kind="stable")
        pks = pk[order]; gs = h["gid"][order]; qs = h["q"][order]
        w_s = h["wire"][order]; t_s = h["time"][order]
        uniq, starts = np.unique(pks, return_index=True)
        ends = np.concatenate([starts[1:], [len(pks)]])
        # alignment coverage on this plane's pixels (dedup by pixel)
        keys = pix_token_keys(gid_plane, w_s[starts], t_s[starts], TOFF[pl])
        qpix_arr = np.add.reduceat(qs, starts)
        c = coverage(tok, keys, qpix_arr); covm += np.array(c); nv += 1
        if tok_mismatch is not None:
            covx += np.array(coverage(tok_mismatch, keys, qpix_arr))
        pres_bits = np.stack([np.fromiter((k in tok for k in keys[b].tolist()), bool, len(uniq))
                              for b in range(4)], 1)
        for j, (s, e) in enumerate(zip(starts, ends)):
            q = qs[s:e]; g = gs[s:e]; Q = q.sum()
            if Q < QTOT_MIN:
                continue
            # merge same-gid entries at one pixel
            gg, ginv = np.unique(g, return_inverse=True)
            qg = np.bincount(ginv, weights=q)
            fr = qg / Q
            lead = int(gg[np.argmax(qg)])
            C = np.stack([cen[int(x)] for x in gg])
            mu = (C * fr[:, None]).sum(0)
            D = float(np.sqrt((fr * ((C - mu) ** 2).sum(1)).sum()))
            pix["gidp"].append(gid_plane); pix["wire"].append(int(w_s[s])); pix["t"].append(int(t_s[s]))
            pix["qtot"].append(float(Q)); pix["F"].append(float(fr.max())); pix["D"].append(D)
            pix["b1"].append(cen[lead]); pix["theta"].append(ang[lead][0]); pix["phi"].append(ang[lead][1])
            pix["pres"].append(pres_bits[j]); pix["dom"].append(bool(fr.max() >= 0.5)); pix["lgid"].append(lead)
        # group table for X (charge-weighted mean member time per group)
        gsum = {}
        for gi, ti, qi in zip(h["gid"], h["time"], h["q"]):
            a = gsum.setdefault(int(gi), [0.0, 0.0]); a[0] += qi * ti; a[1] += qi
        for gi, (tw, qw) in gsum.items():
            grp_tab["vol"].append(v); grp_tab["plane"].append(PLANES.index(pl))
            grp_tab["gid"].append(gi); grp_tab["tmean"].append(tw / max(qw, 1e-9))
    rep["cov_matched"] = (covm / nv).tolist()
    if tok_mismatch is not None:
        rep["cov_mismatched"] = (covx / nv).tolist()
    arr = {k: np.array(v) for k, v in pix.items()}
    arr.update({f"grp_{k}": np.array(v) for k, v in grp_tab.items()})
    np.savez_compressed(os.path.join(OUT, f"pl_{ev:05d}.npz"), **arr)
    rep["n_pix"] = len(pix["qtot"])
    return rep, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="30000-30009")
    a = ap.parse_args()
    lo, hi = map(int, a.events.split("-"))
    os.makedirs(OUT, exist_ok=True)
    run, _ = run_of(lo)
    assert run_of(hi)[0] == run, "keep one run per invocation"
    idx_h = ShardIndex("hits", run, "hits"); idx_s = ShardIndex("step", run, "step")
    print(f"run {run}: nominal shard indexing (verified per-event via coverage)", flush=True)
    tok_mm = token_sets(os.path.join(CACHE, f"ev_{lo + 7:05d}.npz"))   # mismatch null
    allD = []; allF = []; allq = []; ncand = []
    for ev in range(lo, hi + 1):
        rep, arr = process_event(ev, idx_h, idx_s, tok_mismatch=tok_mm if ev == lo else None)
        print(json.dumps(rep), flush=True)
        allD.append(arr["D"]); allF.append(arr["F"]); allq.append(arr["qtot"])
        # X candidate counts: U groups vs V groups within +-5 ticks, same volume
        for v in range(2):
            tu = arr["grp_tmean"][(arr["grp_vol"] == v) & (arr["grp_plane"] == 0)]
            tv = np.sort(arr["grp_tmean"][(arr["grp_vol"] == v) & (arr["grp_plane"] == 1)])
            if len(tu) and len(tv):
                c = np.searchsorted(tv, tu + 5) - np.searchsorted(tv, tu - 5)
                ncand.extend(c.tolist())
    D = np.concatenate(allD); F = np.concatenate(allF)
    print("\n=== DISTRIBUTION CHECKS (expect ~red-team values) ===")
    print(f"D pctl[50,75,90,95,99] = {[round(x,2) for x in np.percentile(D,[50,75,90,95,99])]} mm "
          f"(expect ~[0.15,0.36,0.92,2.94,21.9])")
    print(f"D>5mm: {(D>5).mean()*100:.2f}% (expect ~3.45%)   D>20mm: {(D>20).mean()*100:.2f}% (~1.09%)")
    print(f"F pctl[5,25,50,75,95] = {[round(x,3) for x in np.percentile(F,[5,25,50,75,95])]} "
          f"(expect ~[0.164,0.315,0.512,1.0,1.0])  F>=0.5: {(F>=0.5).mean()*100:.1f}% (~51.3%)")
    print(f"X candidates (U->V, +-5 ticks): median {np.median(ncand):.0f} mean {np.mean(ncand):.0f} (expect ~119/196)")
    q = np.concatenate(allq)
    print(f"qtot>=250 pixel count: {len(q)}  min qtot {q.min():.1f} (floor should be >=250 by cut)")


if __name__ == "__main__":
    main()
