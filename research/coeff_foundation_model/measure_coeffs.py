"""Data measurements for the sparse wavelet-coefficient model design.

Uses the FAST GPU production pipeline end to end:
  pimm_data extract (sparse clean)  ->  GPU densify  ->  GPU coherent+intrinsic noise
  ->  digitize  ->  torch DWT (coif3 L4, periodization)  ->  production per-band-sigma
  hard threshold (kappa=1, threshold_approx=True), applied in torch on the GPU coeffs.

Threshold (stated): VisuShrink hard, per-band MAD sigma (sigma_b = median(|c_b|)/0.6745
over the whole band), t_b = kappa * sigma_b * sqrt(2 ln len_b), kappa=1.0, applied to
ALL bands incl. A4 (= helix.tpc production threshold_spec). "Active" = |coeff| >= t_b.

Pipeline note: torch periodization DWT pads T (4321) to a multiple of 2^4 -> 4336, so
band lengths are A4=D4=271, D3=542, D2=1084, D1=2168 (vs numpy no-pad 271/271/541/1081/
2161). This is the pipeline the GPU model will actually consume.

    python research/measure_coeffs.py --scan 200 --dump /tmp/typical_event_coeffs.npz
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from statistics import median

import numpy as np


DATA_ROOT = "/sdf/data/neutrino/doraemon/wire_test_00_00_02"   # moved from omara/JAXTPC_Wire
SPLIT, DATASET_NAME = "run_0027575766", "sim_wire"
GEOM_JSON = ("/sdf/group/neutrino/omara/particle-imaging-models/"
             "libs/pimm-data/src/pimm_data/data/cubic_wireplane_geometry.json")
WAVELET, LEVEL = "coif3", 4
BAND_NAMES = ["A4", "D4", "D3", "D2", "D1"]   # coeffs order: [cA, cD4, cD3, cD2, cD1]


def load_geom():
    import json
    from pimm_data.jaxtpc import canonical_plane_id
    d = json.load(open(GEOM_JSON)); nts = int(d["num_time_steps"])
    geom = {}
    for label, e in d["planes"].items():
        nw = int(e["n_wires"])
        wl = np.asarray(e["wire_lengths_m"], np.float32)
        wl = wl[:nw] if wl.size >= nw else np.full(nw, 2.33, np.float32)
        geom[canonical_plane_id(label)] = {
            "label": label, "n_wires": nw, "n_ticks": nts,
            "pedestal": int(e["pedestal"]), "wire_lengths": wl}
    return geom, nts


def build_batch(ds, ev0, B):
    """Collate B events, all 6 planes, into the dense-path batch dict (CPU torch)."""
    import torch
    wires, times, vals, gids, counts, names = [], [], [], [], [], []
    for i in range(B):
        s = ds.get_data(ev0 + i)["sensor"]
        wires.append(s["wire"]); times.append(s["time"]); vals.append(s["value"])
        gids.append(s["plane_gid"]); counts.append(len(s["wire"])); names.append(s["name"])
    return {
        "wire": torch.as_tensor(np.concatenate(wires), dtype=torch.int32),
        "time": torch.as_tensor(np.concatenate(times), dtype=torch.int32),
        "value": torch.as_tensor(np.concatenate(vals), dtype=torch.float32),
        "plane_gid": torch.as_tensor(np.concatenate(gids), dtype=torch.int32),
        "offset": torch.tensor(np.cumsum(counts), dtype=torch.int64), "name": names}


def prod_threshold(bands, kappa=1.0):
    """Production per-band-sigma hard threshold (in torch). Returns thresholded bands."""
    out = []
    for c in bands:
        sigma = c.abs().median() / 0.6745
        t = kappa * float(sigma) * math.sqrt(2.0 * math.log(max(c.shape[-1], 2)))
        out.append(c * (c.abs() >= t))
    return out


SMART_KGATE, SMART_KSIG, GS = 4.0, 3.0, 64


def smart_gate_bands(bands, kgate=SMART_KGATE, ksig=SMART_KSIG):
    """GPU twin of research/coherent_coeffs/smart.py::smart_removal, FUSED in DWT space.

    The DWT is linear, so the cleaned image's bands == band - gated_block_common_mode;
    no waverec/re-wavedec round-trip needed. Per band: robust per-(64-block,pos) common
    mode (ksig-masked mean over wires) -> coherent scale sigc=MAD over blocks -> gate
    (keep |m|<kgate*sigc as coherent, subtract; drop large=signal) -> subtract from band.
    Returns coherent-removed bands (pre-threshold)."""
    import torch
    out = []
    for b in bands:                                   # (W, Lb)
        W, Lb = b.shape
        ngf = W // GS
        Ms = []
        if ngf > 0:
            bf = b[:ngf * GS].reshape(ngf, GS, Lb)
            med = bf.quantile(0.5, dim=1)                            # (ngf, Lb)
            resid = bf - med.unsqueeze(1)
            sg = (resid.abs().reshape(ngf, -1).quantile(0.5, dim=1) / 0.6745).clamp_min(1e-6)
            uf = resid.abs() <= ksig * sg[:, None, None]
            nuf = uf.sum(1)
            mean = (bf * uf).sum(1) / nuf.clamp_min(1)
            Ms.append(torch.where(nuf > 0, mean, med))              # (ngf, Lb)
        rem = W - ngf * GS
        if rem > 0:
            blk = b[ngf * GS:]
            med = blk.quantile(0.5, dim=0)
            resid = blk - med
            sg = (resid.abs().reshape(-1).quantile(0.5) / 0.6745).clamp_min(1e-6)
            uf = resid.abs() <= ksig * sg
            nuf = uf.sum(0)
            mean = (blk * uf).sum(0) / nuf.clamp_min(1)
            Ms.append(torch.where(nuf > 0, mean, med).unsqueeze(0))
        M = torch.cat(Ms, dim=0)                                     # (ng, Lb)
        sigc = (M.abs().median() / 0.6745).clamp_min(1e-6)
        Mc = torch.where(M.abs() < kgate * sigc, M, torch.zeros_like(M))
        idx = (torch.arange(W, device=b.device) // GS).clamp(max=M.shape[0] - 1)
        out.append(b - Mc[idx])
    return out


def event_coeffs(ds, ev, geom, dwt, stages, kappa=1.0):
    """One event -> {gid: [A4,D4,D3,D2,D1]} thresholded coeff tensors (nw,len) on GPU."""
    import torch
    from pimm_data.batch_transforms import move_to_device
    b = move_to_device(build_batch(ds, ev, 1), "cuda")
    for st in stages:
        st(b)                                          # self-seeds from batch['name']
    out = {}
    for gid, g in b["dense"].items():                  # g: (1, W, T)
        x = torch.nn.functional.pad(g, (0, dwt["pad"]))
        bands = dwt["ops"]._wavedec(x[0], WAVELET, LEVEL)   # list (W,len)
        bands = smart_gate_bands(bands)                     # coeff-space coherent removal (kgate=4)
        out[gid] = prod_threshold(bands, kappa)
    return out


# ── Item 2: wavelet-tree occupancy ─────────────────────────────────────────
def tree_occupancy(coeffs_by_plane):
    import torch
    num = {"d3_act": 0, "d3_tot": 0, "d2_act": 0, "d2_tot": 0,
           "d3_ch_act": 0, "d3_ch_slot": 0, "d2_gc_act": 0, "d2_gc_slot": 0}
    for bands in coeffs_by_plane.values():
        d4 = (bands[1] != 0); d3 = (bands[2] != 0); d2 = (bands[3] != 0)
        L4 = d4.shape[1]
        num["d3_act"] += int(d3.sum()); num["d3_tot"] += d3.numel()
        num["d2_act"] += int(d2.sum()); num["d2_tot"] += d2.numel()
        tau = torch.arange(L4, device=d4.device)
        # D3 children 2tau,2tau+1
        ch = torch.stack([2 * tau, 2 * tau + 1], 1)            # (L4,2)
        vch = ch < d3.shape[1]
        d3c = d3[:, ch.clamp(max=d3.shape[1] - 1)] & vch[None]  # (W,L4,2)
        par = d4[:, :, None] & vch[None]
        num["d3_ch_act"] += int((d3c & d4[:, :, None]).sum()); num["d3_ch_slot"] += int(par.sum())
        # D2 grandchildren 4tau..4tau+3
        gc = torch.stack([4 * tau + k for k in range(4)], 1)    # (L4,4)
        vgc = gc < d2.shape[1]
        d2g = d2[:, gc.clamp(max=d2.shape[1] - 1)] & vgc[None]
        parg = d4[:, :, None] & vgc[None]
        num["d2_gc_act"] += int((d2g & d4[:, :, None]).sum()); num["d2_gc_slot"] += int(parg.sum())
    return {
        "P(D3)": num["d3_act"] / num["d3_tot"],
        "P(D3|D4)": num["d3_ch_act"] / max(num["d3_ch_slot"], 1),
        "P(D2)": num["d2_act"] / num["d2_tot"],
        "P(D2|D4)": num["d2_gc_act"] / max(num["d2_gc_slot"], 1)}


# ── Items 3/4: token grid on A4∪D4 (stride-16 coarse coords) ────────────────
# coarse index c -> footprint per band (start:stop): A4/D4 [c, c+1); D3 [2c,2c+2); D2 [4c,4c+4); D1 [8c,8c+8)
def token_stats(bands, wires_per_patch, ctime_per_patch=4):
    """Returns (n_occupied_tokens, list_of_(D2+D3)_counts_per_occupied_token)."""
    import torch
    A4, D4, D3, D2, D1 = [b != 0 for b in bands]
    W, L4 = A4.shape
    cw, ct = wires_per_patch, ctime_per_patch
    nWp = (W + cw - 1) // cw
    nTp = (L4 + ct - 1) // ct
    occ = 0; fanout = []
    for wp in range(nWp):
        ws, we = wp * cw, min((wp + 1) * cw, W)
        for tp in range(nTp):
            c0, c1 = tp * ct, min((tp + 1) * ct, L4)            # coarse range
            any_act = (A4[ws:we, c0:c1].any() or D4[ws:we, c0:c1].any()
                       or D3[ws:we, 2 * c0:2 * c1].any() or D2[ws:we, 4 * c0:4 * c1].any()
                       or D1[ws:we, 8 * c0:8 * c1].any())
            if bool(any_act):
                occ += 1
                fan = int(D3[ws:we, 2 * c0:2 * c1].sum()) + int(D2[ws:we, 4 * c0:4 * c1].sum())
                fanout.append(fan)
    return occ, fanout


# ── timing ──────────────────────────────────────────────────────────────────
def bench(fn, warmup=5, iters=20):
    import torch
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e))
    return median(ts), torch.cuda.max_memory_allocated() / 1e6


def time_subm(n_active, time_len, bwd):
    import torch, spconv.pytorch as spconv
    Cin, Cout = 32, 64
    coords = torch.stack([torch.randint(0, 1969, (n_active,)),
                          torch.randint(0, time_len, (n_active,)),
                          torch.zeros(n_active, dtype=torch.long)], 1)
    coords = torch.unique(coords, dim=0).int()
    idx = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], 1).cuda()
    feat = torch.randn(coords.shape[0], Cin, device="cuda", requires_grad=bwd)
    conv = spconv.SubMConv3d(Cin, Cout, (5, 3, 1), indice_key="k").cuda()
    def step():
        x = spconv.SparseConvTensor(feat, idx, [1969, time_len, 1], 1)
        y = conv(x)
        if bwd:
            y.features.sum().backward()
    return bench(step)


def time_attn(seqlen, bwd):
    import torch
    from flash_attn import flash_attn_func
    H, D = 12, 64
    q = torch.randn(1, seqlen, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=bwd)
    k = torch.randn(1, seqlen, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=bwd)
    v = torch.randn(1, seqlen, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=bwd)
    def step():
        o = flash_attn_func(q, k, v)
        if bwd:
            o.sum().backward()
    return bench(step)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan", type=int, default=200, help="events to scan for activity ranking")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--patch-wires", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--dump", type=str, default=None)
    args = ap.parse_args()

    import torch
    from pimm_data import JAXTPCDataset
    from pimm_data.detector_transforms import Densify, AddNoise, Digitize
    from helix.core import backend
    backend.set_backend("torch")
    ops = backend.ops("helix.core.wavelet_ops")

    geom, nts = load_geom()
    pad = (-nts) % (2 ** LEVEL)
    dwt = {"ops": ops, "pad": pad}
    stages = [Densify(geom),
              AddNoise(geom=geom, coherent=True, incoherent=True),
              Digitize(geom=geom)]
    ds = JAXTPCDataset(data_root=DATA_ROOT, split=SPLIT,
                       modalities=("sensor",), dataset_name=DATASET_NAME)
    gids = sorted(geom.keys())
    band_len = [b.shape[-1] for b in event_coeffs(ds, 0, geom, dwt, stages)[gids[0]]]

    print("=" * 78)
    print("ITEM 0 — geometry")
    for g in gids:
        e = geom[g]
        print(f"  gid {g}  {e['label']:14s}  n_wires={e['n_wires']}  pedestal={e['pedestal']}")
    print(f"  n_time_ticks (raw, pre-DWT) = {nts}; DWT pads to {nts + pad}")
    print(f"  band lengths (A4,D4,D3,D2,D1) = {band_len}")

    # ---- scan: total active coeffs per event (activity) --------------------
    print(f"\nscanning {args.scan} events for activity ranking (GPU pipeline)...")
    t0 = time.perf_counter()
    activity, percounts = [], []
    for ev in range(args.scan):
        cf = event_coeffs(ds, ev, geom, dwt, stages, args.kappa)
        tot = sum(int((b != 0).sum()) for bands in cf.values() for b in bands)
        activity.append((tot, ev))
        if ev % 50 == 0:
            print(f"  ev {ev}: total active = {tot:,}  ({(time.perf_counter()-t0):.0f}s)")
    activity.sort()
    print(f"  scan done in {time.perf_counter()-t0:.0f}s")
    acts = np.array([a for a, _ in activity])
    print(f"  total-active across {args.scan} events: "
          f"min={acts.min():,} p5={int(np.percentile(acts,5)):,} "
          f"p50={int(np.percentile(acts,50)):,} p95={int(np.percentile(acts,95)):,} max={acts.max():,}")
    quiet = activity[max(0, int(0.05 * len(activity)))][1]
    typ = activity[int(0.50 * len(activity))][1]
    busy = activity[min(len(activity) - 1, int(0.95 * len(activity)))][1]
    reps = [("quiet", quiet), ("typical", typ), ("busy", busy)]
    print(f"  representatives: quiet=ev{quiet}, typical=ev{typ}, busy=ev{busy}")

    # ---- Item 1: per-band counts -------------------------------------------
    rep_coeffs = {lbl: event_coeffs(ds, ev, geom, dwt, stages, args.kappa) for lbl, ev in reps}
    print("\n" + "=" * 78)
    print("ITEM 1 — per-band active-coefficient counts (kappa={:.1f}, AFTER coeff-space "
          "smart removal kgate={:.1f})".format(args.kappa, SMART_KGATE))
    print(f"{'event':>8} {'plane':>14} " + " ".join(f"{b:>8}" for b in BAND_NAMES) + f" {'total':>9}")
    for lbl, ev in reps:
        for g in gids:
            bands = rep_coeffs[lbl][g]
            counts = [int((b != 0).sum()) for b in bands]
            print(f"{lbl+' '+str(ev):>8} {geom[g]['label']:>14} "
                  + " ".join(f"{c:>8,}" for c in counts) + f" {sum(counts):>9,}")

    # ---- Item 2: tree occupancy (typical, pooled all planes) ---------------
    occ2 = tree_occupancy(rep_coeffs["typical"])
    print("\n" + "=" * 78)
    print("ITEM 2 — wavelet-tree parent->child occupancy (typical, pooled 6 planes)")
    for k in ["P(D3)", "P(D3|D4)", "P(D2)", "P(D2|D4)"]:
        print(f"  {k:>10} = {occ2[k]:.4f}")
    print(f"  lift: D3 {occ2['P(D3|D4)']/max(occ2['P(D3)'],1e-9):.1f}x, "
          f"D2 {occ2['P(D2|D4)']/max(occ2['P(D2)'],1e-9):.1f}x")

    # ---- Item 3: token counts vs patch size (typical, busy) ----------------
    print("\n" + "=" * 78)
    print("ITEM 3 — occupied coarse tokens (>=1 active in any band in footprint)")
    hdr = "".join(f"{f'{pw}x4':>10}" for pw in args.patch_wires)
    print(f"{'event':>8} {'plane':>14}" + hdr)
    typ_fanout = None
    for lbl in ("typical", "busy"):
        for g in gids:
            occs = []
            for pw in args.patch_wires:
                occ, fan = token_stats(rep_coeffs[lbl][g], pw)
                occs.append(occ)
                if lbl == "typical" and pw == 8:
                    typ_fanout = (typ_fanout or []) + fan
            print(f"{lbl:>8} {geom[g]['label']:>14}" + "".join(f"{o:>10,}" for o in occs))

    # ---- Item 4: fine fan-out per 8x4 token (typical, all planes pooled) ----
    fo = np.array(typ_fanout)
    print("\n" + "=" * 78)
    print("ITEM 4 — active (D2+D3) per occupied 8x4 token (typical, pooled)")
    print(f"  tokens={fo.size:,}  mean={fo.mean():.1f}  p95={int(np.percentile(fo,95))}  max={int(fo.max())}")

    # ---- Item 5: per-band value stats (typical, pooled) --------------------
    print("\n" + "=" * 78)
    print("ITEM 5 — |coeff| percentiles over ACTIVE coeffs (typical, pooled)")
    print(f"{'band':>5} {'p1':>9} {'p50':>9} {'p99':>9}  {'coexist(>10x neigh)':>20}")
    for bi, bn in enumerate(BAND_NAMES[:4]):   # A4,D4,D3,D2
        vals = []
        coex_num = coex_den = 0
        for g in gids:
            c = rep_coeffs["typical"][g][bi]
            a = c[c != 0].abs()
            vals.append(a.cpu().numpy())
            # coexist: active coeff whose same-band 8-neighborhood has another active with |ratio|>10
            m = (c != 0)
            ac = c.abs()
            big = ac.clone(); big[~m] = 0
            import torch.nn.functional as F
            mx = F.max_pool2d(big[None, None], 3, 1, 1)[0, 0]
            mn = ac.clone(); mn[~m] = float("inf")
            mnp = -F.max_pool2d(-mn[None, None], 3, 1, 1)[0, 0]
            ratio = mx / mnp.clamp_min(1e-9)
            coex_num += int((m & (ratio > 10)).sum()); coex_den += int(m.sum())
        v = np.concatenate(vals)
        frac = coex_num / max(coex_den, 1)
        print(f"{bn:>5} {np.percentile(v,1):>9.3f} {np.percentile(v,50):>9.3f} "
              f"{np.percentile(v,99):>9.3f}  {frac*100:>18.1f}%")

    # ---- Timing block ------------------------------------------------------
    typ_a4 = int((rep_coeffs["typical"][gids[0]][0] != 0).sum())
    busy_a4 = int((rep_coeffs["busy"][gids[0]][0] != 0).sum())
    typ_d2 = int((rep_coeffs["typical"][gids[0]][3] != 0).sum())
    # token counts (8x4, typical): S1 = mean single plane, S2 = sum over 6 planes (global stage)
    per_plane_tok = [token_stats(rep_coeffs["typical"][g], 8)[0] for g in gids]
    s1 = int(round(np.mean(per_plane_tok)))
    s2 = int(sum(per_plane_tok))
    print("\n" + "=" * 78)
    print(f"TIMING (A100 bf16/TF32) | sizes: A4 typ={typ_a4:,} busy={busy_a4:,} "
          f"D2 typ={typ_d2:,} | tokens/plane={per_plane_tok} S1(mean)={s1:,} S2(6-plane sum)={s2:,}")
    print(f"{'operator':>34} {'fwd ms':>8} {'fwd MB':>8} {'fwd+bwd ms':>11} {'fwd+bwd MB':>11}")
    rows = [
        (f"SubM(5x3) A4 typ N={typ_a4}", lambda b: time_subm(typ_a4, band_len[0], b)),
        (f"SubM(5x3) A4 busy N={busy_a4}", lambda b: time_subm(busy_a4, band_len[0], b)),
        (f"SubM(5x3) D2 typ N={typ_d2}", lambda b: time_subm(typ_d2, band_len[3], b)),
        (f"flash attn S1={s1} (1 plane)", lambda b: time_attn(s1, b)),
        (f"flash attn S2={s2} (6 planes)", lambda b: time_attn(s2, b)),
    ]
    for name, f in rows:
        fm, fmb = f(False); bm, bmb = f(True)
        print(f"{name:>34} {fm:>8.3f} {fmb:>8.0f} {bm:>11.3f} {bmb:>11.0f}")

    # ---- dump typical event active coeffs ----------------------------------
    if args.dump:
        P, Bn, Wr, Ti, Vl = [], [], [], [], []
        for g in gids:
            for bi, bands in enumerate(rep_coeffs["typical"][g]):
                nz = torch.nonzero(bands, as_tuple=False)
                if nz.numel():
                    P.append(np.full(nz.shape[0], g, np.int16))
                    Bn.append(np.full(nz.shape[0], bi, np.int8))
                    Wr.append(nz[:, 0].cpu().numpy().astype(np.int32))
                    Ti.append(nz[:, 1].cpu().numpy().astype(np.int32))
                    Vl.append(bands[nz[:, 0], nz[:, 1]].cpu().numpy().astype(np.float32))
        np.savez_compressed(args.dump, plane_gid=np.concatenate(P), band_id=np.concatenate(Bn),
                            wire=np.concatenate(Wr), time_index=np.concatenate(Ti),
                            value=np.concatenate(Vl),
                            legend=np.array(["band_id: 0=A4 1=D4 2=D3 3=D2 4=D1"]),
                            event=np.array([typ]))
        print(f"\ndumped typical event (ev{typ}) active coeffs -> {args.dump}")


if __name__ == "__main__":
    import torch  # noqa
    for p in ("/sdf/group/neutrino/omara/helix/.pylibs",
              "/sdf/group/neutrino/omara/particle-imaging-models/libs/pimm-data/src",
              "/sdf/group/neutrino/omara/helix"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import hdf5plugin  # noqa
    main()
