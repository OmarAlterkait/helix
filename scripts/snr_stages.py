#!/usr/bin/env python3
"""Where does a faint deposit's signal-to-noise go between the wire and the model?

    python scripts/snr_stages.py --table <noise_vs_hits.npz> --truth <truth_v2> [--events 32] [--workers 16]

Replays the corpus chain on the CLEAN sensor planes of the floor's test events with
fresh forward-model noise (scripts/build_coeff_corpus.py: digitize(clean + incoherent
+ coherent), then helix.tpc.pipeline 'gate': wavedec -> coherent gate -> hard
threshold at kappa x per-band MAD sigma x sqrt(2 ln n_band)), and the same with the
clean image left out (noise only, same realisation). Stages, as DWT coefficients:

  inc        digitize(clean + incoherent)            -- the analytic bound's setting
  coh        digitize(clean + incoherent + coherent) -- no removal
  gate       coherent gate applied to coh
  gate-D1    gate without band D1 (the tokenizer reads A4, D4, D3, D2 only)
  k1.0       gate, thresholded at the corpus's kappa = 1 (3.35-3.9 sigma per band)
  k1.0-D1    that without D1: WHAT THE MODEL READS
  k0.75-D1, k0.5-D1   the same at lower thresholds (2.5-2.9 / 1.7-2.0 sigma)
  s4-D1      a flat hard threshold at 4 band-sigma (no sqrt(2 ln n) factor), no D1

Statistic: a linear matched filter on the deposit's own clean template (the clean
image on its own wires/ticks, DWT'd), band-whitened by the gated noise sigma:
z = sum_b T_b.D_b / s_b^2 / sqrt(sum_b |T_b|^2 / s_b^2) -- for the Gaussian stages
this is ~N(0,1) under noise and ~N(SNR, 1) with signal. Null: the template on the
noise-only planes at its own place and 24 shifted places (wire shifts; time shifts
by multiples of 16 ticks, so the decimated transform stays aligned). The 1%-FPR
threshold is the pooled null's 99th percentile per stage: known-location
efficiency, an upper bound on any detector reading that stage. The model's own
efficiency (window search, from the table) is shown alongside.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KAPPAS = (1.0, 0.75, 0.5)
SIGMAS = (4.0,)
STAGES = ("inc", "coh", "gate", "gate-D1", "k1.0", "k1.0-D1", "k0.75-D1", "k0.5-D1") \
    + tuple(f"s{s:g}-D1" for s in SIGMAS)
NULL_SHIFTS = 24


def process_event(job):
    from helix.core import backend
    backend.set_backend("numpy")
    import h5py
    from helix.core.coeff_io import read_coeff_event
    from helix.core.wavelet import ThresholdSpec, threshold_bands, wavedec
    from helix.probe import resolution as R
    from helix.tpc.config import DetectorConfig
    from helix.tpc.coherent_gate import coherent_gate
    from helix.tpc.geometry import load_plane_registry
    from helix.tpc.io import read_sensor_plane
    from helix.tpc.noise import digitize, generate_noise
    from scripts.build_denoise_truth import event_cells

    f, deps, source, geom, seed = job
    cfg = DetectorConfig()
    reg = load_plane_registry(geom)
    rng = np.random.default_rng(seed)
    z = np.load(f, allow_pickle=True)
    shard, event = str(z["shard"]), int(z["event"])
    with h5py.File(shard, "r") as fh:
        pos = int(np.searchsorted(fh["ident"]["event"][:], event))
    ce = read_coeff_event(shard, pos)
    run = os.path.basename(os.path.dirname(shard))
    sensor = os.path.join(source, "sensor", run, ce.source_file)
    tk, tq = event_cells(os.path.join(source, "hits", run, ce.source_file.replace("_sensor_", "_hits_")), ce.event)
    tg, tw, tt = R.unkey(tk)
    level = cfg.dwt_level
    npad = lambda x: np.pad(x, ((0, 0), (0, (-x.shape[1]) % (1 << level))))
    dec = lambda x: [np.asarray(b) for b in wavedec(npad(np.asarray(x, np.float32)), wavelet=cfg.wavelet,
                                                     level=level, mode=cfg.dwt_mode)[0]]
    noD1 = lambda bands: bands[:-1] + [np.zeros_like(bands[-1])]

    def stages(img_inc, img_coh):
        g = [np.asarray(b) for b in coherent_gate(dec(img_coh), group_size=cfg.group_size, kgate=cfg.gate_kgate,
                                                  ksig=cfg.gate_ksig, npass=cfg.gate_npass, tau=cfg.gate_tau,
                                                  gate_approx=True)]
        out, kept = dict(inc=dec(img_inc), coh=dec(img_coh), gate=g, **{"gate-D1": noD1(g)}), {}
        for k in KAPPAS:
            spec = ThresholdSpec(method="universal", func=cfg.threshold_mode, scale=k, per_band_sigma=True,
                                 threshold_approx=cfg.threshold_include_approx)
            t, _, _, _ = threshold_bands(g, spec)
            t = [np.asarray(b) for b in t]
            if k == 1.0:
                out["k1.0"] = t
            out[f"k{k}-D1"] = noD1(t)
            kept[f"k{k}"] = sum(int(np.count_nonzero(b)) for b in t[:-1]) / sum(b.size for b in t[:-1])
        # a flat threshold at s band-sigma (MAD, as threshold_bands measures it)
        sg = [max(float(np.median(np.abs(b))) / 0.6745, 1e-6) for b in g]
        for s in SIGMAS:
            t = [np.where(np.abs(b) > s * sb_, b, 0.0) for b, sb_ in zip(g, sg)]
            out[f"s{s:g}-D1"] = noD1(t)
            kept[f"s{s:g}"] = sum(int(np.count_nonzero(b)) for b in t[:-1]) / sum(b.size for b in t[:-1])
        return out, kept

    rows, info = [], {}
    for g in sorted({d["g"] for d in deps}):
        clean = np.asarray(read_sensor_plane(sensor, ce.event, reg[g]["label"]), np.float32)
        nw, nt = clean.shape
        L = np.asarray(reg[g]["wire_lengths"]); ped = reg[g]["pedestal"]
        n_inc = generate_noise((nw, nt), rng=rng, wire_lengths_m=L, incoherent=True, coherent=False)
        n_coh = generate_noise((nw, nt), rng=rng, incoherent=False, coherent=True, group_size=cfg.group_size)
        sig, kept = stages(digitize(clean + n_inc, ped), digitize(clean + n_inc + n_coh, ped))
        nul, kept0 = stages(digitize(n_inc, ped), digitize(n_inc + n_coh, ped))
        info[g] = dict(kept=kept, kept_null=kept0, corpus=int(np.sum(ce.plane_gid == g)),
                       emulated=int(sum(np.count_nonzero(b) for b in sig["k1.0"])))
        s_gate = [float(np.std(b)) for b in nul["gate"]]
        sb = {s: ([float(np.std(b)) for b in nul[s]] if s in ("inc", "coh") else s_gate) for s in STAGES}
        for d in [d for d in deps if d["g"] == g]:
            inw = (tg == g) & (tw >= d["w0"] // R.FW) & (tw < (d["w0"] + R.WW) // R.FW) \
                & (tt >= d["t0"] // R.FT) & (tt < (d["t0"] + R.WT) // R.FT)
            if not inw.any():
                continue
            wa, wb = max(tw[inw].min() * R.FW - 2, 0), min((tw[inw].max() + 1) * R.FW + 2, nw)
            ta, tb = max(tt[inw].min() * R.FT - 32, 0), min((tt[inw].max() + 1) * R.FT + 32, nt)
            own = np.zeros((wb - wa, nt), np.float32); own[:, ta:tb] = clean[wa:wb, ta:tb]
            if not own.any():
                continue
            T = dec(own)
            shifts = [(0, 0)]
            while len(shifts) < NULL_SHIFTS + 1:
                dw, m = int(rng.integers(-400, 401)), int(rng.integers(-60, 61))
                if 0 <= wa + dw and wb + dw <= nw and 0 <= ta + 16 * m and tb + 16 * m <= nt:
                    shifts.append((dw, m))
            rec = dict(d)
            for s in STAGES:
                w = [1.0 / max(x, 1e-6) ** 2 for x in sb[s]]
                if s.endswith("-D1"):
                    w[-1] = 0.0
                norm = np.sqrt(sum(wi * float((Tb ** 2).sum()) for wi, Tb in zip(w, T)))

                def stat(C, dw, m):
                    return sum(wi * float((np.roll(Tb, m * (Tb.shape[1] // T[0].shape[1]), axis=1)
                                           * Cb[wa + dw:wb + dw]).sum()) for wi, Tb, Cb in zip(w, T, C)) / norm

                rec[f"z_{s}"] = stat(sig[s], 0, 0)
                rec[f"null_{s}"] = [stat(nul[s], dw, m) for dw, m in shifts]
                rec[f"resp_{s}"] = rec[f"z_{s}"] - rec[f"null_{s}"][0]     # signal response, same noise
            rows.append(rec)
    return rows, {os.path.basename(f): info}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", required=True, help="noise_vs_hits.py --out npz")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--events", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--snr-max", type=float, default=20.0)
    ap.add_argument("--source", default="/global/cfs/cdirs/m5238/users/oalter/wire_test_00_00_02")
    ap.add_argument("--geom", default="cubic_wireplane_geometry.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    Tb = np.load(a.table)
    ok = (Tb["foreign"] <= 0.25 * Tb["q"]) & (Tb["E"] > 0) & (Tb["snr"] > 0) & (Tb["snr"] < a.snr_max)
    evs = sorted(set(Tb["ev"][ok].tolist()))[:a.events]
    jobs = []
    for ev in evs:
        idx = np.nonzero(ok & (Tb["ev"] == ev))[0]
        deps = [dict(ev=int(ev), g=int(Tb["g"][i]), w0=int(Tb["w0"][i]), t0=int(Tb["t0"][i]), E=float(Tb["E"][i]),
                     q=float(Tb["q"][i]), snr=float(Tb["snr"][i]),
                     model=float(Tb["model"][i]) if "model" in Tb.files else np.nan) for i in idx]
        jobs.append((os.path.join(a.truth, f"ev{ev:03d}.npz"), deps, a.source, a.geom, 1000 + ev))
    print(f"{sum(len(j[1]) for j in jobs)} deposits (analytic SNR < {a.snr_max:g}) in {len(jobs)} events")

    from multiprocessing import Pool
    rows, info = [], {}
    with Pool(a.workers) as pool:
        for r, i in pool.imap_unordered(process_event, jobs):
            rows += r; info.update(i)
    em = np.array([v["emulated"] for i in info.values() for v in i.values()])
    co = np.array([v["corpus"] for i in info.values() for v in i.values()])
    print(f"\nemulation vs corpus: kept coefficients per plane, emulated/corpus median {np.median(em / co):.3f} "
          f"(IQR {np.percentile(em / co, 25):.3f}-{np.percentile(em / co, 75):.3f}, {len(em)} planes)")
    base = np.median([v["kept"]["k1.0"] for i in info.values() for v in i.values()])
    for k in [f"k{k}" for k in KAPPAS] + [f"s{s:g}" for s in SIGMAS]:
        kp = np.array([v["kept"][k] for i in info.values() for v in i.values()])
        k0 = np.array([v["kept_null"][k] for i in info.values() for v in i.values()])
        print(f"  {k:>6}: kept fraction of A4-D2 coefficients {np.median(kp):.4f} (noise only {np.median(k0):.4f}) "
              f"-> {np.median(kp) / base:.1f}x the corpus")

    snr = np.array([r["snr"] for r in rows]); p3 = np.array([r["g"] % 3 for r in rows])
    model = np.array([r["model"] for r in rows])
    Z = {s: np.array([r[f"z_{s}"] for r in rows]) for s in STAGES}
    RS = {s: np.array([r[f"resp_{s}"] for r in rows]) for s in STAGES}
    thr = {s: float(np.quantile(np.concatenate([r[f"null_{s}"] for r in rows]), 0.99)) for s in STAGES}
    print("\nnull 99th percentile per stage: " + ", ".join(f"{s} {thr[s]:.2f}" for s in STAGES) + "  (Gaussian 2.33)")
    print("\n== known-location 1%-FPR efficiency by analytic SNR (upper bound for any detector reading that stage)")
    print(f"  {'SNR':>7} {'n':>4} " + " ".join(f"{s:>9}" for s in STAGES) + "  model(window)")
    for lo, hi in ((1, 3), (3, 5), (5, 7), (7, 10), (10, 15), (15, 20)):
        m = (snr >= lo) & (snr < hi)
        if m.sum() >= 5:
            print(f"  {lo:>3}-{hi:<3} {m.sum():4d} " + " ".join(f"{np.mean(Z[s][m] > thr[s]):9.2f}" for s in STAGES)
                  + f"  {np.nanmean(model[m]):9.2f}")
    print("\n== signal response (z with signal minus z on the same noise without it; ~SNR when nothing is lost), median")
    for lo, hi in ((3, 5), (5, 7), (7, 10), (10, 15)):
        m = (snr >= lo) & (snr < hi)
        if m.sum() >= 5:
            print(f"  SNR {lo:>2}-{hi:<2}: " + "  ".join(f"{s} {np.median(RS[s][m]):5.2f}" for s in STAGES))
    print("\n== per plane, SNR 3-10: known-location efficiency")
    for p in range(3):
        m = (snr >= 3) & (snr < 10) & (p3 == p)
        if m.sum() >= 5:
            print(f"  {'UVY'[p]} n={m.sum():4d}: " + "  ".join(f"{s} {np.mean(Z[s][m] > thr[s]):.2f}" for s in STAGES)
                  + f"   model {np.nanmean(model[m]):.2f}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(rows=[{k: v for k, v in r.items() if not k.startswith("null_")} for r in rows],
                           thr=thr, info={k: {str(g): v for g, v in i.items()} for k, i in info.items()}),
                      fh, default=float)


if __name__ == "__main__":
    main()
