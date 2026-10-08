#!/usr/bin/env python3
"""A real classical detector on the floor windows: a whitened matched-filter bank.

    python scripts/mf_detector.py --events 48 --workers 16 [--kernel-dir baselines/uboone_sp/response]

Unlike the matched-filter BOUND (scripts/noise_vs_hits.py, scripts/snr_stages.py),
this detector does not know the deposit's shape or place: it correlates the image
with a bank of templates -- the simulation's own 2D response (least-squares fit,
baselines/uboone_sp/fit_response.py) convolved with small charge boxes of 1-4 wires
x 4-48 ticks -- after whitening each wire's time series by the forward noise model's
exact spectrum, and scores a window by the MAX over every template and every position
whose charge location falls inside it (z, in noise sigma). The 1%-FPR threshold is
taken on the floor's noise windows (far: kind bg) and on near-activity ones (bgn).

Inputs, all with the same noise realisation -- a detection ladder:
  syn_float+inc   ONLY the isolated deposits' own hits charge folded with the fitted
                  response (no neighbours), float, + incoherent noise: the bound WITH search
  syn_zs+inc      the same stored like the sensor (integer ADC, |v| < 2 dropped)
  syn_zs+gate     + coherent noise and the coherent gate (inverse DWT)
  syn_zs+corpus   + hard threshold (kappa 1) and band D1 dropped
  inc             the REAL clean plane (neighbours present) + incoherent noise
  gate            real + coherent noise, gated
  corpus          real, gated, thresholded, no D1: what the model reads
On the synthetic planes near-activity windows are pure noise (no activity is
simulated), so far and near thresholds agree there; on the real ones the near
threshold carries every neighbour's response.

This is the best a hand-built detector does on each input; compare with the model
(--model-windows: an eval_denoise *_windows.json) on the same events.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUTS = ("syn_float+inc", "syn_zs+inc", "syn_zs+gate", "syn_zs+corpus", "inc", "gate", "corpus")
BOX_W, BOX_T = (1, 2, 4), (4, 16, 48)
WPAD, NT_PAD = 32, 4608


def process_event(job):
    from helix.core import backend
    backend.set_backend("numpy")
    import h5py
    from helix.core.coeff_io import read_coeff_event
    from helix.core.wavelet import SparseResult, ThresholdSpec, reconstruct, threshold_bands, wavedec
    from helix.probe import resolution as R
    from helix.tpc.config import DetectorConfig
    from helix.tpc.coherent_gate import coherent_gate
    from helix.tpc.geometry import load_plane_registry
    from helix.tpc.io import read_sensor_plane
    from helix.tpc.noise import DEFAULT_ENC, DEFAULT_SAMPLING_RATE_HZ, _series_spectrum_shape, digitize, generate_noise
    from helix.probe.truth import PLANES, decode_hits_plane

    ev, truth, near, kdir, source, geom = job
    cfg = DetectorConfig()
    reg = load_plane_registry(geom)
    rng = np.random.default_rng(1000 + ev)
    z = np.load(os.path.join(truth, f"ev{ev:03d}.npz"), allow_pickle=True)
    wins = json.loads(str(z["meta"]))
    zn = np.load(os.path.join(near, f"ev{ev:03d}.npz"), allow_pickle=True)
    wins += json.loads(str(zn["meta"]))
    shard, event = str(z["shard"]), int(z["event"])
    with h5py.File(shard, "r") as fh:
        pos = int(np.searchsorted(fh["ident"]["event"][:], event))
    ce = read_coeff_event(shard, pos)
    run = os.path.basename(os.path.dirname(shard))
    sensor = os.path.join(source, "sensor", run, ce.source_file)
    hf = h5py.File(os.path.join(source, "hits", run, ce.source_file.replace("_sensor_", "_hits_")), "r")
    level = cfg.dwt_level
    X2 = DEFAULT_ENC[0] ** 2 + 1 / 12
    spec = _series_spectrum_shape(NT_PAD, "microboone", DEFAULT_SAMPLING_RATE_HZ) ** 2
    wts = np.full(NT_PAD // 2 + 1, 2.0); wts[0] = wts[-1] = 1.0
    spec = spec / (wts * spec).sum() * NT_PAD           # mean over the two-sided spectrum = 1

    def dwt(img):
        x = np.pad(np.asarray(img, np.float32), ((0, 0), (0, (-img.shape[1]) % (1 << level))))
        return [np.asarray(b) for b in wavedec(x, wavelet=cfg.wavelet, level=level, mode=cfg.dwt_mode)[0]]

    def idwt(bands, nt):
        return np.asarray(reconstruct(SparseResult(coeffs=bands, n_kept=0, n_total=0, sigma_per_band=None,
                                                   wavelet=cfg.wavelet, level=level, mode=cfg.dwt_mode), nt))

    out = []
    for g in sorted({int(m["g"]) for m in wins}):
        p = g % 3
        kz = np.load(os.path.join(kdir, f"fit_{'UVY'[p]}.npz"))
        Rk, wo, tzb = kz["R_avg"].astype(np.float64), kz["wire_offsets"].astype(np.int64), int(kz["time_zero_bin"])
        clean = np.asarray(read_sensor_plane(sensor, ce.event, reg[g]["label"]), np.float32)
        nw, nt = clean.shape
        L = np.asarray(reg[g]["wire_lengths"]); ped = reg[g]["pedestal"]
        n_inc = generate_noise((nw, nt), rng=rng, wire_lengths_m=L, incoherent=True, coherent=False)
        n_coh = generate_noise((nw, nt), rng=rng, incoherent=False, coherent=True, group_size=cfg.group_size)

        def chain(signal):
            """-> (gated image, corpus image) of digitize(signal + incoherent + coherent)."""
            gb = [np.asarray(b) for b in coherent_gate(dwt(digitize(signal + n_inc + n_coh, ped)), group_size=cfg.group_size,
                                                       kgate=cfg.gate_kgate, ksig=cfg.gate_ksig, npass=cfg.gate_npass,
                                                       tau=cfg.gate_tau, gate_approx=True)]
            tb, _, _, _ = threshold_bands(gb, ThresholdSpec(method="universal", func=cfg.threshold_mode, scale=1.0,
                                                            per_band_sigma=True, threshold_approx=cfg.threshold_include_approx))
            tb = [np.asarray(b) for b in tb]; tb[-1] = np.zeros_like(tb[-1])
            return idwt(gb, nt), idwt(tb, nt)

        # synthetic plane: ONLY the isolated deposits' own hits charge, folded with the fitted
        # response -- no neighbours at all; float, and stored like the sensor (integer, |v| < 2 dropped)
        dec = decode_hits_plane(hf[f"event_{ce.event:03d}"][f"volume_{g // 3}"][PLANES[p]])
        Hs = np.zeros((nw, nt))
        for m in [m for m in wins if int(m["g"]) == g and m["kind"] == "iso"]:
            w0, t0 = int(m["w0"]), int(m["t0"])
            sel = (dec["wire"] >= w0) & (dec["wire"] < w0 + R.WW) & (dec["tick"] >= t0) & (dec["tick"] < t0 + R.WT)
            np.add.at(Hs, (dec["wire"][sel], dec["tick"][sel]), dec["q"][sel])
        Kfull = np.zeros((nw, nt)); kk = np.arange(Rk.shape[0])
        for j, o in enumerate(wo):
            Kfull[int(o) % nw, (kk - tzb) % nt] += Rk[:, j]
        syn = np.fft.irfft2(np.fft.rfft2(Hs) * np.fft.rfft2(Kfull), s=(nw, nt))
        syn_z = np.round(syn); syn_z[np.abs(syn_z) < 2] = 0.0
        sg, sc = chain(syn_z.astype(np.float32))
        g_real, c_real = chain(clean)
        imgs = {"syn_float+inc": digitize(syn.astype(np.float32) + n_inc, ped),
                "syn_zs+inc": digitize(syn_z.astype(np.float32) + n_inc, ped),
                "syn_zs+gate": sg, "syn_zs+corpus": sc,
                "inc": digitize(clean + n_inc, ped), "gate": g_real, "corpus": c_real}

        # whitening along time: per-sample noise spectrum of the median wire
        s_ser = DEFAULT_ENC[1] + DEFAULT_ENC[2] * float(np.median(L))
        psd = X2 + s_ser ** 2 * spec                      # E|N_f|^2 / NT_PAD
        wf = 1.0 / np.sqrt(psd)
        H, Wd = nw + WPAD, NT_PAD

        def whiten(x):
            X = np.zeros((H, Wd)); X[:x.shape[0], :x.shape[1]] = x
            return np.fft.irfft(np.fft.rfft(X, axis=1) * wf, n=Wd, axis=1)

        # templates: response x charge box, placed so the correlation peaks at the charge
        base = np.zeros((H, Wd))
        k = np.arange(Rk.shape[0])
        for j, o in enumerate(wo):
            base[int(o) % H, (k - tzb) % Wd] += Rk[:, j]
        Bf = np.fft.rfft2(base)
        tmpl_f = []
        for bw in BOX_W:
            for bt in BOX_T:
                box = np.zeros((H, Wd)); box[:bw, :bt] = 1.0
                box = np.roll(box, (-(bw // 2), -(bt // 2)), axis=(0, 1))          # centred on the charge
                T = np.fft.irfft2(np.fft.rfft2(box) * Bf, s=(H, Wd))
                Tw = np.fft.irfft(np.fft.rfft(T, axis=1) * wf, n=Wd, axis=1)
                Tw /= np.sqrt((Tw ** 2).sum())
                tmpl_f.append(np.conj(np.fft.rfft2(Tw)))
        maps = {}
        for name, img in imgs.items():
            Df = np.fft.rfft2(whiten(img))
            M = np.full((nw, nt), -np.inf)
            for Tf in tmpl_f:
                M = np.maximum(M, np.fft.irfft2(Df * Tf, s=(H, Wd))[:nw, :nt])
            maps[name] = M
        for m in [m for m in wins if int(m["g"]) == g]:
            w0, t0 = int(m["w0"]), int(m["t0"])
            rec = dict(ev=ev, g=g, w0=w0, t0=t0, kind=m["kind"], E=float(np.sum(m["E"])) if m["E"] else 0.0)
            for name, M in maps.items():
                reg_ = M[w0:w0 + R.WW, t0:t0 + R.WT]
                rec[name] = float(reg_.max()) if reg_.size else -np.inf
            out.append(rec)
    hf.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    W = "/pscratch/sd/o/oalter/helix_work"
    ap.add_argument("--truth", default=f"{W}/resolution/truth_v2")
    ap.add_argument("--near", default=f"{W}/resolution/truth_v2_near")
    ap.add_argument("--annot", default=f"{W}/denoise/noise_vs_hits.npz")
    ap.add_argument("--kernel-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                         "baselines", "uboone_sp", "response"))
    ap.add_argument("--model-windows", nargs="*", default=[f"{W}/denoise/results/dn_near_M4n_ft_full_near_pres_trained_windows.json"])
    ap.add_argument("--events", type=int, default=48)
    ap.add_argument("--first", type=int, default=260)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--source", default="/global/cfs/cdirs/m5238/users/oalter/wire_test_00_00_02")
    ap.add_argument("--geom", default="cubic_wireplane_geometry.json")
    ap.add_argument("--out", default=f"{W}/denoise/mf_detector.json")
    a = ap.parse_args()

    from multiprocessing import Pool
    evs = list(range(a.first, a.first + a.events))
    jobs = [(ev, a.truth, a.near, a.kernel_dir, a.source, a.geom) for ev in evs]
    rows = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(process_event, jobs):
            rows += r
    json.dump(rows, open(a.out, "w"))

    from helix.probe.resolution import ENERGY_BINS
    A = np.load(a.annot)
    ann = {k: (float(s) if pk > 0 else 0.0, bool(fo <= 0.25 * q))
           for k, s, pk, fo, q in zip(zip(A["ev"].tolist(), A["g"].tolist(), A["w0"].tolist(), A["t0"].tolist()),
                                      A["snr"], A["peak"], A["foreign"], A["q"])}
    dets = {name: {(r["ev"], r["g"], r["w0"], r["t0"], r["kind"]): r[name] for r in rows} for name in INPUTS}
    keyset = [(r["ev"], r["g"], r["w0"], r["t0"], r["kind"]) for r in rows]
    for f in a.model_windows:
        Wm = json.load(open(f))
        tag = os.path.basename(f).replace("dn_near_", "").replace("_trained_windows.json", "")
        dets[tag] = {(w["ev"], w["g"], w["w0"], w["t0"], w["kind"]): w["score"] for w in Wm if w["ev"] in evs}
    print(f"{len(evs)} events: iso {sum(k[4] == 'iso' for k in keyset)}, bg {sum(k[4] == 'bg' for k in keyset)}, "
          f"bgn {sum(k[4] == 'bgn' for k in keyset)}")
    E = {k: r["E"] for k, r in zip(keyset, rows)}
    print("\n1%-FPR efficiency, threshold from far (bg) | near-activity (bgn) noise windows")
    eb_names = ["<0.1", "0.1-0.2", "0.2-0.5", "0.5-1"]
    snr_bins = ((0, 1e-9), (1e-9, 3), (3, 5), (5, 7), (7, 10), (10, 15), (15, 1e9))
    for name, D in dets.items():
        bg = np.array([v for k, v in D.items() if k[4] == "bg"]); bn = np.array([v for k, v in D.items() if k[4] == "bgn"])
        tf, tn = np.quantile(bg, 0.99), np.quantile(bn, 0.99)
        iso = [(k, v) for k, v in D.items() if k[4] == "iso"]
        line = f"  {name:30s} thr far {tf:9.2f} near {tn:9.2f} | E: "
        for i, nm in enumerate(eb_names):
            s = np.array([v for k, v in iso if ENERGY_BINS[i] <= E.get(k, -1) < ENERGY_BINS[i + 1]])
            line += f"{nm} {np.mean(s > tf):.2f}|{np.mean(s > tn):.2f}  "
        print(line)
        line = f"  {'':30s} near/far FPR {np.mean(bn > tf):.2f}          | SNR: "
        for lo, hi in snr_bins:
            s = np.array([v for k, v in iso if k[:4] in ann and ann[k[:4]][1] and lo <= ann[k[:4]][0] < hi])
            lab = "nosig" if hi <= 1e-9 else f"{lo:g}-{hi:g}" if hi < 1e9 else f">{lo:g}"
            if len(s):
                line += f"{lab} {np.mean(s > tf):.2f}|{np.mean(s > tn):.2f}  "
        print(line)


if __name__ == "__main__":
    main()
