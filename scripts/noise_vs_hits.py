#!/usr/bin/env python3
"""First principles: what is `hits` in, how big is the noise next to it, and how
close does a denoiser get to the best any detector could do.

    python scripts/noise_vs_hits.py --truth <truth_v2> [--windows <eval_denoise windows json>] [--events 128]

For every isolated-deposit window of the floor evaluation's test events, on the
deposit's OWN region (the wires and ticks of its hits cells, +-2 wires, +-32 ticks):
  units    q / E: hits charge per MeV the particle deposited (step `de`).
           Electrons give 1/W_ion x recombination = 42,400 x ~0.5-0.7 per MeV.
  gain     peak |clean ADC| per 1000 e-; ENC = noise sigma / that.
  zs       the clean sensor is stored as integer ADC with |v| < 2 suppressed:
           the fraction of deposits with NO clean signal left at all.
  SNR      whitened matched filter of the deposit's own clean waveform against the
           forward noise model's exact spectrum, SNR^2 = sum_w sum_f |S(f)|^2/P(f).
           Coherent noise taken as perfectly removed and the waveform shape as
           known: an upper bound for any detector given this sensor image.
  z99      that ideal detector's 1%-FPR threshold when it searches a 32 x 128
           window (max over placements of its whitened template on pure noise).
  ideal    efficiency at 1% FPR = mean Phi(SNR - z99), against a trained model's
           window detections at its own 1%-FPR threshold, window by window.
Windows with foreign charge (other particles) above 25% of their own within +-8
wires / +-64 ticks of the own region are excluded: their clean image is not theirs.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PL = "UVY"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--truth", required=True)
    ap.add_argument("--windows", default=None, help="eval_denoise *_windows.json of a model to compare")
    ap.add_argument("--events", type=int, default=128)
    ap.add_argument("--source", default="/global/cfs/cdirs/m5238/users/oalter/wire_test_00_00_02")
    ap.add_argument("--geom", default="cubic_wireplane_geometry.json")
    ap.add_argument("--foreign", type=float, default=0.25)
    ap.add_argument("--out", default=None, help="per-window table (.npz)")
    a = ap.parse_args()

    import h5py
    from scipy.stats import norm
    from helix.core.coeff_io import read_coeff_event
    from helix.probe import resolution as R
    from helix.tpc.geometry import load_plane_registry
    from helix.tpc.io import read_sensor_plane
    from helix.tpc.noise import DEFAULT_ENC, DEFAULT_SAMPLING_RATE_HZ, _series_spectrum_shape
    from scripts.build_denoise_truth import event_cells

    reg = load_plane_registry(a.geom)
    X2 = DEFAULT_ENC[0] ** 2 + 1 / 12                   # white + digitisation (1 ADC LSB)
    TS = 512                                            # segment length (ticks)
    spec = _series_spectrum_shape(TS, "microboone", DEFAULT_SAMPLING_RATE_HZ) ** 2
    wts = np.full(TS // 2 + 1, 2.0); wts[0] = wts[-1] = 1.0
    spec_full = (wts * spec).sum()                      # over the two-sided spectrum

    def psd(L):
        """E|N_k|^2 of numpy's rfft (length TS) for one wire of length L metres."""
        s = DEFAULT_ENC[1] + DEFAULT_ENC[2] * L
        return TS * X2 + s ** 2 * TS ** 2 * spec / spec_full

    rng = np.random.default_rng(0)
    files = sorted(glob.glob(os.path.join(a.truth, "ev*.npz")))
    test = [f for f in files if int(os.path.basename(f)[2:5]) >= 260][:a.events]
    rows, templates = [], {0: [], 1: [], 2: []}
    for f in test:
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        shard, event = str(z["shard"]), int(z["event"])
        with h5py.File(shard, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], event))
        ce = read_coeff_event(shard, pos)
        run = os.path.basename(os.path.dirname(shard))
        sensor = os.path.join(a.source, "sensor", run, ce.source_file)
        tk, tq = event_cells(os.path.join(a.source, "hits", run, ce.source_file.replace("_sensor_", "_hits_")), ce.event)
        tg, tw, tt = R.unkey(tk)
        planes, ev = {}, int(os.path.basename(f)[2:5])
        for m in meta:
            if m["kind"] != "iso":
                continue
            g, w0, t0 = int(m["g"]), int(m["w0"]), int(m["t0"])
            inw = (tg == g) & (tw >= w0 // R.FW) & (tw < (w0 + R.WW) // R.FW) & (tt >= t0 // R.FT) & (tt < (t0 + R.WT) // R.FT)
            if not inw.any():
                continue
            if g not in planes:
                planes[g] = np.asarray(read_sensor_plane(sensor, ce.event, reg[g]["label"]), np.float64)
            img = planes[g]; nw, nt = img.shape
            q = float(tq[inw].sum()); E = float(np.sum(m["E"]))
            wa, wb = max(tw[inw].min() * R.FW - 2, 0), min((tw[inw].max() + 1) * R.FW + 2, nw)
            ta, tb = max(tt[inw].min() * R.FT - 32, 0), min((tt[inw].max() + 1) * R.FT + 32, nt)
            tb = min(tb, ta + TS)
            near = (tg == g) & (tw >= (wa - 8) // R.FW) & (tw < (wb + 8) // R.FW + 1) \
                & (tt >= (ta - 64) // R.FT) & (tt < (tb + 64) // R.FT + 1)
            foreign = float(tq[near].sum()) - q
            S = np.zeros((wb - wa, TS)); S[:, :tb - ta] = img[wa:wb, ta:tb]
            L = np.asarray(reg[g]["wire_lengths"])[wa:wb]
            Sk = np.fft.rfft(S, axis=1)
            P = np.stack([psd(l) for l in L])
            snr = float(np.sqrt((wts * np.abs(Sk) ** 2 / P).sum()))
            sig = float(np.sqrt(X2 + (DEFAULT_ENC[1] + DEFAULT_ENC[2] * L.mean()) ** 2))
            rows.append(dict(ev=ev, g=g, w0=w0, t0=t0, E=E, q=q, foreign=foreign, snr=snr,
                             peak=float(np.abs(S).max()), sig=sig))
            if foreign <= a.foreign * q and S.any() and len(templates[g % 3]) < 24 and 0.05 < E < 0.5:
                templates[g % 3].append(Sk / np.sqrt(P))

    # the ideal detector's 1%-FPR threshold: whitened template, unknown placement in the window
    z99 = {}
    for p, tl in templates.items():
        zs = []
        for wk in tl:
            tmpl = np.fft.irfft(wk, n=TS, axis=1); tmpl /= np.sqrt((tmpl ** 2).sum())
            H, Wd = tmpl.shape[0] + R.WW, TS + R.WT                   # room for every placement
            F = np.fft.rfft2(tmpl, s=(H, Wd)).conj()
            mx = [np.fft.irfft2(np.fft.rfft2(rng.standard_normal((H, Wd))) * F, s=(H, Wd))[:R.WW, :R.WT].max()
                  for _ in range(200)]
            zs.append(np.quantile(mx, 0.99))
        z99[p] = float(np.median(zs)) if zs else 4.3
        print(f"[z99] {PL[p]}: ideal detector's 1%-FPR threshold {z99[p]:.2f} sigma "
              f"({len(zs)} templates, {np.min(zs) if zs else 0:.2f}-{np.max(zs) if zs else 0:.2f})")

    T = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    p3 = T["g"] % 3
    T["ideal"] = norm.cdf(T["snr"] - np.array([z99[int(x)] for x in p3]))
    if a.windows:
        W = json.load(open(a.windows))
        thr = np.quantile([w["score"] for w in W if w["kind"] == "bg"], 0.99)
        sc = {(w["ev"], w["g"], w["w0"], w["t0"]): w["score"] for w in W if w["kind"] == "iso"}
        T["model"] = np.array([float(sc[k] > thr) if k in sc else np.nan
                               for k in zip(T["ev"], T["g"], T["w0"], T["t0"])])
        print(f"[model] {os.path.basename(a.windows)}: 1%-FPR threshold {thr:.0f} e-")

    clean = (T["foreign"] <= a.foreign * T["q"]) & (T["E"] > 0)
    print(f"\n{len(T['g'])} isolated deposits in {len(test)} test events; {clean.sum()} with foreign charge "
          f"<= {a.foreign:.0%} of their own (used below)")
    eb = np.searchsorted(R.ENERGY_BINS, T["E"], side="right") - 1
    print("\n== units: hits charge per MeV deposited (electrons: 42,400/MeV x recombination ~0.5-0.7)")
    for p in range(3):
        m = clean & (p3 == p)
        r = T["q"][m] / T["E"][m]
        print(f"  {PL[p]}: median {np.median(r):,.0f} e-/MeV (IQR {np.percentile(r, 25):,.0f}-{np.percentile(r, 75):,.0f}, n={m.sum()})")
    print("\n== gain and noise (deposits 0.1-0.5 MeV with clean signal)")
    for p in range(3):
        m = clean & (p3 == p) & (T["E"] >= 0.1) & (T["E"] < 0.5) & (T["peak"] > 0)
        gpk = np.median(T["peak"][m] / T["q"][m])
        print(f"  {PL[p]}: peak {1000 * gpk:.2f} ADC per 1000 e- (n={m.sum()}); noise sigma {np.median(T['sig'][m]):.2f} ADC/tick "
              f"-> ENC {np.median(T['sig'][m]) / gpk:,.0f} e-; charge at optimal SNR 1: {np.median(T['q'][m] / T['snr'][m]):,.0f} e-")
    names = ["<0.1", "0.1-0.2", "0.2-0.5", "0.5-1", ">1"]
    print("\n== per energy bin [U / V / Y]: median charge, zero-suppressed (no clean signal), optimal SNR, "
          "ideal and model 1%-FPR efficiency")
    for i, nm in enumerate(names):
        m = clean & (eb == i)
        if not m.any():
            continue
        f3 = lambda v, fmt: " / ".join(fmt.format(v(m & (p3 == p))) if (m & (p3 == p)).any() else "-" for p in range(3))
        line = (f"  {nm:>7} MeV n={m.sum():4d}: q {np.median(T['q'][m]):6,.0f} e-; no signal {np.mean(T['peak'][m] == 0):.2f} "
                f"[{f3(lambda k: np.mean(T['peak'][k] == 0), '{:.2f}')}]; SNR {np.median(T['snr'][m]):5.1f} "
                f"[{f3(lambda k: np.median(T['snr'][k]), '{:.1f}')}]; ideal {T['ideal'][m].mean():.2f} "
                f"[{f3(lambda k: T['ideal'][k].mean(), '{:.2f}')}]")
        if "model" in T:
            mm = m & np.isfinite(T["model"])
            line += f"; model {T['model'][mm].mean():.2f} [{f3(lambda k: np.nanmean(T['model'][k & mm]), '{:.2f}')}]"
        print(line)
    if "model" in T:
        print("\n== efficiency vs the deposit's optimal SNR (clean windows)")
        for lo, hi in ((0, 1e-9), (1e-9, 3), (3, 5), (5, 7), (7, 10), (10, 15), (15, 25), (25, 1e9)):
            m = clean & (T["snr"] >= lo) & (T["snr"] < hi) & np.isfinite(T["model"])
            if m.sum() >= 5:
                lab = "no signal" if hi <= 1e-9 else f"{lo:g}-{hi if hi < 1e9 else 'inf'}"
                print(f"  SNR {lab:>9}: n={m.sum():4d}  ideal {T['ideal'][m].mean():.2f}  model {T['model'][m].mean():.2f}  "
                      f"median E {np.median(T['E'][m]):.3f} MeV, q {np.median(T['q'][m]):,.0f} e-")
    if a.out:
        np.savez(a.out, z99=json.dumps(z99), **T)


if __name__ == "__main__":
    main()
