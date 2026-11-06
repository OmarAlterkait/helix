#!/usr/bin/env python3
"""Per isolated deposit (every 'iso' window of the test events): the known-location
matched-filter evidence on each input a detector can read.

    scripts/helix_run.sh python baselines/uboone_sp/dense/snr_deposits.py --events 260-387 \
        --out /global/cfs/cdirs/m5238/users/oalter/helix_runs/causes2/snr

Statistic (scripts/snr_stages.py): the deposit's own clean template (the stored
clean sensor on the deposit's wires/ticks, +-2 wires, +-32 ticks), DWT'd; a
band-whitened linear matched filter  stat = sum_b w_b <T_b, C_b> / sqrt(sum_b w_b |T_b|^2),
w_b = 1 / s_b^2 with s_b the per-band std of the gated NOISE-ONLY planes; D1 weight 0
for the corpus inputs (the tokenizer reads A4, D4, D3, D2 only).

Inputs, per plane:
  gate     the classical chain's input: digitize(clean + incoherent + coherent) with
           v2/classical.py's noise draw (seed 1000+ev, same stream order), coherent gate
  k1       the same realisation through the corpus threshold (kappa 1, universal, per
           band) with D1 dropped: what the model WOULD read on the classical noise draw
           (= the input of model_dump_cn.py up to the A100-vs-2080Ti DSP difference)
  corp     the STORED corpus coefficients of this event (A4-D2): exactly what the model read
Null: the same statistic at NSHIFT random CHARGE-FREE placements of the template in
the same plane of the same input (no truth charge within the shifted box +-1 cell),
so each input's null includes its own noise, thresholding and near-activity leakage.
Also the noise-only response: resp_<s> = stat(signal+noise) - stat(noise only, same
realisation) for gate and k1 (signal surviving each stage).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.dirname(BASE)))
TRUTH = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
PTRUTH = "/pscratch/sd/o/oalter/helix_work/uboone_sp/dense/ptruth"
PS_CORPUS, CFS_CORPUS = "/pscratch/sd/o/oalter/coeff_tpc_r1", "/global/cfs/cdirs/m5238/users/oalter/coeff_tpc_r1"
NSHIFT = 40
FW, FT, WW, WT = 2, 16, 32, 128


def process(job):
    ev, out_dir = job
    out = os.path.join(out_dir, f"ev{ev:03d}.json")
    if os.path.exists(out):
        return ev, "exists"
    from helix.core import backend
    backend.set_backend("numpy")
    import h5py
    from helix.core.coeff_io import read_coeff_event
    from helix.core.wavelet import ThresholdSpec, threshold_bands, wavedec
    from helix.tpc.coherent_gate import coherent_gate
    from helix.tpc.config import DetectorConfig
    from helix.tpc.noise import digitize, generate_noise
    from sp.data import Event, registry

    cfg = DetectorConfig()
    reg = registry()
    E = Event(ev)
    z = np.load(os.path.join(TRUTH, f"ev{ev:03d}.npz"), allow_pickle=True)
    meta = [m for m in json.loads(str(z["meta"])) if m["kind"] == "iso"]
    PT = np.load(os.path.join(PTRUTH, f"ev{ev:03d}.npz"))
    shard = str(z["shard"]).replace(PS_CORPUS, CFS_CORPUS)
    with h5py.File(shard, "r") as fh:
        pos = int(np.searchsorted(fh["ident"]["event"][:], int(z["event"])))
    st = read_coeff_event(shard, pos)
    sb_, sg_, sw_, stau_, sv_ = (np.asarray(x) for x in (st.band, st.plane_gid, st.wire, st.tau, st.value))
    level = cfg.dwt_level
    npad = lambda x: np.pad(x, ((0, 0), (0, (-x.shape[1]) % (1 << level))))
    dec = lambda x: [np.asarray(b) for b in wavedec(npad(np.asarray(x, np.float32)), wavelet=cfg.wavelet,
                                                     level=level, mode=cfg.dwt_mode)[0]]
    spec = ThresholdSpec(method="universal", func=cfg.threshold_mode, scale=1.0, per_band_sigma=True,
                         threshold_approx=cfg.threshold_include_approx)

    def gate(img):
        return [np.asarray(b) for b in coherent_gate(dec(img), group_size=cfg.group_size, kgate=cfg.gate_kgate,
                                                     ksig=cfg.gate_ksig, npass=cfg.gate_npass, tau=cfg.gate_tau,
                                                     gate_approx=True)]

    def k1(G):
        t = [np.asarray(b) for b in threshold_bands(G, spec)[0]]
        t[-1] = np.zeros_like(t[-1])
        return t

    rng = np.random.default_rng(1000 + ev)
    rows = []
    srng = np.random.default_rng(5000 + ev)
    for g in range(6):                                   # noise stream exactly as v2/classical.py
        nw_, nt_ = reg[g]["n_wires"], reg[g]["n_ticks"]
        n_inc = generate_noise((nw_, nt_), rng=rng, wire_lengths_m=np.asarray(reg[g]["wire_lengths"]),
                               incoherent=True, coherent=False)
        n_coh = generate_noise((nw_, nt_), rng=rng, incoherent=False, coherent=True, group_size=64)
        deps = [m for m in meta if m["g"] == g]
        if not deps:
            continue
        clean = E.clean(g, reg)
        nw, nt = clean.shape
        ped = reg[g]["pedestal"]
        G = gate(digitize(clean + n_inc + n_coh, ped))
        G0 = gate(digitize(n_inc + n_coh, ped))
        K, K0 = k1(G), k1(G0)
        S = [np.zeros_like(b) for b in G]
        m = (sg_ == g) & (sb_ < 4)
        for b in range(4):
            mb = m & (sb_ == b)
            S[b][sw_[mb], stau_[mb]] = sv_[mb]
        s_b = [float(np.std(b)) for b in G0]
        ins = dict(gate=(G, s_b, False), k1=(K, s_b, True), corp=(S, s_b, True))
        # truth occupancy (fine cells) for charge-free null placements
        ncol = -(-nt // FT); nwc = -(-nw // FW)
        occ = np.zeros(nwc * ncol, np.float64); occ[PT[f"c_{g}"]] = 1.0
        occ = occ.reshape(nwc, ncol)
        SAT = np.pad(occ, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

        def free(w_lo, w_hi, t_lo, t_hi):          # pixel box -> no charge in its cells +-1
            a, b_ = max(w_lo // FW - 1, 0), min((w_hi - 1) // FW + 2, nwc)
            c, d = max(t_lo // FT - 1, 0), min((t_hi - 1) // FT + 2, ncol)
            return SAT[b_, d] - SAT[a, d] - SAT[b_, c] + SAT[a, c] == 0

        cw, ct = PT[f"c_{g}"] // ncol, PT[f"c_{g}"] % ncol
        for d in deps:
            inw = (cw >= d["w0"] // FW) & (cw < (d["w0"] + WW) // FW) & (ct >= d["t0"] // FT) & (ct < (d["t0"] + WT) // FT)
            if not inw.any():
                continue
            wa, wb = max(cw[inw].min() * FW - 2, 0), min((cw[inw].max() + 1) * FW + 2, nw)
            ta, tb = max(ct[inw].min() * FT - 32, 0), min((ct[inw].max() + 1) * FT + 32, nt)
            own = np.zeros((wb - wa, nt), np.float32); own[:, ta:tb] = clean[wa:wb, ta:tb]
            rec = dict(ev=ev, g=g, w0=d["w0"], t0=d["t0"], E=float(d["E"][0]) if isinstance(d["E"], list) else float(d["E"]),
                       wa=int(wa), wb=int(wb), ta=int(ta), tb=int(tb), own_max=float(np.abs(own).max()))
            if not own.any():
                rows.append(rec); continue
            T = dec(own)
            shifts, tries = [], 0
            while len(shifts) < NSHIFT and tries < 4000:
                tries += 1
                dw, mm = int(srng.integers(-600, 601)), int(srng.integers(-120, 121))
                if 0 <= wa + dw and wb + dw <= nw and 0 <= ta + 16 * mm and tb + 16 * mm <= nt \
                        and free(wa + dw, wb + dw, ta + 16 * mm, tb + 16 * mm):
                    shifts.append((dw, mm))
            rec["n_null"] = len(shifts)
            for name, (C, sb, noD1) in ins.items():
                w = [1.0 / max(x, 1e-6) ** 2 for x in sb]
                if noD1:
                    w[-1] = 0.0
                norm = np.sqrt(sum(wi * float((Tb ** 2).sum()) for wi, Tb in zip(w, T)))

                def stat(Cx, dw, mm):
                    return sum(wi * float((np.roll(Tb, mm * (Tb.shape[1] // T[0].shape[1]), axis=1)
                                           * Cx[wa + dw:wb + dw]).sum()) for wi, Tb, Cx in zip(w, T, Cx)) / norm

                rec[f"s_{name}"] = stat(C, 0, 0)
                rec[f"null_{name}"] = [stat(C, dw, mm) for dw, mm in shifts]
                if name == "gate":
                    rec["resp_gate"] = rec["s_gate"] - stat(G0, 0, 0)
                if name == "k1":
                    rec["resp_k1"] = rec["s_k1"] - stat(K0, 0, 0)
            rows.append(rec)
    json.dump(rows, open(out + ".tmp", "w"))
    os.replace(out + ".tmp", out)
    return ev, f"{len(rows)} deposits"


def main():
    sys.path.insert(0, os.path.join(BASE, "v2"))
    from classical import parse_events
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", default="260-387")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.makedirs(a.out, exist_ok=True)
    from multiprocessing import Pool
    with Pool(a.workers, maxtasksperchild=2) as pool:
        for ev, msg in pool.imap_unordered(process, [(e, a.out) for e in parse_events(a.events)]):
            print(f"ev{ev:03d}: {msg}", flush=True)


if __name__ == "__main__":
    main()
