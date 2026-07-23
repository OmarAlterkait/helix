"""R2 qualification RERUN — addresses every finding of the completeness audit.

Fixes vs qual.py (the n=12 pass):
  1. DENSITY-STRATIFIED events: scan a large event pool, rank by activity
     (nonzero count on U, the F0-limited plane), take an evenly-spaced sample
     across the occupancy range PLUS the densest tail (the known U failure
     mode). The n=12 tie was a regime average of an arbitrary ev0-11 sample.
  2. n >= 100 INDEPENDENT events (the statistical unit; 1 seed/event so cells
     are independent — no seed-clustering SE deflation).
  3. REPRODUCIBLE seeds: blake2b(int) not salted str-hash. results replayable.
  4. NEW ARMS closing catalog gaps: r2_soft (gate_soft clip variant, never
     measured anywhere) and de2_clamp (the record's opt-in refinement, ported
     self-contained — informational: confirms U non-win vs R1 on white noise).
  5. Kept counts labelled pywt-protocol (not the FM padded-torch pipeline).

Arms: raw, r1, r2_k3, r2_k3.5, r2_k4, r2np_k4, r2_soft_k4, de2_clamp, oracle.
Same faithful pimm-data injectors + verbatim smart.py metrics as qual.py.

Usage: python qual2.py [--events 120] [--pool 400] [--noise both]
"""
import argparse
import collections
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, "/sdf/group/neutrino/omara/pimm-data/src")

import numpy as np
import pywt
import torch
from scipy import ndimage

from helix.tpc.io import config_from_file, read_sensor_plane, count_events
from helix.tpc.config import DetectorConfig
from helix.tpc.coherent import remove_coherent
from helix.core.wavelet_ops_torch import _wavedec, _waverec

from pimm_data.noise import coherent_noise, incoherent_noise, digitize
from pimm_data.geometry import load_plane_registry
from pimm_data.jaxtpc import canonical_plane_id

sys.path.insert(0, os.path.join(HERE, "..", "coeff_foundation_model"))
import measure_coeffs as MC                                        # read-only

SHARD = ("/sdf/data/neutrino/doraemon/wire_test_00_00_02/sensor/"
         "run_0027575715/sim_wire_sensor_0000.h5")
NPZ = "/sdf/group/neutrino/omara/JAXTPC/config/noise_spectrum.npz"
PLANES = ("volume_0_U", "volume_0_V", "volume_0_Y")
WAVELET, LEVEL, MODE, KAPPA, GS = "coif3", 4, "periodization", 1.0, 64
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def rng_for(*key):
    h = hashlib.blake2b("|".join(map(str, key)).encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "little"))


# ── smart gate: numpy reference (verbatim smart.py) with hard/soft ──────────

def _block_common_mode(band, nw, ksig=3.0):
    nblk = (nw + GS - 1) // GS
    Mm = np.zeros((nblk, band.shape[-1]), np.float32)
    for g in range(nblk):
        lo, hi = g * GS, min((g + 1) * GS, nw)
        blk = band[lo:hi]
        med = np.median(blk, axis=0)
        resid = blk - med
        sg = max(np.median(np.abs(resid)) / 0.6745, 1e-6)
        uf = np.abs(resid) <= ksig * sg
        nuf = uf.sum(0)
        Mm[g] = np.where(nuf > 0, (blk * uf).sum(0) / np.maximum(nuf, 1), med)
    return Mm


def smart_removal_np(noisy, kgate=4.0, ksig=3.0, soft=False):
    nw, nt = noisy.shape
    bands = pywt.wavedec(noisy.astype(np.float32), WAVELET, level=LEVEL, mode=MODE, axis=-1)
    out = []
    for b in bands:
        Mm = _block_common_mode(b, nw, ksig)
        sigc = max(float(np.median(np.abs(Mm)) / 0.6745), 1e-6)
        t = kgate * sigc
        Mc = (np.sign(Mm) * np.minimum(np.abs(Mm), t)) if soft \
            else np.where(np.abs(Mm) < t, Mm, 0.0)
        idx = np.minimum(np.arange(nw) // GS, Mm.shape[0] - 1)
        out.append(b - Mc[idx])
    return pywt.waverec(out, WAVELET, mode=MODE, axis=-1)[..., :nt].astype(np.float32)


def smart_removal_torch(noisy, kgate):
    nw, nt = noisy.shape
    pad = (-nt) % (1 << LEVEL)
    x = torch.as_tensor(noisy, dtype=torch.float32, device=DEV)
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    gated = MC.smart_gate_bands(_wavedec(x, WAVELET, LEVEL), kgate=kgate)
    return _waverec(gated, WAVELET)[..., :nt].cpu().numpy()


# ── de2_clamp: self-contained port of induction.py final_removal ────────────

def _dilate_t(mask, ticks):
    return ndimage.maximum_filter1d(mask.view(np.uint8), size=ticks, axis=1).astype(bool) \
        if ticks > 1 else mask


def _smart_baseline(noisy):
    return noisy - smart_removal_np(noisy, kgate=4.0)          # coherent estimate, image space


def _mask_hysteresis(noisy, baseline, klo=0.7, khi=3.5, dilate=15):
    sigw = np.maximum(np.median(np.abs(noisy - np.median(noisy, axis=1, keepdims=True)),
                                axis=1) / 0.6745, 1e-6)[:, None]
    z = np.abs(noisy - baseline) / sigw
    lbl, n = ndimage.label(z > klo, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, bool)
    sl = np.unique(lbl[z > khi])
    keep[sl[sl > 0]] = True
    return _dilate_t(keep[lbl], dilate)


def _estimate(noisy, mask, minc=4):
    nw, nt = noisy.shape
    nblk = (nw + GS - 1) // GS
    xs = np.arange(nt)
    coh = np.empty_like(noisy)
    for g in range(nblk):
        lo, hi = g * GS, min((g + 1) * GS, nw)
        blk, uf = noisy[lo:hi], ~mask[lo:hi]
        nuf = uf.sum(0)
        m = (blk * uf).sum(0) / np.maximum(nuf, 1)
        rel = nuf >= minc
        if rel.sum() >= 2 and (~rel).any():
            m[~rel] = np.interp(xs[~rel], xs[rel], m[rel])
        elif rel.sum() < 2:
            m = np.median(blk, axis=0)
        coh[lo:hi] = m[None, :]
    return coh


def de2_clamp(noisy, clamp=4.0, n_iter=4, klo=0.7, khi=3.5, dilate=15):
    smc = _smart_baseline(noisy)
    coh = smc.copy()
    for _ in range(n_iter):
        coh = _estimate(noisy, _mask_hysteresis(noisy, coh, klo, khi, dilate))
    coh = np.clip(coh, smc - clamp, smc + clamp)
    return (noisy - coh).astype(np.float32)


# ── metrics (verbatim smart.py) + production threshold end-to-end ───────────

def removal_metrics(cleaned, signal, coherent, coh_hat):
    sig = np.abs(signal) > 0
    tc = float(np.abs(signal)[sig].sum())
    return dict(
        f0=1.0 - float(np.abs(cleaned - signal)[sig].sum()) / max(tc, 1e-9),
        nrms=float(np.sqrt(np.mean((cleaned - signal)[~sig] ** 2))),
        coh_left=float(np.sqrt(np.mean((coh_hat - coherent) ** 2))))


def end_to_end(cleaned, signal):
    bands = pywt.wavedec(cleaned.astype(np.float32), WAVELET, level=LEVEL, mode=MODE, axis=-1)
    kept = 0
    thr = []
    for b in bands:
        sg = float(np.median(np.abs(b)) / 0.6745)
        t = KAPPA * sg * np.sqrt(2.0 * np.log(max(b.shape[-1], 2)))
        m = np.abs(b) >= t
        thr.append(np.where(m, b, 0.0))
        kept += int(m.sum())
    recon = pywt.waverec(thr, WAVELET, mode=MODE, axis=-1)[..., :cleaned.shape[-1]]
    sig = np.abs(signal) > 0
    tc = float(np.abs(signal)[sig].sum())
    return dict(f0_recon=1.0 - float(np.abs(recon - signal)[sig].sum()) / max(tc, 1e-9),
                n_kept_pywt=kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=120)
    ap.add_argument("--pool", type=int, default=None)     # scan pool for stratification
    ap.add_argument("--noise", choices=("colored", "white", "both"), default="both")
    ap.add_argument("--out", default=os.path.join(HERE, "results2.jsonl"))
    args = ap.parse_args()

    npz = np.load(NPZ, allow_pickle=True)
    spectrum = (npz["spectrum_freqs_hz"], npz["spectrum_shape"])
    registry = load_plane_registry("cubic_wireplane_geometry.json")
    cfg = config_from_file(SHARD)
    r1cfg = DetectorConfig(num_time_steps=cfg.num_time_steps)
    n_avail = count_events(SHARD)
    pool = min(args.pool or max(args.events * 3, 200), n_avail)
    noise_models = ({"colored": spectrum, "white": None} if args.noise == "both"
                    else {args.noise: {"colored": spectrum, "white": None}[args.noise]})

    def wl(label, nw):
        gid = canonical_plane_id(label)
        e = registry.get(gid, {})
        v = np.asarray(e.get("wire_lengths", []), np.float64)
        return v if len(v) == nw else np.full(nw, 2.33)

    # ── stratify: rank pool events by U activity, sample across the range + dense tail
    print(f"scanning {pool}/{n_avail} events for U-activity stratification...", flush=True)
    act = []
    for ev in range(pool):
        u = read_sensor_plane(SHARD, ev, "volume_0_U", cfg.num_time_steps,
                              cfg.pedestals["U"])
        act.append((int((u != 0).sum()), ev))
    act.sort()
    order = [ev for _, ev in act]
    k = args.events
    spaced = [order[int(round(i * (len(order) - 1) / (k - 1)))]
              for i in range(k)]                      # even span low->high occupancy
    dense_tail = order[-max(k // 5, 5):]              # densest 20% (the U failure mode)
    events = sorted(set(spaced) | set(dense_tail))
    dense_set = set(dense_tail)
    print(f"selected {len(events)} events (U nonzero range "
          f"{act[0][0]}..{act[-1][0]}; {len(dense_set)} in dense tail)", flush=True)

    rows = []
    for i, ev in enumerate(events):
        for label in PLANES:
            pt = label.split("_")[-1]
            ped = cfg.pedestals[pt]
            clean = read_sensor_plane(SHARD, ev, label, cfg.num_time_steps, ped)
            nw, nt = clean.shape
            wire_len = wl(label, nw)
            for nm, spec in noise_models.items():
                rng = rng_for(ev, label, nm)
                coh = coherent_noise(nw, nt, rng)
                inc = incoherent_noise((nw, nt), wire_len, rng, series_spectrum=spec)
                noisy = digitize(clean + coh + inc, ped)
                nocoh = digitize(clean + inc, ped)
                arms = {"raw": noisy, "r1": np.asarray(remove_coherent(noisy, r1cfg)),
                        "r2_k3": smart_removal_torch(noisy, 3.0),
                        "r2_k3.5": smart_removal_torch(noisy, 3.5),
                        "r2_k4": smart_removal_torch(noisy, 4.0),
                        "r2np_k4": smart_removal_np(noisy, 4.0),
                        "r2_soft_k4": smart_removal_np(noisy, 4.0, soft=True),
                        "de2_clamp": de2_clamp(noisy),
                        "oracle": nocoh}
                for arm, cl in arms.items():
                    r = dict(ev=ev, plane=pt, noise=nm, arm=arm,
                             dense=(ev in dense_set), u_nz=int((clean != 0).sum()))
                    r.update(removal_metrics(cl, clean, coh, noisy - cl))
                    r.update(end_to_end(cl, clean))
                    rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(events)} events ({len(rows)} rows)", flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # aggregate: mean F0 ± clustered SE (per event), win-rate vs r1, kept/oracle
    print(f"\n{'noise':7s} {'pl':2s} {'arm':11s} {'F0':>7s}±SE   {'coh_l':>6s} "
          f"{'kept/orc':>8s} {'win>r1':>7s}  (dense-only F0 / win)")
    for nm in noise_models:
        for pl in ("U", "V", "Y"):
            cells = collections.defaultdict(dict)
            for r in rows:
                if r["noise"] == nm and r["plane"] == pl:
                    cells[r["ev"]][r["arm"]] = r
            evs = list(cells.values())
            for arm in ("r1", "r2_k3", "r2_k3.5", "r2_k4", "r2_soft_k4", "de2_clamp", "oracle"):
                f0 = np.array([c[arm]["f0"] for c in evs])
                cl = np.mean([c[arm]["coh_left"] for c in evs])
                ko = np.mean([c[arm]["n_kept_pywt"] for c in evs]) / \
                    np.mean([c["oracle"]["n_kept_pywt"] for c in evs])
                win = np.mean([c[arm]["f0"] > c["r1"]["f0"] for c in evs]) if arm != "r1" else np.nan
                dmask = [c for c in evs if c[arm]["dense"]]
                df0 = np.mean([c[arm]["f0"] for c in dmask]) if dmask else float("nan")
                dwin = (np.mean([c[arm]["f0"] > c["r1"]["f0"] for c in dmask])
                        if dmask and arm != "r1" else float("nan"))
                print(f"{nm:7s} {pl:2s} {arm:11s} {f0.mean():7.4f}±{f0.std(ddof=1)/np.sqrt(len(f0)):.4f} "
                      f"{cl:6.3f} {ko:8.3f} {win:7.2f}   {df0:7.4f} / {dwin:.2f}")
        print()


if __name__ == "__main__":
    main()
