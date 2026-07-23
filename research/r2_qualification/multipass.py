"""Does MULTIPASS help the smart gate (R2)? R1 iterates 3x; smart is 1-pass.

Multipass smart (mirrors R1's logic in coefficient space): after gating, detect
signal on the CLEANED bands (large |coeff| vs block scale), accumulate a signal
mask, re-estimate the block common-mode from the ORIGINAL bands EXCLUDING the
accumulated signal wires, re-gate. Compare 1/2/3-pass smart vs R1 vs oracle.

Usage: python multipass.py [--events 40] [--k 3.0]
"""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, "/sdf/group/neutrino/omara/pimm-data/src")
sys.path.insert(0, os.path.join(HERE, "..", "coeff_foundation_model"))

import numpy as np
import torch
from helix.core import backend as _backend
from helix.core.wavelet_ops_torch import _wavedec, _waverec
from helix.tpc.io import config_from_file, read_sensor_plane, count_events
from helix.tpc.config import DetectorConfig
from helix.tpc.coherent import remove_coherent
from pimm_data.dense_ops import _coherent_torch, _incoherent_torch
from pimm_data.noise import (DEFAULT_ENC, DEFAULT_COH_RMS_ADC, DEFAULT_COH_CORNER_FREQ_HZ,
                             DEFAULT_COH_SLOPE, DEFAULT_COH_BETA, DEFAULT_SAMPLING_RATE_HZ, digitize)
from pimm_data.geometry import load_plane_registry
from pimm_data.jaxtpc import canonical_plane_id
import measure_coeffs as MC

SHARD = ("/sdf/data/neutrino/doraemon/wire_test_00_00_02/sensor/"
         "run_0027575715/sim_wire_sensor_0000.h5")
NPZ = "/sdf/group/neutrino/omara/JAXTPC/config/noise_spectrum.npz"
WAVELET, LEVEL, KAPPA, GS, DEV = "coif3", 4, 1.0, 64, "cuda"


def _block_common_mode(b, ksig, sigmask):
    """Per-(64-block, pos) ksig-masked mean over wires, EXCLUDING sigmask entries."""
    W, Lb = b.shape
    ngf = W // GS
    Ms = []
    if ngf > 0:
        bf = b[:ngf * GS].reshape(ngf, GS, Lb)
        smf = sigmask[:ngf * GS].reshape(ngf, GS, Lb)
        med = bf.quantile(0.5, dim=1)
        resid = bf - med.unsqueeze(1)
        sg = (resid.abs().reshape(ngf, -1).quantile(0.5, dim=1) / 0.6745).clamp_min(1e-6)
        uf = (resid.abs() <= ksig * sg[:, None, None]) & (~smf)
        nuf = uf.sum(1)
        Ms.append(torch.where(nuf > 0, (bf * uf).sum(1) / nuf.clamp_min(1), med))
    rem = W - ngf * GS
    if rem > 0:
        blk = b[ngf * GS:]; smr = sigmask[ngf * GS:]
        med = blk.quantile(0.5, dim=0)
        resid = blk - med
        sg = (resid.abs().reshape(-1).quantile(0.5) / 0.6745).clamp_min(1e-6)
        uf = (resid.abs() <= ksig * sg) & (~smr)
        nuf = uf.sum(0)
        Ms.append(torch.where(nuf > 0, (blk * uf).sum(0) / nuf.clamp_min(1), med).unsqueeze(0))
    return torch.cat(Ms, dim=0)


def smart_bands(bands, kgate, ksig=3.0, npass=1):
    out = []
    for b in bands:
        W, Lb = b.shape
        idx = (torch.arange(W, device=b.device) // GS).clamp(max=(W + GS - 1) // GS - 1)
        sigmask = torch.zeros_like(b, dtype=torch.bool)
        Mc_full = None
        for p in range(npass):
            M = _block_common_mode(b, ksig, sigmask)
            sigc = (M.abs().median() / 0.6745).clamp_min(1e-6)
            Mc = torch.where(M.abs() < kgate * sigc, M, torch.zeros_like(M))
            Mc_full = Mc[idx]
            if p < npass - 1:
                cleaned = b - Mc_full            # detect signal on cleaned bands
                cf = cleaned.abs()
                # per-block MAD scale of the cleaned band
                ngf = W // GS
                if ngf > 0:
                    cb = cf[:ngf * GS].reshape(ngf, GS, Lb)
                    csg = (cb.reshape(ngf, -1).quantile(0.5, dim=1) / 0.6745).clamp_min(1e-6)
                    det = cb > ksig * csg[:, None, None]
                    sigmask[:ngf * GS] |= det.reshape(ngf * GS, Lb)
                rem = W - ngf * GS
                if rem > 0:
                    cr = cf[ngf * GS:]
                    csg = (cr.reshape(-1).quantile(0.5) / 0.6745).clamp_min(1e-6)
                    sigmask[ngf * GS:] |= (cr > ksig * csg)
        out.append(b - Mc_full)
    return out


def gate_img(noisy_t, kgate, npass):
    nt = noisy_t.shape[-1]
    pad = (-nt) % (1 << LEVEL)
    x = torch.nn.functional.pad(noisy_t, (0, pad)) if pad else noisy_t
    gated = smart_bands(_wavedec(x, WAVELET, LEVEL), kgate, npass=npass)
    return _waverec(gated, WAVELET)[..., :nt]


def metrics(cleaned, signal, coherent, noisy):
    sig = signal.abs() > 0
    tc = signal.abs()[sig].sum().clamp_min(1e-9)
    W, T = cleaned.shape; nf = W // GS
    c = cleaned[:nf * GS].reshape(nf, GS, T).mean(dim=1)
    off = ~(signal[:nf * GS].reshape(nf, GS, T).abs() > 0).any(dim=1)
    stripe = float((c[off] ** 2).mean().sqrt()) if bool(off.any()) else float("nan")
    return dict(signal_lost=float((cleaned - signal).abs()[sig].sum() / tc),
                coh_left=float(((noisy - cleaned - coherent) ** 2).mean().sqrt()),
                ontrack_rms=float(((cleaned - signal)[sig] ** 2).mean().sqrt()),
                stripe=stripe)


def n_kept(cleaned):
    nt = cleaned.shape[-1]; pad = (-nt) % (1 << LEVEL)
    x = torch.nn.functional.pad(cleaned, (0, pad)) if pad else cleaned
    k = 0
    for b in _wavedec(x, WAVELET, LEVEL):
        sg = b.abs().median() / 0.6745
        k += int((b.abs() >= KAPPA * sg * (2 * np.log(max(b.shape[-1], 2))) ** 0.5).sum())
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=40)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--k", type=float, default=3.0)
    args = ap.parse_args()
    npz = np.load(NPZ, allow_pickle=True); spec = (npz["spectrum_freqs_hz"], npz["spectrum_shape"])
    reg = load_plane_registry("cubic_wireplane_geometry.json")
    cfg = config_from_file(SHARD); r1cfg = DetectorConfig(num_time_steps=cfg.num_time_steps)
    pool = min(args.pool, count_events(SHARD))
    print(f"scan {pool}...", flush=True)
    act = sorted((int((read_sensor_plane(SHARD, ev, "volume_0_U", cfg.num_time_steps,
                                         cfg.pedestals["U"]) != 0).sum()), ev) for ev in range(pool))
    order = [ev for _, ev in act]; k = args.events
    events = sorted(set(order[int(round(i * (len(order) - 1) / (k - 1)))] for i in range(k))
                    | set(order[-max(k // 5, 5):]))
    dense = set(order[-max(k // 5, 5):])
    print(f"{len(events)} events, k={args.k}", flush=True)
    ARMS = ["r1", "smart_1p", "smart_2p", "smart_3p", "oracle"]
    rows = []
    for i, ev in enumerate(events):
        for pl in ("U", "V", "Y"):
            lab = f"volume_0_{pl}"; ped = cfg.pedestals[pl]
            clean_np = read_sensor_plane(SHARD, ev, lab, cfg.num_time_steps, ped)
            nw, nt = clean_np.shape
            clean = torch.as_tensor(clean_np, device=DEV)
            v = np.asarray(reg.get(canonical_plane_id(lab), {}).get("wire_lengths", []), np.float64)
            wlt = torch.as_tensor(v if len(v) == nw else np.full(nw, 2.33), dtype=torch.float32, device=DEV)
            gen = torch.Generator(device=DEV); gen.manual_seed(hash((ev, pl)) & 0xFFFFFFFF)
            coh = _coherent_torch(nw, nt, gen=gen, group_size=GS, rms_adc=DEFAULT_COH_RMS_ADC,
                                  corner_freq_hz=DEFAULT_COH_CORNER_FREQ_HZ, spectral_slope=DEFAULT_COH_SLOPE,
                                  beta=DEFAULT_COH_BETA, sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ, device=DEV)
            inc = _incoherent_torch((nw, nt), wlt, gen=gen, enc=DEFAULT_ENC, series_spectrum=spec,
                                    sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ, device=DEV)
            noisy = digitize((clean + coh + inc).cpu().numpy(), ped)
            nocoh = digitize((clean + inc).cpu().numpy(), ped)
            noisy_t = torch.as_tensor(noisy, device=DEV)
            _backend.set_backend("jax")
            r1 = torch.as_tensor(np.asarray(remove_coherent(noisy, r1cfg)), device=DEV)
            _backend.set_backend("numpy")
            cl = {"r1": r1, "oracle": torch.as_tensor(nocoh, device=DEV),
                  "smart_1p": gate_img(noisy_t, args.k, 1),
                  "smart_2p": gate_img(noisy_t, args.k, 2),
                  "smart_3p": gate_img(noisy_t, args.k, 3)}
            for a in ARMS:
                r = dict(ev=ev, plane=pl, dense=(ev in dense), arm=a, n_kept=n_kept(cl[a]))
                r.update(metrics(cl[a], clean, coh, noisy_t))
                rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(events)}", flush=True)

    # diff smart_1p vs smart_2p vs smart_3p (is multipass a no-op?)
    d12 = np.mean([abs(next(r for r in rows if r["ev"] == e and r["plane"] == p and r["arm"] == "smart_1p")["signal_lost"]
                       - next(r for r in rows if r["ev"] == e and r["plane"] == p and r["arm"] == "smart_2p")["signal_lost"])
                  for e in events for p in ("U", "V", "Y")])
    print(f"\nmean |signal_lost(1p) - signal_lost(2p)| = {d12:.5f}  (0 = multipass is a no-op)")
    print(f"\n{'pl':2s} {'arm':9s} {'sig_lost%':>9s} {'coh_left':>8s} {'stripe':>7s} "
          f"{'ontrk_rms':>9s} {'coeffs':>7s}   dense sig_lost%")
    for pl in ("U", "V", "Y"):
        cells = collections.defaultdict(dict)
        for r in rows:
            if r["plane"] == pl:
                cells[r["ev"]][r["arm"]] = r
        evs = list(cells.values())
        for arm in ARMS:
            g = lambda key: np.mean([c[arm][key] for c in evs])
            dsl = np.mean([c[arm]["signal_lost"] for c in evs if c[arm]["dense"]]) * 100
            print(f"{pl:2s} {arm:9s} {g('signal_lost')*100:9.3f} {g('coh_left'):8.3f} "
                  f"{g('stripe'):7.3f} {g('ontrack_rms'):9.3f} {g('n_kept'):7.0f}   {dsl:8.3f}")
        print()


if __name__ == "__main__":
    main()
