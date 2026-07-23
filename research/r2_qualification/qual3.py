"""R2 qualification — GPU-optimized decisive run (raw/r1/r2_k{3,3.5,4}/oracle).

~50x faster than qual2.py by using the hardware that was idle:
  - noise on GPU (pimm_data.dense_ops torch coherent+incoherent): 1 ms vs 1130 ms
  - R1 multipass on GPU via helix jax backend: 48 ms vs 870 ms (bit-matches numpy)
  - threshold/kept/F0_recon batched on GPU (one _wavedec over all arms): vs 6.5 s pywt
  - gate = canonical measure_coeffs.smart_gate_bands (torch), the packaging target
Metrics = verbatim smart.py formulas, computed as torch reductions.

Faithfulness note: GPU noise is the statistical-parity torch port (same model,
different RNG stream than qual.py's numpy). For a Monte-Carlo over n>=100 noise
realizations this is scientifically equivalent; it is NOT bit-comparable to
qual.py's numbers. The paired design (one noisy image to all arms) is preserved.
de2_clamp / gate_soft are informational, not decision-gating — see qual2.py.

Stratified by U-plane activity (quiet..busy + dense tail = the U failure mode).
Usage: python qual3.py [--events 120] [--pool 200] [--noise both]
"""
import argparse
import collections
import json
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
from pimm_data.noise import (DEFAULT_ENC, DEFAULT_COH_RMS_ADC,
                             DEFAULT_COH_CORNER_FREQ_HZ, DEFAULT_COH_SLOPE,
                             DEFAULT_COH_BETA, DEFAULT_SAMPLING_RATE_HZ, digitize)
from pimm_data.geometry import load_plane_registry
from pimm_data.jaxtpc import canonical_plane_id
import measure_coeffs as MC

SHARD = ("/sdf/data/neutrino/doraemon/wire_test_00_00_02/sensor/"
         "run_0027575715/sim_wire_sensor_0000.h5")
NPZ = "/sdf/group/neutrino/omara/JAXTPC/config/noise_spectrum.npz"
PLANES = ("volume_0_U", "volume_0_V", "volume_0_Y")
WAVELET, LEVEL, KAPPA, GS = "coif3", 4, 1.0, 64
DEV = "cuda"


def gpu_noise(nw, nt, wire_len_t, spec, gen):
    coh = _coherent_torch(nw, nt, gen=gen, group_size=GS, rms_adc=DEFAULT_COH_RMS_ADC,
                          corner_freq_hz=DEFAULT_COH_CORNER_FREQ_HZ,
                          spectral_slope=DEFAULT_COH_SLOPE, beta=DEFAULT_COH_BETA,
                          sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ, device=DEV)
    inc = _incoherent_torch((nw, nt), wire_len_t, gen=gen, enc=DEFAULT_ENC,
                            series_spectrum=spec, sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ,
                            device=DEV)
    return coh, inc


def gate_torch(noisy_t, kgate):
    nt = noisy_t.shape[-1]
    pad = (-nt) % (1 << LEVEL)
    x = torch.nn.functional.pad(noisy_t, (0, pad)) if pad else noisy_t
    gated = MC.smart_gate_bands(_wavedec(x, WAVELET, LEVEL), kgate=kgate)
    return _waverec(gated, WAVELET)[..., :nt]


def removal_f0_coh(cleaned, signal, coherent, noisy):
    """(f0, nrms, coh_left) — verbatim smart.py::metrics, torch."""
    sig = signal.abs() > 0
    tc = signal.abs()[sig].sum().clamp_min(1e-9)
    f0 = 1.0 - (cleaned - signal).abs()[sig].sum() / tc
    nrms = ((cleaned - signal)[~sig] ** 2).mean().sqrt()
    coh_hat = noisy - cleaned
    cl = ((coh_hat - coherent) ** 2).mean().sqrt()
    return float(f0), float(nrms), float(cl)


def threshold_batched(cleaned_stack, signal):
    """cleaned_stack (A, W, T) -> per-arm (n_kept, f0_recon) via one batched GPU DWT.
    Production threshold: per-band MAD sigma, kappa=1, all bands. Matches prod_threshold."""
    A, W, T = cleaned_stack.shape
    pad = (-T) % (1 << LEVEL)
    x = torch.nn.functional.pad(cleaned_stack, (0, pad)) if pad else cleaned_stack
    bands = _wavedec(x, WAVELET, LEVEL)                # list of (A, W, Lb)
    kept = torch.zeros(A, device=DEV)
    thr = []
    for b in bands:
        Lb = b.shape[-1]
        sg = b.reshape(A, -1).abs().median(dim=1).values / 0.6745   # per-arm MAD
        t = (KAPPA * sg * (2.0 * np.log(max(Lb, 2))) ** 0.5).view(A, 1, 1)
        m = b.abs() >= t
        kept += m.reshape(A, -1).sum(1)
        thr.append(torch.where(m, b, torch.zeros_like(b)))
    recon = _waverec(thr, WAVELET)[..., :T]
    sig = signal.abs() > 0
    tc = signal.abs()[sig].sum().clamp_min(1e-9)
    f0r = [float(1.0 - (recon[a] - signal).abs()[sig].sum() / tc) for a in range(A)]
    return kept.cpu().numpy(), np.array(f0r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=120)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--noise", choices=("colored", "white", "both"), default="both")
    ap.add_argument("--out", default=os.path.join(HERE, "results3.jsonl"))
    args = ap.parse_args()

    npz = np.load(NPZ, allow_pickle=True)
    spec_np = (npz["spectrum_freqs_hz"], npz["spectrum_shape"])
    registry = load_plane_registry("cubic_wireplane_geometry.json")
    cfg = config_from_file(SHARD)
    r1cfg = DetectorConfig(num_time_steps=cfg.num_time_steps)
    n_avail = count_events(SHARD)
    pool = min(args.pool, n_avail)
    nmodels = ({"colored": spec_np, "white": None} if args.noise == "both"
               else {args.noise: {"colored": spec_np, "white": None}[args.noise]})

    def wl_t(label, nw):
        e = registry.get(canonical_plane_id(label), {})
        v = np.asarray(e.get("wire_lengths", []), np.float64)
        v = v if len(v) == nw else np.full(nw, 2.33)
        return torch.as_tensor(v, dtype=torch.float32, device=DEV)

    # stratify by U activity
    print(f"scan {pool}/{n_avail} for U-activity...", flush=True)
    act = sorted((int((read_sensor_plane(SHARD, ev, "volume_0_U", cfg.num_time_steps,
                                         cfg.pedestals["U"]) != 0).sum()), ev)
                 for ev in range(pool))
    order = [ev for _, ev in act]
    k = args.events
    spaced = {order[int(round(i * (len(order) - 1) / (k - 1)))] for i in range(k)}
    dense = set(order[-max(k // 5, 5):])
    events = sorted(spaced | dense)
    print(f"{len(events)} events (U-nz {act[0][0]}..{act[-1][0]}; {len(dense)} dense)", flush=True)

    ARMS = ["raw", "r1", "r2_k3", "r2_k3.5", "r2_k4", "oracle"]
    rows = []
    for i, ev in enumerate(events):
        for label in PLANES:
            pt = label.split("_")[-1]
            ped = cfg.pedestals[pt]
            clean_np = read_sensor_plane(SHARD, ev, label, cfg.num_time_steps, ped)
            nw, nt = clean_np.shape
            clean = torch.as_tensor(clean_np, device=DEV)
            wlt = wl_t(label, nw)
            for nm, spec in nmodels.items():
                gen = torch.Generator(device=DEV)
                gen.manual_seed((hash((ev, pt, nm)) & 0xFFFFFFFF))
                coh, inc = gpu_noise(nw, nt, wlt, spec, gen)
                noisy = digitize((clean + coh + inc).cpu().numpy(), ped)
                nocoh = digitize((clean + inc).cpu().numpy(), ped)
                noisy_t = torch.as_tensor(noisy, device=DEV)
                # removals
                _backend.set_backend("jax")
                r1 = torch.as_tensor(np.asarray(remove_coherent(noisy, r1cfg)), device=DEV)
                _backend.set_backend("numpy")
                cleaned = {"raw": noisy_t, "r1": r1,
                           "r2_k3": gate_torch(noisy_t, 3.0),
                           "r2_k3.5": gate_torch(noisy_t, 3.5),
                           "r2_k4": gate_torch(noisy_t, 4.0),
                           "oracle": torch.as_tensor(nocoh, device=DEV)}
                stack = torch.stack([cleaned[a] for a in ARMS])
                kept, f0r = threshold_batched(stack, clean)
                for a_i, a in enumerate(ARMS):
                    f0, nrms, cl = removal_f0_coh(cleaned[a], clean, coh, noisy_t)
                    rows.append(dict(ev=ev, plane=pt, noise=nm, arm=a,
                                     dense=(ev in dense), f0=f0, nrms=nrms,
                                     coh_left=cl, n_kept_pywt=int(kept[a_i]),
                                     f0_recon=float(f0r[a_i])))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(events)} ({len(rows)} rows)", flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'noise':7s} {'pl':2s} {'arm':8s} {'F0':>7s}±SE   {'coh_l':>6s} "
          f"{'kept/orc':>8s} {'win>r1':>7s}   dense:{'F0':>7s}/{'win':>4s} (n_dense)")
    for nm in nmodels:
        for pl in ("U", "V", "Y"):
            cells = collections.defaultdict(dict)
            for r in rows:
                if r["noise"] == nm and r["plane"] == pl:
                    cells[r["ev"]][r["arm"]] = r
            evs = list(cells.values())
            nd = sum(c["r1"]["dense"] for c in evs)
            for arm in ARMS[1:]:
                f0 = np.array([c[arm]["f0"] for c in evs])
                cl = np.mean([c[arm]["coh_left"] for c in evs])
                ko = np.mean([c[arm]["n_kept_pywt"] for c in evs]) / \
                    np.mean([c["oracle"]["n_kept_pywt"] for c in evs])
                win = np.nan if arm == "r1" else np.mean([c[arm]["f0"] > c["r1"]["f0"] for c in evs])
                dm = [c for c in evs if c[arm]["dense"]]
                df0 = np.mean([c[arm]["f0"] for c in dm]) if dm else np.nan
                dwin = np.nan if arm == "r1" else (np.mean([c[arm]["f0"] > c["r1"]["f0"] for c in dm]) if dm else np.nan)
                se = f0.std(ddof=1) / np.sqrt(len(f0))
                print(f"{nm:7s} {pl:2s} {arm:8s} {f0.mean():7.4f}±{se:.4f} {cl:6.3f} "
                      f"{ko:8.3f} {win:7.2f}   {df0:7.4f}/{dwin:4.2f} ({nd})")
        print()


if __name__ == "__main__":
    main()
