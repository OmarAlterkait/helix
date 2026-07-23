"""Smart-gate kgate sweep — rich per-k metrics over many events.

Uses the SMART gate (measure_coeffs.smart_gate_bands, the canonical R2), swept
over a fine kgate grid, density-stratified events, both noise models. Reports
the quantities that drive the operating-point choice, at each k, per plane:

  signal_lost   = 1 - F0            (removal stage; frac of signal CHARGE lost)
  signal_lost_w = 1 - F0_recon      (after wavelet threshold; the FM basis)
  coeffs        = n_kept            (surviving coefficients, pywt protocol)
  coeffs/oracle = n_kept / no-coherent-oracle kept   (compression headroom)
  noise_kept    = off-track RMS     (residual noise away from signal)
  coh_left      = RMS(coh_hat-coh)  (leftover coherent)
  ontrack_rms   = on-support RMS(cleaned-signal)  (signal-region distortion)

GPU-optimized (dense_ops noise, jax R1 baseline, batched GPU threshold).
Usage: python ksweep.py [--events 100] [--pool 200] [--noise both]
Writes ksweep.jsonl + ksweep.png.
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
from pimm_data.noise import (DEFAULT_ENC, DEFAULT_COH_RMS_ADC, DEFAULT_COH_CORNER_FREQ_HZ,
                             DEFAULT_COH_SLOPE, DEFAULT_COH_BETA, DEFAULT_SAMPLING_RATE_HZ, digitize)
from pimm_data.geometry import load_plane_registry
from pimm_data.jaxtpc import canonical_plane_id
import measure_coeffs as MC

SHARD = ("/sdf/data/neutrino/doraemon/wire_test_00_00_02/sensor/"
         "run_0027575715/sim_wire_sensor_0000.h5")
NPZ = "/sdf/group/neutrino/omara/JAXTPC/config/noise_spectrum.npz"
PLANES = ("volume_0_U", "volume_0_V", "volume_0_Y")
WAVELET, LEVEL, KAPPA, GS, DEV = "coif3", 4, 1.0, 64, "cuda"
KGRID = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]


def gate_torch(noisy_t, kgate):
    nt = noisy_t.shape[-1]
    pad = (-nt) % (1 << LEVEL)
    x = torch.nn.functional.pad(noisy_t, (0, pad)) if pad else noisy_t
    gated = MC.smart_gate_bands(_wavedec(x, WAVELET, LEVEL), kgate=kgate)
    return _waverec(gated, WAVELET)[..., :nt]


def metrics_img(cleaned, signal, coherent, noisy):
    sig = signal.abs() > 0
    tc = signal.abs()[sig].sum().clamp_min(1e-9)
    return dict(
        signal_lost=float((cleaned - signal).abs()[sig].sum() / tc),
        noise_kept=float(((cleaned - signal)[~sig] ** 2).mean().sqrt()),
        ontrack_rms=float(((cleaned - signal)[sig] ** 2).mean().sqrt()),
        coh_left=float(((noisy - cleaned - coherent) ** 2).mean().sqrt()))


def threshold_metrics(cleaned_stack, signal):
    A, W, T = cleaned_stack.shape
    pad = (-T) % (1 << LEVEL)
    x = torch.nn.functional.pad(cleaned_stack, (0, pad)) if pad else cleaned_stack
    bands = _wavedec(x, WAVELET, LEVEL)
    kept = torch.zeros(A, device=DEV)
    thr = []
    for b in bands:
        sg = b.reshape(A, -1).abs().median(dim=1).values / 0.6745
        t = (KAPPA * sg * (2.0 * np.log(max(b.shape[-1], 2))) ** 0.5).view(A, 1, 1)
        m = b.abs() >= t
        kept += m.reshape(A, -1).sum(1)
        thr.append(torch.where(m, b, torch.zeros_like(b)))
    recon = _waverec(thr, WAVELET)[..., :T]
    sig = signal.abs() > 0
    tc = signal.abs()[sig].sum().clamp_min(1e-9)
    lost_w = [float((recon[a] - signal).abs()[sig].sum() / tc) for a in range(A)]
    return kept.cpu().numpy(), np.array(lost_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=100)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--noise", choices=("colored", "white", "both"), default="both")
    args = ap.parse_args()
    npz = np.load(NPZ, allow_pickle=True)
    spec_np = (npz["spectrum_freqs_hz"], npz["spectrum_shape"])
    reg = load_plane_registry("cubic_wireplane_geometry.json")
    cfg = config_from_file(SHARD)
    r1cfg = DetectorConfig(num_time_steps=cfg.num_time_steps)
    pool = min(args.pool, count_events(SHARD))
    nmodels = ({"colored": spec_np, "white": None} if args.noise == "both"
               else {args.noise: {"colored": spec_np, "white": None}[args.noise]})

    def wl_t(label, nw):
        v = np.asarray(reg.get(canonical_plane_id(label), {}).get("wire_lengths", []), np.float64)
        return torch.as_tensor(v if len(v) == nw else np.full(nw, 2.33), dtype=torch.float32, device=DEV)

    print(f"scan {pool} for stratification...", flush=True)
    act = sorted((int((read_sensor_plane(SHARD, ev, "volume_0_U", cfg.num_time_steps,
                                         cfg.pedestals["U"]) != 0).sum()), ev) for ev in range(pool))
    order = [ev for _, ev in act]
    k = args.events
    events = sorted(set(order[int(round(i * (len(order) - 1) / (k - 1)))] for i in range(k))
                    | set(order[-max(k // 5, 5):]))
    print(f"{len(events)} events", flush=True)

    # arms: r1, oracle, and the k-grid gates
    KARMS = [f"k{g:g}" for g in KGRID]
    ARMS = ["r1", "oracle"] + KARMS
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
                gen = torch.Generator(device=DEV); gen.manual_seed(hash((ev, pt, nm)) & 0xFFFFFFFF)
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
                cleaned = {"r1": r1, "oracle": torch.as_tensor(nocoh, device=DEV)}
                for g in KGRID:
                    cleaned[f"k{g:g}"] = gate_torch(noisy_t, g)
                stack = torch.stack([cleaned[a] for a in ARMS])
                kept, lost_w = threshold_metrics(stack, clean)
                for a_i, a in enumerate(ARMS):
                    r = dict(ev=ev, plane=pt, noise=nm, arm=a, n_kept=int(kept[a_i]),
                             signal_lost_w=float(lost_w[a_i]))
                    r.update(metrics_img(cleaned[a], clean, coh, noisy_t))
                    rows.append(r)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(events)}", flush=True)

    with open(os.path.join(HERE, "ksweep.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # ── tables ──
    def agg(nm, pl, arm, key):
        vs = [r[key] for r in rows if r["noise"] == nm and r["plane"] == pl and r["arm"] == arm]
        return np.mean(vs)
    for nm in nmodels:
        print(f"\n===== {nm} noise =====")
        print(f"{'pl':2s} {'arm':6s} {'sig_lost%':>9s} {'sig_lost_w%':>11s} {'coeffs':>7s} "
              f"{'c/oracle':>8s} {'noise_kept':>10s} {'coh_left':>8s} {'ontrk_rms':>9s}")
        for pl in ("U", "V", "Y"):
            orc = agg(nm, pl, "oracle", "n_kept")
            for arm in ARMS:
                print(f"{pl:2s} {arm:6s} {agg(nm,pl,arm,'signal_lost')*100:9.3f} "
                      f"{agg(nm,pl,arm,'signal_lost_w')*100:11.3f} {agg(nm,pl,arm,'n_kept'):7.0f} "
                      f"{agg(nm,pl,arm,'n_kept')/orc:8.3f} {agg(nm,pl,arm,'noise_kept'):10.3f} "
                      f"{agg(nm,pl,arm,'coh_left'):8.3f} {agg(nm,pl,arm,'ontrack_rms'):9.3f}")
            print()

    # ── plot: metrics vs k, per plane (colored) ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        nm = "colored" if "colored" in nmodels else list(nmodels)[0]
        cols = {"U": "tab:blue", "V": "tab:orange", "Y": "tab:green"}
        for pl in ("U", "V", "Y"):
            sl = [agg(nm, pl, f"k{g:g}", "signal_lost") * 100 for g in KGRID]
            ck = [agg(nm, pl, f"k{g:g}", "n_kept") / agg(nm, pl, "oracle", "n_kept") for g in KGRID]
            nk = [agg(nm, pl, f"k{g:g}", "noise_kept") for g in KGRID]
            cl = [agg(nm, pl, f"k{g:g}", "coh_left") for g in KGRID]
            r1sl = agg(nm, pl, "r1", "signal_lost") * 100
            ax[0, 0].plot(KGRID, sl, "o-", color=cols[pl], label=pl)
            ax[0, 0].axhline(r1sl, color=cols[pl], ls=":", lw=1)
            ax[0, 1].plot(KGRID, ck, "o-", color=cols[pl], label=pl)
            ax[1, 0].plot(KGRID, nk, "o-", color=cols[pl], label=pl)
            ax[1, 1].plot(KGRID, cl, "o-", color=cols[pl], label=pl)
        ax[0, 0].set(title="signal lost % (1-F0)  [dotted = R1]", xlabel="kgate", ylabel="%")
        ax[0, 1].set(title="coeffs / oracle (compression headroom)", xlabel="kgate")
        ax[0, 1].axhline(1.0, color="k", ls="--", lw=0.8)
        ax[1, 0].set(title="noise kept (off-track RMS, ADC)", xlabel="kgate")
        ax[1, 1].set(title="coh_left (residual coherent RMS, ADC)", xlabel="kgate")
        for a in ax.flat:
            a.legend(); a.grid(alpha=0.3); a.axvline(3.0, color="red", ls="--", lw=1, alpha=0.5)
        fig.suptitle(f"Smart-gate kgate sweep ({nm} noise, {len(events)} events) — red = k=3 default",
                     fontweight="bold")
        fig.tight_layout()
        p = os.path.join(HERE, "ksweep.png")
        fig.savefig(p, dpi=130); plt.close(fig)
        print("\nsaved", p)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
