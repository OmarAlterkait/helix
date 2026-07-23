"""R2 (smart-gate) qualification harness — the systematic check before R2
becomes the default coherent removal.

Protocol mirrors research/coherent_coeffs exactly, but self-contained (cc_common
is dead-path-unimportable) and with the FAITHFUL noise injectors from pimm-data
(the package's own tests inject white uncoupled coherent noise — unrepresentative,
see blast-radius report):

  clean   = stored doraemon sensor plane (noise-free production, digitized+thresholded)
  coh     = pimm_data.noise.coherent_noise (1/f spectrum, beta coupling, per-group)
  inc     = pimm_data.noise.incoherent_noise (ENC; colored via noise_spectrum.npz
            or white via series_spectrum=None — BOTH arms, the record is colored-only)
  noisy   = digitize(clean + coh + inc, pedestal)

Arms per (event, plane, seed, noise_model):
  raw            no removal
  r1             classic multipass (helix.tpc.coherent, numpy)
  r2_k{3,3.5,4}  smart gate, canonical torch impl (measure_coeffs.smart_gate_bands)
  r2np_k4        numpy reference (verbatim smart.py port) — cross-impl check
  oracle         coherent never injected (kept-count ceiling, RESULTS.md 6c protocol)

Metrics (copied verbatim from smart.py::metrics + the 6b/6c end-to-end protocol):
  removal quality: F0 (on-support L1), nrms (off-support RMS), coh_left
                   (RMS of coh_hat - coh, coh_hat := noisy - cleaned)
  end-to-end:      production threshold (per-band MAD, kappa=1, approx incl.)
                   -> n_kept (total + per band) + F0_recon
Safety arms: r2_k4 on digitize(clean) and digitize(clean+inc) — what the gate
subtracts when there is no coherent noise ("removal if needed" semantics).

Usage:  python qual.py [--events N] [--seeds N] [--noise colored|white|both]
Writes results.jsonl (one row per cell) + prints the aggregate table.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))                  # helix root
sys.path.insert(0, "/sdf/group/neutrino/omara/pimm-data/src")       # pimm_data

import numpy as np
import pywt
import torch

from helix.tpc.io import config_from_file, read_sensor_plane
from helix.tpc.config import DetectorConfig
from helix.tpc.coherent import remove_coherent
from helix.core.wavelet_ops_torch import _wavedec, _waverec

from pimm_data.noise import coherent_noise, incoherent_noise, digitize
from pimm_data.geometry import load_plane_registry
from pimm_data.jaxtpc import canonical_plane_id

sys.path.insert(0, os.path.join(HERE, "..", "coeff_foundation_model"))
import measure_coeffs as M                                          # noqa: E402 (read-only import)

SHARD = ("/sdf/data/neutrino/doraemon/wire_test_00_00_02/sensor/"
         "run_0027575715/sim_wire_sensor_0000.h5")
NPZ = "/sdf/group/neutrino/omara/JAXTPC/config/noise_spectrum.npz"
PLANES = ("volume_0_U", "volume_0_V", "volume_0_Y")
WAVELET, LEVEL, MODE = "coif3", 4, "periodization"
KAPPA = 1.0
GS = 64
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ── verbatim smart.py reference (numpy, unpadded pywt bands) ────────────────

def _block_common_mode(band, nw, ksig=3.0):
    nblk = (nw + GS - 1) // GS
    L = band.shape[-1]
    Mm = np.zeros((nblk, L), np.float32)
    for g in range(nblk):
        lo, hi = g * GS, min((g + 1) * GS, nw)
        blk = band[lo:hi]
        med = np.median(blk, axis=0)
        resid = blk - med
        sg = max(np.median(np.abs(resid)) / 0.6745, 1e-6)
        uf = np.abs(resid) <= ksig * sg
        nuf = uf.sum(0)
        mean = (blk * uf).sum(0) / np.maximum(nuf, 1)
        Mm[g] = np.where(nuf > 0, mean, med)
    return Mm


def smart_removal_np(noisy, kgate=4.0, ksig=3.0):
    nw, nt = noisy.shape
    bands = pywt.wavedec(noisy.astype(np.float32), WAVELET, level=LEVEL,
                         mode=MODE, axis=-1)
    out = []
    for b in bands:
        Mm = _block_common_mode(b, nw, ksig)
        sigc = max(float(np.median(np.abs(Mm)) / 0.6745), 1e-6)
        Mc = np.where(np.abs(Mm) < kgate * sigc, Mm, 0.0)
        idx = np.minimum(np.arange(nw) // GS, Mm.shape[0] - 1)
        out.append(b - Mc[idx])
    return pywt.waverec(out, WAVELET, mode=MODE, axis=-1)[..., :nt].astype(np.float32)


# ── metrics (verbatim smart.py::metrics) ────────────────────────────────────

def removal_metrics(cleaned, signal, coherent, coh_hat):
    sig = np.abs(signal) > 0
    tc = float(np.abs(signal)[sig].sum())
    f0 = 1.0 - float(np.abs(cleaned - signal)[sig].sum()) / max(tc, 1e-9)
    nrms = float(np.sqrt(np.mean((cleaned - signal)[~sig] ** 2)))
    cl = float(np.sqrt(np.mean((coh_hat - coherent) ** 2)))
    return dict(f0=f0, nrms=nrms, coh_left=cl)


def prod_threshold_np(bands):
    """Production threshold: per-band MAD sigma, kappa=1, ALL bands incl. approx
    (== measure_coeffs.prod_threshold / DetectorConfig.threshold_spec)."""
    out, kept = [], []
    for b in bands:
        sg = float(np.median(np.abs(b)) / 0.6745)
        t = KAPPA * sg * np.sqrt(2.0 * np.log(max(b.shape[-1], 2)))
        m = np.abs(b) >= t
        out.append(np.where(m, b, 0.0))
        kept.append(int(m.sum()))
    return out, kept


def end_to_end_np(cleaned, signal):
    """cleaned image -> pywt bands -> production threshold -> recon F0 + kept."""
    bands = pywt.wavedec(cleaned.astype(np.float32), WAVELET, level=LEVEL,
                         mode=MODE, axis=-1)
    thr, kept = prod_threshold_np(bands)
    recon = pywt.waverec(thr, WAVELET, mode=MODE, axis=-1)[..., :cleaned.shape[-1]]
    sig = np.abs(signal) > 0
    tc = float(np.abs(signal)[sig].sum())
    f0r = 1.0 - float(np.abs(recon - signal)[sig].sum()) / max(tc, 1e-9)
    return dict(f0_recon=f0r, n_kept=int(sum(kept)), kept_bands=kept)


# ── R2 canonical torch path (what would be packaged) ────────────────────────

def smart_removal_torch(noisy, kgate):
    nw, nt = noisy.shape
    pad = (-nt) % (1 << LEVEL)
    x = torch.as_tensor(noisy, dtype=torch.float32, device=DEV)
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    bands = _wavedec(x, WAVELET, LEVEL)
    gated = M.smart_gate_bands(bands, kgate=kgate)
    cleaned = _waverec(gated, WAVELET)[..., :nt]
    return cleaned.cpu().numpy()


# ── the campaign ────────────────────────────────────────────────────────────

def wire_lengths_for(label, n_wires, registry):
    gid = canonical_plane_id(label)
    if gid in registry and len(registry[gid].get("wire_lengths", [])) == n_wires:
        return np.asarray(registry[gid]["wire_lengths"], np.float64)
    return np.full(n_wires, 2.33)          # spectrum calibration length fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--noise", choices=("colored", "white", "both"), default="both")
    ap.add_argument("--out", default=os.path.join(HERE, "results.jsonl"))
    args = ap.parse_args()

    npz = np.load(NPZ, allow_pickle=True)
    spectrum = (npz["spectrum_freqs_hz"], npz["spectrum_shape"])
    registry = load_plane_registry("cubic_wireplane_geometry.json")
    cfg_file = config_from_file(SHARD)
    r1cfg = DetectorConfig(num_time_steps=cfg_file.num_time_steps)
    noise_models = {"colored": spectrum, "white": None}
    if args.noise != "both":
        noise_models = {args.noise: noise_models[args.noise]}

    rows = []
    for ev in range(args.events):
        for label in PLANES:
            ptype = label.split("_")[-1]
            ped = cfg_file.pedestals[ptype]
            clean = read_sensor_plane(SHARD, ev, label, cfg_file.num_time_steps, ped)
            nw, nt = clean.shape
            wl = wire_lengths_for(label, nw, registry)
            for seed in range(args.seeds):
                for nm, spec in noise_models.items():
                    rng = np.random.default_rng(
                        abs(hash((ev, label, seed, nm))) % (1 << 63))
                    coh = coherent_noise(nw, nt, rng)
                    inc = incoherent_noise((nw, nt), wl, rng, series_spectrum=spec)
                    noisy = digitize(clean + coh + inc, ped)
                    nocoh = digitize(clean + inc, ped)

                    arms = {"raw": noisy,
                            "r1": np.asarray(remove_coherent(noisy, r1cfg))}
                    for k in (3.0, 3.5, 4.0):
                        arms[f"r2_k{k:g}"] = smart_removal_torch(noisy, k)
                    arms["r2np_k4"] = smart_removal_np(noisy, 4.0)
                    arms["oracle"] = nocoh                     # coherent never injected

                    for arm, cleaned in arms.items():
                        r = dict(ev=ev, plane=ptype, seed=seed, noise=nm, arm=arm)
                        r.update(removal_metrics(cleaned, clean, coh,
                                                 noisy - cleaned))
                        r.update(end_to_end_np(cleaned, clean))
                        rows.append(r)

                    # safety arms: gate on inputs WITHOUT coherent noise
                    for tag, img in (("safety_cleanonly", digitize(clean, ped)),
                                     ("safety_nocoh", nocoh)):
                        out = smart_removal_torch(img, 4.0)
                        sig = np.abs(clean) > 0
                        r = dict(ev=ev, plane=ptype, seed=seed, noise=nm, arm=tag,
                                 max_delta=float(np.abs(out - img).max()),
                                 off_rms_in=float(np.sqrt(np.mean(img[~sig] ** 2))),
                                 off_rms_out=float(np.sqrt(np.mean(out[~sig] ** 2))))
                        r.update(removal_metrics(out, clean, np.zeros_like(clean),
                                                 img - out))
                        rows.append(r)
        print(f"event {ev} done ({len(rows)} rows)", flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # aggregate table
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["noise"], r["plane"], r["arm"])].append(r)
    print(f"\n{'noise':7s} {'pl':2s} {'arm':16s} {'F0':>7s} {'nrms':>6s} "
          f"{'coh_left':>8s} {'F0_recon':>8s} {'n_kept':>8s}")
    for (nm, pl, arm) in sorted(agg):
        rs = agg[(nm, pl, arm)]
        g = lambda k: np.mean([r[k] for r in rs if k in r])
        print(f"{nm:7s} {pl:2s} {arm:16s} {g('f0'):7.4f} {g('nrms'):6.3f} "
              f"{g('coh_left'):8.4f} {g('f0_recon'):8.4f} {g('n_kept'):8.0f}"
              if arm.startswith(('raw', 'r1', 'r2', 'oracle')) else
              f"{nm:7s} {pl:2s} {arm:16s} {g('f0'):7.4f} {g('nrms'):6.3f}   "
              f"max_delta={g('max_delta'):.3f} off_rms {g('off_rms_in'):.3f}"
              f"->{g('off_rms_out'):.3f}")


if __name__ == "__main__":
    main()
