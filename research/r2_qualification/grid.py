"""5x5 grid of (k_pass1, k_pass2) for the 2-pass smart gate, 100 stratified events.

Pass 1 gates with k1 (protects signal, removes clear coherent); its cleaned bands
are used to detect signal and refine the common-mode estimate; pass 2 re-gates
that purer estimate with k2. Sweeps k1,k2 in {2.5,3,3.5,4,4.5}. Reports, per plane,
the (k1,k2) surface for signal_lost, coeffs/oracle, and residual stripe.

Efficiency: M1 (pass-1 common mode) is k1-independent -> computed once/band;
M2 depends only on k1 (via the detected mask) -> 5 per band; the 25 gates are
cheap; images for all 25 combos are reconstructed in one batched GPU waverec.

Usage: python grid.py [--events 100]
"""
import argparse, collections, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, "/sdf/group/neutrino/omara/pimm-data/src")
sys.path.insert(0, os.path.join(HERE, "..", "coeff_foundation_model"))
import numpy as np, torch
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

SHARD = ("/sdf/data/neutrino/doraemon/wire_test_00_00_02/sensor/run_0027575715/sim_wire_sensor_0000.h5")
NPZ = "/sdf/group/neutrino/omara/JAXTPC/config/noise_spectrum.npz"
WAVELET, LEVEL, KAPPA, GS, DEV = "coif3", 4, 1.0, 64, "cuda"
KGRID = [2.5, 3.0, 3.5, 4.0, 4.5]


def block_cm(b, ksig, sigmask):
    W, Lb = b.shape; ngf = W // GS; Ms = []
    if ngf > 0:
        bf = b[:ngf * GS].reshape(ngf, GS, Lb); smf = sigmask[:ngf * GS].reshape(ngf, GS, Lb)
        med = bf.quantile(0.5, dim=1); resid = bf - med.unsqueeze(1)
        sg = (resid.abs().reshape(ngf, -1).quantile(0.5, dim=1) / 0.6745).clamp_min(1e-6)
        uf = (resid.abs() <= ksig * sg[:, None, None]) & (~smf); nuf = uf.sum(1)
        Ms.append(torch.where(nuf > 0, (bf * uf).sum(1) / nuf.clamp_min(1), med))
    rem = W - ngf * GS
    if rem > 0:
        blk = b[ngf * GS:]; smr = sigmask[ngf * GS:]
        med = blk.quantile(0.5, dim=0); resid = blk - med
        sg = (resid.abs().reshape(-1).quantile(0.5) / 0.6745).clamp_min(1e-6)
        uf = (resid.abs() <= ksig * sg) & (~smr); nuf = uf.sum(0)
        Ms.append(torch.where(nuf > 0, (blk * uf).sum(0) / nuf.clamp_min(1), med).unsqueeze(0))
    return torch.cat(Ms, dim=0)


def detect(cleaned_band, ksig):
    W, Lb = cleaned_band.shape; ngf = W // GS
    sm = torch.zeros_like(cleaned_band, dtype=torch.bool); cf = cleaned_band.abs()
    if ngf > 0:
        cb = cf[:ngf * GS].reshape(ngf, GS, Lb)
        csg = (cb.reshape(ngf, -1).quantile(0.5, dim=1) / 0.6745).clamp_min(1e-6)
        sm[:ngf * GS] = (cb > ksig * csg[:, None, None]).reshape(ngf * GS, Lb)
    rem = W - ngf * GS
    if rem > 0:
        cr = cf[ngf * GS:]; csg = (cr.reshape(-1).quantile(0.5) / 0.6745).clamp_min(1e-6)
        sm[ngf * GS:] = cr > ksig * csg
    return sm


def two_pass_grid(bands, ksig=3.0):
    """-> {(k1,k2): cleaned_bands}. M1 once/band; M2 per k1; gates per (k1,k2)."""
    pb = []                                             # per band: (b, M1, sigc1, idx)
    for b in bands:
        W = b.shape[0]; idx = (torch.arange(W, device=b.device) // GS).clamp(max=(W + GS - 1) // GS - 1)
        M1 = block_cm(b, ksig, torch.zeros_like(b, dtype=torch.bool))
        pb.append((b, M1, (M1.abs().median() / 0.6745).clamp_min(1e-6), idx))
    # per k1: refine mask -> M2, sigc2 per band
    m2 = {}
    for k1 in KGRID:
        rows = []
        for (b, M1, sigc1, idx) in pb:
            Mc1 = torch.where(M1.abs() < k1 * sigc1, M1, torch.zeros_like(M1))
            sm = detect(b - Mc1[idx], ksig)
            M2 = block_cm(b, ksig, sm)
            rows.append((M2, (M2.abs().median() / 0.6745).clamp_min(1e-6)))
        m2[k1] = rows
    out = {}
    for k1 in KGRID:
        for k2 in KGRID:
            cb = []
            for (b, _, _, idx), (M2, sigc2) in zip(pb, m2[k1]):
                Mc2 = torch.where(M2.abs() < k2 * sigc2, M2, torch.zeros_like(M2))
                cb.append(b - Mc2[idx])
            out[(k1, k2)] = cb
    return out


def kept_of_bands(bands):
    k = 0
    for b in bands:
        sg = b.abs().median() / 0.6745
        k += int((b.abs() >= KAPPA * sg * (2 * np.log(max(b.shape[-1], 2))) ** 0.5).sum())
    return k


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--events", type=int, default=100)
    ap.add_argument("--pool", type=int, default=200); args = ap.parse_args()
    npz = np.load(NPZ, allow_pickle=True); spec = (npz["spectrum_freqs_hz"], npz["spectrum_shape"])
    reg = load_plane_registry("cubic_wireplane_geometry.json")
    cfg = config_from_file(SHARD); r1cfg = DetectorConfig(num_time_steps=cfg.num_time_steps)
    pool = min(args.pool, count_events(SHARD))
    print(f"scan {pool}...", flush=True)
    act = sorted((int((read_sensor_plane(SHARD, ev, "volume_0_U", cfg.num_time_steps, cfg.pedestals["U"]) != 0).sum()), ev) for ev in range(pool))
    order = [ev for _, ev in act]; k = args.events
    events = sorted(set(order[int(round(i * (len(order) - 1) / (k - 1)))] for i in range(k)) | set(order[-max(k // 5, 5):]))
    print(f"{len(events)} events", flush=True)
    rows = []
    for i, ev in enumerate(events):
        for pl in ("U", "V", "Y"):
            lab = f"volume_0_{pl}"; ped = cfg.pedestals[pl]
            clean_np = read_sensor_plane(SHARD, ev, lab, cfg.num_time_steps, ped); nw, nt = clean_np.shape
            clean = torch.as_tensor(clean_np, device=DEV)
            v = np.asarray(reg.get(canonical_plane_id(lab), {}).get("wire_lengths", []), np.float64)
            wlt = torch.as_tensor(v if len(v) == nw else np.full(nw, 2.33), dtype=torch.float32, device=DEV)
            gen = torch.Generator(device=DEV); gen.manual_seed(hash((ev, pl)) & 0xFFFFFFFF)
            coh = _coherent_torch(nw, nt, gen=gen, group_size=GS, rms_adc=DEFAULT_COH_RMS_ADC, corner_freq_hz=DEFAULT_COH_CORNER_FREQ_HZ,
                                  spectral_slope=DEFAULT_COH_SLOPE, beta=DEFAULT_COH_BETA, sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ, device=DEV)
            inc = _incoherent_torch((nw, nt), wlt, gen=gen, enc=DEFAULT_ENC, series_spectrum=spec, sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ, device=DEV)
            noisy = digitize((clean + coh + inc).cpu().numpy(), ped)
            nocoh = digitize((clean + inc).cpu().numpy(), ped)
            noisy_t = torch.as_tensor(noisy, device=DEV)
            npad = (-nt) % (1 << LEVEL); xin = torch.nn.functional.pad(noisy_t, (0, npad)) if npad else noisy_t
            grid = two_pass_grid(_wavedec(xin, WAVELET, LEVEL))
            sig = clean.abs() > 0; tc = clean.abs()[sig].sum().clamp_min(1e-9)
            nf = nw // GS
            off = ~(clean[:nf * GS].reshape(nf, GS, nt).abs() > 0).any(dim=1)
            xo = torch.as_tensor(nocoh, device=DEV); xo = torch.nn.functional.pad(xo, (0, npad)) if npad else xo
            orc_kept = kept_of_bands(_wavedec(xo, WAVELET, LEVEL))
            for c, cb in grid.items():                              # one combo at a time (memory)
                im = _waverec(cb, WAVELET)[..., :nt]
                cm = im[:nf * GS].reshape(nf, GS, nt).mean(dim=1)
                rows.append(dict(ev=ev, plane=pl, k1=c[0], k2=c[1],
                                 signal_lost=float((im - clean).abs()[sig].sum() / tc),
                                 coh_left=float(((noisy_t - im - coh) ** 2).mean().sqrt()),
                                 stripe=float((cm[off] ** 2).mean().sqrt()),
                                 kept=kept_of_bands(cb), kept_oracle=orc_kept))
                del im
            del grid
            torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(events)}", flush=True)

    with open(os.path.join(HERE, "grid.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

    # per-plane 5x5 tables for signal_lost%, coeffs/oracle, stripe
    def cell(pl, k1, k2, key):
        vs = [r[key] for r in rows if r["plane"] == pl and r["k1"] == k1 and r["k2"] == k2]
        return np.mean(vs)
    def cell_ratio(pl, k1, k2):
        num = [r["kept"] for r in rows if r["plane"] == pl and r["k1"] == k1 and r["k2"] == k2]
        den = [r["kept_oracle"] for r in rows if r["plane"] == pl and r["k1"] == k1 and r["k2"] == k2]
        return np.mean(num) / np.mean(den)
    for pl in ("U", "V", "Y"):
        for title, fn in (("signal_lost %  (rows=k1 pass1, cols=k2 pass2)", lambda a, b: cell(pl, a, b, "signal_lost") * 100),
                          ("coeffs / oracle", lambda a, b: cell_ratio(pl, a, b)),
                          ("residual stripe (ADC)", lambda a, b: cell(pl, a, b, "stripe"))):
            print(f"\n=== {pl} : {title} ===")
            print("k1\\k2  " + "".join(f"{k2:8.1f}" for k2 in KGRID))
            for k1 in KGRID:
                print(f"{k1:5.1f}  " + "".join(f"{fn(k1, k2):8.3f}" for k2 in KGRID))


if __name__ == "__main__":
    main()
