"""P5 Stage B: coherent noise. Does the per-wire transform need explicit
cross-wire coherent removal, or can it cope alone?

For each plane + best method (DWT, KLT, learned-wavelet) evaluate the R-D on:
  (raw)     coherent-noisy input straight into the transform
  (removed) helix multi-pass coherent removal FIRST, then the transform
Coherent noise is group-correlated (same waveform across 64 wires) so per-wire
transforms should fail on (raw) and recover on (removed).
"""
import json
import time
import numpy as np
import torch

import sys
sys.path.insert(0, '/sdf/group/neutrino/omara/helix')   # repo root for `import helix`

import common as C
import baselines as B
import klt as K
import learned_wavelet as LW

# helix coherent removal (production front-end)
from helix.core.backend import set_backend
set_backend('jax')
from helix.tpc.coherent import remove_coherent
from helix.tpc.config import DetectorConfig

BEST_DWT = {'Y': ('db8', 4), 'U': ('coif3', 4), 'V': ('db8', 4)}
KAPPAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
N_TEST = 4
SEED = 1


def coherent_remove(noisy_stack, ptype):
    """Apply helix multi-pass coherent removal per event (sigma auto-estimated)."""
    cfg = DetectorConfig(group_size=64, mask_threshold_nsigma=3.0, num_time_steps=C.N_TICKS,
                         temporal_dilation_ticks=11, n_passes=3)
    out = np.empty_like(noisy_stack)
    for i in range(noisy_stack.shape[0]):
        out[i] = np.asarray(remove_coherent(noisy_stack[i], cfg, sigma_per_wire=None))
    return out


def dwt_curve(noisy, clean, pt):
    w, lv = BEST_DWT[pt]
    return [dict(B.dwt_rd_point(noisy, clean, w, lv, k), method='dwt', kappa=k) for k in KAPPAS]


def klt_curve(noisy, clean, pt):
    Bmat = np.load(f'artifacts/klt_basis_{pt}.npy')
    return [dict(K.klt_rd_point(noisy, clean, Bmat, k), method='klt', kappa=k) for k in KAPPAS]


def lwave_curve(noisy, clean, pt):
    model = LW.LiftingWavelet().to(LW.DEV)
    model.load_state_dict(torch.load(f'artifacts/lwave_{pt}.pt')); model.eval()
    pts = []
    with torch.no_grad():
        for scale in KAPPAS:
            recon = np.empty_like(noisy); nk = nt = 0
            for i in range(noisy.shape[0]):
                xn = LW._pad(torch.from_numpy(noisy[i]).float().to(LW.DEV))
                rec, k = model.denoise(xn, LW._sigma(xn), scale=scale, hard=True, count=True)
                recon[i] = rec[:, :C.N_TICKS].cpu().numpy(); nk += k; nt += xn.numel()
            f0, nr = C.aggregate(clean, recon)
            pts.append(dict(method='lwave', scale=scale, f0=f0, noise_rms=nr,
                            compression=nt / max(nk, 1)))
    return pts


def at10(pts):
    cand = [p for p in pts if 7 <= p['compression'] <= 14]
    return max(cand, key=lambda p: p['f0']) if cand else min(pts, key=lambda p: abs(p['compression'] - 10))


def run():
    results = {}
    for pt in ['Y', 'U', 'V']:
        t0 = time.perf_counter()
        clean = C.load_clean(C.train_test_events()[1][:N_TEST], pt)
        noisy = C.make_noisy(clean, pt, seed=SEED, coherent=True)
        removed = coherent_remove(noisy, pt)
        raw_f0, raw_nr = C.aggregate(clean, noisy)
        rem_f0, rem_nr = C.aggregate(clean, removed)
        curves = {}
        for variant, inp in [('raw', noisy), ('removed', removed)]:
            curves[variant] = dict(
                dwt=dwt_curve(inp, clean, pt),
                klt=klt_curve(inp, clean, pt),
                lwave=lwave_curve(inp, clean, pt),
            )
        results[pt] = dict(raw_f0=raw_f0, raw_nr=raw_nr, removed_f0=rem_f0, removed_nr=rem_nr,
                           curves=curves)
        print(f"[{pt}] coherent raw F0={raw_f0:.3f} nrms={raw_nr:.2f} | after removal F0={rem_f0:.3f} nrms={rem_nr:.2f}")
        for variant in ['raw', 'removed']:
            cells = []
            for m in ['dwt', 'klt', 'lwave']:
                p = at10(curves[variant][m])
                cells.append(f"{m} {p['compression']:.0f}x/F0{p['f0']:.3f}/n{p['noise_rms']:.2f}")
            print(f"    {variant:8s}: " + "  ".join(cells))
        print(f"    ({time.perf_counter()-t0:.0f}s)", flush=True)
    with open('artifacts/stage_b.json', 'w') as f:
        json.dump(results, f, indent=1)
    print('saved artifacts/stage_b.json')


if __name__ == '__main__':
    run()
