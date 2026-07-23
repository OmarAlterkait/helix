"""P1 baselines: per-wire DWT VisuShrink R-D sweep + time-domain threshold floor.

Each method -> a rate-distortion curve (compression, F0, noise_rms) per plane,
evaluated on held-out test events with intrinsic-only noise (Stage A).
"""
import json
import time
import numpy as np
import pywt

import common as C

WAVELETS = ['coif3', 'sym8', 'db8', 'coif5']
LEVELS = [4, 8]
KAPPAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
N_TEST = 6
SEED = 1


def _hard(x, t):
    return np.where(np.abs(x) >= t, x, 0.0)


def dwt_rd_point(noisy_stack, clean_stack, wavelet, level, kappa):
    """One R-D point: per-wire DWT, VisuShrink-hard with sigma=MAD(finest band)."""
    recon = np.empty_like(noisy_stack)
    n_kept = 0
    n_total = 0
    for i in range(noisy_stack.shape[0]):
        coeffs = pywt.wavedec(noisy_stack[i], wavelet, mode='periodization', level=level, axis=-1)
        sigma = np.median(np.abs(coeffs[-1]), axis=-1, keepdims=True) / 0.6745  # (nw,1) finest band MAD
        n_kept += int(coeffs[0].size)          # approx kept fully
        n_total += int(coeffs[0].size)
        for b in range(1, len(coeffs)):
            Lb = coeffs[b].shape[-1]
            t = kappa * sigma * np.sqrt(2.0 * np.log(max(Lb, 2)))
            coeffs[b] = _hard(coeffs[b], t)
            n_kept += int(np.count_nonzero(coeffs[b]))
            n_total += int(coeffs[b].size)
        recon[i] = pywt.waverec(coeffs, wavelet, mode='periodization', axis=-1)[:, :C.N_TICKS]
    f0, nrms = C.aggregate(clean_stack, recon)
    comp = n_total / max(n_kept, 1)
    return dict(f0=f0, noise_rms=nrms, compression=comp, n_kept=n_kept, n_total=n_total)


def time_threshold_point(noisy_stack, clean_stack, kappa):
    """Floor: hard-threshold in the time domain (transform = identity)."""
    # sigma per wire from MAD of the signal itself (robust to sparse pulses)
    recon = np.empty_like(noisy_stack)
    n_kept = 0
    for i in range(noisy_stack.shape[0]):
        sigma = np.median(np.abs(noisy_stack[i]), axis=-1, keepdims=True) / 0.6745
        t = kappa * sigma
        recon[i] = _hard(noisy_stack[i], t)
        n_kept += int(np.count_nonzero(recon[i]))
    f0, nrms = C.aggregate(clean_stack, recon)
    n_total = noisy_stack.size
    return dict(f0=f0, noise_rms=nrms, compression=n_total / max(n_kept, 1),
                n_kept=n_kept, n_total=n_total)


def run():
    _, test = C.train_test_events()
    results = {}
    for pt in ['Y', 'U', 'V']:
        t0 = time.perf_counter()
        clean = C.load_clean(test[:N_TEST], pt)
        noisy = C.make_noisy(clean, pt, seed=SEED, coherent=False)
        raw_f0, raw_nrms = C.aggregate(clean, noisy)
        pts = []
        for w in WAVELETS:
            for lv in LEVELS:
                for k in KAPPAS:
                    p = dwt_rd_point(noisy, clean, w, lv, k)
                    p.update(method=f'dwt:{w}:L{lv}', wavelet=w, level=lv, kappa=k)
                    pts.append(p)
        for k in KAPPAS:
            p = time_threshold_point(noisy, clean, k)
            p.update(method='time-threshold', kappa=k)
            pts.append(p)
        results[pt] = dict(raw_f0=raw_f0, raw_noise_rms=raw_nrms, points=pts)
        # best DWT config near compression ~10x
        dwt = [p for p in pts if p['method'].startswith('dwt')]
        near10 = min(dwt, key=lambda p: abs(p['compression'] - 10))
        best_at_10 = max([p for p in dwt if 8 <= p['compression'] <= 13], key=lambda p: p['f0'], default=near10)
        print(f"[{pt}] raw F0={raw_f0:.4f} nrms={raw_nrms:.2f} | "
              f"best DWT @~10x: {best_at_10['method']} k={best_at_10['kappa']} "
              f"comp={best_at_10['compression']:.1f}x F0={best_at_10['f0']:.4f} nrms={best_at_10['noise_rms']:.3f} "
              f"({time.perf_counter()-t0:.0f}s)")
    with open('artifacts/baselines.json', 'w') as f:
        json.dump(results, f, indent=1)
    print('saved artifacts/baselines.json')
    return results


if __name__ == '__main__':
    run()
