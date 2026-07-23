"""P7: 2D (wire x time) separable wavelet on COHERENT-noisy data.

Tests the Stage-B conclusion: coherent noise is cross-wire correlated, so a 2D
transform (which decorrelates along wires too) might suppress it WITHOUT the
explicit removal step that per-wire transforms need. Compare 2D-DWT on raw
coherent-noisy vs per-wire DWT (raw) and vs coherent-removal+per-wire DWT (Stage B).
"""
import json
import numpy as np
import pywt

import common as C

WAVELET = 'coif3'
LEVEL = 4
KAPPAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
N_TEST = 4
SEED = 1


def dwt2_rd_point(noisy_stack, clean_stack, kappa):
    recon = np.empty_like(noisy_stack)
    n_kept = n_total = 0
    for i in range(noisy_stack.shape[0]):
        co = pywt.wavedec2(noisy_stack[i], WAVELET, mode='periodization', level=LEVEL)
        # sigma from finest diagonal subband MAD
        sigma = np.median(np.abs(co[-1][2])) / 0.6745
        n_kept += int(co[0].size); n_total += int(co[0].size)
        new = [co[0]]
        for lvl in co[1:]:
            sub = []
            for band in lvl:                      # (cH, cV, cD)
                Lb = band.size
                t = kappa * sigma * np.sqrt(2.0 * np.log(max(Lb, 2)))
                b = np.where(np.abs(band) >= t, band, 0.0)
                sub.append(b)
                n_kept += int(np.count_nonzero(b)); n_total += int(b.size)
            new.append(tuple(sub))
        recon[i] = pywt.waverec2(new, WAVELET, mode='periodization')[:noisy_stack.shape[1], :C.N_TICKS]
    f0, nrms = C.aggregate(clean_stack, recon)
    return dict(method='dwt2d', kappa=kappa, f0=f0, noise_rms=nrms,
                compression=n_total / max(n_kept, 1))


def run():
    results = {}
    for pt in ['Y', 'U', 'V']:
        clean = C.load_clean(C.train_test_events()[1][:N_TEST], pt)
        noisy = C.make_noisy(clean, pt, seed=SEED, coherent=True)
        pts = [dwt2_rd_point(noisy, clean, k) for k in KAPPAS]
        results[pt] = dict(points=pts)
        cand = [p for p in pts if 7 <= p['compression'] <= 14]
        best = max(cand, key=lambda p: p['f0']) if cand else min(pts, key=lambda p: abs(p['compression'] - 10))
        print(f"[{pt}] 2D-DWT on RAW coherent @~10x: comp={best['compression']:.0f}x "
              f"F0={best['f0']:.3f} nrms={best['noise_rms']:.2f}", flush=True)
    json.dump(results, open('artifacts/stage_c_2d.json', 'w'), indent=1)
    print('saved artifacts/stage_c_2d.json')


if __name__ == '__main__':
    run()
