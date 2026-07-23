"""P2: per-plane KLT/PCA transform learned on clean signal patches.

Learn the optimal *orthonormal* basis (energy-compacting) for each plane from
clean time-patches, then denoise noisy patches by VisuShrink-hard in that basis.
Tests whether a data-optimal linear basis beats the fixed wavelet.

Patches: length-P windows along time, non-overlapping per wire (P | tiling).
Rate = kept coeffs / total coeffs (orthonormal => directly comparable).
"""
import json
import time
import numpy as np

import common as C

P = 64            # patch length along time
KAPPAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
N_TRAIN = 16
N_TEST = 6
SEED = 1


def to_patches(img, p=P):
    """(nw, T) -> (nw * nT, p) non-overlapping time patches (T trimmed to multiple of p)."""
    nw, T = img.shape
    nT = T // p
    return img[:, :nT * p].reshape(nw, nT, p).reshape(-1, p), nw, nT


def from_patches(patches, nw, nT, T, p=P):
    out = np.zeros((nw, T), np.float32)
    out[:, :nT * p] = patches.reshape(nw, nT, p).reshape(nw, nT * p)
    return out


def learn_klt(clean_stack, p=P):
    """KLT basis (p,p) from clean patches (eigvecs of patch covariance, desc)."""
    pts = []
    for i in range(clean_stack.shape[0]):
        pa, _, _ = to_patches(clean_stack[i], p)
        pts.append(pa[np.abs(pa).max(1) > 0])     # signal-bearing patches only
    X = np.concatenate(pts, 0).astype(np.float64)
    cov = (X.T @ X) / max(len(X), 1)
    w, V = np.linalg.eigh(cov)
    return V[:, ::-1].astype(np.float32)           # (p, p), columns = basis, desc variance


def klt_rd_point(noisy_stack, clean_stack, B, kappa, p=P):
    recon = np.empty_like(noisy_stack)
    n_kept = 0
    n_total = 0
    for i in range(noisy_stack.shape[0]):
        pa, nw, nT = to_patches(noisy_stack[i], p)
        c = pa @ B                                 # (Npatch, p) coeffs
        sigma = np.median(np.abs(c), axis=1, keepdims=True) / 0.6745   # per-patch noise
        t = kappa * sigma * np.sqrt(2.0 * np.log(p))
        c = np.where(np.abs(c) >= t, c, 0.0)
        rec = c @ B.T
        recon[i] = from_patches(rec, nw, nT, C.N_TICKS, p)
        n_kept += int(np.count_nonzero(c))
        n_total += int(c.size)
    f0, nrms = C.aggregate(clean_stack, recon)
    return dict(f0=f0, noise_rms=nrms, compression=n_total / max(n_kept, 1),
                n_kept=n_kept, n_total=n_total)


def run():
    train, test = C.train_test_events()
    results = {}
    for pt in ['Y', 'U', 'V']:
        t0 = time.perf_counter()
        ctr = C.load_clean(train[:N_TRAIN], pt)
        B = learn_klt(ctr)
        np.save(f'artifacts/klt_basis_{pt}.npy', B)
        cte = C.load_clean(test[:N_TEST], pt)
        noisy = C.make_noisy(cte, pt, seed=SEED, coherent=False)
        pts = []
        for k in KAPPAS:
            p = klt_rd_point(noisy, cte, B, k)
            p.update(method='klt', kappa=k)
            pts.append(p)
        results[pt] = dict(points=pts)
        near10 = max([p for p in pts if 8 <= p['compression'] <= 13], key=lambda x: x['f0'],
                     default=min(pts, key=lambda x: abs(x['compression'] - 10)))
        print(f"[{pt}] KLT @~10x: comp={near10['compression']:.1f}x F0={near10['f0']:.4f} "
              f"nrms={near10['noise_rms']:.3f}  ({time.perf_counter()-t0:.0f}s)")
    with open('artifacts/klt.json', 'w') as f:
        json.dump(results, f, indent=1)
    print('saved artifacts/klt.json')


if __name__ == '__main__':
    run()
