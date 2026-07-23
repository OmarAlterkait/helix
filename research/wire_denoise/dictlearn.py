"""P3: per-plane dictionary learning (sparse coding) on 1D time patches.

Learn an overcomplete dictionary from clean signal patches (MiniBatchDictionaryLearning),
then denoise noisy patches by sparse coding (OMP, sweep n_nonzero -> R-D).

Caveat: dictionary is overcomplete (n_atoms > P), so a kept coeff also needs an
atom index (~log2(n_atoms) bits) that orthonormal transforms don't. We report
compression = n_pixels / n_kept (index overhead ignored) and flag it in RESULTS.
"""
import json
import time
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning, SparseCoder

import common as C
from klt import to_patches, from_patches, P

N_ATOMS = 128
N_TRAIN = 12
N_TEST = 4
MAX_TRAIN_PATCHES = 60000
NNZ = [1, 2, 3, 4, 6]
SEED = 1


def signal_patches(clean_stack, p=P):
    out = []
    for i in range(clean_stack.shape[0]):
        pa, _, _ = to_patches(clean_stack[i], p)
        out.append(pa[np.abs(pa).max(1) > 0])
    return np.concatenate(out, 0)


def learn_dict(clean_stack, rng):
    X = signal_patches(clean_stack).astype(np.float64)
    if len(X) > MAX_TRAIN_PATCHES:
        X = X[rng.choice(len(X), MAX_TRAIN_PATCHES, replace=False)]
    dl = MiniBatchDictionaryLearning(n_components=N_ATOMS, alpha=1.0, max_iter=25,
                                     batch_size=1024, transform_algorithm='omp',
                                     random_state=0, n_jobs=-1)
    dl.fit(X)
    return dl.components_.astype(np.float64), len(X)


def dict_rd_point(noisy_stack, clean_stack, D, nnz, p=P):
    coder = SparseCoder(dictionary=D, transform_algorithm='omp',
                        transform_n_nonzero_coefs=nnz, n_jobs=-1)
    recon = np.empty_like(noisy_stack)
    n_kept = 0
    n_total = 0
    for i in range(noisy_stack.shape[0]):
        pa, nw, nT = to_patches(noisy_stack[i], p)
        c = coder.transform(pa.astype(np.float64))     # (Npatch, n_atoms) sparse
        rec = c @ D
        recon[i] = from_patches(rec.astype(np.float32), nw, nT, C.N_TICKS, p)
        n_kept += int(np.count_nonzero(c))
        n_total += int(pa.shape[0] * p)                # vs pixels (orthonormal-equiv rate)
    f0, nrms = C.aggregate(clean_stack, recon)
    return dict(f0=f0, noise_rms=nrms, compression=n_total / max(n_kept, 1),
                n_kept=n_kept, n_total=n_total, nnz=nnz)


def run():
    train, test = C.train_test_events()
    rng = np.random.default_rng(0)
    results = {}
    for pt in ['Y', 'U', 'V']:
        t0 = time.perf_counter()
        ctr = C.load_clean(train[:N_TRAIN], pt)
        D, npatch = learn_dict(ctr, rng)
        np.save(f'artifacts/dict_{pt}.npy', D)
        cte = C.load_clean(test[:N_TEST], pt)
        noisy = C.make_noisy(cte, pt, seed=SEED, coherent=False)
        pts = []
        for k in NNZ:
            p = dict_rd_point(noisy, cte, D, k)
            p.update(method='dict')
            pts.append(p)
            print(f"  [{pt}] nnz={k} comp={p['compression']:.1f}x F0={p['f0']:.4f} nrms={p['noise_rms']:.3f}", flush=True)
        results[pt] = dict(points=pts, n_train_patches=npatch)
        print(f"[{pt}] dict done ({time.perf_counter()-t0:.0f}s)", flush=True)
    with open('artifacts/dict.json', 'w') as f:
        json.dump(results, f, indent=1)
    print('saved artifacts/dict.json')


if __name__ == '__main__':
    run()
