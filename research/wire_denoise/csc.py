"""P9: convolutional sparse coding with LEARNED pulse templates (matched filters).

Rationale: the signal is sparse pulses; transform-thresholding spends ~tens of
coeffs per pulse (energy spread across scales). A shift-invariant template bank +
greedy convolutional matching pursuit encodes each pulse with ~1 atom
(amplitude+location), and amplitudes are least-squares (unbiased) -> far better
F0-per-coefficient. Templates learned per plane from clean pulse snippets (PCA).

Rate currency = number of atoms (each = 1 amplitude + 1 position index + tiny
template id), directly comparable to DWT's kept (value+index) coefficients.
"""
import json
import time
import numpy as np
import torch

import common as C

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
L = 64            # template length
K = 8             # number of templates
M = 30            # max atoms per wire
N_TRAIN = 16
N_TEST = 6
SEED = 1
TAUS = [3.0, 4.0, 5.0, 7.0, 10.0, 14.0, 20.0]    # MP stopping threshold (ADC) -> rate knob


def learn_templates(clean_stack, k=K, ell=L, thr=20.0):
    """PCA pulse templates from aligned clean snippets (peak-centered windows)."""
    snips = []
    for e in range(clean_stack.shape[0]):
        img = clean_stack[e]
        a = np.abs(img)
        # local maxima above thr, per wire
        loc = (a[:, 1:-1] > thr) & (a[:, 1:-1] >= a[:, :-2]) & (a[:, 1:-1] > a[:, 2:])
        ws, ts = np.nonzero(loc); ts = ts + 1
        keep = (ts >= ell // 2) & (ts < img.shape[1] - ell // 2)
        for w, t in zip(ws[keep], ts[keep]):
            snips.append(img[w, t - ell // 2:t + ell // 2])
    X = np.asarray(snips, np.float64)
    U, S, Vt = np.linalg.svd(X - 0, full_matrices=False)   # keep DC of pulse shape
    D = Vt[:k]
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    return D.astype(np.float32), len(X)


def encode(noisy_img, D, tau, m=M):
    """Batched convolutional matching pursuit over all wires of one image.
    Returns (recon_img, n_atoms)."""
    R = torch.from_numpy(noisy_img).float().to(DEV)            # (nw, T)
    Dt = torch.from_numpy(D).float().to(DEV)                   # (K, L)
    nw, T = R.shape; k, ell = Dt.shape
    recon = torch.zeros_like(R)
    filt = Dt[:, None, :]                                      # (K,1,L)
    rows = torch.arange(nw, device=DEV)
    arL = torch.arange(ell, device=DEV)
    n_atoms = 0
    for _ in range(m):
        corr = torch.nn.functional.conv1d(R[:, None, :], filt)  # (nw, K, T-L+1)
        Tc = corr.shape[2]
        flat = corr.reshape(nw, -1)
        val, idx = flat.abs().max(dim=1)
        active = val > tau
        if not active.any():
            break
        amp = flat[rows, idx]                                  # signed amplitude (D unit-norm)
        kk = idx // Tc                                          # template id
        tt = idx % Tc                                          # start position
        pos = tt[:, None] + arL[None, :]                       # (nw, L)
        upd = amp[:, None] * Dt[kk]                            # (nw, L)
        a_rows = rows[active]
        ri = a_rows[:, None].expand(-1, ell)
        pi = pos[active]
        recon.index_put_((ri, pi), upd[active], accumulate=True)
        R.index_put_((ri, pi), -upd[active], accumulate=True)
        n_atoms += int(active.sum())
    return recon.cpu().numpy(), n_atoms


def run():
    train, test = C.train_test_events()
    results = {}
    for pt in ['Y', 'U', 'V']:
        t0 = time.perf_counter()
        ctr = C.load_clean(train[:N_TRAIN], pt)
        D, nsnip = learn_templates(ctr)
        np.save(f'artifacts/csc_templates_{pt}.npy', D)
        cte = C.load_clean(test[:N_TEST], pt)
        noisy = C.make_noisy(cte, pt, seed=SEED, coherent=False)
        pts = []
        for tau in TAUS:
            recon = np.empty_like(noisy); natoms = 0
            for i in range(noisy.shape[0]):
                r, na = encode(noisy[i], D, tau); recon[i] = r; natoms += na
            f0, nr = C.aggregate(cte, recon)
            comp = noisy.size / max(natoms, 1)
            # bias: mean residual on signal pixels (want ~0)
            sig = np.abs(cte) > 0
            bias = float((recon - cte)[sig].mean())
            pts.append(dict(method='csc', tau=tau, f0=f0, noise_rms=nr, compression=comp,
                            n_atoms=natoms, bias=bias))
            print(f"  [{pt}] tau={tau:4.0f} comp={comp:6.0f}x F0={f0:.4f} nrms={nr:.3f} bias={bias:+.3f}", flush=True)
        results[pt] = dict(points=pts, n_snippets=nsnip)
        print(f"[{pt}] csc done ({time.perf_counter()-t0:.0f}s, {nsnip} snippets)", flush=True)
    json.dump(results, open('artifacts/csc.json', 'w'), indent=1)
    print('saved artifacts/csc.json')


if __name__ == '__main__':
    run()
