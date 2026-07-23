"""P4: neural learned wavelet via the lifting scheme (per plane).

A lifting wavelet is invertible *by construction* (perfect reconstruction for any
filter weights), so we can learn the predict/update filters + per-level soft
thresholds end-to-end to maximize denoising fidelity, then read off the learned
"optimal wavelet" per plane.

Forward (one level, even-length x):
    xe, xo = x[::2], x[1::2]
    d = xo - predict(xe)          # detail
    s = xe + update(d)            # approx
Inverse: xe = s - update(d); xo = d + predict(xe); interleave. Exactly invertible.

Multi-level: recurse on s. Soft-threshold details (learnable per-level threshold
scaled by per-signal MAD sigma). Train: noisy wire signals -> clean, L1-on-signal
+ MSE-off-signal loss. Eval: sweep threshold scale -> rate-distortion.
"""
import json
import time
import numpy as np
import torch
import torch.nn as nn

import common as C

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N_LEVELS = 6
PAD_LEN = 4352            # next multiple of 2^6 above 4321 (reflect-padded)
KSIZE = 4
EPOCHS = 500
BATCH = 2048             # wire-signals per step (low GPU mem; ~40s/plane)
LR = 3e-3
N_TRAIN = 16
N_TEST = 6
SEED = 1


def _fir(x, w):
    """Length-preserving circular FIR conv along last dim. x (B,L), w (k,) -> (B,L)."""
    k = w.numel()
    left, right = k // 2, (k - 1) - (k // 2)         # total pad = k-1 -> 'same' length
    pieces = [x[:, -left:]] if left else []
    pieces.append(x)
    if right:
        pieces.append(x[:, :right])
    xp = torch.cat(pieces, dim=1)
    return torch.nn.functional.conv1d(xp[:, None, :], w.flip(0)[None, None, :])[:, 0, :]


class LiftingWavelet(nn.Module):
    def __init__(self, n_levels=N_LEVELS, ksize=KSIZE):
        super().__init__()
        # per-level predict/update FIR filters; init ~ Haar/linear (small smooth)
        self.P = nn.ParameterList([nn.Parameter(torch.full((ksize,), 0.5 / ksize)) for _ in range(n_levels)])
        self.U = nn.ParameterList([nn.Parameter(torch.full((ksize,), 0.25 / ksize)) for _ in range(n_levels)])
        self.log_tau = nn.Parameter(torch.zeros(n_levels))   # per-level threshold scale (×sigma)
        self.n_levels = n_levels

    def analysis(self, x):
        details = []
        s = x
        for l in range(self.n_levels):
            xe, xo = s[:, ::2], s[:, 1::2]
            d = xo - _fir(xe, self.P[l])
            s = xe + _fir(d, self.U[l])
            details.append(d)
        return s, details

    def synthesis(self, s, details):
        for l in reversed(range(self.n_levels)):
            d = details[l]
            xe = s - _fir(d, self.U[l])
            xo = d + _fir(xe, self.P[l])
            x = torch.empty(xe.shape[0], xe.shape[1] * 2, device=xe.device, dtype=xe.dtype)
            x[:, ::2] = xe
            x[:, 1::2] = xo
            s = x
        return s

    def denoise(self, x, sigma, scale=1.0, hard=False, count=False):
        s, details = self.analysis(x)
        n_kept = s.shape[0] * s.shape[1]      # approx kept
        thr = []
        for l, d in enumerate(details):
            t = scale * torch.exp(self.log_tau[l]) * sigma * np.sqrt(2.0 * np.log(max(d.shape[1], 2)))
            if hard:
                dd = torch.where(d.abs() >= t, d, torch.zeros_like(d))
            else:
                dd = torch.sign(d) * torch.clamp(d.abs() - t, min=0.0)
            thr.append(dd)
            if count:
                n_kept += int((dd != 0).sum().item())
        rec = self.synthesis(s, thr)
        return (rec, n_kept) if count else rec


def _pad(x):   # (B,T) -> (B,PAD_LEN) reflect
    return torch.nn.functional.pad(x, (0, PAD_LEN - x.shape[1]), mode='reflect')


def _sigma(x):  # per-signal MAD (B,1)
    return (x.abs().median(dim=1, keepdim=True).values / 0.6745).clamp_min(1e-3)


def wires_tensor(stack):
    """(E,nw,T) -> (E*nw, T) torch on device, dropping all-zero (dead) wires kept (they're fine)."""
    a = stack.reshape(-1, stack.shape[-1])
    return torch.from_numpy(a).float()


def train_plane(pt, log):
    train, _ = C.train_test_events()
    clean = C.load_clean(train[:N_TRAIN], pt)
    Xc = wires_tensor(clean)                       # (Nwire, T) clean
    info = C.PLANES[pt]
    model = LiftingWavelet().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)
    N = Xc.shape[0]
    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        idx = rng.choice(N, BATCH, replace=False)
        xc = Xc[idx].to(DEV)
        # add intrinsic noise on the fly (per-wire white + colored approx via torch)
        noise = _intrinsic_torch(xc.shape, pt, rng).to(DEV)
        xn = torch.round(xc + noise).clamp(-info['pedestal'], 4095 - info['pedestal'])  # ~digitize
        xcp, xnp = _pad(xc), _pad(xn)
        sig = _sigma(xnp)
        rec = model.denoise(xnp, sig, scale=1.0, hard=False)
        rec = rec[:, :C.N_TICKS]
        xc_t = xcp[:, :C.N_TICKS]
        smask = (xc_t.abs() > 0).float()
        l_sig = ((rec - xc_t).abs() * smask).sum() / smask.sum().clamp_min(1)
        l_off = (((rec - xc_t) * (1 - smask)) ** 2).mean()
        loss = l_sig + 0.5 * l_off
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0 or ep == EPOCHS - 1:
            log(f"    [{pt}] ep{ep} loss={loss.item():.4f} l_sig={l_sig.item():.4f} l_off={l_off.item():.4f}")
    torch.save(model.state_dict(), f'artifacts/lwave_{pt}.pt')
    print(f"[{pt}] trained ({time.perf_counter()-t0:.0f}s)")
    return model


# ---- on-the-fly intrinsic noise in torch (matches common.intrinsic_noise statistics) ----
_SPEC_T = {}
def _intrinsic_torch(shape, pt, rng):
    B, T = shape
    if T not in _SPEC_T:
        _SPEC_T[T] = torch.from_numpy(C._spectrum(T)).float().to(DEV)
    spec = _SPEC_T[T]
    n_freq = spec.shape[0]
    lo, hi = C.PLANES[pt]['wire_len']
    series_rms = float((C.NOISE_Y + C.NOISE_Z * 0.5 * (lo + hi)))   # mid wire length
    r = torch.randn(B, n_freq, device=DEV) * spec
    im = torch.randn(B, n_freq, device=DEV) * spec
    cpx = torch.complex(r, im)
    cpx[:, 0] = torch.complex(cpx[:, 0].real, torch.zeros_like(cpx[:, 0].real))
    shaped = torch.fft.irfft(cpx, n=T, dim=1)
    shaped = shaped / shaped.std(dim=1, keepdim=True).clamp_min(1e-6) * series_rms
    white = torch.randn(B, T, device=DEV) * C.NOISE_X
    return (shaped + white).float()


def eval_plane(pt, model):
    _, test = C.train_test_events()
    cte = C.load_clean(test[:N_TEST], pt)
    noisy = C.make_noisy(cte, pt, seed=SEED, coherent=False)
    info = C.PLANES[pt]
    pts = []
    model.eval()
    with torch.no_grad():
        for scale in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
            recon = np.empty_like(noisy)
            n_kept = 0; n_total = 0
            for i in range(noisy.shape[0]):
                xn = _pad(torch.from_numpy(noisy[i]).float().to(DEV))
                sig = _sigma(xn)
                rec, nk = model.denoise(xn, sig, scale=scale, hard=True, count=True)
                recon[i] = rec[:, :C.N_TICKS].cpu().numpy()
                n_kept += nk
                n_total += xn.shape[0] * xn.shape[1]
            f0, nrms = C.aggregate(cte, recon)
            pts.append(dict(method='lwave', scale=scale, f0=f0, noise_rms=nrms,
                            compression=n_total / max(n_kept, 1), n_kept=n_kept, n_total=n_total))
    return pts


def run():
    logs = []
    def log(m): logs.append(m); print(m, flush=True)
    results = {}
    for pt in ['Y', 'U', 'V']:
        model = train_plane(pt, log)
        pts = eval_plane(pt, model)
        results[pt] = dict(points=pts)
        near10 = max([p for p in pts if 8 <= p['compression'] <= 13], key=lambda x: x['f0'],
                     default=min(pts, key=lambda x: abs(x['compression'] - 10)))
        print(f"[{pt}] lwave @~10x: comp={near10['compression']:.1f}x F0={near10['f0']:.4f} nrms={near10['noise_rms']:.3f}")
    with open('artifacts/lwave.json', 'w') as f:
        json.dump(dict(results=results, log=logs), f, indent=1)
    print('saved artifacts/lwave.json')


if __name__ == '__main__':
    run()
