"""P8: group-aware LEARNED coherent removal vs hand-crafted helix removal.

Coherent noise is one waveform shared across each 64-wire group. A CNN estimates
that per-group waveform from robust across-wire statistics (median/mean/std — it
can refine the median, which is already ~the coherent) plus temporal context, then
subtracts it. We compare (learned removal -> DWT) against (helix removal -> DWT)
and (no removal -> DWT) at matched compression. True coherent is known at train
time (we synthesize it), so the target is supervised.
"""
import json
import time
import numpy as np
import torch
import torch.nn as nn

import common as C
import baselines as Bl

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
GS = 64
EPOCHS = 600
BATCH = 256          # group-blocks per step
LR = 2e-3
N_TRAIN = 16
N_TEST = 4
SEED = 1
BEST_DWT = {'Y': ('db8', 4), 'U': ('coif3', 4), 'V': ('db8', 4)}


class CoherentNet(nn.Module):
    """(B, GS, T) group block -> (B, T) coherent-waveform estimate."""
    def __init__(self, nfeat=3, ch=32, k=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(nfeat, ch, k, padding=k // 2), nn.ReLU(),
            nn.Conv1d(ch, ch, k, padding=k // 2), nn.ReLU(),
            nn.Conv1d(ch, ch, k, padding=k // 2), nn.ReLU(),
            nn.Conv1d(ch, 1, 1))

    def forward(self, x):                      # x (B, GS, T)
        med = x.median(dim=1).values
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        f = torch.stack([med, mean, std], dim=1)   # (B,3,T) — median≈coherent already
        return self.net(f)[:, 0, :]                # (B,T)


def _coh_wave(rng, T=C.N_TICKS):
    """One per-group coherent waveform (T,) via tools.coherent_noise (1 group)."""
    from tools.coherent_noise import generate_group_waveforms
    return generate_group_waveforms(1, T, beta=0.15, rms_adc=2.5, rng=rng)[0].astype(np.float32)


def train_plane(pt):
    train, _ = C.train_test_events()
    clean = C.load_clean(train[:N_TRAIN], pt)        # (E,nw,T)
    E, nw, T = clean.shape
    ng = nw // GS
    blocks = clean[:, :ng * GS, :].reshape(E, ng, GS, T).reshape(-1, GS, T)   # (E*ng, GS, T) clean
    blocks = torch.from_numpy(blocks).float()
    ped = C.PLANES[pt]['pedestal']
    net = CoherentNet().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)
    Nb = blocks.shape[0]
    pool = torch.from_numpy(np.stack([_coh_wave(rng) for _ in range(4000)])).to(DEV)  # coherent-wave pool
    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        idx = rng.choice(Nb, BATCH, replace=False)
        cg = blocks[idx].to(DEV)                              # (B,GS,T) clean groups
        # synth noise: intrinsic per wire + one coherent waveform per group (sampled from pool)
        intr = _intr(cg.shape, pt).to(DEV)
        coh = pool[rng.choice(pool.shape[0], BATCH)]          # (B,T)
        noisy = torch.round(cg + intr + coh[:, None, :]).clamp(-ped, 4095 - ped)
        est = net(noisy)                                     # (B,T)
        loss = ((est - coh) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 150 == 0 or ep == EPOCHS - 1:
            # baseline median-removal MSE for reference
            med_mse = ((noisy.median(dim=1).values - coh) ** 2).mean().item()
            print(f"    [{pt}] ep{ep} loss={loss.item():.3f} (median-only {med_mse:.3f})", flush=True)
    torch.save(net.state_dict(), f'artifacts/grpnet_{pt}.pt')
    print(f"[{pt}] trained ({time.perf_counter()-t0:.0f}s)")
    return net


_SPEC = {}
def _intr(shape, pt):
    B, gs, T = shape
    if T not in _SPEC:
        _SPEC[T] = torch.from_numpy(C._spectrum(T)).float().to(DEV)
    spec = _SPEC[T]; nf = spec.shape[0]
    lo, hi = C.PLANES[pt]['wire_len']
    srms = float(C.NOISE_Y + C.NOISE_Z * 0.5 * (lo + hi))
    r = torch.randn(B, gs, nf, device=DEV) * spec
    im = torch.randn(B, gs, nf, device=DEV) * spec
    cpx = torch.complex(r, im); cpx[:, :, 0] = torch.complex(cpx[:, :, 0].real, torch.zeros_like(cpx[:, :, 0].real))
    shaped = torch.fft.irfft(cpx, n=T, dim=2)
    shaped = shaped / shaped.std(dim=2, keepdim=True).clamp_min(1e-6) * srms
    white = torch.randn(B, gs, T, device=DEV) * C.NOISE_X
    return (shaped + white).float()


def learned_remove(noisy_stack, net):
    """Apply per-group learned coherent estimate + subtract -> removed image stack."""
    out = np.empty_like(noisy_stack)
    net.eval()
    with torch.no_grad():
        for i in range(noisy_stack.shape[0]):
            img = noisy_stack[i]; nw, T = img.shape
            ng = (nw + GS - 1) // GS
            pad = ng * GS - nw
            x = np.pad(img, ((0, pad), (0, 0)), mode='reflect') if pad else img
            blk = torch.from_numpy(x.reshape(ng, GS, T)).float().to(DEV)
            est = net(blk).cpu().numpy()                    # (ng, T)
            rem = (x.reshape(ng, GS, T) - est[:, None, :]).reshape(ng * GS, T)
            out[i] = rem[:nw]
    return out


def run():
    results = {}
    for pt in ['Y', 'U', 'V']:
        net = train_plane(pt)
        _, test = C.train_test_events()
        clean = C.load_clean(test[:N_TEST], pt)
        noisy = C.make_noisy(clean, pt, seed=SEED, coherent=True)
        removed = learned_remove(noisy, net)
        rf0, rnr = C.aggregate(clean, removed)
        w, lv = BEST_DWT[pt]
        # DWT R-D on learned-removed
        pts = [dict(Bl.dwt_rd_point(removed, clean, w, lv, k), kappa=k) for k in Bl.KAPPAS]
        cand = [p for p in pts if 7 <= p['compression'] <= 14]
        best = max(cand, key=lambda p: p['f0']) if cand else min(pts, key=lambda p: abs(p['compression'] - 10))
        results[pt] = dict(removed_f0=rf0, removed_nr=rnr, points=pts, best10=best)
        print(f"[{pt}] LEARNED removal: residual F0={rf0:.3f} nrms={rnr:.2f} | "
              f"+DWT @~10x: comp={best['compression']:.0f}x F0={best['f0']:.3f} nrms={best['noise_rms']:.2f}", flush=True)
    json.dump(results, open('artifacts/group_removal.json', 'w'), indent=1)
    print('saved artifacts/group_removal.json')


if __name__ == '__main__':
    run()
