"""Stage B frontier plot + learned-wavelet interpretability (effective mother
wavelet per plane vs coif3, and learned per-level thresholds)."""
import json
import numpy as np
import torch
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import common as C
import learned_wavelet as LW


def stage_b_plot():
    R = json.load(open('artifacts/stage_b.json'))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, pt in zip(axes, ['Y', 'U', 'V']):
        for variant, ls in [('raw', '--'), ('removed', '-')]:
            for m, col in [('dwt', 'C0'), ('klt', 'C1'), ('lwave', 'C2')]:
                pts = sorted(R[pt]['curves'][variant][m], key=lambda p: p['compression'])
                ax.plot([p['compression'] for p in pts], [p['f0'] for p in pts],
                        ls, color=col, ms=3, marker='o',
                        label=f'{m} ({variant})' if pt == 'Y' else None)
        ax.axhline(R[pt]['raw_f0'], color='red', ls=':', lw=1)
        ax.set(xscale='log', xlabel='compression x', ylabel='F0',
               title=f'{pt} — coherent noise: raw(--) vs removal(–)')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig('figures/stage_b_frontiers.png', dpi=110); plt.close(fig)
    print('saved figures/stage_b_frontiers.png')


def effective_wavelet(model, level=3, T=LW.PAD_LEN):
    """Impulse in detail band `level` -> synthesis -> the model's effective wavelet."""
    with torch.no_grad():
        x = torch.zeros(1, T, device=LW.DEV)
        s, details = model.analysis(x)
        s = torch.zeros_like(s)
        dz = [torch.zeros_like(d) for d in details]
        dz[level][0, dz[level].shape[1] // 2] = 1.0
        w = model.synthesis(s, dz)[0].cpu().numpy()
    nz = np.nonzero(np.abs(w) > 1e-4)[0]
    return w[nz.min():nz.max() + 1] if len(nz) else w


def interpret():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    print("\n=== learned per-level threshold scales exp(log_tau) ===")
    for pt, col in [('Y', 'C0'), ('U', 'C1'), ('V', 'C2')]:
        m = LW.LiftingWavelet().to(LW.DEV)
        m.load_state_dict(torch.load(f'artifacts/lwave_{pt}.pt', weights_only=True)); m.eval()
        w = effective_wavelet(m, level=3)
        w = w / (np.abs(w).max() + 1e-9)
        ax[0].plot(w, color=col, label=f'{pt} (learned)')
        taus = torch.exp(m.log_tau).detach().cpu().numpy()
        print(f"  {pt}: " + " ".join(f"L{i}:{t:.2f}" for i, t in enumerate(taus)))
    # reference coif3 wavelet (level-1 detail filter, normalized)
    wav = pywt.Wavelet('coif3')
    psi = np.array(wav.wavefun(level=5)[1]); psi = psi / (np.abs(psi).max() + 1e-9)
    ax[1].plot(psi, 'k'); ax[1].set(title='coif3 mother wavelet (reference)')
    ax[0].set(title='Learned effective wavelet (lifting, detail L3)'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig('figures/learned_wavelets.png', dpi=110); plt.close(fig)
    print('saved figures/learned_wavelets.png')


if __name__ == '__main__':
    stage_b_plot()
    interpret()
