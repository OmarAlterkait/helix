"""2x2 image panels per plane: clean | noisy(+coherent) | learned-removed+DWT | diff,
on a signal-rich crop. Symlog ADC style matching scripts/plot_steps.py (SymLogNorm
linthresh=2, base=10, RdBu_r) + 64-wire group lines (the CNN operates per group)."""
import sys; sys.path.insert(0, '/sdf/group/neutrino/omara/helix')
import numpy as np, torch, pywt
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import common as C
from group_removal import CoherentNet, learned_remove, BEST_DWT, GS

KAP = {'Y': 1.5, 'U': 1.5, 'V': 1.5}


def dwt_recon(img, wavelet, level, kappa):
    co = pywt.wavedec(img, wavelet, mode='periodization', level=level, axis=-1)
    sig = np.median(np.abs(co[-1]), axis=-1, keepdims=True) / 0.6745
    for b in range(1, len(co)):
        t = kappa * sig * np.sqrt(2 * np.log(max(co[b].shape[-1], 2)))
        co[b] = np.where(np.abs(co[b]) >= t, co[b], 0.0)
    return pywt.waverec(co, wavelet, mode='periodization', axis=-1)[:, :C.N_TICKS]


def best_crop(clean, hw=90, ht=220):
    en = np.abs(clean); nw, T = en.shape
    wc = en.sum(1); tc = en.sum(0)
    wi = max(0, min(int(np.argmax(np.convolve(wc, np.ones(hw), 'same'))) - hw // 2, nw - hw))
    ti = max(0, min(int(np.argmax(np.convolve(tc, np.ones(ht), 'same'))) - ht // 2, T - ht))
    return slice(wi, wi + hw), slice(ti, ti + ht)


def symlog_im(ax, img, title, ws, ts, vmax, cbar=True):
    """img is (wire, tick) crop; matches scripts/plot_steps.py symlog_im style."""
    norm = SymLogNorm(linthresh=2.0, vmin=-vmax, vmax=vmax, base=10)
    extent = [ws.start, ws.stop, ts.start, ts.stop]
    im = ax.imshow(img.T, aspect='auto', origin='lower', cmap='RdBu_r', norm=norm, extent=extent)
    ax.set_title(title, fontsize=10); ax.set_xlabel('wire'); ax.set_ylabel('tick')
    first = ((ws.start // GS) + 1) * GS                 # 64-wire group boundaries
    for g in range(first, ws.stop, GS):
        ax.axvline(g, color='gray', lw=0.5, ls='--', alpha=0.5)
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label='ADC')
    return im


for pt in ['Y', 'U', 'V']:
    clean = C.load_clean(C.train_test_events(n_train=24, n_test=14)[1][:1], pt)[0]
    noisy = C.make_noisy(clean[None], pt, seed=11, coherent=True)
    net = CoherentNet().to('cuda'); net.load_state_dict(torch.load(f'artifacts/grpnet_{pt}.pt', weights_only=True))
    removed = learned_remove(noisy, net)[0]
    w, lv = BEST_DWT[pt]; recon = dwt_recon(removed, w, lv, KAP[pt])
    ws, ts = best_crop(clean)
    cl, no, rc = clean[ws, ts], noisy[0][ws, ts], recon[ws, ts]
    diff = rc - cl
    vmax = max(float(np.abs(cl).max()), 20.0)           # shared across clean/noisy/recon
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    symlog_im(ax[0, 0], cl, 'clean (truth)', ws, ts, vmax)
    symlog_im(ax[0, 1], no, 'noisy: +coherent +intrinsic', ws, ts, vmax)
    symlog_im(ax[1, 0], rc, 'learned removal + DWT', ws, ts, vmax)
    symlog_im(ax[1, 1], diff, 'diff (recon - clean)', ws, ts, max(vmax * 0.5, 20.0))
    fig.suptitle(f'{pt} plane - group-aware learned removal (coherent noise), symlog ADC',
                 fontweight='bold')
    fig.tight_layout(); fig.savefig(f'figures/panel_{pt}.png', dpi=110); plt.close(fig)
    print(f'saved figures/panel_{pt}.png  vmax {vmax:.0f}  wires {ws.start}-{ws.stop} ticks {ts.start}-{ts.stop}')
