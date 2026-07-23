"""2x2 panels (clean | coherent-noisy | helix-removal + best-DWT recon | diff) per
plane, using the Pareto-best config bior4.4 L8. ALL CPU (numpy helix removal +
pywt) so it does not touch the GPUs. Symlog ADC style (matches plot_steps.py)."""
import os, sys
os.environ['HELIX_BACKEND'] = 'numpy'                    # force CPU helix removal
import numpy as np, pywt
sys.path.insert(0, '/sdf/group/neutrino/omara/helix')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import common as C
from helix.core.backend import set_backend; set_backend('numpy')
from helix.tpc.coherent import remove_coherent
from helix.tpc.config import DetectorConfig

GS = 64
CFG = DetectorConfig(group_size=64, mask_threshold_nsigma=3.0, num_time_steps=C.N_TICKS,
                     temporal_dilation_ticks=11, n_passes=3)
BEST = {'Y': ('bior4.4', 8, 1.25), 'U': ('bior4.4', 8, 2.0), 'V': ('bior4.4', 8, 1.5)}


def dwt_recon(img, w, lv, k):
    co = pywt.wavedec(img, w, mode='periodization', level=lv, axis=-1)
    sg = np.median(np.abs(co[-1]), axis=-1, keepdims=True) / 0.6745
    nk = co[0].size
    for b in range(1, len(co)):
        t = k * sg * np.sqrt(2 * np.log(max(co[b].shape[-1], 2)))
        co[b] = np.where(np.abs(co[b]) >= t, co[b], 0.0); nk += int(np.count_nonzero(co[b]))
    return pywt.waverec(co, w, mode='periodization', axis=-1)[:, :C.N_TICKS], img.size / max(nk, 1)


def crop(clean, hw=90, ht=220):
    en = np.abs(clean); nw, T = en.shape
    wi = max(0, min(int(np.argmax(np.convolve(en.sum(1), np.ones(hw), 'same'))) - hw // 2, nw - hw))
    ti = max(0, min(int(np.argmax(np.convolve(en.sum(0), np.ones(ht), 'same'))) - ht // 2, T - ht))
    return slice(wi, wi + hw), slice(ti, ti + ht)


def syml(ax, img, title, ws, ts, vmax):
    norm = SymLogNorm(linthresh=2.0, vmin=-vmax, vmax=vmax, base=10)
    im = ax.imshow(img.T, aspect='auto', origin='lower', cmap='RdBu_r', norm=norm,
                   extent=[ws.start, ws.stop, ts.start, ts.stop])
    ax.set_title(title, fontsize=10); ax.set_xlabel('wire'); ax.set_ylabel('tick')
    for g in range(((ws.start // GS) + 1) * GS, ws.stop, GS):
        ax.axvline(g, color='gray', lw=0.5, ls='--', alpha=0.5)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label='ADC')


for pt in ['Y', 'U', 'V']:
    w, lv, k = BEST[pt]
    clean = C.load_clean(C.train_test_events(n_train=24, n_test=20)[1][:1], pt)[0]
    noisy = C.make_noisy(clean[None], pt, seed=11, coherent=True)[0]
    removed = np.asarray(remove_coherent(noisy, CFG, sigma_per_wire=None))
    recon, comp = dwt_recon(removed, w, lv, k)
    ws, ts = crop(clean)
    cl, no, rc = clean[ws, ts], noisy[ws, ts], recon[ws, ts]; diff = rc - cl
    vmax = max(float(np.abs(cl).max()), 20.0)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    syml(ax[0, 0], cl, 'clean (truth)', ws, ts, vmax)
    syml(ax[0, 1], no, 'noisy: +coherent +intrinsic', ws, ts, vmax)
    syml(ax[1, 0], rc, f'helix removal + {w} L{lv} (k{k:g})  ~{comp:.0f}x', ws, ts, vmax)
    syml(ax[1, 1], diff, 'diff (recon - clean)', ws, ts, max(vmax * 0.5, 20.0))
    fig.suptitle(f'{pt} plane - coherent: helix removal + {w} L{lv} DWT  (symlog ADC)', fontweight='bold')
    fig.tight_layout(); fig.savefig(f'figures/panel_best_{pt}.png', dpi=110); plt.close(fig)
    print(f'saved figures/panel_best_{pt}.png  ({w} L{lv} k{k:g}, ~{comp:.0f}x, vmax {vmax:.0f})', flush=True)

    # ---- full-plane (zoomed-out) version: whole event, no crop / no group lines ----
    vmf = max(float(np.abs(clean).max()), 20.0)
    fwn, fwt = slice(0, clean.shape[0]), slice(0, C.N_TICKS)
    figf, axf = plt.subplots(2, 2, figsize=(15, 9))
    for a, (img, ttl) in zip(axf.ravel(), [(clean, 'clean (truth)'),
                                           (noisy, 'noisy: +coherent +intrinsic'),
                                           (recon, f'helix removal + {w} L{lv} (k{k:g})  ~{comp:.0f}x'),
                                           (recon - clean, 'diff (recon - clean)')]):
        vm = vmf if 'diff' not in ttl else max(vmf * 0.5, 20.0)
        im = a.imshow(img.T, aspect='auto', origin='lower', cmap='RdBu_r',
                      norm=SymLogNorm(linthresh=2.0, vmin=-vm, vmax=vm, base=10))
        a.set_title(ttl, fontsize=10); a.set_xlabel('wire'); a.set_ylabel('tick')
        plt.colorbar(im, ax=a, fraction=0.045, pad=0.02, label='ADC')
    figf.suptitle(f'{pt} plane FULL ({clean.shape[0]} wires x {C.N_TICKS} ticks) - '
                  f'helix removal + {w} L{lv} DWT  (symlog ADC)', fontweight='bold')
    figf.tight_layout(); figf.savefig(f'figures/panel_best_{pt}_full.png', dpi=120); plt.close(figf)
    print(f'saved figures/panel_best_{pt}_full.png', flush=True)
