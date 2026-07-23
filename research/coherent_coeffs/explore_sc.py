"""Understand signal vs coherent per-tick in a block (sample space), 1 event.

The coherent is IDENTICAL across the 64 wires of a block (= waveforms[block][tick]),
so at each tick the 64 wire values are: a dense cluster at coherent (clean wires,
spread sigma_int) + a minority pushed away by signal. A human reads the coherent as
that dense cluster, ignoring the track. We test which robust location estimator
recovers the KNOWN coherent best -- especially behind signal (high occupancy).

Estimators per (block,tick): median, masked-mean (helix-style), trimmed (25%),
Huber M (IRLS), low-quantile q25, shorth (mean of shortest half = densest cluster).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cc_common as cc

GS = cc.GROUP_SIZE


def est_median(x):
    return np.median(x)


def est_masked(x, k=3.0):
    m = np.median(x); r = x - m; s = max(np.median(np.abs(r)) / 0.6745, 1e-6)
    uf = np.abs(r) <= k * s
    return x[uf].mean() if uf.any() else m


def est_trim(x, frac=0.25):
    n = len(x); lo = int(frac * n); hi = n - lo
    return np.sort(x)[lo:hi].mean() if hi > lo else np.median(x)


def est_huber(x, c=1.345, iters=5):
    mu = np.median(x); s = max(np.median(np.abs(x - mu)) / 0.6745, 1e-6)
    for _ in range(iters):
        r = (x - mu) / s
        w = np.where(np.abs(r) <= c, 1.0, c / np.maximum(np.abs(r), 1e-9))
        mu = (w * x).sum() / w.sum()
    return mu


def est_q25(x):
    return np.percentile(x, 25)


def est_shorth(x):
    """Mean of the shortest contiguous half (densest cluster) -> robust to >50% one-sided."""
    s = np.sort(x); n = len(s); h = n // 2
    widths = s[h:] - s[:n - h]
    i = int(np.argmin(widths))
    return s[i:i + h + 1].mean()


ESTS = {'median': est_median, 'masked': est_masked, 'trim': est_trim,
        'huber': est_huber, 'q25': est_q25, 'shorth': est_shorth}


def main():
    plane = sys.argv[1] if len(sys.argv) > 1 else 'Y'
    event = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    s, c, i = cc.components(plane, event)
    nw, nt = s.shape
    sig_int = float(np.median(np.abs(i)) / 0.6745)
    # busiest block
    ng = cc.full_groups(nw)
    energies = [float(np.sum(s[g * GS:(g + 1) * GS] ** 2)) for g in range(ng)]
    g0 = int(np.argmax(energies))
    lo = g0 * GS; hi = lo + GS
    noisy = (s + c + i)[lo:hi]                 # (64, nt) float (pre-digitize for clarity)
    truth = c[lo, :]                           # (nt,) true coherent for this block
    occ = (np.abs(s[lo:hi]) > 5.0).sum(0)      # per-tick signal occupancy (#wires)

    print(f"=== plane {plane} event {event}, block {g0} (sigma_int~{sig_int:.2f}) ===")
    print(f"per-tick occupancy: max {occ.max()}/64, mean {occ.mean():.1f}, "
          f"ticks with >0 signal {100*(occ>0).mean():.1f}%, >32 wires {100*(occ>32).mean():.2f}%")
    # error of each estimator vs true coherent, overall and by occupancy band
    errs = {k: np.array([fn(noisy[:, t]) for t in range(nt)]) - truth for k, fn in ESTS.items()}
    print(f"\n   {'estimator':>9}  {'RMS err (all)':>13}  {'occ=0':>8}  {'1-16':>8}  {'17-32':>8}  {'>32':>8}")
    bands = [(occ == 0), (occ >= 1) & (occ <= 16), (occ >= 17) & (occ <= 32), (occ > 32)]
    for k in ESTS:
        e = errs[k]
        row = [np.sqrt(np.mean(e[b] ** 2)) if b.any() else np.nan for b in bands]
        print(f"   {k:>9}  {np.sqrt(np.mean(e**2)):>13.4f}  " +
              "  ".join(f"{v:>8.4f}" for v in row))

    # ---- figures ----
    # find a tick window around the densest signal crossing
    tcen = int(np.argmax(occ)); t0 = max(0, tcen - 250); t1 = min(nt, t0 + 500)
    tw = slice(t0, t1); tt = np.arange(t0, t1)
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].plot(tt, truth[tw], 'k', lw=2.2, label='true coherent', zorder=5)
    for k, col in zip(['median', 'masked', 'huber', 'shorth'],
                      ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']):
        ax[0].plot(tt, (errs[k] + truth)[tw], col, lw=1.0, alpha=0.8, label=k)
    ax[0].set_title(f'{plane} block {g0}: estimators vs true coherent (around a track crossing)')
    ax[0].legend(ncol=5, fontsize=8); ax[0].set_ylabel('ADC')
    ax2 = ax[0].twinx(); ax2.fill_between(tt, occ[tw], color='gray', alpha=0.15)
    ax2.set_ylabel('occupancy (#wires)', color='gray')
    # error vs occupancy
    for k, col in zip(['median', 'masked', 'huber', 'shorth', 'q25'],
                      ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']):
        oc = np.arange(0, 64)
        me = [np.sqrt(np.mean(errs[k][occ == o] ** 2)) if (occ == o).any() else np.nan for o in oc]
        ax[1].plot(oc, me, col, marker='.', ms=3, label=k)
    ax[1].set_xlabel('per-tick occupancy (#wires with signal)')
    ax[1].set_ylabel('RMS coherent error (ADC)')
    ax[1].set_title('estimator error vs occupancy — which recovers coherent behind signal')
    ax[1].legend(fontsize=8); ax[1].set_ylim(0, 6)
    fig.suptitle(f'{plane} plane event {event} — recovering coherent behind signal', fontweight='bold')
    fig.tight_layout()
    p = os.path.join(cc.FIGDIR, f'fig13_explore_{plane}.png')
    fig.savefig(p, dpi=130); plt.close(fig)
    print('\nsaved', p)

    # histograms at low/med/high occupancy ticks
    levels = [(0, 'occ=0'), (10, 'occ~10'), (25, 'occ~25'), (40, 'occ~40')]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    for j, (target, lab) in enumerate(levels):
        cand = np.where(np.abs(occ - target) <= 1)[0]
        if len(cand) == 0:
            ax[j].set_title(f'{lab}: none'); continue
        t = cand[len(cand) // 2]
        x = noisy[:, t]
        ax[j].hist(x, bins=24, color='lightgray', edgecolor='gray')
        ax[j].axvline(truth[t], color='k', lw=2, label='true coherent')
        for k, col in zip(['median', 'huber', 'shorth'], ['tab:blue', 'tab:green', 'tab:red']):
            ax[j].axvline(ESTS[k](x), color=col, ls='--', lw=1.3, label=k)
        ax[j].set_title(f'{lab} (tick {t})', fontsize=10); ax[j].set_xlabel('wire value ADC')
        if j == 0:
            ax[j].legend(fontsize=7)
    fig.suptitle(f'{plane} block {g0}: 64-wire value distribution at increasing signal occupancy',
                 fontweight='bold')
    fig.tight_layout()
    p2 = os.path.join(cc.FIGDIR, f'fig14_hist_{plane}.png')
    fig.savefig(p2, dpi=130); plt.close(fig)
    print('saved', p2)


if __name__ == '__main__':
    main()
