"""Per-level wavelet coefficient visualization for each plane type.

For each plane (U, V, Y):
  - Run full pipeline: noise → coherent removal → wavelet sparsify
  - Reconstruct from each DWT band independently (approx, d4, d3, d2, d1)
  - Plot full reconstruction + per-band reconstructions (full view and zoomed)

Figures:
  wavelet_levels_U_full / wavelet_levels_U_zoom
  wavelet_levels_V_full / wavelet_levels_V_zoom
  wavelet_levels_Y_full / wavelet_levels_Y_zoom
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

sys.path.insert(0, '/home/oalterka/desktop_linux/pimm-data/src')
sys.path.insert(0, '/home/oalterka/desktop_linux/JAXTPC')
os.chdir('/home/oalterka/desktop_linux/helix')
sys.path.insert(0, '.')

from pimm_data.jaxtpc import JAXTPCDataset
from tools.coherent_noise import generate_group_waveforms, broadcast_to_wires
from helix._numpy_ops import (
    group_median, broadcast_groups, signal_mask,
    temporal_dilate, masked_group_mean,
    per_wire_dwt, per_wire_idwt, estimate_subband_sigma, hard_threshold,
)
from helix.config import DetectorConfig

OUT_DIR = '/home/oalterka/desktop_linux/helix/figures'
DATA_ROOT = '/home/oalterka/desktop_linux/JAXTPC/sample_edepsim/test_00_00_01/run_0026628546/output_200'
N_TICKS = 4321
GS = 64
NSIGMA = 3.0
DILATION = 11

PLANES = {
    'U': {'key': 'volume_0_U', 'nw': 1969, 'ped': 1843, 'wl': (0.42, 4.63), 'evt': 2,
           'zw_full': slice(900, 1700), 'zt_full': slice(800, 2400),
           'zw_zoom': slice(1216, 1536), 'zt_zoom': slice(1200, 1800)},
    'V': {'key': 'volume_0_V', 'nw': 1969, 'ped': 1843, 'wl': (0.42, 4.63), 'evt': 1,
           'zw_full': slice(850, 1650), 'zt_full': slice(1600, 3200),
           'zw_zoom': slice(1050, 1450), 'zt_zoom': slice(1950, 2550)},
    'Y': {'key': 'volume_0_Y', 'nw': 1443, 'ped': 410, 'wl': (2.33, 2.33), 'evt': 2,
           'zw_full': slice(400, 1200), 'zt_full': slice(800, 2400),
           'zw_zoom': slice(600, 920), 'zt_zoom': slice(1200, 1800)},
}

BAND_NAMES = ['approx', 'detail_4', 'detail_3', 'detail_2', 'detail_1']
BAND_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

cfg = DetectorConfig(group_size=GS, mask_threshold_nsigma=NSIGMA,
                     temporal_dilation_ticks=DILATION, n_passes=3,
                     num_time_steps=N_TICKS)

ds = JAXTPCDataset(data_root=DATA_ROOT, split='all', modalities=('sensor',))


def run_pipeline(plane_key, nw, ped, wl_range, evt):
    sp = ds[evt]['sensor']['raw'][plane_key]
    clean = np.zeros((nw, N_TICKS), dtype=np.float32)
    clean[sp['wire'], sp['time']] = sp['value']

    wl = np.linspace(wl_range[0], wl_range[1], nw)
    sigma_w = np.sqrt(0.90**2 + (0.79 + 0.22 * wl)**2).astype(np.float32)

    rng = np.random.default_rng(42)
    ng = (nw + 63) // 64
    wf = generate_group_waveforms(ng, N_TICKS, beta=0.15, rms_adc=2.5,
                                  rng=np.random.default_rng(999))
    coh = broadcast_to_wires(wf, nw, GS)
    intrinsic = sigma_w[:, None] * rng.standard_normal((nw, N_TICKS)).astype(np.float32)
    noisy = np.clip(np.round(clean + coh + intrinsic + ped), 0, 4095).astype(np.float32) - ped

    # 3-pass coherent removal
    gm = group_median(noisy, GS)
    gm_full = broadcast_groups(gm, nw, GS)
    residual = noisy - gm_full
    mask = signal_mask(residual, sigma_w, NSIGMA)
    mask = temporal_dilate(mask, DILATION)
    est, nuf = masked_group_mean(noisy, mask, GS)
    alpha = broadcast_groups(nuf / float(GS), nw, GS)
    cleaned = noisy - alpha * broadcast_groups(est, nw, GS)

    for _ in range(2):
        detect = signal_mask(cleaned, sigma_w, NSIGMA)
        detect = temporal_dilate(detect, DILATION)
        mask = mask | detect
        est, nuf = masked_group_mean(noisy, mask, GS)
        alpha = broadcast_groups(nuf / float(GS), nw, GS)
        cleaned = noisy - alpha * broadcast_groups(est, nw, GS)

    # Wavelet: DWT → threshold
    coeffs = per_wire_dwt(np.array(cleaned, dtype=np.float32, copy=True),
                          cfg.wavelet, cfg.dwt_level)
    sigma_bands = estimate_subband_sigma(coeffs)
    tc = hard_threshold(coeffs, sigma_bands, cfg.threshold_kappa, cfg.threshold_include_approx)

    # Full reconstruction
    recon = per_wire_idwt(tc, cfg.wavelet, N_TICKS)

    # Per-band reconstructions
    n_bands = len(tc)
    band_recons = []
    for b in range(n_bands):
        isolated = [np.zeros_like(c) for c in tc]
        isolated[b] = tc[b].copy()
        band_recons.append(per_wire_idwt(isolated, cfg.wavelet, N_TICKS))

    alive_counts = [int(np.count_nonzero(tc[b])) for b in range(n_bands)]
    total_counts = [tc[b].size for b in range(n_bands)]

    return recon, band_recons, alive_counts, total_counts


def plot_plane(pname, pinfo, zoomed):
    suffix = 'zoom' if zoomed else 'full'
    zw = pinfo[f'zw_{suffix}']
    zt = pinfo[f'zt_{suffix}']
    ext = [zw.start, zw.stop, zt.start, zt.stop]

    recon, band_recons, alive_counts, total_counts = run_pipeline(
        pinfo['key'], pinfo['nw'], pinfo['ped'], pinfo['wl'], pinfo['evt'])

    n_bands = len(band_recons)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    vmax = max(float(np.abs(recon[zw, zt]).max()), 20.0)
    norm = SymLogNorm(linthresh=2.0, vmin=-vmax, vmax=vmax, base=10)

    total_alive = sum(alive_counts)
    total_all = sum(total_counts)

    # (0,0): full reconstruction
    im = axes[0, 0].imshow(recon[zw, zt].T, aspect='auto', origin='lower',
                           cmap='RdBu_r', norm=norm, extent=ext)
    axes[0, 0].set_title(f'{pname}: full reconstruction ({total_alive:,} coefficients)',
                         fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=axes[0, 0], fraction=0.045, pad=0.02, label='ADC')

    # Per-band panels
    panel_order = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for b in range(n_bands):
        r, c = panel_order[b]
        img = band_recons[b][zw, zt]
        alive = alive_counts[b]
        total = total_counts[b]
        pct = 100 * alive / max(total, 1)

        im = axes[r, c].imshow(img.T, aspect='auto', origin='lower',
                               cmap='RdBu_r', norm=norm, extent=ext)
        axes[r, c].set_title(
            f'{BAND_NAMES[b]}  ({alive:,} / {total:,} alive, {pct:.1f}%)',
            fontsize=10, color=BAND_COLORS[b], fontweight='bold')
        plt.colorbar(im, ax=axes[r, c], fraction=0.045, pad=0.02, label='ADC')

    for ax in axes.flat:
        ax.set_xlabel('wire')
        ax.set_ylabel('tick')

    fig.suptitle(
        f'{pname} plane — per-band wavelet reconstruction  '
        f'(coif3, level 4, κ=1.0, event {pinfo["evt"]})',
        fontsize=13, fontweight='medium')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(OUT_DIR, f'wavelet_levels_{pname}_{suffix}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wavelet_levels_{pname}_{suffix}')


print('Generating wavelet level decomposition plots...\n')
for pname, pinfo in PLANES.items():
    plot_plane(pname, pinfo, zoomed=False)
    plot_plane(pname, pinfo, zoomed=True)

print('\nDone.')
