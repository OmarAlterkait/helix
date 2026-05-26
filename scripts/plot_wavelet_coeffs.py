"""Per-level wavelet coefficient 2D maps for each plane type.

Plots the raw thresholded DWT coefficients at their native resolution.
Each level has different time resolution (approx/d4 ~ 1/16, d3 ~ 1/4,
d2 ~ 1/2, d1 ~ 1/1), shown with nearest-neighbor interpolation so the
block structure is visible at coarse levels.

Figures:
  wavelet_coeffs_U_full / wavelet_coeffs_U_zoom
  wavelet_coeffs_V_full / wavelet_coeffs_V_zoom
  wavelet_coeffs_Y_full / wavelet_coeffs_Y_zoom
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
    per_wire_dwt, estimate_subband_sigma, hard_threshold,
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
BAND_DOWNSAMPLE = [16, 16, 8, 4, 2]

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

    coeffs = per_wire_dwt(np.array(cleaned, dtype=np.float32, copy=True),
                          cfg.wavelet, cfg.dwt_level)
    sigma_bands = estimate_subband_sigma(coeffs)
    tc = hard_threshold(coeffs, sigma_bands, cfg.threshold_kappa, cfg.threshold_include_approx)

    return tc


def plot_plane(pname, pinfo, zoomed):
    suffix = 'zoom' if zoomed else 'full'
    zw = pinfo[f'zw_{suffix}']
    zt = pinfo[f'zt_{suffix}']

    tc = run_pipeline(pinfo['key'], pinfo['nw'], pinfo['ped'], pinfo['wl'], pinfo['evt'])

    n_bands = len(tc)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                             gridspec_kw={'wspace': 0.25, 'hspace': 0.30})

    # Global vmax across all bands for consistent color scale
    vmax_all = 0.0
    for b in range(n_bands):
        band = tc[b]
        ds_factor = BAND_DOWNSAMPLE[b]
        n_coeffs_t = band.shape[1]
        c_start = max(int(zt.start / ds_factor) - 1, 0)
        c_stop = min(int(np.ceil(zt.stop / ds_factor)) + 1, n_coeffs_t)
        sub = band[zw, c_start:c_stop]
        vmax_all = max(vmax_all, float(np.abs(sub).max()))
    vmax_all = max(vmax_all, 5.0)
    norm = SymLogNorm(linthresh=1.0, vmin=-vmax_all, vmax=vmax_all, base=10)

    # First panel: compact summary
    axes[0, 0].axis('off')
    total_alive = sum(int(np.count_nonzero(tc[b])) for b in range(n_bands))
    total_all = sum(tc[b].size for b in range(n_bands))
    lines = []
    for b in range(n_bands):
        alive = int(np.count_nonzero(tc[b]))
        ds_f = BAND_DOWNSAMPLE[b]
        lines.append(f'{BAND_NAMES[b]:>9s}  {ds_f:>2d}x  {alive:>6,} alive')
    lines.append(f'{"total":>9s}       {total_alive:,} / {total_all:,}')
    axes[0, 0].text(0.5, 0.5, '\n'.join(lines), transform=axes[0, 0].transAxes,
                    fontsize=11, verticalalignment='center', horizontalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='#f0f0f0', alpha=0.8))

    panel_order = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for b in range(n_bands):
        r, c = panel_order[b]
        band = tc[b]
        ds_factor = BAND_DOWNSAMPLE[b]
        n_coeffs_t = band.shape[1]

        # Map tick range to coefficient indices
        c_start = max(int(zt.start / ds_factor) - 1, 0)
        c_stop = min(int(np.ceil(zt.stop / ds_factor)) + 1, n_coeffs_t)

        sub = band[zw, c_start:c_stop]
        # Extent: wire range stays physical, time extent maps coefficients back to ticks
        ext = [zw.start, zw.stop, c_start * ds_factor, c_stop * ds_factor]

        alive = int(np.count_nonzero(tc[b]))
        total = tc[b].size
        pct = 100 * alive / max(total, 1)

        im = axes[r, c].imshow(sub.T, aspect='auto', origin='lower',
                               cmap='RdBu_r', norm=norm, extent=ext,
                               interpolation='nearest')
        axes[r, c].set_title(
            f'{BAND_NAMES[b]}  ({n_coeffs_t} coeffs/wire, ×{ds_factor} ds, '
            f'{alive:,} alive, {pct:.1f}%)',
            fontsize=9, color=BAND_COLORS[b], fontweight='bold')
        axes[r, c].set_xlim(zw.start, zw.stop)
        axes[r, c].set_ylim(zt.start, zt.stop)
        plt.colorbar(im, ax=axes[r, c], fraction=0.045, pad=0.02, label='coeff')
        axes[r, c].set_xlabel('wire')
        axes[r, c].set_ylabel('tick')

    fig.suptitle(
        f'{pname} plane — wavelet coefficients at native resolution  '
        f'(coif3, level 4, event {pinfo["evt"]})',
        fontsize=13, fontweight='medium', y=0.98)
    fig.subplots_adjust(top=0.92)
    path = os.path.join(OUT_DIR, f'wavelet_coeffs_{pname}_{suffix}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wavelet_coeffs_{pname}_{suffix}')


print('Generating wavelet coefficient 2D maps...\n')
for pname, pinfo in PLANES.items():
    plot_plane(pname, pinfo, zoomed=False)
    plot_plane(pname, pinfo, zoomed=True)

print('\nDone.')
