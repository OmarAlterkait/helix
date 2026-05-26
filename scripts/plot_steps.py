"""Step-by-step figures for the 3-pass coherent removal + wavelet pipeline.

Uses JAXTPC-matched shaped noise (FFT series + white).

Figures:
  step1  -- (a) input, (b) group median, (c) residual
  step2  -- (a) residual distribution + threshold, (b) clean truth, (c) mask accuracy
  step3  -- (a) clean truth, (b) mask seed + dilation, (c) mask accuracy
  step4  -- (a) coherent estimate, (b) alpha map, (c) pass-1 result
  step5  -- (a) cleaned1 with detections, (b) augmented mask, (c) pass-2 result
  step6  -- (a) cleaned2, (b) final mask, (c) pass-3 result
  step7  -- (a) coherent output, (b) wavelet reconstruction, (c) removed
  step8  -- 2x2: (a) clean, (b) noisy, (c) output, (d) error  x3 (1/2/3 pass)
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
NOISE_X, NOISE_Y, NOISE_Z = 0.90, 0.79, 0.22

EVT = 2
ZW = slice(1216, 1536)
ZT = slice(1200, 1800)
EXT = [ZW.start, ZW.stop, ZT.start, ZT.stop]

_noise_npz = np.load('/home/oalterka/desktop_linux/JAXTPC/config/noise_spectrum.npz')

def get_noise_spectrum(n_ticks):
    n_freq = n_ticks // 2 + 1
    freqs = np.fft.rfftfreq(n_ticks, d=1.0 / 2e6)
    spectrum = np.interp(freqs, _noise_npz['spectrum_freqs_hz'], _noise_npz['spectrum_shape'])
    energy = np.sum(spectrum**2)
    if energy > 0:
        spectrum = spectrum / np.sqrt(energy) * np.sqrt(n_freq)
    return spectrum.astype(np.float32)

SPECTRUM = get_noise_spectrum(N_TICKS)

def generate_intrinsic_noise(nw, n_ticks, series_rms, rng):
    n_freq = n_ticks // 2 + 1
    r = rng.standard_normal((nw, n_freq)).astype(np.float32) * SPECTRUM
    i = rng.standard_normal((nw, n_freq)).astype(np.float32) * SPECTRUM
    cpx = r + 1j * i
    cpx[:, 0] = cpx[:, 0].real
    if n_ticks % 2 == 0:
        cpx[:, -1] = cpx[:, -1].real
    shaped = np.fft.irfft(cpx, n=n_ticks, axis=1)
    cur_rms = np.maximum(np.std(shaped, axis=1, keepdims=True), 1e-10)
    shaped = shaped / cur_rms * series_rms[:, None]
    white = rng.standard_normal((nw, n_ticks)).astype(np.float32) * NOISE_X
    return (shaped + white).astype(np.float32)


def symlog_im(ax, img, title, vmax=None, extent=None, cbar=True):
    if vmax is None:
        vmax = max(float(np.abs(img).max()), 20.0)
    norm = SymLogNorm(linthresh=2.0, vmin=-vmax, vmax=vmax, base=10)
    im = ax.imshow(img.T, aspect='auto', origin='lower', cmap='RdBu_r',
                   norm=norm, extent=extent)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('wire')
    ax.set_ylabel('tick')
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label='ADC')
    return im


def mask_rgb(mask_z, truth_z):
    caught = truth_z & mask_z
    missed = truth_z & ~mask_z
    false_alarm = mask_z & ~truth_z
    rgb = np.full((*mask_z.shape, 3), 1.0, dtype=np.float32)
    rgb[caught] = [0.15, 0.70, 0.25]
    rgb[missed] = [0.90, 0.20, 0.20]
    rgb[false_alarm] = [1.0, 0.65, 0.15]
    return rgb


def mask_stats(mask, truth):
    tp = int((truth & mask).sum())
    fn = int((truth & ~mask).sum())
    fp = int((mask & ~truth).sum())
    recall = 100 * tp / max(tp + fn, 1)
    return tp, fn, fp, recall


def glines(ax):
    first = ((ZW.start // GS) + 1) * GS
    for g in range(first, ZW.stop, GS):
        ax.axvline(g, color='gray', lw=0.5, ls='--', alpha=0.5)


def save(fig, name):
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(OUT_DIR, f'{name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  {name}')


def show_mask(ax, mask_z, truth_z, title):
    rgb = mask_rgb(mask_z, truth_z)
    tp, fn, fp, recall = mask_stats(mask_z, truth_z)
    ax.imshow(np.transpose(rgb, (1, 0, 2)), aspect='auto', origin='lower', extent=EXT)
    ax.set_title(f'{title}\ncaught {recall:.0f}% (green), missed {100-recall:.0f}% (red), '
                 f'false alarm {fp} (orange)', fontsize=9)
    ax.set_xlabel('wire')
    ax.set_ylabel('tick')


def setup():
    ds = JAXTPCDataset(data_root=DATA_ROOT, split='all', modalities=('sensor',))
    sp = ds[EVT]['sensor']['raw']['volume_0_U']
    nw = 1969
    clean = np.zeros((nw, N_TICKS), dtype=np.float32)
    clean[sp['wire'], sp['time']] = sp['value']

    wl = np.linspace(0.42, 4.63, nw)
    sigma_w = np.sqrt(NOISE_X**2 + (NOISE_Y + NOISE_Z * wl)**2).astype(np.float32)
    series_rms = (NOISE_Y + NOISE_Z * wl).astype(np.float32)

    seed = 1000 + EVT * 100
    ng = (nw + 63) // 64
    wf = generate_group_waveforms(ng, N_TICKS, beta=0.15, rms_adc=2.5,
                                  rng=np.random.default_rng(seed + 999))
    coh = broadcast_to_wires(wf, nw, GS)
    intrinsic = generate_intrinsic_noise(nw, N_TICKS, series_rms, np.random.default_rng(seed))
    noisy = np.clip(np.round(clean + coh + intrinsic + 1843), 0, 4095).astype(np.float32) - 1843

    # Pass 1
    gm1 = group_median(noisy, GS)
    gm1_full = broadcast_groups(gm1, nw, GS)
    residual1 = noisy - gm1_full
    mask1_raw = signal_mask(residual1, sigma_w, NSIGMA)
    mask1 = temporal_dilate(mask1_raw, DILATION)
    est1, nuf1 = masked_group_mean(noisy, mask1, GS)
    est1_full = broadcast_groups(est1, nw, GS)
    alpha1 = broadcast_groups(nuf1 / float(GS), nw, GS)
    cleaned1 = noisy - alpha1 * est1_full

    # Pass 2
    detect2 = signal_mask(cleaned1, sigma_w, NSIGMA)
    detect2 = temporal_dilate(detect2, DILATION)
    mask2 = mask1 | detect2
    est2, nuf2 = masked_group_mean(noisy, mask2, GS)
    est2_full = broadcast_groups(est2, nw, GS)
    alpha2 = broadcast_groups(nuf2 / float(GS), nw, GS)
    cleaned2 = noisy - alpha2 * est2_full

    # Pass 3
    detect3 = signal_mask(cleaned2, sigma_w, NSIGMA)
    detect3 = temporal_dilate(detect3, DILATION)
    mask3 = mask2 | detect3
    est3, nuf3 = masked_group_mean(noisy, mask3, GS)
    est3_full = broadcast_groups(est3, nw, GS)
    alpha3 = broadcast_groups(nuf3 / float(GS), nw, GS)
    cleaned3 = noisy - alpha3 * est3_full

    # Single-pass for comparison
    est1p, nuf1p = masked_group_mean(noisy, mask1, GS)
    est1p_full = broadcast_groups(est1p, nw, GS)
    alpha1p = broadcast_groups(nuf1p / float(GS), nw, GS)
    cleaned1p = noisy - alpha1p * est1p_full

    truth_mask = np.abs(clean) > 0

    # Wavelet on 3-pass output
    cfg = DetectorConfig(group_size=64, num_time_steps=N_TICKS)
    coeffs3 = per_wire_dwt(np.array(cleaned3, dtype=np.float32, copy=True),
                           cfg.wavelet, cfg.dwt_level)
    sb3 = estimate_subband_sigma(coeffs3)
    tc3 = hard_threshold(coeffs3, sb3, cfg.threshold_kappa, cfg.threshold_include_approx)
    recon3 = per_wire_idwt(tc3, cfg.wavelet, N_TICKS)

    # Wavelet on 1-pass and 2-pass for comparison
    coeffs1p = per_wire_dwt(np.array(cleaned1p, dtype=np.float32, copy=True),
                            cfg.wavelet, cfg.dwt_level)
    sb1p = estimate_subband_sigma(coeffs1p)
    tc1p = hard_threshold(coeffs1p, sb1p, cfg.threshold_kappa, cfg.threshold_include_approx)
    recon1p = per_wire_idwt(tc1p, cfg.wavelet, N_TICKS)

    coeffs2p = per_wire_dwt(np.array(cleaned2, dtype=np.float32, copy=True),
                            cfg.wavelet, cfg.dwt_level)
    sb2p = estimate_subband_sigma(coeffs2p)
    tc2p = hard_threshold(coeffs2p, sb2p, cfg.threshold_kappa, cfg.threshold_include_approx)
    recon2p = per_wire_idwt(tc2p, cfg.wavelet, N_TICKS)

    print(f'Region: wires [{ZW.start},{ZW.stop}), ticks [{ZT.start},{ZT.stop})')

    return dict(
        clean=clean, noisy=noisy, coh=coh, sigma_w=sigma_w, nw=nw,
        gm1_full=gm1_full, residual1=residual1,
        mask1_raw=mask1_raw, mask1=mask1,
        est1_full=est1_full, alpha1=alpha1, cleaned1=cleaned1,
        detect2=detect2, mask2=mask2, est2_full=est2_full, cleaned2=cleaned2,
        detect3=detect3, mask3=mask3, est3_full=est3_full, cleaned3=cleaned3,
        truth_mask=truth_mask,
        recon1p=recon1p, recon2p=recon2p, recon3=recon3,
    )


print('Setting up (shaped noise)...')
d = setup()

# -- Step 1: Input / Median / Residual --------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Step 1: Group median subtraction (shaped noise)', fontsize=13)
symlog_im(axes[0], d['noisy'][ZW, ZT], '(a) noisy input', extent=EXT); glines(axes[0])
symlog_im(axes[1], d['gm1_full'][ZW, ZT], '(b) group median', extent=EXT); glines(axes[1])
symlog_im(axes[2], d['residual1'][ZW, ZT], '(c) residual', extent=EXT); glines(axes[2])
save(fig, 'step1_median')

# -- Step 2: Threshold + Mask -----------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Step 2: Signal detection (3-sigma threshold)', fontsize=13)
ax = axes[0]
res_flat = d['residual1'][ZW, ZT].ravel()
ax.hist(res_flat, bins=200, density=True, alpha=0.7, range=(-15, 15))
thresh = NSIGMA * np.median(d['sigma_w'][ZW])
ax.axvline(thresh, color='r', ls='--', label=f'+{NSIGMA}sigma={thresh:.1f}')
ax.axvline(-thresh, color='r', ls='--', label=f'-{NSIGMA}sigma')
ax.set_xlabel('Residual (ADC)')
ax.set_title('(a) residual distribution')
ax.legend(fontsize=8)
symlog_im(axes[1], d['clean'][ZW, ZT], '(b) clean truth', extent=EXT)
show_mask(axes[2], d['mask1'][ZW, ZT], d['truth_mask'][ZW, ZT], '(c) mask accuracy')
save(fig, 'step2_mask')

# -- Step 3: Dilation --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Step 3: Temporal dilation ({DILATION} ticks)', fontsize=13)
symlog_im(axes[0], d['clean'][ZW, ZT], '(a) clean truth', extent=EXT)
m_raw = d['mask1_raw'][ZW, ZT]
m_dil = d['mask1'][ZW, ZT]
seed_only = m_raw & ~m_dil  # won't exist, dilation expands
dil_only = m_dil & ~m_raw
rgb = np.full((*m_raw.shape, 3), 1.0, dtype=np.float32)
rgb[m_raw] = [0.1, 0.2, 0.6]
rgb[dil_only] = [0.4, 0.6, 0.9]
axes[1].imshow(np.transpose(rgb, (1, 0, 2)), aspect='auto', origin='lower', extent=EXT)
axes[1].set_title('(b) seed (dark) + dilation (light)', fontsize=10)
axes[1].set_xlabel('wire'); axes[1].set_ylabel('tick')
show_mask(axes[2], d['mask1'][ZW, ZT], d['truth_mask'][ZW, ZT], '(c) mask accuracy')
save(fig, 'step3_dilation')

# -- Step 4: Pass 1 result ---------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Step 4: Pass 1 coherent estimate + subtraction', fontsize=13)
symlog_im(axes[0], d['est1_full'][ZW, ZT], '(a) coherent estimate', extent=EXT); glines(axes[0])
im = axes[1].imshow(d['alpha1'][ZW, ZT].T, aspect='auto', origin='lower', extent=EXT, cmap='viridis')
axes[1].set_title('(b) alpha (unflagged fraction)'); axes[1].set_xlabel('wire'); axes[1].set_ylabel('tick')
plt.colorbar(im, ax=axes[1], fraction=0.045, pad=0.02)
glines(axes[1])
symlog_im(axes[2], d['cleaned1'][ZW, ZT], '(c) pass-1 result', extent=EXT)
save(fig, 'step4_pass1')

# -- Step 5: Pass 2 result ---------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Step 5: Pass 2 (augmented mask, re-estimate from original)', fontsize=13)
symlog_im(axes[0], d['cleaned1'][ZW, ZT], '(a) cleaned1 input', extent=EXT)
show_mask(axes[1], d['mask2'][ZW, ZT], d['truth_mask'][ZW, ZT], '(b) augmented mask')
symlog_im(axes[2], d['cleaned2'][ZW, ZT], '(c) pass-2 result', extent=EXT)
save(fig, 'step5_pass2')

# -- Step 6: Pass 3 result ---------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Step 6: Pass 3 (final mask, final estimate)', fontsize=13)
symlog_im(axes[0], d['cleaned2'][ZW, ZT], '(a) cleaned2 input', extent=EXT)
show_mask(axes[1], d['mask3'][ZW, ZT], d['truth_mask'][ZW, ZT], '(b) final mask')
symlog_im(axes[2], d['cleaned3'][ZW, ZT], '(c) pass-3 result', extent=EXT)
save(fig, 'step6_pass3')

# -- Step 7: Wavelet ---------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Step 7: Wavelet sparsification (coif3, level 4)', fontsize=13)
symlog_im(axes[0], d['cleaned3'][ZW, ZT], '(a) coherent output', extent=EXT)
symlog_im(axes[1], d['recon3'][ZW, ZT], '(b) wavelet reconstruction', extent=EXT)
symlog_im(axes[2], (d['cleaned3'] - d['recon3'])[ZW, ZT], '(c) removed by wavelet', extent=EXT)
save(fig, 'step7_wavelet')

# -- Step 8: 1/2/3 pass comparison (2x2 each) -------------------------
clean_z = d['clean'][ZW, ZT]
noisy_z = d['noisy'][ZW, ZT]
vmax = max(float(np.abs(clean_z).max()), 20.0)
norm = SymLogNorm(linthresh=2.0, vmin=-vmax, vmax=vmax, base=10)
sig = np.abs(clean_z) > 0
tc = max(np.abs(clean_z[sig]).sum(), 1.0)

for n_pass, recon in [(1, d['recon1p']), (2, d['recon2p']), (3, d['recon3'])]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                             gridspec_kw={'wspace': 0.25, 'hspace': 0.25})
    recon_z = recon[ZW, ZT]
    error_z = recon_z - clean_z
    f0 = 1.0 - np.abs(error_z[sig]).sum() / tc
    bias = error_z[sig].mean()
    rms = np.sqrt((error_z[sig]**2).mean())

    for ax, img, title in [
        (axes[0, 0], clean_z, '(a) clean truth'),
        (axes[0, 1], noisy_z, '(b) noisy input'),
        (axes[1, 0], recon_z, f'(c) {n_pass}-pass output'),
        (axes[1, 1], error_z, '(d) error (c - a)'),
    ]:
        im = ax.imshow(img.T, aspect='auto', origin='lower', cmap='RdBu_r',
                       norm=norm, extent=EXT)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('wire'); ax.set_ylabel('tick')
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label='ADC')

    fig.suptitle(f'{n_pass}-pass pipeline (shaped noise)    '
                 f'[F0={f0:.4f}  bias={bias:+.3f}  rms={rms:.3f}]',
                 fontsize=13, fontweight='medium')
    fig.subplots_adjust(top=0.93)
    save(fig, f'step8_{n_pass}pass')

print('\nDone.')
