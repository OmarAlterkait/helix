"""Event gallery with JAXTPC-matched shaped noise.

For each event and plane, produces two 2x2 figures:
  1. Coherent removal only: (a) clean, (b) noisy, (c) coherent output, (d) error
  2. Full pipeline: (a) clean, (b) noisy, (c) full output, (d) error

Uses spectrally-shaped intrinsic noise matching JAXTPC's model.
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

OUT_DIR = '/home/oalterka/desktop_linux/helix/figures/gallery'
os.makedirs(OUT_DIR, exist_ok=True)
DATA_ROOT = '/home/oalterka/desktop_linux/JAXTPC/sample_edepsim/test_00_00_01/run_0026628546/output_200'
N_TICKS = 4321
GS = 64
NSIGMA = 3.0
DILATION = 11

NOISE_X = 0.90
NOISE_Y = 0.79
NOISE_Z = 0.22

_noise_npz = np.load('/home/oalterka/desktop_linux/JAXTPC/config/noise_spectrum.npz')

def get_noise_spectrum(n_ticks, sampling_rate_hz=2e6):
    n_freq = n_ticks // 2 + 1
    freqs = np.fft.rfftfreq(n_ticks, d=1.0 / sampling_rate_hz)
    spectrum = np.interp(freqs, _noise_npz['spectrum_freqs_hz'],
                         _noise_npz['spectrum_shape'])
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

cfg = DetectorConfig(group_size=GS, mask_threshold_nsigma=NSIGMA,
                     temporal_dilation_ticks=DILATION, n_passes=3,
                     num_time_steps=N_TICKS)

PLANE_DEFS = {
    'U': {'key': 'volume_0_U', 'nw': 1969, 'ped': 1843, 'wl': (0.42, 4.63)},
    'V': {'key': 'volume_0_V', 'nw': 1969, 'ped': 1843, 'wl': (0.42, 4.63)},
    'Y': {'key': 'volume_0_Y', 'nw': 1443, 'ped': 410,  'wl': (2.33, 2.33)},
}

EVENTS = [
    {'evt': 2, 'planes': {
        'U': slice(1000, 1500), 'V': slice(700, 1200), 'Y': slice(500, 1000)},
     'ticks': {
        'U': slice(1000, 1800), 'V': slice(1000, 1800), 'Y': slice(1000, 1800)}},
    {'evt': 6, 'planes': {
        'U': slice(800, 1300), 'V': slice(500, 1000), 'Y': slice(500, 1000)},
     'ticks': {
        'U': slice(600, 1400), 'V': slice(200, 1000), 'Y': slice(200, 1000)}},
    {'evt': 15, 'planes': {
        'U': slice(700, 1200), 'V': slice(800, 1300), 'Y': slice(700, 1200)},
     'ticks': {
        'U': slice(3400, 4200), 'V': slice(400, 1200), 'Y': slice(3400, 4200)}},
    {'evt': 16, 'planes': {
        'U': slice(400, 900), 'V': slice(1000, 1500), 'Y': slice(400, 900)},
     'ticks': {
        'U': slice(2000, 2800), 'V': slice(2000, 2800), 'Y': slice(2200, 3000)}},
]

ds = JAXTPCDataset(data_root=DATA_ROOT, split='all', modalities=('sensor',))


def run_event_plane(evt, ptype):
    pdef = PLANE_DEFS[ptype]
    nw, ped = pdef['nw'], pdef['ped']
    sp = ds[evt]['sensor']['raw'].get(pdef['key'])
    if sp is None:
        return None

    clean = np.zeros((nw, N_TICKS), dtype=np.float32)
    clean[sp['wire'], sp['time']] = sp['value']

    wl = np.linspace(pdef['wl'][0], pdef['wl'][1], nw)
    sigma_w = np.sqrt(NOISE_X**2 + (NOISE_Y + NOISE_Z * wl)**2).astype(np.float32)
    series_rms = (NOISE_Y + NOISE_Z * wl).astype(np.float32)

    seed = 1000 + evt * 100
    ng = (nw + 63) // 64
    wf = generate_group_waveforms(ng, N_TICKS, beta=0.15, rms_adc=2.5,
                                  rng=np.random.default_rng(seed + 999))
    coh = broadcast_to_wires(wf, nw, GS)

    rng = np.random.default_rng(seed)
    intrinsic = generate_intrinsic_noise(nw, N_TICKS, series_rms, rng)
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

    # Wavelet
    coeffs = per_wire_dwt(np.array(cleaned, dtype=np.float32, copy=True),
                          cfg.wavelet, cfg.dwt_level)
    sigma_bands = estimate_subband_sigma(coeffs)
    tc = hard_threshold(coeffs, sigma_bands, cfg.threshold_kappa, cfg.threshold_include_approx)
    recon = per_wire_idwt(tc, cfg.wavelet, N_TICKS)

    return clean, noisy, cleaned, recon


def make_figure(clean, noisy, output, zw, zt, evt, ptype, stage_label, fname):
    ext = [zw.start, zw.stop, zt.start, zt.stop]
    error = output[zw, zt] - clean[zw, zt]

    vmax = max(float(np.abs(clean[zw, zt]).max()), 20.0)
    norm = SymLogNorm(linthresh=2.0, vmin=-vmax, vmax=vmax, base=10)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                             gridspec_kw={'wspace': 0.25, 'hspace': 0.25})

    panels = [
        (axes[0, 0], clean[zw, zt], '(a) clean truth', norm),
        (axes[0, 1], noisy[zw, zt], '(b) noisy input', norm),
        (axes[1, 0], output[zw, zt], f'(c) {stage_label} output', norm),
        (axes[1, 1], error, '(d) error (c - a)', norm),
    ]

    for ax, img, title, n in panels:
        im = ax.imshow(img.T, aspect='auto', origin='lower', cmap='RdBu_r',
                       norm=n, extent=ext, interpolation='nearest')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('wire')
        ax.set_ylabel('tick')
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label='ADC')

    sig_mask = np.abs(clean[zw, zt]) > 0
    n_sig = sig_mask.sum()
    if n_sig > 0:
        err_sig = error[sig_mask]
        bias = err_sig.mean()
        rms = np.sqrt((err_sig**2).mean())
        total_charge = np.abs(clean[zw, zt][sig_mask]).sum()
        f0 = 1.0 - np.abs(error[sig_mask]).sum() / max(total_charge, 1.0)
        metrics_str = f'F0={f0:.4f}  bias={bias:+.3f}  rms={rms:.3f}'
    else:
        metrics_str = 'no signal in region'

    fig.suptitle(
        f'Event {evt} - {ptype} plane - {stage_label}    [{metrics_str}]',
        fontsize=13, fontweight='medium')
    fig.subplots_adjust(top=0.93)
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  {fname}')


print('Generating event gallery (shaped noise)...\n')

for edef in EVENTS:
    evt = edef['evt']
    print(f'Event {evt}:')
    for ptype in ['U', 'V', 'Y']:
        zw = edef['planes'][ptype]
        zt = edef['ticks'][ptype]

        result = run_event_plane(evt, ptype)
        if result is None:
            print(f'  {ptype}: no data, skipping')
            continue
        clean, noisy, cleaned, recon = result

        make_figure(clean, noisy, cleaned, zw, zt, evt, ptype,
                    'coherent removal', f'evt{evt}_{ptype}_coherent.png')
        make_figure(clean, noisy, recon, zw, zt, evt, ptype,
                    'full pipeline', f'evt{evt}_{ptype}_full.png')

print('\nDone.')
