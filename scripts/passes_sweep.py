"""Sweep n_passes (1-5) for different (sigma, dilation) pairs.

Coherent-only metrics (no wavelet). Per-plane.
3 sigma values × 3 dilation values = 9 curves, each with 5 points (1-5 passes).

Plots:
  1. |Bias| vs sig_rms — per plane
  2. (reserved for sparsity — skip since no wavelet)
  3. noise_rms vs sig_rms — per plane
"""

import sys, os, numpy as np, time

sys.path.insert(0, '/home/oalterka/desktop_linux/pimm-data/src')
sys.path.insert(0, '/home/oalterka/desktop_linux/JAXTPC')
os.chdir('/home/oalterka/desktop_linux/helix')
sys.path.insert(0, '.')

import helix._backend as _b
_b._backend = 'jax'
import jax, jax.numpy as jnp
from functools import partial

from pimm_data.jaxtpc import JAXTPCDataset
from tools.coherent_noise import generate_group_waveforms, broadcast_to_wires
from helix._jax_ops import (
    group_median, broadcast_groups, signal_mask,
    temporal_dilate, masked_group_mean,
    pad_to_groups, pad_mask_to_groups,
)

print(f'JAX: {jax.devices()}')

DATA_ROOT = '/home/oalterka/desktop_linux/JAXTPC/sample_edepsim/test_00_00_01/run_0026628546/output_200'
N_TICKS = 4321
GS = 64

PLANES = [
    ('volume_0_U', 1969, 1843, (0.42, 4.63), 'U (induction)'),
    ('volume_0_V', 1969, 1843, (0.42, 4.63), 'V (induction)'),
    ('volume_0_Y', 1443, 410, (2.33, 2.33), 'Y (collection)'),
]

SIGMA_CACHE = {}
for _, nw, _, wlr, _ in PLANES:
    if nw not in SIGMA_CACHE:
        wl = np.linspace(wlr[0], wlr[1], nw)
        SIGMA_CACHE[nw] = jnp.array(np.sqrt(0.90**2 + (0.79 + 0.22 * wl)**2).astype(np.float32))


@partial(jax.jit, static_argnums=(1, 2))
def add_noise(clean_plus_coh, nw, ped, sw, key):
    intr = sw[:, None] * jax.random.normal(key, (nw, N_TICKS), dtype=jnp.float32)
    return jnp.round(clean_plus_coh + intr + ped).clip(0, 4095) - ped


@partial(jax.jit)
def compute_metrics(clean_j, output_j):
    sig = jnp.abs(clean_j) > 0
    nsig = jnp.maximum(jnp.sum(sig).astype(jnp.float32), 1.0)
    err = (output_j - clean_j) * sig
    bias = jnp.sum(err) / nsig
    sig_rms = jnp.sqrt(jnp.sum(err**2) / nsig)
    npix = ~sig
    nn = jnp.maximum(jnp.sum(npix).astype(jnp.float32), 1.0)
    noi_rms = jnp.sqrt(jnp.sum((output_j - clean_j)**2 * npix) / nn)
    return bias, sig_rms, noi_rms


def remove_coherent_custom(image_j, sigma_j, nsigma, dilation, n_passes, nw):
    image_p = pad_to_groups(image_j, GS)
    gm = group_median(image_p, GS)
    gm_full = broadcast_groups(gm, nw, GS)
    residual = image_j - gm_full
    mask = signal_mask(residual, sigma_j, nsigma)
    mask = temporal_dilate(mask, dilation)
    mask_p = pad_mask_to_groups(mask, GS)
    image_p = pad_to_groups(image_j, GS)
    est, nuf = masked_group_mean(image_p, mask_p, GS)
    est_full = broadcast_groups(est, nw, GS)
    alpha = broadcast_groups(nuf / float(GS), nw, GS)
    cleaned = image_j - alpha * est_full

    for _ in range(n_passes - 1):
        detect = signal_mask(cleaned, sigma_j, nsigma)
        detect = temporal_dilate(detect, dilation)
        mask = mask | detect
        mask_p = pad_mask_to_groups(mask, GS)
        image_p = pad_to_groups(image_j, GS)
        est, nuf = masked_group_mean(image_p, mask_p, GS)
        est_full = broadcast_groups(est, nw, GS)
        alpha = broadcast_groups(nuf / float(GS), nw, GS)
        cleaned = image_j - alpha * est_full

    return cleaned


ds = JAXTPCDataset(data_root=DATA_ROOT, split='all', modalities=('sensor',))
N_EVENTS = 30
N_SEEDS = 3

SIGMAS = [2.5, 3.0, 3.5]
DILATIONS = [7, 9, 11]
PASSES = [1, 2, 3, 4, 5]

results = {}
for ns in SIGMAS:
    for dil in DILATIONS:
        for np_ in PASSES:
            for pk, _, _, _, _ in PLANES:
                key = f's{ns}_d{dil}_p{np_}_{pk}'
                results[key] = {'bias': [], 'sig_rms': [], 'noi_rms': []}

t0 = time.perf_counter()
for evt_idx in range(N_EVENTS):
    sample = ds[evt_idx]
    raw = sample['sensor']['raw']

    for pk, nw, ped, _, pl in PLANES:
        if pk not in raw:
            continue
        sw = SIGMA_CACHE[nw]
        sp = raw[pk]
        clean_np = np.zeros((nw, N_TICKS), dtype=np.float32)
        clean_np[sp['wire'], sp['time']] = sp['value']
        clean_j = jnp.array(clean_np)

        for seed_idx in range(N_SEEDS):
            seed = 160000 + evt_idx * 100 + seed_idx
            ng = (nw + 63) // 64
            wf = generate_group_waveforms(ng, N_TICKS, beta=0.15, rms_adc=2.5,
                                          rng=np.random.default_rng(seed + 999))
            coh_j = jnp.array(broadcast_to_wires(wf, nw, GS))
            noisy_j = add_noise(clean_j + coh_j, nw, ped, sw, jax.random.PRNGKey(seed))

            for ns in SIGMAS:
                for dil in DILATIONS:
                    for np_ in PASSES:
                        cleaned = remove_coherent_custom(noisy_j, sw, ns, dil, np_, nw)
                        jax.block_until_ready(cleaned)
                        b, sr, nr = compute_metrics(clean_j, cleaned)
                        key = f's{ns}_d{dil}_p{np_}_{pk}'
                        results[key]['bias'].append(float(b))
                        results[key]['sig_rms'].append(float(sr))
                        results[key]['noi_rms'].append(float(nr))

    if (evt_idx + 1) % 10 == 0:
        elapsed = time.perf_counter() - t0
        eta = elapsed / (evt_idx + 1) * (N_EVENTS - evt_idx - 1)
        print(f'  event {evt_idx+1}/{N_EVENTS}  (elapsed {elapsed:.0f}s, eta {eta:.0f}s)', flush=True)

print(f'\nTotal: {time.perf_counter() - t0:.0f}s')

# ── Plot ──────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/home/oalterka/desktop_linux/helix/figures'

# Colors: one per (sigma, dilation) pair
colors_map = {}
cmap = plt.cm.tab10
idx = 0
for ns in SIGMAS:
    for dil in DILATIONS:
        colors_map[(ns, dil)] = cmap(idx)
        idx += 1

markers_pass = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v'}

for plot_idx, (y_metric, y_label, x_metric, x_label, fname) in enumerate([
    ('sig_rms', 'Signal RMS (ADC)', 'bias', '|Signal Bias| (ADC)', 'passes_bias_vs_rms'),
    ('noi_rms', 'Noise RMS (ADC)', 'sig_rms', 'Signal RMS (ADC)', 'passes_noise_vs_signal'),
    ('noi_rms', 'Noise RMS (ADC)', 'bias', '|Signal Bias| (ADC)', 'passes_bias_vs_noise'),
]):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (pk, _, _, _, pl) in zip(axes, PLANES):
        for ns in SIGMAS:
            for dil in DILATIONS:
                col = colors_map[(ns, dil)]
                xs, ys = [], []
                for np_ in PASSES:
                    key = f's{ns}_d{dil}_p{np_}_{pk}'
                    r = results[key]
                    if x_metric == 'bias':
                        x = abs(np.mean(r['bias']))
                    else:
                        x = np.mean(r[x_metric])
                    y = np.mean(r[y_metric])
                    xs.append(x)
                    ys.append(y)

                ax.plot(xs, ys, '-', color=col, alpha=0.6, linewidth=1.5)
                for i, np_ in enumerate(PASSES):
                    ax.plot(xs[i], ys[i], markers_pass[np_], color=col,
                            markersize=6, alpha=0.8)

                # Label the line
                ax.annotate(f'σ={ns} d={dil}', (xs[-1], ys[-1]),
                            textcoords='offset points', xytext=(4, 2),
                            fontsize=6, color=col, alpha=0.7)

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(pl, fontsize=12)
        ax.grid(True, alpha=0.3)

    # Legend for pass markers
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=markers_pass[p], color='gray', linestyle='',
                       markersize=6, label=f'{p} pass{"es" if p > 1 else ""}')
               for p in PASSES]
    fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f'{OUT}/{fname}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved {fname}.png')
