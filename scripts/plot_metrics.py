"""Key metrics plots for the 3-mode HELIX pipeline with JAXTPC-matched shaped noise.

Reads metrics_200evt_5seed_3mode.npz (200 events x 6 planes x 5 seeds x 3 modes).
Groups planes by type (U/V/Y) averaging over volumes.

Three modes: coherent_only, wavelet_only, full pipeline.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = '/home/oalterka/desktop_linux/helix/figures/metrics_200evt_5seed_3mode.npz'
OUT_DIR = '/home/oalterka/desktop_linux/helix/figures'

data = np.load(DATA_PATH)
plane_types = data['plane_types']
n_events = int(data['n_events'])
n_seeds = int(data['n_seeds'])
MODES = list(data['modes'])

TYPE_INDICES = {
    'U': [i for i, t in enumerate(plane_types) if t == 'U'],
    'V': [i for i, t in enumerate(plane_types) if t == 'V'],
    'Y': [i for i, t in enumerate(plane_types) if t == 'Y'],
}
PLANE_LABELS = {'U': 'U (induction)', 'V': 'V (induction)', 'Y': 'Y (collection)'}
PLANE_COLORS = {'U': '#4C72B0', 'V': '#DD8452', 'Y': '#55A868'}
PLANE_ORDER = ['U', 'V', 'Y']
MODE_LABELS = {'coherent_only': 'Coherent removal only',
               'wavelet_only': 'Wavelet only',
               'full': 'Full pipeline'}
MODE_STYLES = {'coherent_only': '--', 'wavelet_only': ':', 'full': '-'}


def get_flat(mode, metric, ptype):
    indices = TYPE_INDICES[ptype]
    return data[f'{mode}_{metric}'][:, indices, :].ravel()


def get_per_event(mode, metric, ptype):
    indices = TYPE_INDICES[ptype]
    return data[f'{mode}_{metric}'][:, indices, :].mean(axis=(1, 2))


def savefig(fig, name):
    fig.savefig(f'{OUT_DIR}/{name}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  {name}')


print(f'Loaded: {n_events} events x {len(plane_types)} planes x {n_seeds} seeds x {len(MODES)} modes\n')


# -- 1. F0 comparison across modes ------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, pt in zip(axes, PLANE_ORDER):
    for mode in MODES:
        f0 = get_flat(mode, 'f0', pt)
        ax.hist(f0, bins=50, alpha=0.5, label=f'{MODE_LABELS[mode]} (med={np.median(f0):.4f})',
                density=True, histtype='stepfilled',
                linestyle=MODE_STYLES[mode])
    ax.set_xlabel('F0')
    ax.set_ylabel('Density')
    ax.set_title(f'{PLANE_LABELS[pt]}', fontsize=12)
    ax.legend(fontsize=8)

fig.suptitle('F0 distribution by pipeline mode (shaped noise)', fontsize=13, fontweight='medium')
fig.tight_layout(rect=(0, 0, 1, 0.94))
savefig(fig, 'f0_by_mode')


# -- 2. Signal error: bias + RMS per mode -----------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                         gridspec_kw={'hspace': 0.30, 'wspace': 0.25})

for col, pt in enumerate(PLANE_ORDER):
    for mode in MODES:
        bias = get_flat(mode, 'signal_bias', pt)
        rms = get_flat(mode, 'signal_rms', pt)
        col_c = PLANE_COLORS[pt]
        ls = MODE_STYLES[mode]

        axes[0, col].hist(bias, bins=60, alpha=0.4, density=True,
                          label=f'{MODE_LABELS[mode]} (mu={bias.mean():+.2f})',
                          range=(-2, 1))
        axes[1, col].hist(rms, bins=60, alpha=0.4, density=True,
                          label=f'{MODE_LABELS[mode]} (mu={rms.mean():.2f})',
                          range=(0, 6))

    axes[0, col].axvline(0, color='k', ls='--', lw=0.8)
    axes[0, col].set_xlabel('Signal bias (ADC/pixel)')
    axes[0, col].set_title(f'{PLANE_LABELS[pt]} - Bias')
    axes[0, col].legend(fontsize=7)

    axes[1, col].set_xlabel('Signal RMS (ADC/pixel)')
    axes[1, col].set_title(f'{PLANE_LABELS[pt]} - RMS')
    axes[1, col].legend(fontsize=7)

axes[0, 0].set_ylabel('Density')
axes[1, 0].set_ylabel('Density')
fig.suptitle('Signal error by pipeline mode (shaped noise)', fontsize=13, fontweight='medium')
savefig(fig, 'signal_error_by_mode')


# -- 3. Bias vs RMS scatter per mode -----------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, mode in zip(axes, MODES):
    for pt in PLANE_ORDER:
        bias = get_per_event(mode, 'signal_bias', pt)
        rms = get_per_event(mode, 'signal_rms', pt)
        ax.scatter(rms, bias, s=8, alpha=0.5, color=PLANE_COLORS[pt],
                   label=PLANE_LABELS[pt])
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('Signal RMS (ADC/pixel)')
    ax.set_ylabel('Signal bias (ADC/pixel)')
    ax.set_title(MODE_LABELS[mode], fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Signal bias vs RMS per event (shaped noise)', fontsize=13, fontweight='medium')
fig.tight_layout(rect=(0, 0, 1, 0.94))
savefig(fig, 'bias_vs_rms_by_mode')


# -- 4. F0 vs activity per mode ---------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, mode in zip(axes, MODES):
    for pt in PLANE_ORDER:
        indices = TYPE_INDICES[pt]
        n_sig = data[f'{mode}_n_signal'][:, indices, :].mean(axis=(1, 2))
        nw = 1969 if pt != 'Y' else 1443
        activity = n_sig / (nw * 4321) * 100
        f0 = data[f'{mode}_f0'][:, indices, :].mean(axis=(1, 2))
        ax.scatter(activity, f0, s=8, alpha=0.5, color=PLANE_COLORS[pt],
                   label=PLANE_LABELS[pt])
    ax.set_xlabel('Signal activity (%)')
    ax.set_ylabel('F0')
    ax.set_title(MODE_LABELS[mode], fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('F0 vs event activity (shaped noise)', fontsize=13, fontweight='medium')
fig.tight_layout(rect=(0, 0, 1, 0.94))
savefig(fig, 'f0_vs_activity_by_mode')


# -- 5. Noise performance (full pipeline only) ------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for pt in PLANE_ORDER:
    nrms = get_flat('full', 'noise_rms', pt)
    crej = get_flat('full', 'coh_rejection', pt)
    col = PLANE_COLORS[pt]

    axes[0].hist(nrms, bins=50, alpha=0.6, color=col,
                 label=f'{PLANE_LABELS[pt]} (mu={nrms.mean():.3f})', density=True)
    axes[1].hist(crej, bins=50, alpha=0.6, color=col,
                 label=f'{PLANE_LABELS[pt]} (mu={crej.mean():.1f})', density=True)

axes[0].set_xlabel('Noise RMS (ADC)')
axes[0].set_ylabel('Density')
axes[0].set_title('Residual noise on empty pixels')
axes[0].legend(fontsize=8)

axes[1].set_xlabel('Coherent rejection ratio')
axes[1].set_ylabel('Density')
axes[1].set_title('Coherent noise suppression')
axes[1].legend(fontsize=8)

plt.tight_layout()
savefig(fig, 'noise_performance')


# -- 6. Compression (full pipeline only) ------------------------------

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for pt in PLANE_ORDER:
    nsig = get_flat('full', 'n_signal', pt) / 1000
    nk = get_flat('full', 'n_kept', pt) / 1000
    sp = get_flat('full', 'sparsity', pt) * 100
    col = PLANE_COLORS[pt]

    axes[0].hist(nsig, bins=40, alpha=0.6, color=col,
                 label=f'{PLANE_LABELS[pt]} (mu={nsig.mean():.1f}k)', density=True)
    axes[1].hist(nk, bins=40, alpha=0.6, color=col,
                 label=f'{PLANE_LABELS[pt]} (mu={nk.mean():.1f}k)', density=True)
    axes[2].hist(sp, bins=40, alpha=0.6, color=col,
                 label=f'{PLANE_LABELS[pt]} (mu={sp.mean():.2f}%)', density=True)

axes[0].set_xlabel('Signal pixels (thousands)')
axes[0].set_title('True signal pixel count')
axes[0].legend(fontsize=8)
axes[1].set_xlabel('Coefficients kept (thousands)')
axes[1].set_title('Wavelet coefficient count')
axes[1].legend(fontsize=8)
axes[2].set_xlabel('Sparsity (%)')
axes[2].set_title('Wavelet sparsity')
axes[2].legend(fontsize=8)
for ax in axes:
    ax.set_ylabel('Density')

plt.tight_layout()
savefig(fig, 'compression')


# -- 7. Compression ratio (full pipeline) -----------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for pt in PLANE_ORDER:
    indices = TYPE_INDICES[pt]
    nsig = data['full_n_signal'][:, indices, :].mean(axis=(1, 2)) / 1000
    nk = data['full_n_kept'][:, indices, :].mean(axis=(1, 2)) / 1000
    ratio = nsig / np.maximum(nk, 0.001)
    col = PLANE_COLORS[pt]

    axes[0].scatter(nsig, nk, s=8, alpha=0.5, color=col, label=PLANE_LABELS[pt])
    axes[1].hist(ratio, bins=np.linspace(0, 6, 50), alpha=0.6, color=col,
                 label=f'{PLANE_LABELS[pt]} (mu={ratio.mean():.2f}x)', density=True)

all_nsig = data['full_n_signal'][:, :, :].mean(axis=2).ravel() / 1000
all_nk = data['full_n_kept'][:, :, :].mean(axis=2).ravel() / 1000
xhi = np.percentile(all_nsig, 99) * 1.1
yhi = np.percentile(all_nk, 99) * 1.1
axes[0].plot([0, xhi], [0, xhi], 'k--', lw=0.8, alpha=0.5, label='1:1')
axes[0].plot([0, xhi], [0, xhi / 5], '-', color='#888888', lw=1.2, alpha=0.7, label='5x compression')
axes[0].set_xlim(0, xhi)
axes[0].set_ylim(0, yhi)
axes[0].set_xlabel('Signal pixels (thousands)')
axes[0].set_ylabel('Coefficients kept (thousands)')
axes[0].set_title('Wavelet coefficients vs signal pixels')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Compression ratio (signal pixels / coefficients)')
axes[1].set_ylabel('Density')
axes[1].set_title('Wavelet compression of signal')
axes[1].legend(fontsize=8)

plt.tight_layout()
savefig(fig, 'compression_ratio')


# -- 8. F0 stability across events (full pipeline) --------------------

fig, ax = plt.subplots(1, 1, figsize=(10, 4))

for pt in PLANE_ORDER:
    f0_per_evt = get_per_event('full', 'f0', pt)
    ax.plot(range(n_events), f0_per_evt, '-', alpha=0.7, linewidth=0.8,
            color=PLANE_COLORS[pt], label=PLANE_LABELS[pt])

ax.set_xlabel('Event index')
ax.set_ylabel('F0 (mean over volumes x seeds)')
ax.set_title('F0 stability across events (full pipeline, shaped noise)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig(fig, 'f0_stability')


# -- 9. Mode comparison bar chart (summary) ----------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

metrics_to_plot = [
    ('f0', 'F0 (median)', np.median),
    ('signal_bias', '|Bias| (ADC)', lambda x: abs(x.mean())),
    ('signal_rms', 'Signal RMS (ADC)', lambda x: x.mean()),
]

for ax, (metric, ylabel, agg_fn) in zip(axes, metrics_to_plot):
    x = np.arange(len(PLANE_ORDER))
    width = 0.25
    for i, mode in enumerate(MODES):
        vals = [agg_fn(get_flat(mode, metric, pt)) for pt in PLANE_ORDER]
        bars = ax.bar(x + i * width, vals, width, label=MODE_LABELS[mode], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([PLANE_LABELS[pt] for pt in PLANE_ORDER], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Pipeline mode comparison (shaped noise, 200 events x 5 seeds)',
             fontsize=13, fontweight='medium')
fig.tight_layout(rect=(0, 0, 1, 0.94))
savefig(fig, 'mode_comparison')


# -- 10. Summary table -------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')

headers = ['Mode', 'Plane', 'F0 (med)', 'F0 (IQR)', 'Bias', 'RMS',
           'Noise RMS', 'Coh Rej', 'Coefficients']
rows = []

for mode in MODES:
    for pt in PLANE_ORDER:
        f0 = get_flat(mode, 'f0', pt)
        bias = get_flat(mode, 'signal_bias', pt)
        rms = get_flat(mode, 'signal_rms', pt)
        nrms = get_flat(mode, 'noise_rms', pt)
        crej = get_flat(mode, 'coh_rejection', pt)
        nk = get_flat(mode, 'n_kept', pt)

        rows.append([
            MODE_LABELS[mode],
            PLANE_LABELS[pt],
            f'{np.median(f0):.4f}',
            f'{np.percentile(f0, 25):.4f}-{np.percentile(f0, 75):.4f}',
            f'{bias.mean():+.3f}',
            f'{rms.mean():.3f}',
            f'{nrms.mean():.3f}',
            f'{crej.mean():.1f}x',
            f'{nk.mean()/1000:.1f}k' if nk.mean() > 0 else '-',
        ])

table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.5)

for j in range(len(headers)):
    table[0, j].set_facecolor('#E8E8E8')
    table[0, j].set_text_props(fontweight='bold')

for i in range(1, len(rows) + 1):
    if (i - 1) % 3 == 0 and i > 1:
        for j in range(len(headers)):
            table[i, j].set_facecolor('#F5F5FF')

ax.set_title('HELIX pipeline summary (shaped noise, 3-pass, sigma=3.0, d=11, 200 events x 5 seeds)',
             fontsize=11, fontweight='medium', pad=20)
plt.tight_layout()
savefig(fig, 'summary_table')

print('\nDone.')
