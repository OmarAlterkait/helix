"""Visualization for the optical wavelet decomposition.

Multi-level decomposition of a 1-D signal with physically-correct dyadic
time–scale placement (level-j coefficient k spans [k·2ʲ, (k+1)·2ʲ]). Two styles:
  style="bars" : per-level coefficient bars (bar width = cell width), per-level
                 amplitude autoscale.
  style="map"  : dyadic colored coefficient map.
"""
from __future__ import annotations

import numpy as np
import pywt


def decompose(signal, wavelet: str = "coif3", level: int = 8, mode: str = "periodization"):
    """Full DWT of a 1-D signal. Returns (bands, scales, names) with
    bands = [cA_L, cD_L, …, cD_1] and scales in samples."""
    lev = min(level, pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len))
    bands = pywt.wavedec(np.asarray(signal, float), wavelet, level=lev, mode=mode)
    scales = [2 ** lev] + [2 ** (lev - k) for k in range(lev)]
    names = [f"A{lev}"] + [f"D{lev - k}" for k in range(lev)]
    return bands, scales, names


def _cells(band, scale, s0, s1, tick_ns):
    """Coeff centers (µs), width (µs), values for `band` within sample window [s0,s1]."""
    k0, k1 = max(0, int(np.floor(s0 / scale))), min(len(band), int(np.ceil(s1 / scale)))
    k = np.arange(k0, k1)
    return (k + 0.5) * scale * tick_ns / 1000.0, scale * tick_ns / 1000.0, band[k0:k1]


def prompt_window(signal, tick_ns=1.0, before_ns=250, width_ns=4000):
    """Sample window centered on the prompt pulse (global most-negative sample)."""
    i = int(np.argmin(signal))
    s0 = max(0, i - int(before_ns / tick_ns))
    return s0, min(len(signal), s0 + int(width_ns / tick_ns))


def plot_decomposition(signal, *, tick_ns=1.0, wavelet="coif3", level=8, mode="periodization",
                       window=None, style="bars", title=None, fig=None):
    """Plot the multi-level decomposition of one signal. style='bars'|'map'."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    bands, scales, names = decompose(signal, wavelet, level, mode)
    nb = len(bands)
    s0, s1 = window or prompt_window(signal, tick_ns)
    t = np.arange(s0, s1) * tick_ns / 1000.0
    fig = fig or plt.figure(figsize=(9, 1.0 * (nb + 1)))

    if style == "bars":
        axes = fig.subplots(nb + 1, 1, sharex=True, gridspec_kw={"hspace": 0.0})
        axes[0].plot(t, np.asarray(signal)[s0:s1], color="#16324f", lw=0.9)
        axes[0].set_ylabel("signal", rotation=0, ha="right", va="center", fontsize=8)
        cmap = plt.cm.viridis(np.linspace(0.15, 0.92, nb))
        for i, (b, sc, nm) in enumerate(zip(bands, scales, names)):
            ax = axes[i + 1]
            cen, w, val = _cells(b, sc, s0, s1, tick_ns)
            ax.bar(cen, val, width=w * 0.9, color=cmap[i], edgecolor="none")
            ax.axhline(0, color="k", lw=0.3); ax.margins(y=0.15); ax.tick_params(labelleft=False)
            ax.set_ylabel(f"{nm}\n{sc*tick_ns:.0f}ns", rotation=0, ha="right", va="center", fontsize=7.5)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].spines[["top", "right"]].set_visible(False)
        axes[-1].set_xlabel("Time [µs]")
    elif style == "map":
        gs = fig.add_gridspec(nb + 1, 1, height_ratios=[2.2] + [1] * nb, hspace=0.0)
        axs = fig.add_subplot(gs[0]); axs.plot(t, np.asarray(signal)[s0:s1], color="#16324f", lw=0.9)
        axs.set_xlim(t[0], t[-1]); axs.set_xticklabels([])
        axs.set_ylabel("signal", rotation=0, ha="right", va="center", fontsize=8)
        axs.spines[["top", "right"]].set_visible(False)
        vmax = max(np.abs(b).max() for b in bands)
        norm = SymLogNorm(linthresh=max(vmax * 0.002, 1.0), vmin=-vmax, vmax=vmax)
        im = None
        for i, (b, sc, nm) in enumerate(zip(bands, scales, names)):
            ax = fig.add_subplot(gs[i + 1])
            _, _, val = _cells(b, sc, s0, s1, tick_ns)
            im = ax.imshow(val[None, :], aspect="auto", origin="lower", cmap="RdBu_r", norm=norm,
                           extent=[t[0], t[-1], 0, 1], interpolation="nearest")
            ax.set_yticks([]); ax.set_ylabel(nm, rotation=0, ha="right", va="center", fontsize=8)
            ax.set_xticklabels([]) if i < nb - 1 else ax.set_xlabel("Time [µs]")
        fig.colorbar(im, ax=fig.axes, fraction=0.02, pad=0.01, label="wavelet coefficient")
    else:
        raise ValueError("style must be 'bars' or 'map'")
    if title:
        fig.suptitle(title, fontsize=12)
    return fig
