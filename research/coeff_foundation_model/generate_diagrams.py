#!/usr/bin/env python
"""Professional architecture diagrams for the wavelet-coefficient tokenizer.
Two figures: optical (1D per-sensor) and TPC (2D wire-plane). PNG + PDF.

Pipeline depicted (both modalities), with measured numbers:
  sensors -> forward noise (on-the-fly GPU) -> coif3 DWT + threshold (levels)
  -> asinh normalize -> patchify (per-band/hybrid) -> linear embed -> trunk.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# ---- palette -------------------------------------------------------------
INK = "#1f2933"; MUT = "#5b6770"; FAINT = "#9aa5b1"
CARD = "#f5f7f9"; CARDE = "#c2ccd6"
PAL = {
    "input":  "#2c7da0",   # teal  - data
    "noise":  "#bc4b51",   # red   - forward model
    "dwt":    "#6a4c93",   # purple- transform / levels
    "norm":   "#3a7d44",   # green - normalize
    "patch":  "#e09f3e",   # amber - tokenize
    "embed":  "#d6852b",   # dk amber - learned embed
    "trunk":  "#8b97a3",   # grey  - deferred (part two)
}
# dyadic band ramp (coarse -> fine), purple-to-blue
BANDRAMP = ["#3d2b56", "#4d3a73", "#5d4a8a", "#5566a3", "#4f7bb0",
            "#4f93b8", "#58a7bd", "#6fbac2", "#8fcac8", "#b3d9cf", "#d4e7da"]


def rcard(ax, x, y, w, h, fill, edge, lw=1.4, r=0.12, alpha=1.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, mutation_scale=1,
                 boxstyle=f"round,pad=0.02,rounding_size={r}",
                 fc=fill, ec=edge, lw=lw, alpha=alpha, ls=ls, zorder=2))


def varrow(ax, x, y0, y1, color=MUT):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                 mutation_scale=15, lw=2.0, color=color, zorder=1,
                 shrinkA=0, shrinkB=0))


def shape_badge(ax, x, y, text, color):
    ax.add_patch(FancyBboxPatch((x - 1.55, y - 0.21), 3.1, 0.42,
                 boxstyle="round,pad=0.02,rounding_size=0.2",
                 fc="white", ec=color, lw=1.1, zorder=4))
    ax.text(x, y, text, ha="center", va="center", fontsize=7.4,
            color=color, family="monospace", zorder=5)


def stage(ax, i, x, y, w, h, accent, title, op, detail):
    rcard(ax, x, y, w, h, CARD, accent, lw=1.7)
    ax.add_patch(Rectangle((x, y), 0.16, h, fc=accent, ec="none", zorder=3))
    # number disc
    ax.add_patch(Circle((x - 0.02, y + h - 0.02), 0.34, fc=accent, ec="white",
                 lw=1.6, zorder=6))
    ax.text(x - 0.02, y + h - 0.02, str(i), ha="center", va="center",
            color="white", fontsize=11, fontweight="bold", zorder=7)
    ax.text(x + 0.42, y + h - 0.42, title, ha="left", va="top", fontsize=11.5,
            fontweight="bold", color=INK)
    ax.text(x + 0.42, y + h - 0.95, op, ha="left", va="top", fontsize=9.3,
            color=PAL.get(accent, INK) if isinstance(accent, str) and accent.startswith("#") else INK)
    ax.text(x + 0.42, y + 0.30, detail, ha="left", va="bottom", fontsize=8.3,
            color=MUT, style="italic")


# ---- glyphs --------------------------------------------------------------

def band_ladder(ax, x, y, w, h, names, lengths, kept_top, drop_bot, title2d=False):
    """Dyadic band ladder: bar per level, width ~ relative length."""
    n = len(names)
    bh = h / n * 0.82
    gap = h / n * 0.18
    maxw = w
    lw = np.array(lengths, float)
    lw = maxw * (lw / lw.max())
    for k in range(n):
        yy = y + h - (k + 1) * (bh + gap) + gap
        col = BANDRAMP[int(round(k / max(n - 1, 1) * (len(BANDRAMP) - 1)))]
        is_kept = (k == 0 and kept_top)
        is_drop = (k == n - 1 and drop_bot)
        ec = "#2f9e44" if is_kept else (FAINT if is_drop else "#3a4750")
        ax.add_patch(Rectangle((x, yy), lw[k], bh, fc=col, ec=ec,
                     lw=1.6 if (is_kept or is_drop) else 0.8,
                     alpha=0.32 if is_drop else 1.0, ls="--" if is_drop else "-",
                     zorder=3))
        tag = "  KEPT (untouched)" if is_kept else ("  DROPPED (noise floor)" if is_drop else "")
        ax.text(x - 0.12, yy + bh / 2, names[k], ha="right", va="center",
                fontsize=7.6, color=INK if not is_drop else FAINT,
                family="monospace")
        if tag:
            ax.text(x + lw[k] + 0.12, yy + bh / 2, tag, ha="left", va="center",
                    fontsize=6.8, color=ec, style="italic")
    ax.annotate("", xy=(x - 0.95, y), xytext=(x - 0.95, y + h),
                arrowprops=dict(arrowstyle="-|>", color=MUT, lw=1.3))
    ax.text(x - 1.2, y + h / 2, "scale (level)", rotation=90, ha="center",
            va="center", fontsize=7.5, color=MUT)
    ax.text(x + w / 2, y + h + 0.18, "coeffs per band  ->",
            ha="center", va="bottom", fontsize=7.0, color=MUT)


def token_glyph_optical(ax, x, y, w, h):
    ax.text(x + w / 2, y + h - 0.05, "hybrid tokens", ha="center", va="top",
            fontsize=8.2, fontweight="bold", color=PAL["patch"])
    # column tokens (coarse A10..D4): stacked multi-band squares
    cx = x + 0.2
    for t in range(3):
        bx = cx + t * 1.05
        for b in range(6):
            col = BANDRAMP[b]
            ax.add_patch(Rectangle((bx, y + h - 1.0 - b * 0.16), 0.85, 0.14,
                         fc=col, ec="white", lw=0.4, zorder=3))
        ax.add_patch(Rectangle((bx - 0.04, y + h - 1.02), 0.93, 6 * 0.16 + 0.04,
                     fc="none", ec=PAL["patch"], lw=1.3, zorder=4))
    ax.text(cx + 1.6, y + h - 2.15, "A10..D4  (anchor 1024)", ha="center",
            va="top", fontsize=6.6, color=MUT)
    # per-band fine tokens (D3,D2): 1D windows
    fy = y + 0.55
    for r, (nm, col) in enumerate([("D3", BANDRAMP[8]), ("D2", BANDRAMP[10])]):
        for t in range(4):
            ax.add_patch(Rectangle((cx + t * 0.82, fy + r * 0.52), 0.72, 0.4,
                         fc=col, ec=PAL["patch"], lw=1.1, zorder=3))
        ax.text(cx + 4 * 0.82 + 0.15, fy + r * 0.52 + 0.2, nm, ha="left",
                va="center", fontsize=7.0, color=INK, family="monospace")
    ax.text(cx + 1.6, fy - 0.12, "per-band  P=64 windows", ha="center",
            va="top", fontsize=6.6, color=MUT)


def token_glyph_tpc(ax, x, y, w, h):
    ax.text(x + w / 2, y + h - 0.05, "per-band 2D patches", ha="center",
            va="top", fontsize=8.2, fontweight="bold", color=PAL["patch"])
    cx = x + 0.35
    for r, (nm, col) in enumerate([("A4", BANDRAMP[0]), ("D4", BANDRAMP[3]),
                                   ("D3", BANDRAMP[6]), ("D2", BANDRAMP[9])]):
        gy = y + h - 1.0 - r * 1.05
        # a 16x8 patch shown as small grid (downsampled to 8x4 cells)
        for wi in range(8):
            for ti in range(4):
                on = (wi + ti) % 3 == 0
                ax.add_patch(Rectangle((cx + ti * 0.16, gy + wi * 0.07),
                             0.15, 0.066, fc=col if on else "#e9edf1",
                             ec="white", lw=0.25, zorder=3))
        ax.add_patch(Rectangle((cx - 0.03, gy - 0.02), 4 * 0.16 + 0.04,
                     8 * 0.07 + 0.03, fc="none", ec=PAL["patch"], lw=1.2, zorder=4))
        ax.text(cx + 0.75, gy + 0.28, nm, ha="left", va="center", fontsize=7.4,
                color=INK, family="monospace")
        # dead-wire row marker
        ax.add_patch(Rectangle((cx, gy + 5 * 0.07), 4 * 0.16, 0.066,
                     fc="none", ec="#bc4b51", lw=1.0, ls=":", zorder=5))
    ax.text(cx + 1.5, y + 0.28, "16 wires x 8 band-ticks\n+ dead-wire bits (red)",
            ha="center", va="bottom", fontsize=6.5, color=MUT)


def sensor_glyph_optical(ax, x, y, w, h):
    # channel stack -> 1D trace with chunks
    for i in range(8):
        a = 1.0 - i * 0.09
        ax.plot([x, x + 1.4], [y + h - 0.3 - i * 0.18] * 2,
                color=PAL["input"], lw=1.2, alpha=a)
    ax.text(x + 0.7, y + h - 0.05, "162 ch", ha="center", va="bottom",
            fontsize=6.8, color=MUT)
    # waveform
    tx = np.linspace(0, 1, 200)
    base = y + 0.9
    wav = base + 0.0 * tx
    for c, amp in [(0.25, -0.7), (0.55, -0.45), (0.78, -0.3)]:
        wav = wav - amp * np.exp(-((tx - c) ** 2) / (2 * 0.012)) * np.exp(-(tx - c) / 0.25 * (tx > c))
    noise = 0.04 * np.sin(tx * 220)
    ax.plot(x + 2.0 + tx * 3.0, wav + noise, color=PAL["input"], lw=1.1)
    ax.plot([x + 2.0, x + 5.0], [base, base], color=FAINT, lw=0.7, ls=":")
    ax.text(x + 3.5, y + 0.2, "stored chunks ~36k samp", ha="center",
            va="bottom", fontsize=6.6, color=MUT)


def sensor_glyph_tpc(ax, x, y, w, h):
    labels = [("vol0 U", "ind"), ("vol0 V", "ind"), ("vol0 Y", "col"),
              ("vol1 U", "ind"), ("vol1 V", "ind"), ("vol1 Y", "col")]
    pw, ph = 1.5, 1.0
    for k, (nm, kind) in enumerate(labels):
        r, c = divmod(k, 3)
        px = x + 0.2 + c * (pw + 0.25)
        py = y + h - 1.2 - r * (ph + 0.45)
        col = "#2c7da0" if kind == "ind" else "#2f9e44"
        ax.add_patch(Rectangle((px, py), pw, ph, fc=col, ec="white", lw=1.0,
                     alpha=0.22, zorder=3))
        ax.add_patch(Rectangle((px, py), pw, ph, fc="none", ec=col, lw=1.2, zorder=4))
        # a faint track streak
        ax.plot([px + 0.2, px + pw - 0.2], [py + 0.25, py + ph - 0.25],
                color=col, lw=1.3, alpha=0.8, zorder=5)
        ax.text(px + pw / 2, py + ph + 0.04, nm, ha="center", va="bottom",
                fontsize=6.6, color=INK)
        ax.text(px + pw / 2, py + 0.08, kind, ha="center", va="bottom",
                fontsize=5.8, color=col, style="italic")


def trunk_glyph(ax, x, y, w, h, mode):
    ax.text(x + w / 2, y + h - 0.05, "VGGT-style trunk", ha="center", va="top",
            fontsize=8.2, fontweight="bold", color=PAL["trunk"])
    rows = [("within-plane / within-sensor", PAL["input"]),
            ("within-plane / within-sensor", PAL["input"]),
            ("within-plane / within-sensor", PAL["input"]),
            (mode, PAL["noise"])]
    for r, (nm, col) in enumerate(rows):
        yy = y + h - 1.0 - r * 0.72
        rcard(ax, x + 0.2, yy, w - 0.4, 0.5, "white", col, lw=1.3, r=0.08, ls="--")
        ax.text(x + w / 2, yy + 0.25, nm, ha="center", va="center",
                fontsize=7.0, color=col)
    ax.text(x + w / 2, y + 0.15, "3 : 1  local : global  (~3x attn cut)",
            ha="center", va="bottom", fontsize=6.6, color=MUT, style="italic")


# ---- figure builder ------------------------------------------------------

def build(modality):
    fig, ax = plt.subplots(figsize=(13.2, 17.6))
    ax.set_xlim(0, 16); ax.set_ylim(0, 24); ax.axis("off")
    fig.patch.set_facecolor("white")

    opt = modality == "optical"
    title = ("Wavelet-Coefficient Tokenizer  —  OPTICAL (PMT, 1D per-sensor)"
             if opt else
             "Wavelet-Coefficient Tokenizer  —  TPC (LArTPC wire planes, 2D)")
    ax.text(0.5, 23.6, title, ha="left", va="center", fontsize=15.5,
            fontweight="bold", color=INK)
    ax.text(0.5, 23.15, "data flow top -> bottom   |   on-the-fly GPU transforms, no pre-stored datasets   |   numbers = measured (typical event)",
            ha="left", va="center", fontsize=8.6, color=MUT)
    ax.plot([0.5, 15.5], [22.9, 22.9], color=CARDE, lw=1.0)

    cx, cw = 0.7, 9.3
    gx, gw = 10.55, 5.0
    h = 2.4
    pitch = 3.0
    top = 20.4

    if opt:
        stages = [
            ("input", "PMT optical waveforms", "162 channels (81 east / 81 west) | 1 ns ticks",
             "stored gap-compressed chunks ~36k samp  |  clean truth on disk (doraemon 20k ev)"),
            ("noise", "Forward noise  (on-the-fly, GPU)", "+ white Gaussian  sigma = 2.6 ADC  (verified)",
             "per-event resample each epoch = free augmentation  |  ~159k samp/chunk batch"),
            ("dwt", "Wavelet transform + threshold", "coif3, level-10 DWT (time)  ->  bands A10 .. D1",
             "per-chunk sigma (db1 MAD) | VisuShrink  |c| >= 1.2 sigma sqrt(2 ln N) | A10 kept, D1 dropped"),
            ("norm", "Normalize", "asinh( c / sigma_band )",
             "signed, compressive  (handles 1350x per-band value range)"),
            ("patch", "Patchify  (hybrid)", "column cells A10-D4 (anchor 1024)  +  per-band P=64 windows for D3/D2",
             "~5.8k tokens / event  (~27x sequence compression; event-independent +-10%)"),
            ("embed", "Linear patch embedding", "Linear([values, occupancy bits]) -> d_model  +  type emb  +  (phys-time, log-scale) PE",
             "0.20M params  |  NO deep encoder, NO attention pooling  |  reaches the per-patch PCA floor"),
            ("trunk", "Trunk  (part two)  +  decode head", "alternating within-sensor / global self-attention (VGGT-style)",
             "decode: Linear -> slots (occupancy + clean value)  =  denoising autoencoder"),
        ]
    else:
        stages = [
            ("input", "Wire-plane images", "6 planes = 2 volumes x {U, V, Y} | U/V 1969 wires (induction, bipolar), Y 1443 (collection, unipolar)",
             "4321 ticks (0.5 us) | shared cathode, direct same-tick cross-volume | ~580k events on disk"),
            ("noise", "Forward noise + digitize  (on-the-fly, GPU)", "coherent (per-64-wire group) + intrinsic  ->  12-bit digitize",
             "per-event resample = augmentation  |  clean truth = stored sensor (noise added at load)"),
            ("dwt", "Wavelet transform + smart removal + threshold", "coif3, level-4 DWT (time) -> A4,D4,D3,D2,D1 | coeff-space coherent removal (kgate=4)",
             "per-band sigma hard threshold (kappa=1) | A4 thresholded | D1 dropped | ~321k coeffs/event"),
            ("norm", "Normalize", "asinh( c / sigma_band )",
             "signed, compressive  |  per-(plane,band) sigma (induction vs collection differ)"),
            ("patch", "Patchify  (per-band 2D)", "16 wires x 8 band-ticks, in each band's native grid  |  occupied patches only",
             "~25.4k tokens / event  (~12.6x compression)  |  wire is the strongest axis (r=0.685)"),
            ("embed", "Linear patch embedding + dead-wire handling", "Linear([values, occ, dead bits]) -> d_model + band/plane emb + (time, wire) PE",
             "wire-kill augmentation (sim->real) | tree op kept (cross-scale value coupling, -15%)"),
            ("trunk", "Trunk  (part two)  +  decode head", "within-plane self-attn  ->  cross-plane self-attn (correspondence)",
             "decode: Linear -> slots (occupancy + clean value)  =  denoising autoencoder"),
        ]

    ys = [top - i * pitch for i in range(len(stages))]
    for i, (acc, ti, op, det) in enumerate(stages):
        y = ys[i]
        col = PAL[acc]
        alpha = 1.0
        if acc == "trunk":
            rcard(ax, cx, y, cw, h, "#eef1f4", col, lw=1.5, ls="--")
            ax.add_patch(Rectangle((cx, y), 0.16, h, fc=col, ec="none", zorder=3))
            ax.add_patch(Circle((cx - 0.02, y + h - 0.02), 0.34, fc=col, ec="white", lw=1.6, zorder=6))
            ax.text(cx - 0.02, y + h - 0.02, str(i), ha="center", va="center", color="white", fontsize=11, fontweight="bold", zorder=7)
            ax.text(cx + 0.42, y + h - 0.42, ti, ha="left", va="top", fontsize=11.5, fontweight="bold", color=INK)
            ax.text(cx + 0.42, y + h - 0.95, op, ha="left", va="top", fontsize=9.3, color=col)
            ax.text(cx + 0.42, y + 0.30, det, ha="left", va="bottom", fontsize=8.3, color=MUT, style="italic")
            ax.text(cx + cw - 0.2, y + h - 0.3, "deferred", ha="right", va="top",
                    fontsize=8, color=col, style="italic", fontweight="bold")
        else:
            stage(ax, i, cx, y, cw, h, col, ti, op, det)
        if i < len(stages) - 1:
            varrow(ax, cx + cw + 0.55, y - 0.06, y - pitch + h + 0.06)

    # data-shape badges on the arrows
    bx = cx + cw + 0.55
    if opt:
        badges = ["clean chunks", "noisy waveform", "sparse coeffs ~159k",
                  "asinh coeffs", "tokens ~5.8k", "embedded tokens"]
    else:
        badges = ["clean planes 6x", "noisy 12-bit", "coeffs ~321k",
                  "asinh coeffs", "tokens ~25.4k", "embedded tokens"]
    for i, b in enumerate(badges):
        ym = (ys[i] + (ys[i + 1] + h)) / 2
        shape_badge(ax, bx, ym, b, PAL[stages[i + 1][0]] if stages[i+1][0] != "trunk" else PAL["embed"])

    # glyph panel
    gpe = "#dde3e9"
    rcard(ax, gx - 0.25, 1.7, gw + 0.5, top + h - 1.7, "#fbfcfd", gpe, lw=1.0, r=0.05)
    ax.text(gx + gw / 2, top + h - 0.18, "structural detail", ha="center",
            va="top", fontsize=8.4, color=MUT, fontweight="bold")
    (sensor_glyph_optical if opt else sensor_glyph_tpc)(ax, gx, ys[0] - 0.55, gw, h)
    band_ladder(ax, gx + 1.3, ys[2] - 0.15, gw - 2.0, h + 0.35,
                (["A10"] + [f"D{j}" for j in range(10, 0, -1)]) if opt else ["A4", "D4", "D3", "D2", "D1"],
                ([1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if opt else [271, 271, 542, 1084, 2168]),
                kept_top=True, drop_bot=True)
    (token_glyph_optical if opt else token_glyph_tpc)(ax, gx, ys[4] - 0.15, gw, h + 0.35)
    trunk_glyph(ax, gx, ys[6] - 0.15, gw, h + 0.35,
                "across sensors (light)" if opt else "across planes (correspondence)")

    # headline result box
    res = ("RESULT  —  linear tokenizer denoising MSE = 0.0351  (asinh space),  BELOW the classical "
           "kept-coefficient baseline 0.0388.\nTokens are strictly better than what production keeps; "
           "the tokenizer is provably not the limiting stage.   +2 attn blocks -> 0.027."
           if opt else
           "RESULT  —  linear tokenizer (per-band 2D, +wire mixing) denoises far below the deep-substrate "
           "baseline 2.48.\nWire axis dominates (r=0.685); cross-scale value coupling kept via tree op.  "
           "Target = classical baseline 0.56.   [acceptance run in progress]")
    rcard(ax, cx, 0.35, gx + gw - 0.25 - cx, 0.95, "#fff8ec", PAL["patch"], lw=1.5, r=0.08)
    ax.text(cx + 0.3, 0.82, res, ha="left", va="center", fontsize=8.6, color="#7a5a18")

    # legend
    leg = [("input", "data"), ("noise", "forward model"), ("dwt", "transform / levels"),
           ("patch", "tokenize (learned)"), ("trunk", "deferred (part two)")]
    lx = cx + 0.1
    ax.text(lx, 1.62, "stage type:", fontsize=7.6, color=MUT, fontweight="bold")
    for k, (acc, nm) in enumerate(leg):
        ax.add_patch(Rectangle((lx + 1.3 + k * 1.85, 1.5, ), 0.28, 0.22, fc=PAL[acc], ec="none"))
        ax.text(lx + 1.64 + k * 1.85, 1.61, nm, fontsize=6.8, color=INK, va="center")

    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"architecture_{modality}.{ext}"),
                    dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote architecture_{modality}.png/.pdf")


if __name__ == "__main__":
    build("optical")
    build("tpc")
