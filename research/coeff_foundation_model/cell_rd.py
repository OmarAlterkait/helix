#!/usr/bin/env python
"""Closed-form rate-distortion of the tokenizer bottleneck (no training).

The bottleneck compresses one anchor cell's slot-vector (asinh values, zeros
at inactive slots) into d dims. Its LINEAR rate-distortion curve is the PCA
eigenspectrum of the cell-content distribution — computable for every
(anchor, d) at once from the existing dumps:

  distortion(d)        = sum_{i>d} lambda_i / n_slot      (per-slot MSE)
  per-band residual(d) = mean over the band's slots of sum_{i>d} lambda_i u_is^2

This replaces full 20k-step runs for tracing the (tokens <-> reconstruction)
tradeoff: PCA gives curve shapes and (anchor, d) ordering in seconds; tiny
per-cell AEs measure the nonlinear gap at selected points; ONE full-model run
confirms the chosen operating point. Caveats: linear floor (trained nets can
beat it), all-slot MSE (includes the easy zeros — per-band shapes are the
comparison currency, absolute values are not the trained metric).

Run:  python cell_rd.py
"""
import sys, os, json

sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
SIGMA = 2.6
D_GRID = [32, 64, 128, 256, 512, 1024]
MAX_CELLS = 60000

OPT_LEVELS = np.array([10] + list(range(10, 1, -1)))      # band_id 0..9 (D1 dropped)
TPC_LEV = np.array([4, 4, 3, 2])
TPC_LENS = np.array([271, 271, 542, 1084])
OPT_NAMES = ["A10", "D10", "D9", "D8", "D7", "D6", "D5", "D4", "D3", "D2"]
TPC_NAMES = ["A4", "D4", "D3", "D2"]


def slot_layout_opt(anchor):
    A = int(np.log2(anchor))
    widths = [1 << max(0, A - int(j)) for j in OPT_LEVELS]
    off = np.concatenate([[0], np.cumsum(widths)])
    return A, widths, off, int(off[-1])


def cells_optical(anchor):
    d = np.load(os.path.join(ART, "typical_event_coeffs_doraemon.npz"))
    keep = d["band_id"] < 10
    band = d["band_id"][keep].astype(np.int64)
    idx = d["idx"][keep].astype(np.int64)
    val = np.arcsinh(d["value"][keep] / SIGMA).astype(np.float32)
    chunk = d["chunk_id"][keep].astype(np.int64)
    A, widths, off, n_slot = slot_layout_opt(anchor)
    j = OPT_LEVELS[band]
    sh = A - j
    shp = np.maximum(sh, 0)
    cell_loc = np.where(sh >= 0, idx >> shp, idx << np.maximum(-sh, 0))
    within = np.where(sh >= 0, idx & ((1 << shp) - 1), 0)
    slot = off[band] + within
    key = chunk * (1 << 22) + cell_loc
    uniq, cell = np.unique(key, return_inverse=True)
    M = np.zeros((len(uniq), n_slot), np.float32)
    M[cell, slot] = val
    return M, band, slot, n_slot, off, widths


def cells_tpc(pw):
    d = np.load(os.path.join(ART, "typical_event_coeffs_smart.npz"))
    keep = d["band_id"] < 4
    band = d["band_id"][keep].astype(np.int64)
    gid = d["plane_gid"][keep].astype(np.int64)
    wire = d["wire"][keep].astype(np.int64)
    tau = d["time_index"][keep].astype(np.int64)
    val = d["value"][keep].astype(np.float32)
    # per (gid, band) MAD-sigma normalization (matches star_tpc convention)
    for g in range(6):
        for b in range(4):
            m = (gid == g) & (band == b)
            if m.any():
                s = np.median(np.abs(val[m])) / 0.6745
                val[m] = val[m] * (SIGMA / max(s, 1e-6))
    val = np.arcsinh(val / SIGMA).astype(np.float32)
    wband = np.array([4, 4, 8, 16])
    off = np.concatenate([[0], np.cumsum(wband * pw)])
    n_slot = int(off[-1])
    j = TPC_LEV[band]
    c4 = tau >> (4 - j)
    cell_loc = (wire // pw) * 68 + (c4 >> 2)
    tpos = tau % (4 << (4 - j))
    slot = off[band] + (wire % pw) * wband[band] + tpos
    key = gid * (1 << 26) + cell_loc
    uniq, cell = np.unique(key, return_inverse=True)
    M = np.zeros((len(uniq), n_slot), np.float32)
    M[cell, slot] = val
    return M, band, slot, n_slot, off, wband * pw


def rd(M, off, widths, names, label, tokens, out):
    if len(M) > MAX_CELLS:
        M = M[np.random.default_rng(0).choice(len(M), MAX_CELLS, replace=False)]
    mu = M.mean(0)
    X = M - mu
    C = (X.T @ X) / len(X)
    lam, U = np.linalg.eigh(C)
    lam, U = lam[::-1], U[:, ::-1]
    n_slot = M.shape[1]
    row = {"tokens_per_event_basis": tokens, "n_slot": n_slot,
           "n_cells_sampled": int(len(M)), "total_var": float(lam.sum())}
    print(f"\n== {label}: n_slot={n_slot}, cells={len(M):,} (one typical event) ==")
    print("  d:        " + "  ".join(f"{d:>7}" for d in D_GRID if d <= n_slot))
    print("  distortion" + "  ".join(f"{lam[d:].sum()/n_slot:7.4f}" for d in D_GRID if d <= n_slot)
          + "   (per-slot MSE, linear floor)")
    row["distortion_vs_d"] = {str(d): float(lam[d:].sum() / n_slot)
                              for d in D_GRID if d <= n_slot}
    # per-band residual at d=256 (or max available)
    dref = min(256, n_slot - 1)
    res_slot = (U[:, dref:] ** 2 * lam[dref:][None, :]).sum(1)
    print(f"  per-band residual at d={dref} (mean over the band's slots):")
    pb = {}
    for b, nm in enumerate(names):
        s0, s1 = off[b], off[b] + widths[b]
        pb[nm] = float(res_slot[s0:s1].mean())
        print(f"    {nm:>4}: {pb[nm]:.4f}")
    row["per_band_residual_at_dref"] = pb
    row["dref"] = dref
    out[label] = row


def main():
    out = {}
    print("===== OPTICAL (doraemon clean targets, one typical event) =====")
    for anchor, tok in ((2048, "2.9k"), (1024, "5.8k"), (512, "11.5k"), (256, "23.1k")):
        M, band, slot, n_slot, off, widths = cells_optical(anchor)
        rd(M, off, widths, OPT_NAMES, f"optical_anchor{anchor}", tok, out)
    print("\n===== TPC (smart-removed typical event) =====")
    for pw, tok in ((16, "~14k"), (8, "25.4k"), (4, "~45k")):
        M, band, slot, n_slot, off, widths = cells_tpc(pw)
        rd(M, off, widths, TPC_NAMES, f"tpc_pw{pw}", tok, out)
    jp = os.path.join(ART, "cell_rd.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\n-> {jp}")


if __name__ == "__main__":
    main()
