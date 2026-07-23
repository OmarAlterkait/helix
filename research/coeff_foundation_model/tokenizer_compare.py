#!/usr/bin/env python
"""Column vs per-band-patch tokenization — the systematic comparison (D-15 protocol).

For both modalities, both designs, builds patch/cell matrices straight from
the dumps and traces the DISTORTION vs TOTAL-DIMS-PER-EVENT frontier with
three estimators per point:
  pca    : closed-form linear floor (eigenspectrum)
  linear : trained Linear encode/decode (sanity vs pca)
  mlp    : 2-layer MLP encode/decode (the nonlinear gap)

Designs:
  column   : mixed-band cell vectors (optical anchor 1024 -> 512 slots;
             TPC 8x4 patch -> 256 slots), d in {32..256}
  perband  : per-band patches in the band's native grid (optical P=64 -> 64
             slots; TPC 8 wires x 8 band-ticks -> 64 slots), uniform d_b in
             {4..64} across bands

Metric: per-band MSE on ACTIVE slots (the trained-AE currency), eval on a
held-out cell split. Frontier x-axis: total dims/event = n_tokens x d.

Run:  python tokenizer_compare.py
"""
import sys, os, json

sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIGMA = 2.6
OPT_LEVELS = np.array([10] + list(range(10, 1, -1)))
OPT_NAMES = ["A10", "D10", "D9", "D8", "D7", "D6", "D5", "D4", "D3", "D2"]
TPC_NAMES = ["A4", "D4", "D3", "D2"]
TPC_LEV = np.array([4, 4, 3, 2])


# ---------------------------------------------------------------- data ------

def load_opt():
    d = np.load(os.path.join(ART, "typical_event_coeffs_doraemon.npz"))
    keep = d["band_id"] < 10
    return (d["band_id"][keep].astype(np.int64), d["idx"][keep].astype(np.int64),
            np.arcsinh(d["value"][keep] / SIGMA).astype(np.float32),
            d["chunk_id"][keep].astype(np.int64))


def load_tpc():
    t = np.load(os.path.join(ART, "typical_event_coeffs_smart.npz"))
    k = t["band_id"] < 4
    band = t["band_id"][k].astype(np.int64)
    gid = t["plane_gid"][k].astype(np.int64)
    wire = t["wire"][k].astype(np.int64)
    tau = t["time_index"][k].astype(np.int64)
    val = t["value"][k].astype(np.float32)
    for g in range(6):
        for b in range(4):
            m = (gid == g) & (band == b)
            if m.any():
                s = np.median(np.abs(val[m])) / 0.6745
                val[m] *= SIGMA / max(s, 1e-6)
    return band, gid, wire, tau, np.arcsinh(val / SIGMA).astype(np.float32)


def mat_from(key, slot, val, n_slot):
    uniq, cell = np.unique(key, return_inverse=True)
    M = np.zeros((len(uniq), n_slot), np.float32)
    O = np.zeros((len(uniq), n_slot), bool)
    M[cell, slot] = val
    O[cell, slot] = True
    return M, O


def opt_column():
    band, idx, val, chunk = load_opt()
    A = 10
    widths = np.array([1 << max(0, A - int(j)) for j in OPT_LEVELS])
    off = np.concatenate([[0], np.cumsum(widths)])
    j = OPT_LEVELS[band]
    cell = chunk * (1 << 22) + (idx >> (A - j))
    slot = off[band] + (idx & ((1 << (A - j)) - 1))
    M, O = mat_from(cell, slot, val, int(off[-1]))
    bb = np.zeros(int(off[-1]), np.int64)
    for b in range(10):
        bb[off[b]:off[b] + widths[b]] = b
    return M, O, bb, OPT_NAMES


def opt_perband(P=64):
    band, idx, val, chunk = load_opt()
    out = {}
    for b in range(10):
        s = band == b
        key = chunk[s] * (1 << 22) + (idx[s] // P)
        out[b] = mat_from(key, idx[s] % P, val[s], P)
    return out, OPT_NAMES


def tpc_column():
    band, gid, wire, tau, val = load_tpc()
    wb = np.array([4, 4, 8, 16])
    off = np.concatenate([[0], np.cumsum(wb * 8)])
    j = TPC_LEV[band]
    c4 = tau >> (4 - j)
    cell = gid * (1 << 26) + (wire // 8) * (1 << 14) + (c4 >> 2)
    slot = off[band] + (wire % 8) * wb[band] + tau % (4 << (4 - j))
    M, O = mat_from(cell, slot, val, int(off[-1]))
    bb = np.zeros(int(off[-1]), np.int64)
    for b in range(4):
        bb[off[b]:off[b] + wb[b] * 8] = b
    return M, O, bb, TPC_NAMES


def tpc_perband(Pt=8):
    band, gid, wire, tau, val = load_tpc()
    out = {}
    for b in range(4):
        s = band == b
        key = gid[s] * (1 << 26) + (wire[s] // 8) * (1 << 14) + (tau[s] // Pt)
        out[b] = mat_from(key, (wire[s] % 8) * Pt + tau[s] % Pt, val[s], 8 * Pt)
    return out, TPC_NAMES


# ------------------------------------------------------------ estimators ----

def split(M, O, frac=0.8, seed=0):
    n = len(M)
    o = np.random.default_rng(seed).permutation(n)
    k = int(n * frac)
    return (M[o[:k]], O[o[:k]]), (M[o[k:]], O[o[k:]])


def pca_eval(tr, te, d):
    mu = tr[0].mean(0)
    X = tr[0] - mu
    C = (X.T @ X) / len(X)
    lam, U = np.linalg.eigh(C)
    U = U[:, ::-1][:, :d]
    R = (te[0] - mu) - ((te[0] - mu) @ U) @ U.T
    return R


def net_eval(tr, te, d, hidden=0, steps=1500, lr=3e-3):
    Xtr = torch.as_tensor(tr[0], device=DEV)
    Xte = torch.as_tensor(te[0], device=DEV)
    n_slot = Xtr.shape[1]
    if hidden:
        enc = torch.nn.Sequential(torch.nn.Linear(n_slot, hidden), torch.nn.GELU(),
                                  torch.nn.Linear(hidden, d)).to(DEV)
        dec = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(),
                                  torch.nn.Linear(hidden, n_slot)).to(DEV)
    else:
        enc = torch.nn.Linear(n_slot, d).to(DEV)
        dec = torch.nn.Linear(d, n_slot).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    bs = min(8192, len(Xtr))
    for i in range(steps):
        bi = torch.randint(0, len(Xtr), (bs,), device=DEV)
        x = Xtr[bi]
        loss = torch.nn.functional.mse_loss(dec(enc(x)), x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
    with torch.no_grad():
        R = (dec(enc(Xte)) - Xte).cpu().numpy()
    return R


def active_mse_per_band(R, O, band_of_slot=None, band_fixed=None, n_bands=10):
    out = np.full(n_bands, np.nan)
    if band_fixed is not None:
        m = O
        out[band_fixed] = float((R[m] ** 2).mean()) if m.any() else np.nan
        return out
    for b in range(n_bands):
        cols = np.nonzero(band_of_slot == b)[0]
        m = O[:, cols]
        if m.any():
            out[b] = float((R[:, cols][m] ** 2).mean())
    return out


# ---------------------------------------------------------------- main ------

def run_column(label, M, O, bb, names, d_grid, out):
    tr, te = split(M, O)
    n_tok = len(M)
    nb = len(names)
    print(f"\n== {label} (column): {n_tok:,} tokens, {M.shape[1]} slots ==")
    for d in d_grid:
        if d >= M.shape[1]:
            continue
        for est, fn in (("pca", lambda: pca_eval(tr, te, d)),
                        ("linear", lambda: net_eval(tr, te, d, 0)),
                        ("mlp", lambda: net_eval(tr, te, d, 512))):
            pb = active_mse_per_band(fn(), te[1], band_of_slot=bb, n_bands=nb)
            prim = np.nanmean(pb)
            out.append(dict(label=label, design="column", est=est, d=d,
                            total_dims=n_tok * d, primary=float(prim),
                            per_band={names[b]: float(pb[b]) for b in range(nb)}))
            print(f"  d={d:>4} {est:>6}: primary {prim:7.4f}  dims/ev {n_tok*d/1e6:5.2f}M  "
                  + " ".join(f"{names[b]}:{pb[b]:.3f}" for b in range(nb) if not np.isnan(pb[b])))


def run_perband(label, mats, names, d_grid, out):
    nb = len(names)
    n_tok = sum(len(m[0]) for m in mats.values())
    print(f"\n== {label} (per-band): {n_tok:,} tokens total ==")
    for d in d_grid:
        for est in ("pca", "linear", "mlp"):
            pb = np.full(nb, np.nan)
            for b, (M, O) in mats.items():
                if d >= M.shape[1] or len(M) < 200:
                    continue
                tr, te = split(M, O)
                R = (pca_eval(tr, te, d) if est == "pca"
                     else net_eval(tr, te, d, 0 if est == "linear" else 512))
                pb[b] = active_mse_per_band(R, te[1], band_fixed=0, n_bands=1)[0]
            prim = np.nanmean(pb)
            out.append(dict(label=label, design="perband", est=est, d=d,
                            total_dims=n_tok * d, primary=float(prim),
                            per_band={names[b]: float(pb[b]) for b in range(nb)}))
            print(f"  d={d:>4} {est:>6}: primary {prim:7.4f}  dims/ev {n_tok*d/1e6:5.2f}M  "
                  + " ".join(f"{names[b]}:{pb[b]:.3f}" for b in range(nb) if not np.isnan(pb[b])))


def main():
    torch.manual_seed(0)
    out = []
    M, O, bb, nm = opt_column()
    run_column("optical_anchor1024", M, O, bb, nm, [64, 128, 256], out)
    mats, nm = opt_perband(64)
    run_perband("optical_P64", mats, nm, [8, 16, 32], out)
    M, O, bb, nm = tpc_column()
    run_column("tpc_8x4", M, O, bb, nm, [64, 128], out)
    mats, nm = tpc_perband(8)
    run_perband("tpc_8x8band", mats, nm, [8, 16, 32], out)
    with open(os.path.join(ART, "tokenizer_compare.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\n-> artifacts/tokenizer_compare.json")


if __name__ == "__main__":
    main()
