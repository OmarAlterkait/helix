#!/usr/bin/env python
"""M1 — cost microbench (EXECUTION_PLAN.md T1.1-T1.3). A100, no training.

T1.1 tree-op cost per topology on the packed coefficient set:
     C0 adjacent V-cycle (up+down gathers), C1 dilated (D=1,2,4,8),
     C2 ancestor-chain attention (down) + hierarchical up,
     C3 cell-pooled scale-axial attention. fwd and fwd+bwd, d_s in {32,64,128}.
T1.2 per-band dense Conv1d vs gather(+/-1)+MLP on active sites (optical bands
     at their true occupancies) — gates the dense-vs-sparse choice (S6).
T1.3 batched stems: TPC SubMConv2d 24 calls (per plane x band) vs 4 calls
     (per band, planes batched). Skipped cleanly if spconv is unavailable.

Substrates: artifacts/typical_event_coeffs_doraemon.npz (optical, L=10) and
artifacts/typical_event_coeffs_smart.npz (TPC, L=4).
Run from this folder:  python microbench_treeops.py
"""
import sys, os, json

sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

OPT_LEVELS = [10] + list(range(10, 0, -1))   # band_id 0..10
TPC_LEVELS = [4, 4, 3, 2, 1]
TPC_LENS = [271, 271, 542, 1084, 2168]


def timer(fn, *args, warmup=5, iters=20, bwd=False):
    for _ in range(warmup):
        out = fn(*args)
        if bwd:
            out.sum().backward()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        out = fn(*args)
        if bwd:
            out.sum().backward()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


# ------------------------------------------------------- packed structures --

def build_packed_optical(path):
    d = np.load(path)
    cl = d["chunk_len"]
    band, chunk, idx = d["band_id"].astype(np.int64), d["chunk_id"].astype(np.int64), d["idx"].astype(np.int64)
    N = len(band)
    rows_by_band = [np.nonzero(band == b)[0] for b in range(11)]
    # per (chunk, band) dense index grids -> parent/dilated/ancestor maps
    Lp = np.ceil(cl / 1024).astype(np.int64) * 1024
    grids = {}
    for b in range(11):
        j = OPT_LEVELS[b]
        n_tot = int((Lp >> j).sum())
        off = np.zeros(len(cl) + 1, np.int64)
        off[1:] = np.cumsum(Lp >> j)
        g = np.full(n_tot, -1, np.int64)
        r = rows_by_band[b]
        g[off[chunk[r]] + idx[r]] = r
        grids[b] = (g, off)
    def lookup(b, ch, tau):
        g, off = grids[b]
        n_b = np.diff(off)[ch]
        ok = (tau >= 0) & (tau < n_b)
        out = np.full(len(tau), -1, np.int64)
        out[ok] = g[off[ch[ok]] + tau[ok]]
        return out
    maps = {}
    # parent maps at distance D (band b -> b-D), detail bands only
    for D in (1, 2, 4, 8):
        m = np.full(N, -1, np.int64)
        for b in range(1 + D, 11):
            r = rows_by_band[b]
            m[r] = lookup(b - D, chunk[r], idx[r] >> D)
        maps[f"par{D}"] = m
    # ancestor chains (up to 9 ancestors for the finest band) for C2
    max_anc = 9
    anc = np.full((N, max_anc), -1, np.int64)
    for b in range(2, 11):
        r = rows_by_band[b]
        for k, bb in enumerate(range(b - 1, 0, -1)):
            anc[r, k] = lookup(bb, chunk[r], idx[r] >> (b - bb))
    maps["anc"] = anc
    # cell map for C3 (1024-tick cells)
    cell_off = np.zeros(len(cl) + 1, np.int64)
    cell_off[1:] = np.cumsum(Lp >> 10)
    cell = cell_off[chunk] + ((idx << np.array([OPT_LEVELS[b] for b in band])) >> 10)
    maps["cell"] = cell
    maps["n_cells"] = int(cell_off[-1])
    return N, band, rows_by_band, maps


def build_packed_tpc(path):
    d = np.load(path)
    band = d["band_id"].astype(np.int64)
    gid, wire, t = d["plane_gid"].astype(np.int64), d["wire"].astype(np.int64), d["time_index"].astype(np.int64)
    N = len(band)
    rows_by_band = [np.nonzero(band == b)[0] for b in range(5)]
    nw = np.array([1969, 1969, 1443, 1969, 1969, 1443])
    woff = np.zeros(7, np.int64)
    woff[1:] = np.cumsum(nw)
    grids = {}
    for b in range(5):
        L = TPC_LENS[b]
        g = np.full(int(woff[-1]) * L, -1, np.int64)
        r = rows_by_band[b]
        g[(woff[gid[r]] + wire[r]) * L + t[r]] = r
        grids[b] = g
    maps = {}
    for D in (1, 2):
        m = np.full(N, -1, np.int64)
        for b in range(1 + D, 5):
            r = rows_by_band[b]
            tau = t[r] >> D
            ok = tau < TPC_LENS[b - D]
            mm = np.full(len(r), -1, np.int64)
            mm[ok] = grids[b - D][(woff[gid[r][ok]] + wire[r][ok]) * TPC_LENS[b - D] + tau[ok]]
            m[r] = mm
        maps[f"par{D}"] = m
    anc = np.full((N, 3), -1, np.int64)
    for b in range(2, 5):
        r = rows_by_band[b]
        for k, bb in enumerate(range(b - 1, 0, -1)):
            tau = t[r] >> (b - bb)
            ok = tau < TPC_LENS[bb]
            mm = np.full(len(r), -1, np.int64)
            mm[ok] = grids[bb][(woff[gid[r][ok]] + wire[r][ok]) * TPC_LENS[bb] + tau[ok]]
            anc[r, k] = mm
    maps["anc"] = anc
    return N, band, rows_by_band, maps, (gid, wire, t)


# -------------------------------------------------------------- operators ---

class EdgeMLP(torch.nn.Module):
    def __init__(self, din, d):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(din), torch.nn.Linear(din, d),
            torch.nn.GELU(), torch.nn.Linear(d, d))
    def forward(self, x):
        return self.net(x)


def bench_topologies(name, N, band, rows_by_band, maps, n_bands, results):
    print(f"\n--- {name}: tree-op cost (N={N:,} coeffs) ---")
    for d in (32, 64, 128):
        X0 = torch.randn(N, d, device=DEV)
        par1 = torch.from_numpy(maps["par1"]).to(DEV)
        ok1 = par1 >= 0
        mlp_u = EdgeMLP(d, d).to(DEV)
        mlp_d = EdgeMLP(d, d).to(DEV)

        def c0(X):
            Y = X.clone()
            m = mlp_u(Y)
            Y = Y.index_add(0, par1[ok1], m[ok1])          # up
            g = torch.zeros_like(Y)
            g[ok1] = Y[par1[ok1]]
            return Y + mlp_d(g)                            # down

        dil = {D: (torch.from_numpy(maps[f"par{D}"]).to(DEV)) for D in (1, 2, 4, 8) if f"par{D}" in maps}
        mlps = {D: EdgeMLP(d, d).to(DEV) for D in dil}
        def c1(X):
            Y = X.clone()
            for D, m in dil.items():
                ok = m >= 0
                g = torch.zeros_like(Y)
                g[ok] = X[m[ok]]
                Y = Y + mlps[D](g)
            return Y

        anc = torch.from_numpy(maps["anc"]).to(DEV)
        amask = anc >= 0
        q_proj = torch.nn.Linear(d, d).to(DEV)
        kv_proj = torch.nn.Linear(d, 2 * d).to(DEV)
        def c2(X):
            Y = X.clone()
            m = mlp_u(Y)
            Y = Y.index_add(0, par1[ok1], m[ok1])          # hierarchical up
            A = torch.zeros(N, anc.shape[1], d, device=DEV)
            A[amask] = Y[anc[amask]]                       # gather chains
            q = q_proj(Y).unsqueeze(1)
            kv = kv_proj(A)
            k, v = kv.chunk(2, -1)
            att = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=amask.unsqueeze(1))
            return Y + att.squeeze(1)

        res = {"d": d,
               "C0_fwd": timer(c0, X0), "C1_fwd": timer(c1, X0), "C2_fwd": timer(c2, X0)}
        if "cell" in maps:
            cell = torch.from_numpy(maps["cell"]).to(DEV)
            lev = torch.from_numpy(np.asarray(band)).to(DEV)
            ncells = maps["n_cells"]
            slot = cell * n_bands + lev
            cattn = torch.nn.MultiheadAttention(d, 4, batch_first=True).to(DEV)
            def c3(X):
                S = torch.zeros(ncells * n_bands, d, device=DEV)
                S = S.index_add(0, slot, X).view(ncells, n_bands, d)   # pool
                S2, _ = cattn(S, S, S, need_weights=False)             # LxL per cell
                return X + (S + S2).view(-1, d)[slot]                  # scatter back
            res["C3_fwd"] = timer(c3, X0)
        Xg = X0.clone().requires_grad_(True)
        res["C0_fb"] = timer(c0, Xg, bwd=True)
        res["C2_fb"] = timer(c2, Xg, bwd=True)
        results.setdefault(name, []).append(res)
        line = f"  d={d:>3}: " + "  ".join(f"{k}={v:6.2f}ms" for k, v in res.items() if k != "d")
        print(line)


def bench_dense_vs_gather(rows_by_band, maps, band_arr, results):
    """T1.2 per-band dense Conv1d vs gather+MLP, optical occupancies (doraemon)."""
    d = np.load(os.path.join(ART, "typical_event_coeffs_doraemon.npz"))
    cl = d["chunk_len"]
    n_chunks = len(cl)
    Lp = int(np.ceil(cl.max() / 1024) * 1024)
    print(f"\n--- T1.2 dense Conv1d vs gather(+/-1)+linear per band ({n_chunks} chunks, pad {Lp}) ---")
    rows = []
    for ds in (32, 64):
        for b in range(1, 11):
            j = OPT_LEVELS[b]
            n_active = int((d["band_id"] == b).sum())
            len_b = Lp >> j
            dense = torch.randn(n_chunks, ds, len_b, device=DEV)
            conv = torch.nn.Conv1d(ds, ds, 3, padding=1, groups=ds).to(DEV)
            t_dense = timer(lambda x: conv(x), dense)
            act = torch.randn(max(n_active, 1), ds, device=DEV)
            nb = torch.randint(0, max(n_active, 1), (max(n_active, 1), 2), device=DEV)
            lin = torch.nn.Linear(3 * ds, ds).to(DEV)
            def gath(x):
                g = x[nb]                                  # (n,2,ds)
                return lin(torch.cat([x, g.flatten(1)], 1))
            t_gath = timer(gath, act)
            occ = n_active / (n_chunks * len_b)
            rows.append(dict(d=ds, band=f"D{j}", n_active=n_active, len_b=len_b,
                             occ=occ, dense_ms=t_dense, gather_ms=t_gath))
            if ds == 64:
                print(f"  D{j:<2} occ={100*occ:5.2f}%  n={n_active:>7,}  "
                      f"dense {t_dense:6.2f}ms  gather {t_gath:6.2f}ms  "
                      f"-> {'DENSE' if t_dense < t_gath else 'gather'}")
    results["dense_vs_gather"] = rows


def bench_stems(tpc_coords, results):
    """T1.3 TPC SubMConv2d: 24 separate vs 4 band-batched calls."""
    try:
        import spconv.pytorch as spconv
    except ImportError:
        print("\n--- T1.3 skipped: spconv not available in this env ---")
        return
    gid, wire, t = tpc_coords
    d = np.load(os.path.join(ART, "typical_event_coeffs_smart.npz"))
    band = d["band_id"].astype(np.int64)
    print("\n--- T1.3 TPC stems: 24 per-(plane,band) calls vs 4 band-batched calls ---")
    for ds in (32, 64):
        nets = {b: spconv.SubMConv2d(ds, ds, (5, 3), padding=(2, 1), bias=False).to(DEV)
                for b in range(4)}
        sep_tensors, bat_tensors = [], {}
        for b in range(4):                                 # A4,D4,D3,D2 (D1 dropped)
            sel = band == b
            for g in range(6):
                s = sel & (gid == g)
                n = int(s.sum())
                if n == 0:
                    continue
                co = torch.stack([torch.zeros(n, dtype=torch.int32, device=DEV),
                                  torch.from_numpy(wire[s]).int().to(DEV),
                                  torch.from_numpy(t[s]).int().to(DEV)], 1)
                feats = torch.randn(n, ds, device=DEV)
                sep_tensors.append((b, spconv.SparseConvTensor(
                    feats, co, [1969, TPC_LENS[b]], 1)))
            sb = sel
            co = torch.stack([torch.from_numpy(gid[sb]).int().to(DEV),
                              torch.from_numpy(wire[sb]).int().to(DEV),
                              torch.from_numpy(t[sb]).int().to(DEV)], 1)
            bat_tensors[b] = spconv.SparseConvTensor(
                torch.randn(int(sb.sum()), ds, device=DEV), co, [1969, TPC_LENS[b]], 6)
        def run_sep():
            return sum(nets[b](x).features.sum() for b, x in sep_tensors)
        def run_bat():
            return sum(nets[b](x).features.sum() for b, x in bat_tensors.items())
        t_sep = timer(run_sep, iters=10)
        t_bat = timer(run_bat, iters=10)
        print(f"  d={ds}: 24 calls {t_sep:6.2f}ms   4 batched {t_bat:6.2f}ms   "
              f"amortization x{t_sep/max(t_bat,1e-9):.1f}")
        results.setdefault("stems_tpc", []).append(
            dict(d=ds, separate_ms=t_sep, batched_ms=t_bat))


def main():
    assert DEV == "cuda", "GPU required"
    torch.backends.cuda.matmul.allow_tf32 = True
    results = {"gpu": torch.cuda.get_device_name(0)}

    p_opt = os.path.join(ART, "typical_event_coeffs_doraemon.npz")
    if os.path.exists(p_opt):
        N, band, rb, maps = build_packed_optical(p_opt)
        bench_topologies("optical_doraemon", N, band, rb, maps, 11, results)
        bench_dense_vs_gather(rb, maps, band, results)

    p_tpc = os.path.join(ART, "typical_event_coeffs_smart.npz")
    if os.path.exists(p_tpc):
        N, band, rb, maps, coords = build_packed_tpc(p_tpc)
        bench_topologies("tpc_smart", N, band, rb, maps, 5, results)
        bench_stems(coords, results)

    jp = os.path.join(ART, "microbench_treeops.json")
    with open(jp, "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"\n-> {jp}")


if __name__ == "__main__":
    main()
