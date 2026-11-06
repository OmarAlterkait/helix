#!/usr/bin/env python3
"""T1 of results/m7_tests_plan.md: what the model receives, tested on the floor events.

    CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python baselines/uboone_sp/causes/t1_inputs.py --events 260-307 --out <json>

T1a  every kept corpus coefficient (band < n_bands) is in the token grid with value arcsinh(v / sigma), and
     the floor's corpus image is built from exactly those coefficients (count check).
T1c  exact-kernel Tikhonov of the corpus image, with the coefficient values as the decoder's first layer sees
     them under bf16 autocast: v -> arcsinh(v/sigma) -> bf16 -> sigma * sinh. 'orig' must reproduce floor.py.
T1e  locality: the same inversion using ONLY the coefficients of one 16-wire token group (the decoder's direct
     view of a cell is its covering token per band; everything else reaches it through encoder features),
     output kept inside the group. Scored per truth bin and per within-token wire position (fw % 8).
Model calo (M7dc, products) is scored per within-token wire position alongside.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(BASE))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "v2")); sys.path.insert(0, HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
R = "/global/cfs/cdirs/m5238/users/oalter/helix_runs"
FLOOR = f"{R}/bright/floor"
BINS = [(16e3, 64e3), (64e3, 256e3), (256e3, np.inf)]
LAMS = (1e-4, 1e-3)
PL = "UVY"


def stats(r):
    if len(r) < 20:
        return dict(n=int(len(r)))
    a, m, b = np.percentile(r, [16, 50, 84])
    return dict(n=int(len(r)), med=float(m), hw68=float((b - a) / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="260-307")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import h5py
    import torch
    import denoise_products as DP
    import floor as FL
    import build as BD
    from helix.core.coeff_io import read_coeff_event
    from helix.model.tokenize import PatchConfig, assemble, sigma_for_rows
    from sp.data import Event, registry, PLANES
    from sp.pipeline import fine_cells
    from sp.response import load_kernel, kernel_full
    dev = torch.device("cuda")
    reg = registry()
    cfg = PatchConfig(cell_t="grid_center")
    prm = json.load(open(f"{R}/products/params_M7dc.json"))["M7dc"]
    lo, hi = map(int, a.events.split("-"))
    evs = [e for e in range(lo, hi + 1) if os.path.exists(f"{FLOOR}/ev{e:03d}.npz")]
    PADW, PADT = 24, 400
    variants = ["orig", "bf16"]
    cols = ["Tf", "calo", "fwpos"] + [f"{v}_tik{l:g}" for v in variants for l in LAMS] + [f"local_tik{l:g}" for l in LAMS]
    rows = {p: [] for p in range(3)}
    t1a = []
    repro = []
    for ev in evs:
        t0 = time.time()
        E = Event(ev)
        shard = E.shard.replace(BD.PS_CORPUS, BD.CFS_CORPUS)
        with h5py.File(shard, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], E.event))
            bl = fh["config"]["band_lengths"][:]
            C = {k: fh["config"][k][:] for k in ("gids", "n_wires", "norm_sigma")}
        ce = read_coeff_event(shard, pos)
        # T1a
        tok = assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, ce.value, gids=C["gids"], n_wires=C["n_wires"],
                       band_lengths=bl, norm_sigma=C["norm_sigma"], cfg=cfg)
        k = ce.band < cfg.n_bands
        sig = np.maximum(sigma_for_rows(ce.plane_gid[k], ce.band[k], C["gids"], C["norm_sigma"]), 1e-6)
        want = np.arcsinh(ce.value[k].astype(np.float32) / sig).astype(np.float32)
        got = tok["inp"][tok["cell"], tok["slot"]]
        t1a.append(dict(ev=ev, n_kept=int(k.sum()), n_tok_slots=int(tok["occ"].sum()),
                        max_abs_diff=float(np.abs(got - want).max()) if len(want) else 0.0,
                        n_dropped_band=int((~k).sum())))
        # bf16 round trip of the values the model reads
        vb = np.zeros_like(ce.value, dtype=np.float64)
        sig_all = np.maximum(sigma_for_rows(ce.plane_gid, np.minimum(ce.band, cfg.n_bands - 1), C["gids"], C["norm_sigma"]), 1e-6)
        x = torch.as_tensor(np.arcsinh(ce.value.astype(np.float32) / sig_all)).to(torch.bfloat16).float().numpy()
        vb = (np.sinh(x.astype(np.float64)) * sig_all).astype(np.float32)
        import types
        ceb = types.SimpleNamespace(band=ce.band, plane_gid=ce.plane_gid, wire=ce.wire, tau=ce.tau, value=vb)
        F = np.load(f"{FLOOR}/ev{ev:03d}.npz")
        Zm = np.load(f"{R}/products/model_test_m7/ev{ev:03d}.npz"); to = int(Zm["toff"])
        for g in range(6):
            p = g % 3
            nw, nt = reg[g]["n_wires"], reg[g]["n_ticks"]
            R_, wo, tzb = load_kernel(PLANES[p], "exact")
            NW, NT = nw + 2 * PADW, nt + 2 * PADT
            Kf = torch.as_tensor(np.fft.rfft2(kernel_full(R_, wo, tzb, NW, NT)), device=dev)
            den = {l: Kf.conj() / (Kf.abs() ** 2 + l * (Kf.abs() ** 2).max()) for l in LAMS}
            pad = lambda X: np.pad(X, ((PADW, PADW), (PADT, PADT)))
            crop = lambda X: X[PADW:PADW + nw, PADT:PADT + nt]

            def tik(S, l):
                Y = torch.fft.rfft2(torch.as_tensor(pad(S), device=dev))
                return crop(torch.fft.irfft2(Y * den[l], s=(NW, NT)).cpu().numpy())

            Tf = F[f"truth_g{g}"].astype(np.float64)
            maps = {}
            S0 = FL.corpus_image(ce, bl, g, nw, nt)
            Sb = FL.corpus_image(ceb, bl, g, nw, nt)
            for l in LAMS:
                maps[f"orig_tik{l:g}"] = fine_cells(tik(S0, l))
                maps[f"bf16_tik{l:g}"] = fine_cells(tik(Sb, l))
                ref = F[f"corpus_tik{l:g}_g{g}"].astype(np.float64)
                b = Tf >= 16e3
                repro.append(float(np.abs(maps[f"orig_tik{l:g}"][b] - ref[b]).max() / max(np.abs(ref[b]).max(), 1)))
                # T1e: per 16-wire token group, its own coefficients only
                S0t = torch.as_tensor(pad(S0), device=dev)
                Lm = np.zeros((nw, nt))
                for w0 in range(0, nw, 16):
                    Sk = torch.zeros_like(S0t); Sk[PADW + w0:PADW + w0 + 16] = S0t[PADW + w0:PADW + w0 + 16]
                    X = torch.fft.irfft2(torch.fft.rfft2(Sk) * den[l], s=(NW, NT))
                    Lm[w0:w0 + 16] = X[PADW + w0:PADW + min(w0 + 16, nw), PADT:PADT + nt].cpu().numpy()
                maps[f"local_tik{l:g}"] = fine_cells(Lm)
            q, pp, cov = (Zm[f"{n}_M7dc_g{g}"] if n != "cov" else Zm[f"cov_g{g}"] for n in ("q", "p", "cov"))
            calo = DP.products_maps(q.astype(np.float64), pp.astype(np.float64), cov, p, prm)["calo"][:Tf.shape[0], to:to + Tf.shape[1]]
            fwpos = (np.arange(Tf.shape[0]) % 8)[:, None] * np.ones((1, Tf.shape[1]))
            Sx = np.stack([Tf.ravel(), calo.ravel(), fwpos.ravel()] + [maps[c].ravel() for c in cols[3:]], 1)
            rows[p].append(Sx[Sx[:, 0] >= 16e3])
        print(f"ev{ev:03d} {time.time() - t0:.0f}s", flush=True)
    out = dict(events=evs, t1a=t1a, t1a_ok=all(r["max_abs_diff"] == 0 and r["n_kept"] == r["n_tok_slots"] for r in t1a),
               orig_reproduces_floor_maxrel=max(repro), cols=cols, by_bin={}, by_fwpos={})
    for p in range(3):
        A = np.concatenate(rows[p]); Tf = A[:, 0]
        out["by_bin"][PL[p]] = {f"{lo_:.0f}-{hi_:.0f}": {c: stats(A[(Tf >= lo_) & (Tf < hi_), j] / Tf[(Tf >= lo_) & (Tf < hi_)] - 1)
                                                         for j, c in enumerate(cols) if c not in ("Tf", "fwpos")}
                                for lo_, hi_ in BINS}
        out["by_fwpos"][PL[p]] = {}
        for lo_, hi_ in ((16e3, 64e3), (64e3, np.inf)):
            for k_ in range(8):
                m = (A[:, 2] == k_) & (Tf >= lo_) & (Tf < hi_)
                out["by_fwpos"][PL[p]][f"{lo_:.0f}+_pos{k_}"] = {c: stats(A[m, j] / Tf[m] - 1) for j, c in enumerate(cols)
                                                                 if c in ("calo", "orig_tik0.001", "local_tik0.001", "orig_tik0.0001", "local_tik0.0001")}
    json.dump(out, open(a.out, "w"), indent=1)
    print("T1a ok:", out["t1a_ok"], "orig reproduces floor (max rel diff):", out["orig_reproduces_floor_maxrel"])
    for p in PL:
        for b, d in out["by_bin"][p].items():
            print(p, b, {c: (round(v.get("med", 0), 4), round(v.get("hw68", 0), 4)) for c, v in d.items()})


if __name__ == "__main__":
    main()
