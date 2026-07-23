"""Deconvolution probe: can a FROZEN MAE encoder linearly predict the true CHARGE
coefficients? Three arms, per-band R² (asinh space):
  rep   - Linear(d, n_slot) on frozen encoder features            (the representation)
  raw   - Linear(2*n_slot, n_slot) on raw [inp,occ] tokens        (does the rep beat raw signal?)
  (scratch arm = train.py with charge target; separate, longer)

Target = true-charge coif3 coeff at each wire-token slot (charge cache), asinh(charge/S_band).
Encoder frozen; only the probe head trains. Run after charge_cache.py.

Run:  python probe.py --ckpt ckpt_dual44M.pt --events 380 --steps 3000
"""
import sys, os, glob, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
import data as D
from data import DEV, N_SLOT, N_BAND
from model import FMModel


def move(B, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in B.items()}


@torch.no_grad()
def band_scales(batches):
    """per-band robust scale S_band (std of charge over active rows) for asinh normalization."""
    sq = np.zeros(N_BAND); n = np.zeros(N_BAND)
    for B in batches:
        c = B["target_charge"].cpu().numpy(); bd = B["band_id"][B["cell"]].cpu().numpy()
        np.add.at(sq, bd, c**2); np.add.at(n, bd, 1)
    return np.sqrt(sq / np.maximum(n, 1)) + 1e-6


def tgt_asinh(B, S):
    c = B["target_charge"]; bd = B["band_id"][B["cell"]]
    return torch.asinh(c / S[bd])


@torch.no_grad()
def r2_per_band(pred, B, S):
    """R² per band over active rows in asinh space."""
    t = tgt_asinh(B, S); bd = B["band_id"][B["cell"]].cpu().numpy()
    e = ((pred - t) ** 2).float().cpu().numpy(); tt = t.float().cpu().numpy()
    return bd, e, tt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_dual44M.pt")
    ap.add_argument("--events", type=int, default=380)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--charge_dir", default=None)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    D.init_pipeline_cpu()
    here = os.path.dirname(os.path.abspath(__file__))
    cdir = os.path.abspath(args.cache_dir or os.path.join(here, "..", "artifacts", "fm_cache_tpc"))
    qdir = os.path.abspath(args.charge_dir or os.path.join(here, "..", "artifacts", "fm_charge_tpc"))

    # frozen encoder from checkpoint
    ck = torch.load(os.path.join(here, args.ckpt), map_location=DEV)
    d, blocks, dec = ck["d"], ck["blocks"], ck.get("dec_blocks", 4)
    model = FMModel(N_SLOT, N_BAND, 6, n_wirefeat=1, d=d, blocks=blocks, dec_blocks=dec).to(DEV)
    model.load_state_dict(ck["model"]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"frozen encoder: d={d} enc={blocks} dec={dec} (ckpt step {ck.get('step')})", flush=True)

    # events with charge target (only those with both caches), CPU-staged
    have = [i for i in range(args.events)
            if os.path.exists(f"{qdir}/ev_charge_{i:05d}.npz") and os.path.exists(f"{cdir}/ev_{i:05d}.npz")]
    print(f"{len(have)} events with charge target", flush=True)
    allb = [move(D.get_cached_charge(f"{cdir}/ev_{i:05d}.npz", f"{qdir}/ev_charge_{i:05d}.npz"), "cpu") for i in have]
    nte = max(1, int(len(allb) * args.test_frac))
    test_b, train_b = allb[:nte], allb[nte:]
    S = torch.tensor(band_scales(train_b), dtype=torch.float32, device=DEV)
    print("per-band charge scale S:", S.cpu().numpy().round(1), f"| train={len(train_b)} test={len(test_b)}", flush=True)

    # PRECOMPUTE frozen encoder features once (fp16, CPU) -> head training is then trivial.
    print("precomputing frozen features...", flush=True); t0 = time.time()
    def precompute(batches):
        zb = []
        with torch.no_grad():
            for B in batches:
                z = model.encode(move(B, DEV)).half().cpu()
                zb.append(z)
        return zb
    ztr, zte = precompute(train_b), precompute(test_b)
    print(f"  precomputed {len(ztr)+len(zte)} events in {time.time()-t0:.0f}s", flush=True)

    heads = {"rep": nn.Linear(d, N_SLOT).to(DEV), "raw": nn.Linear(2 * N_SLOT, N_SLOT).to(DEV)}
    opt = torch.optim.AdamW([p for h in heads.values() for p in h.parameters()], lr=args.lr, weight_decay=1e-5)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        i = step % len(train_b); B = move(train_b[i], DEV)
        f = {"rep": ztr[i].to(DEV).float(), "raw": torch.cat([B["inp"], B["occ"]], -1)}
        tgt = tgt_asinh(B, S); cell, slot = B["cell"], B["slot"]
        loss = sum(((h(f[k])[cell, slot] - tgt) ** 2).mean() for k, h in heads.items())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 1000 == 0 or step == 1:
            print(f"  step {step}: loss {float(loss):.4f} {(time.time()-t0)/step*1000:.0f} ms/step", flush=True)

    # eval per-band R² on held-out
    print("\n=== deconvolution probe: per-band R² (charge, asinh space) ===")
    for k, h in heads.items():
        se = np.zeros(N_BAND); st = np.zeros(N_BAND); n = np.zeros(N_BAND)
        means = np.zeros(N_BAND); cnt = np.zeros(N_BAND)
        # two-pass for proper R² (variance vs per-band mean)
        rows = []
        with torch.no_grad():
            for j, B in enumerate(test_b):
                B = move(B, DEV)
                f = {"rep": zte[j].to(DEV).float(), "raw": torch.cat([B["inp"], B["occ"]], -1)}
                pred = h(f[k])[B["cell"], B["slot"]]
                bd, e, tt = r2_per_band(pred, B, S)
                rows.append((bd, e, tt))
        bd = np.concatenate([r[0] for r in rows]); e = np.concatenate([r[1] for r in rows]); tt = np.concatenate([r[2] for r in rows])
        out = {}
        for b in range(N_BAND):
            m = bd == b
            mse = e[m].mean(); var = tt[m].var() + 1e-9
            out[b] = 1 - mse / var
        ov = np.mean([out[b] for b in range(N_BAND)])
        print(f" {k:>3}: overall R²={ov*100:5.1f}% | " + " ".join(f"{['A4','D4','D3','D2'][b]}={out[b]*100:4.1f}%" for b in range(N_BAND)))


if __name__ == "__main__":
    main()
