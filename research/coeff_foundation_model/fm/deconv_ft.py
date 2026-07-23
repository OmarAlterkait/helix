"""TIER-2 RECIPE DECISION: fine-tuned deconvolution label-efficiency.

Frozen probes proved unreliable for recipe choice (rank-flips x2). The decision metric
is FINE-TUNED downstream at multiple LABEL BUDGETS (pretraining value shows in the
low-label regime). Task = deconv -> clean charge (per-slot). Arms:
  scratch = random d512 (matched-compute baseline; the bar the FM must clear)
  nll     = sc_obj_nll  (NLL-d512 pretrain)   } identical arch, differ ONLY in objective
  mse     = sc_obj_mse  (MSE-d512 pretrain)   } -> this pair PICKS the objective for the big run
LP-FT (Kumar 2022): fit head frozen, then unfreeze all at low LR. Report per-band R2
(D2 = hardest detail band). Held-out eval (0-250, out of pretrain+finetune)."""
import argparse, json, numpy as np, torch, torch.nn as nn
import data as D
D.init_pipeline_cpu()
from model import FMModel
from torch.utils.data import DataLoader
dev = "cuda"; d = 512


def move(B, dv):
    return {k: (v.to(dv, non_blocking=True) if torch.is_tensor(v) else v) for k, v in B.items()}


def wpair(wi):
    return (f"../artifacts/fm_cache_tpc/ev_{wi:05d}.npz",
            f"../artifacts/fm_charge_tpc/ev_charge_{wi:05d}.npz")


def get(wi):
    return D.get_cached_charge(*wpair(wi), device=dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["scratch", "nll", "mse", "ckpt"], required=True)
    ap.add_argument("--ckpt", default=""); ap.add_argument("--nllarch", type=int, default=0)
    ap.add_argument("--budget", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    ckpath = {"ckpt": a.ckpt, "nll": "ckpt_sc_obj_nll.pt", "mse": "ckpt_sc_obj_mse.pt"}.get(a.arm)
    if ckpath:                                         # READ arch from the ckpt (heads/nll may vary per run)
        ck = torch.load(ckpath, map_location=dev)
        heads = ck.get("heads", 4); nll_arch = bool(ck.get("nll", a.arm == "nll"))
        dd, bl, db = ck.get("d", d), ck.get("blocks", 12), ck.get("dec_blocks", 4)
    else:                                              # scratch: standardized arch (head_dim 64 -> 8 heads @ d512)
        heads, nll_arch, dd, bl, db = d // 64, bool(a.nllarch), d, 12, 4
    m = FMModel(128, 4, 6, d=dd, blocks=bl, dec_blocks=db, heads=heads, cond="film",
                dec_mode="cross", nll=nll_arch).to(dev)
    if ckpath: m.load_state_dict(ck["model"])
    head = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, 128)).to(dev)
    FIT = list(range(1000, 1000 + a.budget)); EVAL = list(range(0, 250))
    print(f"arm={a.arm} budget={a.budget} seed={a.seed} fit={len(FIT)} eval={len(EVAL)}", flush=True)

    # async loader over the FIT set: parallel CPU assembly overlapped with GPU (was per-step get()).
    fit_loader = DataLoader(D.CachedTPCCharge([wpair(wi) for wi in FIT]), batch_size=None, shuffle=True,
                            num_workers=8, persistent_workers=True, prefetch_factor=4, pin_memory=True,
                            worker_init_fn=D._worker_init, multiprocessing_context="spawn")
    fit_it = [iter(fit_loader)]
    def next_fit():
        try: return move(next(fit_it[0]), dev)
        except StopIteration:
            fit_it[0] = iter(fit_loader); return move(next(fit_it[0]), dev)

    def run_epochs(params, lr, steps):
        opt = torch.optim.AdamW(params, lr, weight_decay=1e-5)
        for step in range(steps):
            B = next_fit()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats = m.encode(B).float(); pred = head(feats)[B["cell"], B["slot"]]
            loss = ((pred - B["target_charge"].float()) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    # LP: head only (encoder frozen)
    for p in m.parameters(): p.requires_grad_(False)
    run_epochs(head.parameters(), 1e-3, 800)
    # FT: unfreeze all, low encoder LR
    for p in m.parameters(): p.requires_grad_(True)
    ft_steps = min(5000, max(1500, a.budget * 6))
    opt = torch.optim.AdamW([{"params": m.parameters(), "lr": 1e-4},
                             {"params": head.parameters(), "lr": 3e-4}], weight_decay=1e-5)
    for step in range(ft_steps):
        B = next_fit()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = m.encode(B).float(); pred = head(feats)[B["cell"], B["slot"]]
        loss = ((pred - B["target_charge"].float()) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    m.eval(); head.eval()
    ssr = np.zeros(4); sst = np.zeros(4); ssr_t = 0.0; sst_t = 0.0
    with torch.no_grad():
        for wi in EVAL:
            B = get(wi)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats = m.encode(B).float(); pred = head(feats)[B["cell"], B["slot"]].float()
            t = B["target_charge"].float().cpu().numpy(); pr = pred.cpu().numpy()
            bd = B["band_id"][B["cell"]].cpu().numpy()
            ssr_t += ((pr - t) ** 2).sum(); sst_t += (t ** 2).sum()
            for b in range(4):
                mb = bd == b; ssr[b] += ((pr[mb] - t[mb]) ** 2).sum(); sst[b] += (t[mb] ** 2).sum()
    r2 = lambda x, y: round(float((1 - x / max(y, 1e-9)) * 100), 1)
    out = dict(arm=a.arm, budget=a.budget, seed=a.seed, ft_steps=ft_steps, overall=r2(ssr_t, sst_t),
               A4=r2(ssr[0], sst[0]), D4=r2(ssr[1], sst[1]), D3=r2(ssr[2], sst[2]), D2=r2(ssr[3], sst[3]))
    print("FT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
