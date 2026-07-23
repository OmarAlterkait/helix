"""Across-checkpoint evaluation for the MAE snapshots (MSE vs NLL objective).
For each snapshot: (1) RankMe (head-free representation quality), (2) validation on
UNSEEN events (masked-prediction var-explained / MSE / NLL). Reads arch from the ckpt.
Held-out eval events: numeric >=30000 (training used numeric events 1000-20999)."""
import glob, json, os, numpy as np, torch
import data as D; D.init_pipeline_cpu()
from model import FMModel
from data import N_SLOT, N_BAND, N_PLANE
from train import perband_mse, nll_eval, TGT_VAR_BAND
DEV = "cuda"

def rankme(z):
    z = (z - z.mean(0)).float(); s = torch.linalg.svdvals(z); p = s / (s.sum() + 1e-12)
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))

def numsort(g):
    return sorted(g, key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p)))))

EVAL = numsort(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"))[30000:30040]   # 40 held-out events
print(f"{len(EVAL)} held-out eval events (numeric >=30000)", flush=True)

def load(ckpt):
    ck = torch.load(ckpt, map_location=DEV)
    film = tuple(ck.get("film", "band,plane,wire").split(",")) if ck.get("film") else ()
    m = FMModel(N_SLOT, N_BAND, N_PLANE, n_wirefeat=1, d=ck["d"], blocks=ck["blocks"],
                dec_blocks=ck["dec_blocks"], heads=ck["heads"], film=film, nll=ck["nll"],
                ffn_mult=ck.get("ffn_mult", 4), cond=ck.get("cond", "film"),
                dec_mode=ck.get("dec_mode", "cross"), mup=ck.get("mup", True),
                d_base=ck.get("d_base", 128)).to(DEV)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck["step"], bool(ck["nll"]), ck["heads"]

def ve(pbd):
    return float(np.mean([(TGT_VAR_BAND[b] - pbd[f"b{b}"]) / TGT_VAR_BAND[b] for b in range(N_BAND)]))

import sys
_pat = sys.argv[1] if len(sys.argv) > 1 else "ckpt_mae_mse_s0_snap*.pt ckpt_mae_nll_s0_snap*.pt"
ckpts = []
for _p in _pat.split():
    ckpts += numsort(glob.glob(_p))
out = open("eval_ckpts.jsonl", "w")
for c in ckpts:
    obj = os.path.basename(c).replace("ckpt_", "").split("_snap")[0]   # run name, e.g. nll_long
    m, step, isnll, heads = load(c)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        fz = torch.cat([m.encode(D.get_cached(p, device=DEV)).float() for p in EVAL[:20]])
    rme = rankme(fz); eff = int((fz.std(0) > 0.01 * fz.std(0).max()).sum())
    prm, pbd, bse, pvm, pvb = perband_mse(m, EVAL, "random", 0.75, 1)     # validation on unseen
    vnll = nll_eval(m, EVAL, "random", 0.75, 1)
    rec = dict(obj=obj, step=step, heads=heads, rankme=round(rme, 1), eff_dims=eff,
               var_expl=round(ve(pbd), 4), mse=round(prm, 3),
               val_nll=None if vnll is None else round(vnll, 3),
               per_band={k: round(v, 3) for k, v in pbd.items()})
    print(f"  {obj} step {step:>7}: RankMe {rme:5.1f}/{fz.shape[1]} eff {eff:>3}  var_expl {ve(pbd)*100:5.1f}%  "
          f"mse {prm:.3f}  nll {rec['val_nll']}", flush=True)
    out.write(json.dumps(rec) + "\n"); out.flush()
    del m; torch.cuda.empty_cache()
out.close()
print("wrote eval_ckpts.jsonl", flush=True)
