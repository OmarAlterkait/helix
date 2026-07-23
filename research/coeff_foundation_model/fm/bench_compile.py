"""Per-GPU utilization lever #1: torch.compile to fuse the elementwise/LN/RoPE overhead
(~28% of kernel time). Time eager vs compiled MAE step (one event on GPU, mask 0.75)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, glob
import data as D
from data import DEV
from model import FMModel, losses
from train import make_mask, move

D.init_pipeline_cpu()
d, blk, dec = 512, 10, 4
model = FMModel(D.N_SLOT, D.N_BAND, D.N_PLANE, n_wirefeat=1, d=d, blocks=blk, dec_blocks=dec).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
B = move(D.get_cached(sorted(glob.glob("../artifacts/fm_cache_tpc/ev_*.npz"))[0], device="cpu"), DEV)
g = torch.Generator(device=DEV).manual_seed(0); m = make_mask(B, "random", 0.75, 1, g)
print(f"d={d} tokens={int(B['n_cells'])} mask=0.75 (compute-only, event on GPU)")


def step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        occ, mu, lv = model(B, m); bce, val = losses(occ, mu, lv, B, m)
    (bce + val).backward(); opt.step()


def timeit(fn, n=15):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1000


eager = timeit(step)
print(f"eager      : {eager:.0f} ms/step")
cmodel = torch.compile(model, dynamic=True)
def cstep():
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        occ, mu, lv = cmodel(B, m); bce, val = losses(occ, mu, lv, B, m)
    (bce + val).backward(); opt.step()
comp = timeit(cstep, n=15)
print(f"compiled   : {comp:.0f} ms/step  ({eager/comp:.2f}x)")
