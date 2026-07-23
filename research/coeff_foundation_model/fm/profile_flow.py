import sys, os, time, glob, math, torch
sys.path.insert(0, '/sdf/group/neutrino/omara/helix/.pylibs'); sys.path.insert(0, '.')
import data as D
from data import DEV, N_SLOT, N_BAND
from model import FMModel
from flow_head import FlowHead, flow_loss, flow_sample
D.init_pipeline_cpu(); OLD = (2 * math.pi, 47120.)
model = FMModel(N_SLOT, N_BAND, 6, n_wirefeat=1, d=512, blocks=10, dec_blocks=4,
                nll=True, cond='film', lam_t=OLD, lam_w=OLD).to(DEV)
head = FlowHead(512, N_SLOT).to(DEV)
wf = sorted(glob.glob('../artifacts/fm_cache_tpc/ev_*.npz'))[0]
qf = sorted(glob.glob('../artifacts/fm_charge_tpc/ev_charge_*.npz'))[0]
B = D.get_cached_charge(wf, qf); B = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in B.items()}
N = int(B['n_cells']); m = torch.zeros(N, dtype=torch.bool, device=DEV); print(f"event: {N} cells")


def timeit(fn, n=10, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1000


opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-4)
y1 = torch.zeros(N, N_SLOT, device=DEV); sm = torch.zeros(N, N_SLOT, dtype=torch.bool, device=DEV)
y1[B['cell'], B['slot']] = torch.asinh(B['target_charge']); sm[B['cell'], B['slot']] = True


def feat():
    with torch.autocast('cuda', dtype=torch.bfloat16): return model.forward_feat(B, m)


def nll_step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        occ, mu, lv = model(B, m); loss = (mu[B['cell'], B['slot']].float() ** 2).mean()
    loss.backward(); opt.step()


def flow_step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16): z = model.forward_feat(B, m).float()
    flow_loss(head, z, y1, sm).backward(); opt.step()


print(f"forward_feat (transformer body):          {timeit(feat):.0f} ms")
print(f"NLL  train step (fwd+head+bwd):           {timeit(nll_step):.0f} ms")
print(f"FLOW train step (fwd_feat+flow_loss+bwd): {timeit(flow_step):.0f} ms")


@torch.no_grad()
def sample(k, steps):
    with torch.autocast('cuda', dtype=torch.bfloat16): z = model.forward_feat(B, m).float()
    flow_sample(head, z, N_SLOT, steps=steps, k=k)


print("--- sampling/inference (fwd_feat + k*steps head fwds) ---")
for k, st in [(1, 1), (1, 4), (16, 4), (32, 4), (16, 8)]:
    print(f"  k={k:>2} steps={st}: {timeit(lambda: sample(k, st), n=5):.0f} ms/event")
print(f"peak mem: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
