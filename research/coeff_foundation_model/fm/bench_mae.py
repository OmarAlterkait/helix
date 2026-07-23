"""Speed/memory sweep for the Perceiver MAE: ms/event + peak GB across
batch size, gradient-checkpointing on/off, and depth. Picks the run config
the same way the deconv sweep did (B=2 no-ckpt was the sweet spot there).
Runs inside the PIMM container on one GPU.
"""
import sys, time, glob, torch, torch.nn as nn
sys.path.insert(0, '/sdf/group/neutrino/omara/helix/.pylibs'); sys.path.insert(0, '.')
import data as D
from data import DEV, N_SLOT, N_BAND
from perceiver_mae import PerceiverMAE, make_mask
D.init_pipeline_cpu()
files = sorted(glob.glob('../artifacts/fm_cache_tpc/ev_*.npz'))


def move(B, d):
    return {k: (v.to(d) if torch.is_tensor(v) else v) for k, v in B.items()}


evs = [move(D.get_cached(files[i]), DEV) for i in range(8)]
g = torch.Generator(device=DEV).manual_seed(0)
print(f"n_cells: {[int(e['n_cells']) for e in evs]}")
print(f"visible@0.75 mask: {[int((~make_mask(int(e['n_cells']),0.75,DEV,g)).sum()) for e in evs]}", flush=True)


for depth in (24, 48):
    m = PerceiverMAE(N_SLOT, N_BAND, 6, d=512, M=2048, depth=depth, heads=8).to(DEV)
    head = nn.Linear(512, N_SLOT).to(DEV)
    opt = torch.optim.AdamW(list(m.parameters()) + list(head.parameters()), lr=1e-9)
    npar = (sum(p.numel() for p in m.parameters()) + sum(p.numel() for p in head.parameters())) / 1e6

    def bench(Bsz, ckpt):
        def step(i):
            opt.zero_grad(set_to_none=True)
            Bs = [evs[(i * Bsz + j) % 8] for j in range(Bsz)]
            masks = [make_mask(int(B['n_cells']), 0.75, DEV, g) for B in Bs]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                zs = m.forward_batch(Bs, masks, ckpt=ckpt)
            loss = 0
            for z, B, mk in zip(zs, Bs, masks):
                recon = head(z.float())
                loss = loss + ((recon[mk] - B['inp'][mk]) ** 2).mean()
            (loss / Bsz).backward(); opt.step()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for i in range(6): step(i)
        torch.cuda.synchronize(); t = time.time(); n = 12
        for i in range(n): step(i)
        torch.cuda.synchronize()
        return (time.time() - t) / n * 1000 / Bsz, torch.cuda.max_memory_allocated() / 1e9

    print(f"\n=== depth {depth} ({npar:.1f}M) ===  {'config':>14} {'ms/event':>9} {'GB':>7}", flush=True)
    for Bsz, ck in [(2, False), (3, False), (4, False), (2, True), (4, True), (6, True), (8, True)]:
        try:
            ms, mem = bench(Bsz, ck)
            print(f"   B={Bsz} ckpt={str(ck):5}: {ms:8.1f} {mem:7.1f}", flush=True)
        except Exception as ex:
            print(f"   B={Bsz} ckpt={str(ck):5}: FAIL {str(ex)[:45]}", flush=True)
print("\ndone", flush=True)
