"""P12 — is the one-event step launch-bound, and by how much?

A cpu-vs-wall comparison cannot answer this: the step contains 77 host
synchronisations, each of which blocks the CPU until the GPU catches up, so the
two always agree. Two measurements that are not confounded:

  1. the size-independent floor: shrink the token count until GPU work is
     negligible and see what the step still costs;
  2. this node's raw launch rate, times the step's measured launch count.

Also re-checks the sparse-active head under the PLANE mask (a different active
density) and against gradients, not just the loss scalar.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
import kit
from kit import head_bmm
from common import build, emit, load_events, peak_mem, synth_from, timeit, to_device
from helix.model.loss import losses_cat

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=10)
A = ap.parse_args()
dev = "cuda"
torch.set_float32_matmul_precision("high")
R = {}

# --------------------------------------------------- 1. raw launch rate here
x = torch.zeros(64, device=dev)
for _ in range(1000):
    x.add_(1.0)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20000):
    x.add_(1.0)
t1 = time.perf_counter()
torch.cuda.synchronize()
t2 = time.perf_counter()
R["launch"] = dict(us_per_launch_cpu=(t1 - t0) / 20000 * 1e6,
                   us_per_launch_wall=(t2 - t0) / 20000 * 1e6,
                   host=os.uname().nodename)
print(f"raw launch: {R['launch']['us_per_launch_cpu']:.2f} us/kernel (cpu issue), "
      f"{R['launch']['us_per_launch_wall']:.2f} us (wall)")

# --------------------------------------------------- 2. size-independent floor
evs, _ = load_events(2)
src = {k: v for k, v in evs[0].items() if torch.is_tensor(v)}
model = build(device=dev)
opt = torch.optim.AdamW(model.param_groups(0.0, weight_decay=0.05), betas=(0.9, 0.95))

def mk(nc):
    B = to_device(synth_from(src, nc), dev)
    B["n_cells"] = nc
    g = torch.Generator(device=dev); g.manual_seed(0)
    return B, model.make_mask(B, mode="random", gen=g)

rows = []
for nc in (500, 1000, 2000, 4000, 8000, 16000, 32000):
    B, m = mk(nc)
    def st():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            feat = model.forward_feat(B, m)
            occ = model.occ_head(feat) * model.readout_mult
            val = (model.val_head(feat) * model.readout_mult).view(-1, model.n_slot, model.n_bins)
            b, v = losses_cat(occ, val, B, m, model.bin_edges, vis_w=0.0)
        (b + v).backward()
        opt.step()
    t = timeit(st, warmup=3, iters=A.iters)
    rows.append(dict(n_cells=nc, ms=t["ms_med"]))
    print(f"  N={nc:6d}  {t['ms_med']:8.2f} ms", flush=True)
    B = m = None; gc.collect(); torch.cuda.empty_cache()
R["floor"] = rows
a, b = rows[0]["ms"], rows[-1]["ms"]
R["floor_summary"] = dict(floor_ms=a, full_event_ms=b, fixed_frac=a / b)
print(f"  size-independent floor ~{a:.1f} ms of a {b:.1f} ms one-event step "
      f"({100*a/b:.0f}%)")

# --------------------------------------- 3. sparse head under both mask modes
print("\n== sparse-active head vs dense, both mask modes, gradients too ==", flush=True)
B = to_device(src, dev); B["n_cells"] = B["plane_id"].shape[0]
R["head_check"] = {}
for mode in ("random", "plane", "plane_any"):
    g = torch.Generator(device=dev); g.manual_seed(7)
    m = model.make_mask(B, mode=mode, n_planes=1, gen=g)
    act = B["occ"].bool() & B["valid"] & m[:, None]
    def grads(head):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            feat = model.forward_feat(B, m)
            bc, v = head(model, feat, B, m)
        (bc + v).backward()
        return float(bc + v), {n: p.grad.detach().clone() for n, p in model.named_parameters()
                               if p.grad is not None}
    def dense(model, feat, B, m):
        occ = model.occ_head(feat) * model.readout_mult
        val = (model.val_head(feat) * model.readout_mult).view(-1, model.n_slot, model.n_bins)
        return losses_cat(occ, val, B, m, model.bin_edges, vis_w=0.0)
    l0, g0 = grads(dense)
    l1, g1 = grads(head_bmm)
    rel = max(float((g0[k] - g1[k]).abs().max() / g0[k].abs().max().clamp(min=1e-12))
              for k in g0)
    t0 = timeit(lambda: grads(dense), warmup=1, iters=4)
    t1 = timeit(lambda: grads(head_bmm), warmup=1, iters=4)
    R["head_check"][mode] = dict(masked_frac=float(m.float().mean()),
                                 active_frac=float(act.float().mean()),
                                 loss_dense=l0, loss_sparse=l1, d_loss=l1 - l0,
                                 max_rel_grad_diff=rel,
                                 ms_dense=t0["ms_med"], ms_sparse=t1["ms_med"])
    r = R["head_check"][mode]
    print(f"  {mode:10s} masked={r['masked_frac']:.3f} active={r['active_frac']:.4f}  "
          f"dloss={r['d_loss']:+.2e}  max rel grad diff={rel:.2e}  "
          f"{r['ms_dense']:.1f} -> {r['ms_sparse']:.1f} ms", flush=True)
    opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()

emit("p12_floor", R)
