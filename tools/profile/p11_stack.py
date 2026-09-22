"""P11 — the interventions stacked, end to end: ms per EVENT, MiB per EVENT,
and how many events fit. This is the number to quote."""
from __future__ import annotations
import argparse, gc, json, os, sys, traceback
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
import kit
from kit import Pre, forward_feat_ev, grouped_cross_ev, head_bmm, uniform_attn_ev
from common import build, emit, load_events, pack, peak_mem, timeit, to_device
import helix.model.serial as S
from helix.model.loss import losses_cat
from torch.utils.checkpoint import checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=6)
A = ap.parse_args()
dev = "cuda"
torch.set_float32_matmul_precision("high")

evs, _ = load_events(9)
evs_d = [to_device({k: v for k, v in e.items() if torch.is_tensor(v)}, dev) for e in evs]
for e in evs_d:
    e["n_cells"] = e["plane_id"].shape[0]
model = build(device=dev)
masks = []
for i, e in enumerate(evs_d):
    g = torch.Generator(device=dev); g.manual_seed(100 + i)
    masks.append(model.make_mask(e, mode="random", gen=g))
opt = torch.optim.AdamW(model.param_groups(0.0, weight_decay=0.05), betas=(0.9, 0.95))
NC = [int(e["n_cells"]) for e in evs_d]
print("n_cells", NC)
R = {"n_cells": NC, "rows": {}}
_ou, _og, _ar = S.uniform_attn, S.grouped_cross, S.apply_rope


def head_base(model, feat, B, m):
    occ = model.occ_head(feat) * model.readout_mult
    val = (model.val_head(feat) * model.readout_mult).view(-1, model.n_slot, model.n_bins)
    return losses_cat(occ, val, B, m, model.bin_edges, vis_w=0.0)


def ffeat_ckpt(model, B, m, offset=None):
    from p4_ckpt import forward_feat_ckpt
    return forward_feat_ckpt(model, B, m)


def make_step(K, head, rope_bf16, evaware, ckpt):
    if evaware:
        Bp = pack(evs_d[:K]); Bp["n_cells"] = Bp["plane_id"].shape[0]
        mp = torch.cat(masks[:K])
        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16):
                feat = forward_feat_ev(model, Bp, mp, Bp["offset"])
                b, v = head(model, feat, Bp, mp)
            (b + v).backward()
            opt.step()
    else:
        def step():
            opt.zero_grad(set_to_none=True)
            for i in range(K):
                with torch.autocast("cuda", torch.bfloat16):
                    feat = model.forward_feat(evs_d[i], masks[i])
                    b, v = head(model, feat, evs_d[i], masks[i])
                ((b + v) / K).backward()
            opt.step()
    return step


COMBOS = [
    ("A  base (production)",                  1, head_base, False, False),
    ("B  + sparse-active head",               1, head_bmm,  False, False),
    ("C  B + bf16 RoPE tables",               1, head_bmm,  True,  False),
    ("D  C + event-aware pack K=2",           2, head_bmm,  True,  True),
    ("E  C + event-aware pack K=4",           4, head_bmm,  True,  True),
    ("F  C + event-aware pack K=6",           6, head_bmm,  True,  True),
    ("G  C, grad-accum K=4 (no pack)",        4, head_bmm,  True,  False),
]

for name, K, head, rope, ev in COMBOS:
    try:
        S.apply_rope = Pre(cast=torch.bfloat16) if rope else _ar
        S.uniform_attn, S.grouped_cross = ((uniform_attn_ev, grouped_cross_ev) if ev
                                           else (_ou, _og))
        st = make_step(K, head, rope, ev, False)
        t = timeit(st, warmup=2, iters=A.iters)
        with peak_mem() as pm:
            st()
        cells = sum(NC[:K])
        row = dict(K=K, cells=cells, ms=t["ms_med"], ms_per_event=t["ms_med"]/K,
                   peak_MiB=pm["peak_alloc_MiB"], MiB_per_event=pm["peak_alloc_MiB"]/K,
                   tok_per_s=cells/(t["ms_med"]/1e3))
        R["rows"][name] = row
        print(f"  {name:32s} {row['ms']:8.1f} ms  {row['ms_per_event']:7.1f} ms/ev  "
              f"peak {row['peak_MiB']:8.0f} MiB  {row['tok_per_s']:,.0f} tok/s", flush=True)
    except torch.cuda.OutOfMemoryError:
        R["rows"][name] = dict(oom=True); print(f"  {name:32s} OOM", flush=True)
    except Exception as e:
        R["rows"][name] = dict(error=repr(e), tb=traceback.format_exc()[-600:])
        print(f"  {name:32s} ERR {e}", flush=True)
    finally:
        S.apply_rope, S.uniform_attn, S.grouped_cross = _ar, _ou, _og
        opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()

b = R["rows"].get("A  base (production)", {})
for n, r in R["rows"].items():
    if "ms_per_event" in r and "ms_per_event" in b:
        r["speedup_per_event"] = b["ms_per_event"] / r["ms_per_event"]
        r["mem_per_event_vs_base"] = r["MiB_per_event"] / b["MiB_per_event"]
print()
for n, r in R["rows"].items():
    if "speedup_per_event" in r:
        print(f"  {n:32s} x{r['speedup_per_event']:.3f} throughput, "
              f"{r['mem_per_event_vs_base']:.2f}x memory/event")
# ------------------------------------------- is each combo CPU- or GPU-bound?
import time as _t
print("\n== CPU issue vs wall (which side binds) ==", flush=True)
R["bound"] = {}
for name, K, head, rope, ev in COMBOS:
    try:
        S.apply_rope = Pre(cast=torch.bfloat16) if rope else _ar
        S.uniform_attn, S.grouped_cross = ((uniform_attn_ev, grouped_cross_ev) if ev
                                           else (_ou, _og))
        st = make_step(K, head, rope, ev, False)
        for _ in range(3):
            st()
        torch.cuda.synchronize(); t0 = _t.perf_counter()
        for _ in range(A.iters):
            st()
        t1 = _t.perf_counter(); torch.cuda.synchronize(); t2 = _t.perf_counter()
        cpu = (t1 - t0) / A.iters * 1e3
        wall = (t2 - t0) / A.iters * 1e3
        R["bound"][name] = dict(cpu_ms=cpu, wall_ms=wall, K=K,
                                cpu_ms_per_event=cpu / K, wall_ms_per_event=wall / K,
                                bound="GPU" if wall > cpu * 1.05 else "CPU/launch")
        print(f"  {name:32s} cpu {cpu:8.1f}  wall {wall:8.1f}  -> {R['bound'][name]['bound']}",
              flush=True)
    except Exception as e:
        R["bound"][name] = dict(error=repr(e))
        print(f"  {name:32s} ERR {e}")
    finally:
        S.apply_rope, S.uniform_attn, S.grouped_cross = _ar, _ou, _og
        opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()

emit("p11_stack", R)
