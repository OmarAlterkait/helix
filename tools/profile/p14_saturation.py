"""P14 — is the GPU ever idle at batch 1?

If it is, a bigger batch fills it and packing should pay. If it is not, packing
can only amortise whatever is genuinely PER-STEP rather than per-token, and
nothing else. This measures three things directly:

  1. GPU busy fraction (NVML utilisation, sampled; and kernel-time / wall from a
     CUDA-only profile, which carries far less overhead than a CPU+CUDA one).
  2. The per-step-vs-per-token split: fwd+bwd alone against optimizer+clip alone,
     at K=1 and packed, so the packing gain can be PREDICTED and then checked.
  3. Whether two concurrent single-event steps on separate streams take less than
     2x one -- the headroom question with no batching involved.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, threading, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import build, emit, load_events, peak_mem, timeit, to_device
from kit import Pre, forward_feat_ev, grouped_cross_ev, head_bmm, uniform_attn_ev
from common import pack
import helix.model.serial as S
from helix.model.loss import losses_cat

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--sample_s", type=float, default=12.0)
A = ap.parse_args()
dev = "cuda"
torch.set_float32_matmul_precision("high")

evs, _ = load_events(7)
evs_d = [to_device({k: v for k, v in e.items() if torch.is_tensor(v)}, dev) for e in evs]
for e in evs_d:
    e["n_cells"] = e["plane_id"].shape[0]
model = build(device=dev)
masks = []
for i, e in enumerate(evs_d):
    g = torch.Generator(device=dev); g.manual_seed(100 + i)
    masks.append(model.make_mask(e, mode="random", gen=g))
opt = torch.optim.AdamW(model.param_groups(0.0, weight_decay=0.05), betas=(0.9, 0.95))
_ar, _ou, _og = S.apply_rope, S.uniform_attn, S.grouped_cross
B, m = evs_d[0], masks[0]
R = {"n_cells": [int(e["n_cells"]) for e in evs_d]}


def head_base(model, feat, B, m):
    occ = model.occ_head(feat) * model.readout_mult
    val = (model.val_head(feat) * model.readout_mult).view(-1, model.n_slot, model.n_bins)
    return losses_cat(occ, val, B, m, model.bin_edges, vis_w=0.0)


def fb_once(Bx, mx, head, ev=False, off=None):
    with torch.autocast("cuda", torch.bfloat16):
        feat = forward_feat_ev(model, Bx, mx, off) if ev else model.forward_feat(Bx, mx)
        b, v = head(model, feat, Bx, mx)
    (b + v).backward()


def mk(K, head, ev):
    if ev and K > 1:
        Bp = pack(evs_d[:K]); Bp["n_cells"] = Bp["plane_id"].shape[0]
        mp = torch.cat(masks[:K])
        def fb():
            opt.zero_grad(set_to_none=True); fb_once(Bp, mp, head, True, Bp["offset"])
    else:
        def fb():
            opt.zero_grad(set_to_none=True)
            for i in range(K):
                fb_once(evs_d[i], masks[i], head)
    def st():
        fb()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return fb, st


# ---------------------------------------------- 1. GPU busy fraction
def busy(tag, st):
    for _ in range(3):
        st()
    torch.cuda.synchronize()
    stop = threading.Event(); samples = []
    def sampler():
        while not stop.is_set():
            try:
                samples.append(torch.cuda.utilization(0))
            except Exception:
                pass
            time.sleep(0.01)
    th = threading.Thread(target=sampler, daemon=True); th.start()
    t0 = time.perf_counter(); n = 0
    while time.perf_counter() - t0 < A.sample_s:
        st(); n += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    stop.set(); th.join()
    # kernel time / wall, from a CUDA-ONLY profile (much less overhead than CPU+CUDA)
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA], acc_events=True) as pr:
        torch.cuda.synchronize(); p0 = time.perf_counter()
        for _ in range(A.iters):
            st()
        torch.cuda.synchronize(); pwall = (time.perf_counter() - p0) * 1e3
    ker = sum(e.self_device_time_total for e in pr.key_averages()) / 1e3
    row = dict(steps=n, ms_per_step=wall / n * 1e3,
               nvml_util_mean=float(np.mean(samples)) if samples else None,
               nvml_util_p10=float(np.percentile(samples, 10)) if samples else None,
               nvml_samples=len(samples),
               kernel_ms=ker / A.iters, prof_wall_ms=pwall / A.iters,
               busy_frac=ker / pwall)
    R["busy"][tag] = row
    print(f"  {tag:28s} {row['ms_per_step']:8.1f} ms/step  "
          f"NVML util {row['nvml_util_mean']:5.1f}% (p10 {row['nvml_util_p10']:5.1f})  "
          f"kernel/wall {row['busy_frac']*100:5.1f}%", flush=True)
    opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()


print("== GPU busy fraction ==", flush=True)
R["busy"] = {}
fb1, st1 = mk(1, head_base, False)
busy("base K=1", st1)
S.apply_rope = Pre(cast=torch.bfloat16)
fbC, stC = mk(1, head_bmm, False)
busy("optimised (C) K=1", stC)
S.apply_rope = _ar
S.uniform_attn, S.grouped_cross = uniform_attn_ev, grouped_cross_ev
S.apply_rope = Pre(cast=torch.bfloat16)
fb4, st4 = mk(4, head_bmm, True)
busy("optimised pack K=4", st4)
S.apply_rope, S.uniform_attn, S.grouped_cross = _ar, _ou, _og

# ------------------------------- 2. per-step vs per-token, and a prediction
print("\n== what is PER-STEP (amortisable) vs PER-TOKEN (not) ==", flush=True)
R["split"] = {}
for tag, K, head, ev, rope in (("base K=1", 1, head_base, False, False),
                               ("optimised C K=1", 1, head_bmm, False, True),
                               ("optimised pack K=4", 4, head_bmm, True, True),
                               ("optimised seq K=4", 4, head_bmm, False, True)):
    if rope:
        S.apply_rope = Pre(cast=torch.bfloat16)
    if ev and K > 1:
        S.uniform_attn, S.grouped_cross = uniform_attn_ev, grouped_cross_ev
    fb, st = mk(K, head, ev)
    t_fb = timeit(fb, warmup=3, iters=A.iters)
    fb()
    t_opt = timeit(lambda: (torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0),
                            opt.step()), warmup=3, iters=A.iters)
    t_st = timeit(st, warmup=3, iters=A.iters)
    R["split"][tag] = dict(K=K, fwd_bwd_ms=t_fb["ms_med"], opt_clip_ms=t_opt["ms_med"],
                           step_ms=t_st["ms_med"],
                           per_step_frac=t_opt["ms_med"] / t_st["ms_med"])
    print(f"  {tag:22s} fwd+bwd {t_fb['ms_med']:8.1f}  opt+clip {t_opt['ms_med']:6.2f}  "
          f"step {t_st['ms_med']:8.1f}  -> per-step share {100*t_opt['ms_med']/t_st['ms_med']:.1f}%",
          flush=True)
    S.apply_rope, S.uniform_attn, S.grouped_cross = _ar, _ou, _og
    opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()

c = R["split"].get("optimised C K=1")
if c:
    pred = (4 * c["fwd_bwd_ms"] + c["opt_clip_ms"]) / (4 * c["step_ms"])
    R["prediction"] = dict(
        note="if packing amortises ONLY opt+clip and fwd+bwd is exactly linear",
        predicted_speedup_K4=1 / pred,
        measured_speedup_K4=(4 * c["step_ms"]) / R["split"]["optimised pack K=4"]["step_ms"])
    print(f"\n  predicted K=4 speedup if only opt+clip amortises: "
          f"{R['prediction']['predicted_speedup_K4']:.3f}")
    print(f"  measured  K=4 speedup:                            "
          f"{R['prediction']['measured_speedup_K4']:.3f}")

# --------------------------- 3. headroom test: two steps on two streams
print("\n== two concurrent single-event steps on separate streams ==", flush=True)
try:
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    def one(stream, i):
        with torch.cuda.stream(stream):
            with torch.autocast("cuda", torch.bfloat16):
                feat = model.forward_feat(evs_d[i], masks[i])
                b, v = head_base(model, feat, evs_d[i], masks[i])
            (b + v).backward()
    def serial2():
        opt.zero_grad(set_to_none=True); one(torch.cuda.current_stream(), 0)
        opt.zero_grad(set_to_none=True); one(torch.cuda.current_stream(), 1)
    def par2():
        opt.zero_grad(set_to_none=True)
        one(s1, 0); one(s2, 1)
        torch.cuda.current_stream().wait_stream(s1)
        torch.cuda.current_stream().wait_stream(s2)
    t1 = timeit(lambda: (opt.zero_grad(set_to_none=True), one(torch.cuda.current_stream(), 0)),
                warmup=3, iters=A.iters)
    ts = timeit(serial2, warmup=3, iters=A.iters)
    tp = timeit(par2, warmup=3, iters=A.iters)
    R["streams"] = dict(one_ms=t1["ms_med"], two_serial_ms=ts["ms_med"],
                        two_parallel_ms=tp["ms_med"],
                        headroom=ts["ms_med"] / tp["ms_med"])
    print(f"  one event            {t1['ms_med']:8.1f} ms")
    print(f"  two, same stream     {ts['ms_med']:8.1f} ms")
    print(f"  two, two streams     {tp['ms_med']:8.1f} ms   -> "
          f"{ts['ms_med']/tp['ms_med']:.3f}x (1.00 = no idle capacity)")
except Exception as e:
    R["streams"] = dict(error=repr(e))
    print("  stream test failed", e)

emit("p14_saturation", R)
