#!/usr/bin/env python
"""Token budget -> trunk cost -> tokenizer overhead. Pure arithmetic on
MEASURED anchors (no GPU needed; the constants and their provenance are
pinned below). Run:  python compute_budget.py

MEASURED anchors:
- TPC tokens (patch sweep, 200 ev, DESIGN_HANDOFF 4.3): 8x4 -> 25,397 mean
  (p95 27.9k, max 29.3k); 16x8 -> 11,577 (p95 12.1k).
- Optical cells (anchor sweep, info_audit): light_output (SUMMED detector
  readout — the production-representative number): 1024 -> 5,771; 512 ->
  11,542; 256 -> 23,084. doraemon per-interaction chunks inflate cells
  ~4-6x (33.9k at 1024) — an artifact of per-label splitting, NOT the
  detector token count.
- Flash attention f+b (profile_gpu Run A, A100 bf16, d=768 h=12):
  S=4,987 -> 1.90 ms; S=29,924 -> 52.6 ms. Implied sustained ~160 TFLOPS
  for attention-layer f+b math; we use 160 (and 100 pessimistic) for GEMMs.
- Tokenizer ops f+b (microbench_treeops, full typical events, d_s=64):
  C0 tree op 15.5 ms; TPC stems batched 4 calls 4.2 ms fwd (x3 f+b ~12.6);
  optical per-band gather ~0.1-0.3 ms/band fwd.

FLOP model per transformer layer (tokens N, width d):
  linear (qkvo + 2-layer MLP, params 12d^2): fwd 24 d^2 N, f+b 72 d^2 N
  attention matmuls: f+b 12 N_seg^2 d per segment
Hierarchy: 75% of layers within-plane (8 segments of N/8: 6 TPC planes + 2
optical sides), 25% within-volume (2 segments of N/2). Cross-volume summary
tokens: negligible. MAE: encoder sees VIS=30% of tokens; decoder = 8 layers,
d_dec=512, over ALL tokens.
"""
import json, os

TFLOPS = 160e12          # measured-implied sustained (report 100e12 too)
TFLOPS_PESS = 100e12
VIS = 0.30

TPC = {"8x4": 25397, "16x8": 11577}
OPT = {"1024": 5771, "512": 11542, "256": 23084}
MODELS = {  # name: (d, layers, params_M approx 12 d^2 L /1e6)
    "ViT-S": (384, 12), "ViT-B": (768, 12), "ViT-L": (1024, 24), "ViT-H": (1280, 32)}


def trunk_ms(N, d, L, tflops=TFLOPS):
    lin = 72 * d * d * N * L
    n_wp = max(N / 8, 1)
    n_wv = max(N / 2, 1)
    att = 0.75 * L * 8 * 12 * n_wp ** 2 * d + 0.25 * L * 2 * 12 * n_wv ** 2 * d
    return lin / tflops * 1e3, att / tflops * 1e3


def dec_ms(N_all, tflops=TFLOPS):
    d, L = 512, 8
    lin = 72 * d * d * N_all * L
    att = L * 8 * 12 * (N_all / 8) ** 2 * d
    return (lin + att) / tflops * 1e3


def mem_gb(d, L, N_vis, N_all):
    params = 12 * d * d * L
    opt_state = params * 16 / 1e9                 # bf16 w+g + fp32 master+m+v
    act = (N_vis * d * 2 * 12 * L) / 1e9          # bf16, ~12 copies/layer, no ckpt
    dec = (N_all * 512 * 2 * 12 * 8) / 1e9
    return params / 1e6, opt_state, act + dec


def main():
    out = {}
    print("=== 1. TOKEN BUDGET (per event, production-representative) ===")
    print(f"{'config':>14} {'TPC':>7} {'optical':>8} {'joint':>7} {'MAE-vis(30%)':>13}")
    cfgs = {}
    for tp, tn in TPC.items():
        for oc, on in OPT.items():
            j = tn + on
            cfgs[f"{tp}+{oc}"] = j
            print(f"{tp+'+'+oc:>14} {tn:>7,} {on:>8,} {j:>7,} {round(j*VIS):>13,}")
    print("  fixed-shape batching headroom: TPC p95/max at 8x4 = 27.9k/29.3k (+10/15%)")
    print("  CAVEAT: doraemon per-interaction chunks would say 33.9k optical cells at")
    print("  1024 — per-label-splitting artifact; summed readout (light_output) is the")
    print("  production number used above.")
    out["tokens"] = cfgs

    print("\n=== 2. TRUNK COST (fwd+bwd ms/event @160 TFLOPS sustained; [@100]) ===")
    base = cfgs["8x4+1024"]
    rows = {}
    print(f"{'model':>7} {'d':>5} {'L':>3} {'params':>7} | {'MAE lin':>8} {'MAE att':>8} "
          f"{'dec':>6} {'STEP':>7} | {'full-tok step':>13} | {'30ep 10M (GPU-h)':>16}")
    for name, (d, L) in MODELS.items():
        nv = base * VIS
        lin, att = trunk_ms(nv, d, L)
        dc = dec_ms(base)
        step = lin + att + dc
        linf, attf = trunk_ms(base, d, L)
        stepf = linf + attf
        gpuh = step / 1e3 * 10e6 * 30 / 3600
        pm, og, ag = mem_gb(d, L, nv, base)
        rows[name] = dict(d=d, layers=L, params_M=pm, mae_step_ms=step,
                          full_step_ms=stepf, gpu_h_30ep=gpuh,
                          opt_state_GB=og, act_GB=ag)
        print(f"{name:>7} {d:>5} {L:>3} {pm:>6.0f}M | {lin:>7.0f} {att:>8.0f} "
              f"{dc:>6.0f} {step:>6.0f}ms | {stepf:>11.0f}ms | {gpuh:>13,.0f}h")
    print(f"  (@100 TFLOPS pessimistic: multiply times by 1.6)")
    print(f"  token-config sensitivity (ViT-L MAE step): "
          + "  ".join(f"{k}={sum(trunk_ms(v*VIS,1024,24))+dec_ms(v):.0f}ms"
                      for k, v in cfgs.items()))
    out["trunk"] = rows

    print("\n=== 3. TRUNK MEMORY (A100-40GB check, ViT-L, 8x4+1024, MAE) ===")
    pm, og, ag = mem_gb(1024, 24, base * VIS, base)
    print(f"  params {pm:.0f}M | optimizer-state {og:.1f} GB | activations/event "
          f"(no ckpt) {ag:.2f} GB -> per-GPU batch ~{int((40-og-4)//ag)} events "
          f"(more with checkpointing)")

    print("\n=== 4. TOKENIZER OVERHEAD (MEASURED, f+b ms per event, d_s=64) ===")
    tok = {"embed+within-band (TPC stems batched, x2 blocks)": 2 * 12.6,
           "within-band optical (10 gathers x2 blocks)": 2 * 2.5 * 3,
           "tree op C0 (x2 blocks) — DROPPED if rounds0 wins M3": 2 * 15.5,
           "attention-pool to cells + heads": 8.0}
    tot = sum(tok.values())
    for k, v in tok.items():
        print(f"  {k:<58} {v:>6.1f} ms")
    print(f"  {'TOTAL (with tree op)':<58} {tot:>6.1f} ms")
    print(f"  {'TOTAL (without tree op)':<58} {tot-31.0:>6.1f} ms")
    vl = rows["ViT-L"]["mae_step_ms"]
    print(f"  -> overhead vs ViT-L MAE step: {100*tot/(tot+vl):.0f}% with tree op, "
          f"{100*(tot-31)/(tot-31+vl):.0f}% without")
    print(f"  coefficient-stage memory: ~600k coeffs x d_s64 x bf16 x ~12 tensors "
          f"= {600e3*64*2*12/1e9:.1f} GB (trivial)")
    out["tokenizer_ms"] = tok

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "artifacts", "compute_budget.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
