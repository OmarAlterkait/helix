"""P12b — control for P12's 7.5e-3 gradient difference: how much do two runs of
the IDENTICAL dense step differ? SDPA's backward accumulates with atomics, so
some of that number is run-to-run noise, not the sparse head."""
from __future__ import annotations
import os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from kit import head_bmm
from common import build, load_events, emit, to_device
from helix.model.loss import losses_cat

dev = "cuda"
torch.set_float32_matmul_precision("high")
evs, _ = load_events(2)
model = build(device=dev)
B = to_device({k: v for k, v in evs[0].items() if torch.is_tensor(v)}, dev)
B["n_cells"] = B["plane_id"].shape[0]


def dense(model, feat, B, m):
    occ = model.occ_head(feat) * model.readout_mult
    val = (model.val_head(feat) * model.readout_mult).view(-1, model.n_slot, model.n_bins)
    return losses_cat(occ, val, B, m, model.bin_edges, vis_w=0.0)


def grads(head, m):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", torch.bfloat16):
        feat = model.forward_feat(B, m)
        bc, v = head(model, feat, B, m)
    (bc + v).backward()
    return float((bc + v).detach()), {n: p.grad.detach().clone()
                                      for n, p in model.named_parameters() if p.grad is not None}


def cmp(a, b):
    return max(float((a[k] - b[k]).abs().max() / a[k].abs().max().clamp(min=1e-12)) for k in a)


R = {}
for mode in ("random", "plane"):
    g = torch.Generator(device=dev); g.manual_seed(7)
    m = model.make_mask(B, mode=mode, n_planes=1, gen=g)
    l0, g0 = grads(dense, m)
    l0b, g0b = grads(dense, m)
    l1, g1 = grads(head_bmm, m)
    l1b, g1b = grads(head_bmm, m)
    R[mode] = dict(dense_vs_dense=cmp(g0, g0b), sparse_vs_sparse=cmp(g1, g1b),
                   dense_vs_sparse=cmp(g0, g1),
                   d_loss_dense_rerun=l0b - l0, d_loss_sparse=l1 - l0)
    print(f"  {mode:8s} dense-vs-dense {R[mode]['dense_vs_dense']:.3e}   "
          f"sparse-vs-sparse {R[mode]['sparse_vs_sparse']:.3e}   "
          f"dense-vs-sparse {R[mode]['dense_vs_sparse']:.3e}", flush=True)
    print(f"           loss: dense rerun d={R[mode]['d_loss_dense_rerun']:+.2e}  "
          f"sparse d={R[mode]['d_loss_sparse']:+.2e}")
emit("p12b_gradnoise", R)
