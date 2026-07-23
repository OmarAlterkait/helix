#!/usr/bin/env python
"""Measure batching: pack K events with flash-varlen attention (no cross-event
attention via cu_seqlens), sweep K -> fwd+bwd ms, peak mem, throughput, max K.
"""
import sys, time
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, "/sdf/group/neutrino/omara/particle-imaging-models/libs/pimm-data/src")
sys.path.insert(0, "/sdf/group/neutrino/omara/helix")
import hdf5plugin  # noqa
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
import baseline_tpc as bl, star_tpc as stp, vit_tpc as vtp
from star_model import DEV
N_SLOT = vtp.N_SLOT


class VarlenBlock(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, ang_t, ang_w, cu, maxs):
        T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        q = bl.apply_rope(q.view(T, self.h, self.hd), ang_t, ang_w)
        k = bl.apply_rope(k.view(T, self.h, self.hd), ang_t, ang_w)
        v = v.view(T, self.h, self.hd)
        o = flash_attn_varlen_func(q.bfloat16(), k.bfloat16(), v.bfloat16(),
                                   cu, cu, maxs, maxs).to(x.dtype)   # (T,H,hd), block-diag by cu
        x = x + self.proj(o.reshape(T, d))
        return x + self.mlp(self.n2(x))


class BatchedViT(nn.Module):
    def __init__(self, d=192, blocks=4):
        super().__init__()
        self.d = d
        self.embed = nn.Linear(2 * N_SLOT, d)
        self.band_emb = nn.Embedding(bl.NB_T, d); self.gid_emb = nn.Embedding(6, d)
        self.mask_tok = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(VarlenBlock(d) for _ in range(blocks))
        self.dec = nn.Linear(d, 2 * N_SLOT)

    def forward(self, P, tok_mask, cu, maxs):
        x = self.embed(torch.cat([P["inp"], P["occ"]], -1))
        x = torch.where(tok_mask[:, None], self.mask_tok.expand_as(x), x)
        x = x + self.band_emb(P["cell_band"]) + self.gid_emb(P["cell_gid"])
        hd = self.d // 4
        ang_t = bl.rope_angles(P["cell_t"], hd); ang_w = bl.rope_angles(P["cell_wire"], hd)
        for blk in self.blocks:
            x = blk(x, ang_t, ang_w, cu, maxs)
        out = self.dec(x).view(-1, N_SLOT, 2)
        return out[..., 0], out[..., 1]


def pack(events):
    """Concatenate K event-batches into one varlen pack."""
    cells = [e["n_cells"] for e in events]
    cu = torch.tensor(np.concatenate([[0], np.cumsum(cells)]), dtype=torch.int32, device=DEV)
    P = {}
    for k in ("inp", "occ", "cell_band", "cell_gid", "cell_t", "cell_wire"):
        P[k] = torch.cat([e[k] for e in events])
    return P, cu, int(max(cells)), sum(cells)


def main():
    Pp = stp._pipeline(); Pp["nw"] = np.array([Pp["geom"][g]["n_wires"] for g in range(6)], np.int64)
    print("prepping events...", flush=True)
    evs = [bl.event_batch(i) for i in range(32)]                 # cache 32 prepped events
    ntok = [e["n_cells"] for e in evs]
    print(f"tokens/event: mean {np.mean(ntok):.0f} min {min(ntok)} max {max(ntok)}")
    m = BatchedViT(d=192, blocks=4).cuda(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    def run(K, n=8):
        torch.cuda.reset_peak_memory_stats()
        batch = evs[:K]
        P, cu, maxs, T = pack(batch)
        msk = torch.rand(T, device=DEV) < 0.5
        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ol, vp = m(P, msk, cu, maxs)
                loss = F.binary_cross_entropy_with_logits(ol[P["occ"].bool()], P["occ"][P["occ"].bool()]) \
                    + (vp ** 2).mean() * 0.0 + vp.float().mean() * 1e-6
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        for _ in range(2): step()
        torch.cuda.synchronize(); s = time.time()
        for _ in range(n): step()
        torch.cuda.synchronize()
        ms = (time.time() - s) / n * 1000
        return ms, T, torch.cuda.max_memory_allocated() / 1e6

    print(f"\n{'K':>3} {'tokens':>9} {'ms/step':>9} {'peak MB':>9} {'events/s':>9} {'tok/s':>10}")
    for K in (1, 2, 4, 8, 16, 32):
        try:
            ms, T, mem = run(K)
            print(f"{K:>3} {T:>9} {ms:>9.0f} {mem:>9.0f} {1000*K/ms:>9.1f} {1000*T/ms:>10.0f}", flush=True)
        except RuntimeError as e:
            print(f"{K:>3}: {'OOM' if 'memory' in str(e).lower() else str(e)[:50]}"); break


if __name__ == "__main__":
    main()
