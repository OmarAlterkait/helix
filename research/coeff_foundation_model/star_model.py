#!/usr/bin/env python
"""M3 star design substrate — one point-set model family; arms = masks + ties.

Per-chunk autoencoder over active clean-support coefficients (doraemon,
bands A10..D2; D1 dropped per design). Encoder variants differ ONLY in
(topology, weight-tying); bottleneck/decoder identical and pinned to the
production geometry:

  embed: asinh(v/2.6) -> Linear, + sinusoidal PE on physical time
         (group-delay corrected), + band embedding (W2/W7/ceiling arms)
  blocks xK: within-band gather(+-1) mix  ->  cross-level op  ->  MLP
  bottleneck: attention-pool coeffs into anchor cells (1024 ticks, 512 slots)
  decoder: cell token -> MLP -> 512 x (occupancy logit, asinh value);
           skips OFF through the bottleneck (tokens are the only path)

Arms (--arm):
  rounds0  : no cross-level op (the control)
  c0w7     : adjacent V-cycle (level-sequential up+down), per-band weights
  c0w7r2   : same, 2 rounds per block
  c2w7     : hierarchical up + ancestor-chain attention down, per-band weights
  c0w2     : C0, shared weights + FiLM(band)
  c0w0     : C0, fully shared, no band conditioning anywhere
  ceiling  : full attention per chunk (band-embedded, 2x width) — envelope ref

Primary metric (registered): equal-weight per-band asinh-MSE on active slots,
held-out events. Stratified: orphan vs parented actives; per-band tables.
Secondary: occupancy BCE (AP probe separate).

Run from this folder, e.g.:
  python star_model.py --arm c0w7 --fold 0 --seed 0 --steps 3000
"""
import sys, os, json, time, argparse, math

sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DSET = os.path.join(HERE, "artifacts", "coeff_dataset")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

NBANDS = 10                      # A10, D10..D2 (band_id 0..9); D1 (10) dropped
LEVELS = np.array([10] + list(range(10, 1, -1)))          # per band_id
DELTA = np.array([-2.38, .62, .75, .50, 1.0, .50, 0., 0., 0., 1.0])
SLOT_OFF = np.concatenate([[0], np.cumsum([1] + [1 << k for k in range(9)])])
N_SLOT = 512
SIGMA = 2.6
CELL = 1024
LOG_A = 10


def set_anchor(anchor):
    """Re-derive cell geometry for a different anchor (512/1024/2048)."""
    global CELL, LOG_A, SLOT_OFF, N_SLOT
    CELL = anchor
    LOG_A = int(np.log2(anchor))
    widths = [1 << max(0, LOG_A - int(LEVELS[b])) for b in range(NBANDS)]
    SLOT_OFF = np.concatenate([[0], np.cumsum(widths)])
    N_SLOT = int(SLOT_OFF[-1])


# ------------------------------------------------------------------ data ----

def event_files():
    return sorted(f for f in os.listdir(DSET) if f.endswith(".npz"))


def prep_event(fn):
    """npz path: load clean-support dump -> rows + maps."""
    d = np.load(os.path.join(DSET, fn))
    keep = d["band_id"] < NBANDS
    return build_event_struct(
        d["band_id"][keep].astype(np.int64), d["idx"][keep].astype(np.int64),
        d["value"][keep].astype(np.float32), d["chunk_id"][keep].astype(np.int64),
        d["chunk_len"].astype(np.int64))


def build_event_struct(band, idx, val, chunk, cl, val_clean=None):
    """Vectorized per-event preprocessing -> rows + chunk-relative maps."""
    o = np.lexsort((idx, band, chunk))
    band, idx, val, chunk = band[o], idx[o], val[o], chunk[o]
    if val_clean is not None:
        val_clean = val_clean[o]
    n_chunks, N = len(cl), len(band)
    _q = max(CELL, 1024)
    Lp = ((cl + _q - 1) // _q) * _q

    grids = {}
    for b in range(NBANDS):
        sizes = Lp >> LEVELS[b]
        off = np.zeros(n_chunks + 1, np.int64)
        off[1:] = np.cumsum(sizes)
        g = np.full(off[-1], -1, np.int64)
        r = np.nonzero(band == b)[0]
        g[off[chunk[r]] + idx[r]] = r
        grids[b] = (g, off)
    parent = np.full(N, -1, np.int64)
    anc = np.full((N, NBANDS - 2), -1, np.int64)
    for b in range(2, NBANDS):
        r = np.nonzero(band == b)[0]
        if not len(r):
            continue
        for k, pb in enumerate(range(b - 1, 0, -1)):
            g, off = grids[pb]
            m = g[off[chunk[r]] + (idx[r] >> (b - pb))]
            if pb == b - 1:
                parent[r] = m
            anc[r, k] = m
    cstart = np.searchsorted(chunk, np.arange(n_chunks + 1))
    rel = cstart[chunk]
    parent_rel = np.where(parent >= 0, parent - rel, -1)
    anc_rel = np.where(anc >= 0, anc - rel[:, None], -1)

    j = LEVELS[band]
    sh = LOG_A - j
    shp = np.maximum(sh, 0)
    cell_local = np.where(sh >= 0, idx >> shp, idx << np.maximum(-sh, 0))
    within = np.where(sh >= 0, idx & ((1 << shp) - 1), 0)
    slot = SLOT_OFF[band] + within
    tphys = ((idx + DELTA[band]) * (1 << j)).astype(np.float32)
    ev = dict(band=band, idx=idx, val=val, chunk=chunk, parent=parent_rel,
              anc=anc_rel, cell_local=cell_local, slot=slot, tphys=tphys,
              cstart=cstart, cl=cl, n_chunks=n_chunks)
    if val_clean is not None:
        ev["val_clean"] = val_clean
    return ev


class Packer:
    """Pack whole chunks into batches of ~budget coefficients."""

    assemble_fn = None      # class-level override (set by --tokmode band)

    def __init__(self, files, budget=60000, seed=0, sort_size=False, prep=None):
        self.files, self.budget = files, budget
        self.rng = np.random.default_rng(seed)
        self.sort_size = sort_size
        self.prep = prep or prep_event

    def __iter__(self):
        for fi in self.rng.permutation(len(self.files)):
            ev = self.prep(self.files[fi])
            yield from pack_event(ev, self.rng, self.budget, self.sort_size,
                                  assemble_fn=Packer.assemble_fn)


def pack_event(ev, rng, budget, sort_size, assemble_fn=None):
    assemble_fn = assemble_fn or assemble
    sizes = np.diff(ev["cstart"])
    order = (np.argsort(sizes) if sort_size else rng.permutation(ev["n_chunks"]))
    sel, n = [], 0
    for ci in order:
        if sizes[ci] == 0:
            continue
        if n + sizes[ci] > budget and sel:
            yield assemble_fn(ev, sel)
            sel, n = [], 0
        sel.append(ci)
        n += int(sizes[ci])
    if sel:
        yield assemble_fn(ev, sel)


P_BAND = 64


def assemble_band(ev, chunk_ids):
    """Per-band-patch tokens: token = (chunk, band, P_BAND-coeff window)."""
    cs = ev["cstart"]
    rows = np.concatenate([np.arange(cs[c], cs[c + 1]) for c in chunk_ids])
    sizes = np.array([cs[c + 1] - cs[c] for c in chunk_ids])
    boff = np.zeros(len(chunk_ids) + 1, np.int64)
    boff[1:] = np.cumsum(sizes)
    bchunk = np.repeat(np.arange(len(chunk_ids)), sizes)

    band, idx = ev["band"][rows], ev["idx"][rows]
    val, tphys = ev["val"][rows], ev["tphys"][rows]
    target = ev["val_clean"][rows] if "val_clean" in ev else val
    pr, ar = ev["parent"][rows], ev["anc"][rows]
    parent = np.where(pr >= 0, boff[bchunk] + pr, -1)
    anc = np.where(ar >= 0, boff[bchunk][:, None] + ar, -1)

    key = (bchunk << 40) | (band << 32) | (idx // P_BAND)
    uniq, cell = np.unique(key, return_inverse=True)
    n_cells = len(uniq)
    slot = idx % P_BAND
    cell_band = ((uniq >> 32) & 0xFF).astype(np.int64)
    cell_win = (uniq & 0xFFFFFFFF).astype(np.int64)
    cell_chunk = (uniq >> 40).astype(np.int64)

    occ = np.zeros((n_cells, P_BAND), bool)
    tgt = np.zeros((n_cells, P_BAND), np.float32)
    occ[cell, slot] = True
    tgt[cell, slot] = np.arcsinh(target / SIGMA)
    cl = ev["cl"][np.array(chunk_ids)]
    n_valid_b = -(-cl[cell_chunk] // (1 << LEVELS[cell_band]))   # ceil(cl/2^j)
    valid = (cell_win[:, None] * P_BAND + np.arange(P_BAND)[None, :]) < n_valid_b[:, None]

    t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=DEV)
    return dict(
        band=t(band, torch.long), idx=t(idx, torch.long),
        val=t(np.arcsinh(val / SIGMA), torch.float32),
        target=t(np.arcsinh(target / SIGMA), torch.float32),
        tphys=t(tphys, torch.float32), parent=t(parent, torch.long),
        anc=t(anc, torch.long), cell=t(cell, torch.long),
        slot=t(slot, torch.long), chunk=t(bchunk, torch.long),
        n_cells=n_cells, occ=t(occ, torch.float32), tgt=t(tgt, torch.float32),
        valid=t(valid, torch.bool),
        orphan=t((pr < 0) & (band >= 2), torch.bool),
        cell_band=t(cell_band, torch.long))


FINE_BANDS = 8                       # bands >= this go per-band (D3, D2)
HYB_OFF = np.concatenate([[0], np.cumsum([1, 1, 2, 4, 8, 16, 32, 64])])  # 128


def assemble_hybrid(ev, chunk_ids):
    """Hybrid: column tokens for bands 0..7 (anchor 1024), per-band P=64
    windows for D3/D2. Column tokens get type id NBANDS in cell_band."""
    cs = ev["cstart"]
    rows = np.concatenate([np.arange(cs[c], cs[c + 1]) for c in chunk_ids])
    sizes = np.array([cs[c + 1] - cs[c] for c in chunk_ids])
    boff = np.zeros(len(chunk_ids) + 1, np.int64)
    boff[1:] = np.cumsum(sizes)
    bchunk = np.repeat(np.arange(len(chunk_ids)), sizes)

    band, idx = ev["band"][rows], ev["idx"][rows]
    val, tphys = ev["val"][rows], ev["tphys"][rows]
    target = ev["val_clean"][rows] if "val_clean" in ev else val
    pr, ar = ev["parent"][rows], ev["anc"][rows]
    parent = np.where(pr >= 0, boff[bchunk] + pr, -1)
    anc = np.where(ar >= 0, boff[bchunk][:, None] + ar, -1)

    fine = band >= FINE_BANDS
    j = LEVELS[band]
    within = idx & ((1 << (10 - j)) - 1)               # anchor-1024 layout
    cell_loc = idx >> (10 - j)
    key = np.where(
        fine,
        (1 << 55) | (bchunk << 40) | (band << 32) | (idx // P_BAND),
        (bchunk << 40) | cell_loc)
    uniq, cell = np.unique(key, return_inverse=True)
    n_cells = len(uniq)
    slot = np.where(fine, idx % P_BAND, HYB_OFF[np.minimum(band, FINE_BANDS - 1)] + within)
    u_fine = (uniq >> 55) > 0
    cell_band = np.where(u_fine, (uniq >> 32) & 0xFF, NBANDS).astype(np.int64)
    cell_win = (uniq & 0xFFFFFFFF).astype(np.int64)
    cell_chunk = ((uniq >> 40) & 0x7FFF).astype(np.int64)

    NS = 128
    occ = np.zeros((n_cells, NS), bool)
    tgt = np.zeros((n_cells, NS), np.float32)
    inp = np.zeros((n_cells, NS), np.float32)
    occ[cell, slot] = True
    tgt[cell, slot] = np.arcsinh(target / SIGMA)
    inp[cell, slot] = np.arcsinh(val / SIGMA)
    cl = ev["cl"][np.array(chunk_ids)]
    valid = np.zeros((n_cells, NS), bool)
    f = u_fine
    if f.any():                                        # fine tokens: 64 slots
        nv = -(-cl[cell_chunk[f]] // (1 << LEVELS[np.minimum(cell_band[f], NBANDS - 1)]))
        valid[np.nonzero(f)[0][:, None], np.arange(P_BAND)[None, :]] =             (cell_win[f][:, None] * P_BAND + np.arange(P_BAND)[None, :]) < nv[:, None]
    c = ~u_fine
    if c.any():                                        # column tokens: 128 slots
        ci = np.nonzero(c)[0]
        for b in range(FINE_BANDS):
            jb = int(LEVELS[b])
            w = 1 << (10 - jb)
            tt = ((cell_win[c][:, None] * w + np.arange(w)[None, :]) << jb)
            valid[ci[:, None], (HYB_OFF[b] + np.arange(w))[None, :]] = tt < cl[cell_chunk[c]][:, None]

    t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=DEV)
    return dict(
        band=t(band, torch.long), idx=t(idx, torch.long),
        val=t(np.arcsinh(val / SIGMA), torch.float32),
        target=t(np.arcsinh(target / SIGMA), torch.float32),
        tphys=t(tphys, torch.float32), parent=t(parent, torch.long),
        anc=t(anc, torch.long), cell=t(cell, torch.long),
        slot=t(slot, torch.long), chunk=t(bchunk, torch.long),
        n_cells=n_cells, occ=t(occ, torch.float32), tgt=t(tgt, torch.float32),
        valid=t(valid, torch.bool),
        orphan=t((pr < 0) & (band >= 2), torch.bool),
        cell_band=t(cell_band, torch.long),
        inp=t(inp, torch.float32), cell_chunk=t(cell_chunk, torch.long))


def assemble(ev, chunk_ids):
    """Slice selected chunks out of a prepped event -> torch batch."""
    cs = ev["cstart"]
    rows = np.concatenate([np.arange(cs[c], cs[c + 1]) for c in chunk_ids])
    sizes = np.array([cs[c + 1] - cs[c] for c in chunk_ids])
    boff = np.zeros(len(chunk_ids) + 1, np.int64)
    boff[1:] = np.cumsum(sizes)
    bchunk = np.repeat(np.arange(len(chunk_ids)), sizes)

    band, idx = ev["band"][rows], ev["idx"][rows]
    val, tphys = ev["val"][rows], ev["tphys"][rows]
    target = ev["val_clean"][rows] if "val_clean" in ev else val   # denoise if clean known
    pr, ar = ev["parent"][rows], ev["anc"][rows]
    parent = np.where(pr >= 0, boff[bchunk] + pr, -1)
    anc = np.where(ar >= 0, boff[bchunk][:, None] + ar, -1)

    cl = ev["cl"][np.array(chunk_ids)]
    ncell_per = (cl + CELL - 1) // CELL
    cellbase = np.zeros(len(chunk_ids) + 1, np.int64)
    cellbase[1:] = np.cumsum(ncell_per)
    n_cells = int(cellbase[-1])
    cell = cellbase[bchunk] + ev["cell_local"][rows]
    slot = ev["slot"][rows]

    # dense targets + slot validity (vectorized over cells)
    occ = np.zeros((n_cells, N_SLOT), bool)
    tgt = np.zeros((n_cells, N_SLOT), np.float32)
    occ[cell, slot] = True
    tgt[cell, slot] = np.arcsinh(target / SIGMA)
    cell_within = np.concatenate([np.arange(k) for k in ncell_per])
    cell_cl = np.repeat(cl, ncell_per)
    valid = np.zeros((n_cells, N_SLOT), bool)
    for b in range(NBANDS):
        jb = int(LEVELS[b])
        sh = LOG_A - jb
        if sh >= 0:
            w = 1 << sh
            tt = ((cell_within[:, None] * w + np.arange(w)[None, :]) << jb)
            valid[:, SLOT_OFF[b]:SLOT_OFF[b] + w] = tt < cell_cl[:, None]
        else:                       # coarse coeff spans 2^-sh cells: only aligned cells host it
            tt = (cell_within << LOG_A)
            ok = (tt < cell_cl) & (cell_within % (1 << -sh) == 0)
            valid[:, SLOT_OFF[b]] = ok

    t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=DEV)
    return dict(
        band=t(band, torch.long), idx=t(idx, torch.long),
        val=t(np.arcsinh(val / SIGMA), torch.float32),
        target=t(np.arcsinh(target / SIGMA), torch.float32),
        tphys=t(tphys, torch.float32), parent=t(parent, torch.long),
        anc=t(anc, torch.long), cell=t(cell, torch.long),
        slot=t(slot, torch.long), chunk=t(bchunk, torch.long),
        n_cells=n_cells, occ=t(occ, torch.float32), tgt=t(tgt, torch.float32),
        valid=t(valid, torch.bool),
        orphan=t((pr < 0) & (band >= 2), torch.bool))


# ----------------------------------------------------------------- model ----

class BandLinear(nn.Module):
    """Linear with weight mode: 'per' (per-band), 'film' (shared+FiLM), 'shared'."""

    def __init__(self, din, dout, mode, n_bands=NBANDS):
        super().__init__()
        self.mode = mode
        self.n_bands = n_bands
        if mode == "per":
            self.W = nn.Parameter(torch.randn(n_bands, din, dout) * (din ** -0.5))
            self.b = nn.Parameter(torch.zeros(n_bands, dout))
        else:
            self.lin = nn.Linear(din, dout)
            if mode == "film":
                self.film = nn.Embedding(n_bands, 2 * dout)
                nn.init.zeros_(self.film.weight)

    def forward(self, x, band):
        if self.mode == "per":
            out = torch.zeros(x.shape[0], self.W.shape[2], device=x.device, dtype=x.dtype)
            for bb in range(self.n_bands):
                m = band == bb
                if m.any():
                    out[m] = x[m] @ self.W[bb] + self.b[bb]
            return out
        out = self.lin(x)
        if self.mode == "film":
            g, s = self.film(band).chunk(2, -1)
            out = out * (1 + g) + s
        return out


class FullBlock(nn.Module):
    """Pre-LN full-attention block via SDPA (flash/mem-efficient kernels)."""

    def __init__(self, d, heads=4):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, amask):
        Bn, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        sh = (Bn, T, self.h, d // self.h)
        q, k, v = (a.view(sh).transpose(1, 2) for a in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=amask)
        x = x + self.proj(o.transpose(1, 2).reshape(Bn, T, d))
        return x + self.mlp(self.n2(x))


class Block(nn.Module):
    def __init__(self, d, topo, wmode, rounds=1, reach=1, n_bands=NBANDS, n_nbr=None):
        super().__init__()
        self.topo, self.rounds = topo, rounds
        self.n_bands = n_bands
        n_nbr = 2 * reach if n_nbr is None else n_nbr
        self.n1 = nn.LayerNorm(d)
        self.local = BandLinear((n_nbr + 1) * d, d, wmode, n_bands)
        self.n2 = nn.LayerNorm(d)
        self.up = BandLinear(d, d, wmode, n_bands)
        self.down = BandLinear(d, d, wmode, n_bands)
        if topo == "c2":
            self.q = nn.Linear(d, d)
            self.kv = nn.Linear(d, 2 * d)
        self.n3 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, B, nbr):
        h = self.n1(x)
        x = x + self.local(
            torch.cat([h] + [h[nbr[:, k]] for k in range(nbr.shape[1])], -1), B["band"])
        if self.topo != "none":
            for _ in range(self.rounds):
                x = self._cross(x, B)
        return x + self.mlp(self.n3(x))

    def _cross(self, x, B):
        band, parent = B["band"], B["parent"]
        h = self.n2(x)
        for bb in range(self.n_bands - 1, 1, -1):        # up, fine->coarse
            m = (band == bb) & (parent >= 0)
            if m.any():
                x = x.index_add(0, parent[m], self.up(h[m], band[m]))
                h = self.n2(x)
        if self.topo == "c0":
            for bb in range(2, self.n_bands):            # down, coarse->fine
                m = (band == bb) & (parent >= 0)
                if m.any():
                    rows = m.nonzero(as_tuple=True)[0]
                    x = x.index_add(0, rows, self.down(h[parent[m]], band[m]))
                    h = self.n2(x)
        else:                                            # c2: ancestor attention
            anc = B["anc"]
            am = anc >= 0
            A = torch.zeros(*anc.shape, x.shape[1], device=x.device, dtype=x.dtype)
            A[am] = h[anc[am]]
            k, v = self.kv(A).chunk(2, -1)
            att = F.scaled_dot_product_attention(
                self.q(h).unsqueeze(1), k, v, attn_mask=am.unsqueeze(1))
            x = x + att.squeeze(1)
        return x


class StarModel(nn.Module):
    def __init__(self, arm, d=64, d_tok=256, K=2, d_ceil_mult=2, reach=1, pool="attn",
                 n_bands=NBANDS, n_slot=N_SLOT, n_nbr=None):
        super().__init__()
        self.arm, self.reach, self.pool = arm, reach, pool
        self.d_tok = d_tok
        self.n_bands, self.n_slot = n_bands, n_slot
        topo = {"rounds0": "none", "c0w7": "c0", "c0w7r2": "c0", "c2w7": "c2",
                "c0w2": "c0", "c0w0": "c0", "ceiling": "full"}[arm]
        wmode = {"c0w7": "per", "c0w7r2": "per", "c2w7": "per", "rounds0": "per",
                 "c0w2": "film", "c0w0": "shared", "ceiling": "film"}[arm]
        rounds = 2 if arm == "c0w7r2" else 1
        if arm == "ceiling":
            d = d * d_ceil_mult
        self.d = d
        self.use_band_emb = arm != "c0w0"
        self.val_in = nn.Linear(1, d)
        self.band_emb = nn.Embedding(n_bands, d)
        nf = d // 2
        self.register_buffer("freqs", torch.exp(
            torch.linspace(math.log(1.0), math.log(65536.0), nf)))
        self.pe_proj = nn.Linear(2 * nf, d)
        if topo == "full":
            self.full = nn.ModuleList(FullBlock(d) for _ in range(K * 2))
        else:
            self.blocks = nn.ModuleList(Block(d, topo, wmode, rounds, reach, n_bands, n_nbr)
                                        for _ in range(K))
        self.n_q = 4 if pool == "attn4" else 1
        self.pool_q = nn.Parameter(torch.randn(self.n_q, d))
        self.pool_k = nn.Linear(d, d)
        self.pool_v = nn.Linear(d, d_tok)
        self.cell_pe = nn.Linear(2 * nf, d_tok)
        self.tok_band_emb = nn.Embedding(n_bands + 1, d_tok)
        self.dec = nn.Sequential(nn.LayerNorm(d_tok), nn.Linear(d_tok, 1024),
                                 nn.GELU(), nn.Linear(1024, n_slot * 2))
        self.mask_tok = nn.Parameter(torch.zeros(d))
        self.val_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def pe(self, t):
        a = t[:, None] / self.freqs[None, :]
        return torch.cat([torch.sin(a), torch.cos(a)], -1)

    def encode(self, B, mask=None):
        v = self.val_in(B["val"][:, None])
        if mask is not None:
            v = torch.where(mask[:, None], self.mask_tok.expand_as(v), v)
        x = v + self.pe_proj(self.pe(B["tphys"]))
        if self.use_band_emb:
            x = x + self.band_emb(B["band"])
        if self.arm == "ceiling":
            x = self._full(x, B)
        else:
            nbr = self._nbrs(B)
            for blk in self.blocks:
                x = blk(x, B, nbr)
        return x

    def forward_masked(self, B, mask):
        """Masked-coefficient value prediction (Test A: routing, no bottleneck)."""
        x = self.encode(B, mask)
        return self.val_head(x[mask]).squeeze(-1)

    def forward(self, B):
        x = self.encode(B)
        # pool into cells: attn (segment softmax), attn4 (4 queries), or sum
        cell = B["cell"]
        nc = B["n_cells"]
        v = self.pool_v(x)
        if self.pool == "sum":
            tok = torch.zeros(nc, v.shape[1], device=x.device).index_add(0, cell, v)
            cnt = torch.zeros(nc, device=x.device).index_add(
                0, cell, torch.ones(len(cell), device=x.device))
            tok = tok / cnt.clamp_min(1.0).sqrt()[:, None]
        else:
            logits = self.pool_k(x) @ self.pool_q.T / math.sqrt(self.d)   # (N, n_q)
            mx = torch.full((nc, self.n_q), -1e30, device=x.device)
            mx = mx.index_reduce(0, cell, logits, "amax", include_self=True)
            w = torch.exp(logits - mx[cell])
            den = torch.zeros(nc, self.n_q, device=x.device).index_add(0, cell, w)
            w = w / den[cell].clamp_min(1e-30)
            vq = v.view(len(cell), self.n_q, -1) * w[..., None]
            tok = torch.zeros(nc, self.n_q, vq.shape[-1],
                              device=x.device).index_add(0, cell, vq).view(nc, -1)
        cell_t = torch.zeros(B["n_cells"], device=x.device).index_reduce(
            0, cell, B["tphys"], "amax", include_self=True)
        tok = tok + self.cell_pe(self.pe(cell_t))
        if "cell_band" in B:
            tok = tok + self.tok_band_emb(B["cell_band"])
        out = self.dec(tok).view(B["n_cells"], self.n_slot, 2)
        return out[..., 0], out[..., 1]

    def _nbrs(self, B):
        """Nearest active same-band neighbors out to self.reach on each side."""
        band = B["band"]
        key = (B["chunk"] << 36) | (band.to(torch.int64) << 32) | B["idx"]
        order = torch.argsort(key)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(len(order), device=order.device)
        idx_self = torch.arange(len(band), device=band.device)
        cols = []
        for k in range(1, self.reach + 1):
            prev = order[(inv - k).clamp_min(0)]
            nxt = order[(inv + k).clamp_max(len(order) - 1)]
            same_p = (band[prev] == band) & (B["chunk"][prev] == B["chunk"])
            same_n = (band[nxt] == band) & (B["chunk"][nxt] == B["chunk"])
            cols.append(torch.where(same_p, prev, idx_self))
            cols.append(torch.where(same_n, nxt, idx_self))
        return torch.stack(cols, 1)

    def _full(self, x, B):
        chunk = B["chunk"]
        n_ch = int(chunk.max()) + 1
        counts = torch.bincount(chunk, minlength=n_ch)
        mx = int(counts.max())
        pad = torch.zeros(n_ch, mx, x.shape[1], device=x.device)
        mask = torch.ones(n_ch, mx, dtype=torch.bool, device=x.device)
        pos = (torch.cumsum(torch.ones_like(chunk), 0) - 1) - torch.cumsum(
            F.pad(counts, (1, 0))[:-1], 0)[chunk]
        pad[chunk, pos] = x
        mask[chunk, pos] = False
        amask = ~mask[:, None, None, :]                  # (n_ch,1,1,mx) keep-mask
        for blk in self.full:
            pad = blk(pad, amask)
        return pad[chunk, pos]


# ------------------------------------------------------------- train/eval ---

def losses(occ_logit, val_pred, B):
    valid, occ, tgt = B["valid"], B["occ"], B["tgt"]
    bce = F.binary_cross_entropy_with_logits(occ_logit[valid], occ[valid])
    act = occ.bool() & valid
    mse = F.mse_loss(val_pred[act], tgt[act]) if act.any() else occ_logit.sum() * 0
    return bce, mse


@torch.no_grad()
def evaluate_masked(model, files, max_batches=20, sort_size=False, ratio=0.3, packer=None, nb=NBANDS):
    model.eval()
    se = np.zeros(nb); cnt = np.zeros(nb); var = np.zeros(nb)
    se_o = se_p = 0.0; n_o = n_p = 0
    g = torch.Generator(device=DEV).manual_seed(7)
    for bi, B in enumerate((packer or Packer)(files, seed=123, sort_size=sort_size)):
        if bi >= max_batches:
            break
        mask = torch.rand(len(B["band"]), generator=g, device=DEV) < ratio
        pred = model.forward_masked(B, mask)
        err = ((pred - B["target"][mask]) ** 2).cpu().numpy()
        band = B["band"][mask].cpu().numpy()
        orph = B["orphan"][mask].cpu().numpy()
        v = (B["target"][mask] ** 2).cpu().numpy()    # vs predict-zero baseline
        np.add.at(se, band, err); np.add.at(cnt, band, 1); np.add.at(var, band, v)
        se_o += err[orph].sum(); n_o += orph.sum()
        se_p += err[(~orph) & (band >= 2)].sum(); n_p += ((~orph) & (band >= 2)).sum()
    pb = se / np.maximum(cnt, 1)
    vb = var / np.maximum(cnt, 1)
    model.train()
    return dict(primary=float(pb.mean()),
                per_band={f"{b}": float(pb[b]) for b in range(nb)},
                var_baseline={f"{b}": float(vb[b]) for b in range(nb)},
                primary_rel=float((pb / np.maximum(vb, 1e-9)).mean()),
                orphan_mse=float(se_o / max(n_o, 1)),
                parented_mse=float(se_p / max(n_p, 1)))


@torch.no_grad()
def evaluate(model, files, max_batches=60, sort_size=False, packer=None, nb=NBANDS):
    model.eval()
    se = np.zeros(nb); cnt = np.zeros(nb); base = np.zeros(nb)
    se_o = np.zeros(nb); cnt_o = np.zeros(nb)
    se_p = np.zeros(nb); cnt_p = np.zeros(nb)
    bce_t = []
    for bi, B in enumerate((packer or Packer)(files, seed=123, sort_size=sort_size)):
        if bi >= max_batches:
            break
        ol, vp = model(B)
        bce, _ = losses(ol, vp, B)
        bce_t.append(float(bce))
        err = ((vp[B["cell"], B["slot"]] - B["target"]) ** 2).cpu().numpy()
        bl = ((B["val"] - B["target"]) ** 2).cpu().numpy()   # classical baseline
        band = B["band"].cpu().numpy()
        orph = B["orphan"].cpu().numpy()
        np.add.at(se, band, err); np.add.at(cnt, band, 1); np.add.at(base, band, bl)
        np.add.at(se_o, band[orph], err[orph]); np.add.at(cnt_o, band[orph], 1)
        np.add.at(se_p, band[~orph], err[~orph]); np.add.at(cnt_p, band[~orph], 1)
    per_band = se / np.maximum(cnt, 1)
    per_base = base / np.maximum(cnt, 1)
    res = dict(primary=float(per_band.mean()),
               per_band={f"{b}": float(per_band[b]) for b in range(nb)},
               baseline_classical={f"{b}": float(per_base[b]) for b in range(nb)},
               baseline_primary=float(per_base.mean()),
               orphan_mse=float(se_o.sum() / max(cnt_o.sum(), 1)),
               parented_mse=float(se_p[2:].sum() / max(cnt_p[2:].sum(), 1)),
               bce=float(np.mean(bce_t)))
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--budget", type=int, default=60000)
    ap.add_argument("--nev", type=int, default=0, help="cap event file list (avoid partial dumps)")
    ap.add_argument("--task", default="ae", choices=["ae", "masked"])
    ap.add_argument("--data", default="npz", choices=["npz", "onfly"],
                    help="onfly = noisy+clean from doraemon h5, GPU DWT (production transform)")
    ap.add_argument("--dtok", type=int, default=256)
    ap.add_argument("--reach", type=int, default=1)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--pool", default="attn", choices=["attn", "attn4", "sum"])
    ap.add_argument("--anchor", type=int, default=1024)
    ap.add_argument("--tokmode", default="cell", choices=["cell", "band", "hybrid"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.data == "onfly":
        import onfly_optical
        files = onfly_optical.event_keys()
        PK = onfly_optical.OnflyPacker
    else:
        files = event_files()
        PK = Packer
    if args.anchor != 1024:
        set_anchor(args.anchor)
        import star_model as _sm          # script-run: sync the imported copy
        if _sm.__name__ != __name__:
            _sm.set_anchor(args.anchor)
    if args.tokmode in ("band", "hybrid"):
        global N_SLOT
        N_SLOT = P_BAND if args.tokmode == "band" else 128
        fn = "assemble_band" if args.tokmode == "band" else "assemble_hybrid"
        Packer.assemble_fn = globals()[fn]
        import star_model as _sm2
        if _sm2.__name__ != __name__:
            _sm2.N_SLOT = N_SLOT
            _sm2.Packer.assemble_fn = getattr(_sm2, fn)
    if args.nev:
        files = files[:args.nev]
    test = [f for i, f in enumerate(files) if i % 3 == args.fold]
    train = [f for i, f in enumerate(files) if i % 3 != args.fold]
    if args.quick:
        train, test, args.steps = train[:6], test[:3], 50
    print(f"arm={args.arm} fold={args.fold} seed={args.seed} "
          f"train={len(train)}ev test={len(test)}ev steps={args.steps}")

    model = StarModel(args.arm, d=args.d, d_tok=args.dtok, K=args.blocks,
                      reach=args.reach, pool=args.pool, n_slot=N_SLOT).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"params: {nparam/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    sort_size = args.arm == "ceiling"
    step, t0 = 0, time.time()
    while step < args.steps:
        for B in PK(train, budget=args.budget, seed=args.seed + step,
                    sort_size=sort_size):
            if args.task == "masked":
                mask = torch.rand(len(B["band"]), device=DEV) < 0.3
                pred = model.forward_masked(B, mask)
                loss = mse = F.mse_loss(pred, B["target"][mask])
                bce = torch.zeros(())
            else:
                ol, vp = model(B)
                bce, mse = losses(ol, vp, B)
                loss = bce + mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            step += 1
            if step % 200 == 0:
                print(f"  step {step}: bce {bce:.4f} mse {mse:.4f} "
                      f"({(time.time()-t0)/step*1000:.0f} ms/step)", flush=True)
            if step >= args.steps:
                break
    if args.task == "masked":
        res = evaluate_masked(model, test, sort_size=sort_size, packer=PK)
    else:
        res = evaluate(model, test, sort_size=sort_size, packer=PK)
    res["task"] = args.task
    res["dtok"] = args.dtok
    res["data"] = args.data
    res["reach"] = args.reach
    res["blocks"] = args.blocks
    res["pool"] = args.pool
    res["anchor"] = args.anchor
    res["tokmode"] = args.tokmode
    res.update(arm=args.arm, fold=args.fold, seed=args.seed, steps=args.steps,
               lr=args.lr, params=nparam,
               ms_per_step=(time.time() - t0) / max(step, 1) * 1000)
    print(json.dumps(res, indent=1))
    out = os.path.join(HERE, "artifacts", "star_results.jsonl")
    with open(out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
