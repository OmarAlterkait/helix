#!/usr/bin/env python
"""TPC twin of the tokenizer-only denoising AE (Phase 2).

Same substrate as star_model (imports its Blocks/StarModel/pack_event), TPC
geometry: bands [A4, D4, D3, D2] (D1 dropped), tree in TIME only, unit =
(plane gid, 8-wire block), patch token = 8 wires x 4 A4-ticks -> 256 slots
(8 x [4 A4 + 4 D4 + 8 D3 + 16 D2]). Tokens = OCCUPIED patches only (TPC's
A-band is thresholded, so empty patches exist — unlike optical).

Input is the PRODUCTION pipeline on the fly (measure_coeffs helpers):
  pimm extract -> Densify -> GPU coherent+intrinsic noise (AddNoise) -> Digitize
  -> coif3 L4 GPU DWT -> coeff-space smart removal (kgate=4)
  -> per-band-sigma threshold (kappa=1)   == noisy support + values
Clean targets: same event, densify+digitize only -> DWT (no removal, no
threshold), values at the noisy-support coordinates. Values are scaled by
2.6/sigma_band so star_model's asinh(v/2.6) == asinh(v/sigma_band).

Run from this folder:
  python star_tpc.py --arm c0w2 --task masked --steps 500
"""
import sys, os, json, time, argparse

sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, "/sdf/group/neutrino/omara/particle-imaging-models/libs/pimm-data/src")
sys.path.insert(0, "/sdf/group/neutrino/omara/helix")
import hdf5plugin  # noqa: F401
import numpy as np
import torch
import torch.nn as nn

import star_model as sm
from star_model import DEV

# ── TPC geometry ────────────────────────────────────────────────────────────
NB_T = 4                                   # A4, D4, D3, D2
LEV_T = np.array([4, 4, 3, 2])
LENS_T = np.array([271, 271, 542, 1084])   # at pad 4336
DELTA_T = np.array([-2.38, .62, .75, .50])
NTICKS = 4321
PW, PT = 8, 4                              # patch: wires x A4-ticks
N_CT = 68                                  # ceil(271/4) A4-tick patch groups
NW_MAX = 1969
BIGC = (-(-NW_MAX // PW)) * N_CT           # max cells per plane (cell-key stride)
WBAND = np.array([PT, PT, 2 * PT, 4 * PT])           # per-wire slots per band
SOFF_T = np.concatenate([[0], np.cumsum(WBAND * PW)])  # [0,32,64,128,256]
N_SLOT_T = 256
KAPPA_T = 1.0


def set_patch(pw):
    """Re-derive patch geometry for a different wire width (4/8/16)."""
    global PW, SOFF_T, N_SLOT_T, BIGC
    PW = pw
    SOFF_T = np.concatenate([[0], np.cumsum(WBAND * PW)])
    N_SLOT_T = int(SOFF_T[-1])
    BIGC = (-(-NW_MAX // PW)) * N_CT


def build_struct_tpc(band, idx, val, unit, gid, wire, val_clean, val_charge=None):
    """TPC event struct: idx = w_in*len_b + tau; parent same-wire tau>>1.
    val_charge (optional): deconvolution-probe target (true-charge coeff per row)."""
    o = np.lexsort((idx, band, unit))
    band, idx, val, unit = band[o], idx[o], val[o], unit[o]
    gid, wire, val_clean = gid[o], wire[o], val_clean[o]
    val_charge = val_charge[o] if val_charge is not None else None
    n_units = int(unit.max()) + 1 if len(unit) else 0
    N = len(band)

    grids = {}
    for b in range(NB_T):
        sz = NW_MAX * LENS_T[b]
        g = np.full(n_units * sz, -1, np.int64)
        r = np.nonzero(band == b)[0]
        g[unit[r] * sz + idx[r]] = r
        grids[b] = (g, sz)
    parent = np.full(N, -1, np.int64)
    anc = np.full((N, NB_T - 2), -1, np.int64)
    for b in range(2, NB_T):
        r = np.nonzero(band == b)[0]
        if not len(r):
            continue
        w_in = idx[r] // LENS_T[b]
        tau = idx[r] % LENS_T[b]
        for k, pb in enumerate(range(b - 1, 0, -1)):
            g, sz = grids[pb]
            m = g[unit[r] * sz + w_in * LENS_T[pb] + (tau >> (b - pb))]
            if pb == b - 1:
                parent[r] = m
            anc[r, k] = m
    cstart = np.searchsorted(unit, np.arange(n_units + 1))
    rel = cstart[unit]
    parent_rel = np.where(parent >= 0, parent - rel, -1)
    anc_rel = np.where(anc >= 0, anc - rel[:, None], -1)

    j = LEV_T[band]
    w_full = idx // LENS_T[band]
    tau = idx % LENS_T[band]
    c4 = tau >> (4 - j)                                  # A4-tick of coeff
    cell_local = (w_full // PW) * N_CT + (c4 >> 2)       # (wire-block, tick-group)
    tpos = tau % (PT << (4 - j))                         # time slot within patch
    slot = SOFF_T[band] + (w_full % PW) * WBAND[band] + tpos
    tphys = ((tau + DELTA_T[band]) * (1 << j)).astype(np.float32)
    return dict(band=band, idx=idx, val=val, chunk=unit, parent=parent_rel,
                anc=anc_rel, cell_local=cell_local, slot=slot, tphys=tphys,
                cstart=cstart, cl=np.full(n_units, NTICKS, np.int64),
                n_chunks=n_units, val_clean=val_clean, gid=gid, wire=wire,
                val_charge=val_charge)


def assemble_tpc(ev, unit_ids):
    """Occupied-cells-only batch for TPC."""
    cs = ev["cstart"]
    rows = np.concatenate([np.arange(cs[c], cs[c + 1]) for c in unit_ids])
    sizes = np.array([cs[c + 1] - cs[c] for c in unit_ids])
    boff = np.zeros(len(unit_ids) + 1, np.int64)
    boff[1:] = np.cumsum(sizes)
    bchunk = np.repeat(np.arange(len(unit_ids)), sizes)

    band, idx = ev["band"][rows], ev["idx"][rows]
    val, tphys = ev["val"][rows], ev["tphys"][rows]
    target = ev["val_clean"][rows]
    pr, ar = ev["parent"][rows], ev["anc"][rows]
    parent = np.where(pr >= 0, boff[bchunk] + pr, -1)
    anc = np.where(ar >= 0, boff[bchunk][:, None] + ar, -1)

    key = bchunk * BIGC + ev["cell_local"][rows]
    uniq, cell = np.unique(key, return_inverse=True)
    n_cells = len(uniq)
    slot = ev["slot"][rows]

    occ = np.zeros((n_cells, N_SLOT_T), bool)
    tgt = np.zeros((n_cells, N_SLOT_T), np.float32)
    occ[cell, slot] = True
    tgt[cell, slot] = np.arcsinh(target / sm.SIGMA)
    cell_within = (uniq % N_CT).astype(np.int64)         # A4-tick group 0..67
    valid = np.zeros((n_cells, N_SLOT_T), bool)
    for b in range(NB_T):
        jb = int(LEV_T[b])
        w = PT << (4 - jb)                               # time slots per wire
        tt = ((cell_within[:, None] * w + np.arange(w)[None, :]) << jb)
        v = tt < NTICKS                                  # same for all 8 wires
        for wi in range(PW):
            s0 = SOFF_T[b] + wi * WBAND[b]
            valid[:, s0:s0 + WBAND[b]] = v
    t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=DEV)
    return dict(
        band=t(band, torch.long), idx=t(idx, torch.long),
        val=t(np.arcsinh(val / sm.SIGMA), torch.float32),
        target=t(np.arcsinh(target / sm.SIGMA), torch.float32),
        tphys=t(tphys, torch.float32), parent=t(parent, torch.long),
        anc=t(anc, torch.long), cell=t(cell, torch.long),
        slot=t(slot, torch.long), chunk=t(bchunk, torch.long),
        n_cells=n_cells, occ=t(occ, torch.float32), tgt=t(tgt, torch.float32),
        valid=t(valid, torch.bool),
        orphan=t((pr < 0) & (band >= 2), torch.bool),
        gid=t(ev["gid"][rows], torch.long),
        wire=t(ev["wire"][rows].astype(np.float32), torch.float32))


# ── production pipeline loader ─────────────────────────────────────────────
_P = {}


def _pipeline():
    if _P:
        return _P
    from pimm_data import JAXTPCDataset
    from pimm_data.batch_transforms import move_to_device
    from pimm_data.detector_transforms import Densify, AddNoise, Digitize
    from helix.core import backend
    import measure_coeffs as M
    backend.set_backend("torch")
    geom, nts = M.load_geom()
    _P.update(
        M=M, geom=geom, move=move_to_device,            # noise self-seeds from batch['name']
        dwt=dict(ops=backend.ops("helix.core.wavelet_ops"), pad=(-nts) % 16),
        noisy=[Densify(geom), AddNoise(geom=geom, coherent=True, incoherent=True),
               Digitize(geom=geom)],                             # modality=None -> flat 'wire' in, 'dense' out
        clean=[Densify(geom), Digitize(geom=geom)],
        ds=JAXTPCDataset(data_root=M.DATA_ROOT, split=M.SPLIT,
                         modalities=("sensor",), dataset_name=M.DATASET_NAME))
    gids = sorted(geom.keys())
    _P["gids"] = gids
    # fixed normalization table (per gid, band): mean MAD sigma over 2 cal events
    tab = {g: np.zeros(NB_T) for g in gids}
    for ev in (0, 1):
        noisy = _bands_of(ev, _P["noisy"], smart=True, P=_P)
        for g in gids:
            for b in range(NB_T):
                tab[g][b] += float(noisy[g][b].abs().median() / 0.6745) / 2.0
    _P["sigma_tab"] = tab
    return _P


@torch.no_grad()
def _bands_of(ev_idx, stages, smart, P):
    M = P["M"]
    b = P["move"](M.build_batch(P["ds"], ev_idx, 1), "cuda")
    for st in stages:
        st(b)                                            # transforms self-seed from batch['name']
    out = {}
    for gid, g in b["dense"].items():                    # pimm-data: modality=None writes 'dense'
        x = torch.nn.functional.pad(g, (0, P["dwt"]["pad"]))
        bands = P["dwt"]["ops"]._wavedec(x[0], M.WAVELET, M.LEVEL)
        if smart:
            bands = M.smart_gate_bands(bands)
        out[gid] = bands
    return out


@torch.no_grad()
def prep_tpc_rows(ev_idx):
    """Sparse coeff rows AFTER the (expensive, GPU) pipeline+threshold — the thing
    worth caching. Returns dict of numpy arrays."""
    P = _pipeline()
    noisy = _bands_of(ev_idx, P["noisy"], smart=True, P=P)
    clean = _bands_of(ev_idx, P["clean"], smart=False, P=P)
    rows = {k: [] for k in ("band", "idx", "val", "unit", "gid", "wire", "val_clean")}
    for gid in P["gids"]:
        for b in range(NB_T):                            # D1 (list idx 4) dropped
            cn = noisy[gid][b]
            sig = float(cn.abs().median() / 0.6745)
            t = KAPPA_T * sig * float(np.sqrt(2.0 * np.log(max(cn.shape[-1], 2))))
            mask = cn.abs() >= t
            w, tau = mask.nonzero(as_tuple=True)
            scale = sm.SIGMA / max(P["sigma_tab"][gid][b], 1e-6)   # FIXED norm
            rows["band"].append(np.full(len(w), b, np.int64))
            rows["idx"].append((w * LENS_T[b] + tau).cpu().numpy().astype(np.int64))
            rows["unit"].append(np.full(len(w), gid, np.int64))
            rows["gid"].append(np.full(len(w), gid, np.int64))
            rows["wire"].append(w.cpu().numpy().astype(np.int64))
            rows["val"].append((cn[mask] * scale).cpu().numpy().astype(np.float32))
            rows["val_clean"].append((clean[gid][b][mask] * scale).cpu().numpy().astype(np.float32))
    return {k: np.concatenate(v) for k, v in rows.items()}


def rows_to_struct(cat):
    return build_struct_tpc(cat["band"], cat["idx"], cat["val"], cat["unit"],
                            cat["gid"], cat["wire"], cat["val_clean"], cat.get("val_charge"))


@torch.no_grad()
def prep_tpc(ev_idx):
    return rows_to_struct(prep_tpc_rows(ev_idx))


class TPCPacker:
    def __init__(self, events, budget=60000, seed=0, sort_size=False):
        self.events, self.budget = events, budget
        self.rng = np.random.default_rng(seed)
        self.sort_size = sort_size

    def __iter__(self):
        for ei in self.rng.permutation(len(self.events)):
            ev = prep_tpc(self.events[ei])
            yield from sm.pack_event(ev, self.rng, self.budget, self.sort_size,
                                     assemble_fn=assemble_tpc)


# ── model: StarModel + plane/wire embeddings ────────────────────────────────
class StarTPC(sm.StarModel):
    def __init__(self, arm, wire_mix=False, reach=1, **kw):
        n_nbr = 2 * reach + (2 if wire_mix else 0)
        super().__init__(arm, n_bands=NB_T, n_slot=N_SLOT_T, reach=reach,
                         n_nbr=n_nbr, **kw)
        self.wire_mix = wire_mix
        self.gid_emb = nn.Embedding(6, self.d)
        self.wire_proj = nn.Linear(2 * (self.d // 2), self.d)

    def _nbrs(self, B):
        """Time neighbors (per wire) + optional wire neighbors (per time)."""
        nbr = super()._nbrs(B)
        if not self.wire_mix:
            return nbr
        band = B["band"]
        lens = torch.as_tensor(LENS_T, device=band.device)[band]
        tau = B["idx"] % lens
        w_in = B["idx"] // lens
        # sort by (unit, band, tau, wire) -> adjacency = nearest active wire
        key = (B["chunk"] << 28) | (band.to(torch.int64) << 24) | (tau << 12) | w_in
        order = torch.argsort(key)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(len(order), device=order.device)
        idx_self = torch.arange(len(band), device=band.device)
        cols = []
        for s in (-1, 1):
            nb = order[(inv + s).clamp(0, len(order) - 1)]
            same = ((band[nb] == band) & (B["chunk"][nb] == B["chunk"])
                    & ((B["idx"][nb] % lens) == tau))
            cols.append(torch.where(same, nb, idx_self))
        return torch.cat([nbr, torch.stack(cols, 1)], 1)

    def encode(self, B, mask=None):
        v = self.val_in(B["val"][:, None])
        if mask is not None:
            v = torch.where(mask[:, None], self.mask_tok.expand_as(v), v)
        x = v + self.pe_proj(self.pe(B["tphys"]))
        if self.use_band_emb:
            x = x + self.band_emb(B["band"])
        x = x + self.gid_emb(B["gid"]) + self.wire_proj(self.pe(B["wire"] * 2.0))
        if self.arm == "ceiling":
            x = self._full(x, B)
        else:
            nbr = self._nbrs(B)
            for blk in self.blocks:
                x = blk(x, B, nbr)
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="c0w2")
    ap.add_argument("--task", default="masked", choices=["ae", "masked"])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--events", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--dtok", type=int, default=256)
    ap.add_argument("--pool", default="attn4")
    ap.add_argument("--budget", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--wiremix", action="store_true",
                    help="add wire-direction neighbor gathers (full plane)")
    ap.add_argument("--pw", type=int, default=8, help="patch width in wires")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.pw != 8:
        set_patch(args.pw)

    events = list(range(args.events))
    test = [e for i, e in enumerate(events) if i % 3 == args.fold]
    train = [e for i, e in enumerate(events) if i % 3 != args.fold]
    if args.quick:
        train, test, args.steps = train[:4], test[:2], 50

    import torch.nn.functional as F
    model = StarTPC(args.arm, d=args.d, d_tok=args.dtok, pool=args.pool,
                    wire_mix=args.wiremix).to(DEV)
    print(f"arm={args.arm} task={args.task} train={len(train)} test={len(test)} "
          f"steps={args.steps} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    step, t0 = 0, time.time()
    while step < args.steps:
        for B in TPCPacker(train, budget=args.budget, seed=args.seed + step):
            if args.task == "masked":
                mask = torch.rand(len(B["band"]), device=DEV) < 0.3
                pred = model.forward_masked(B, mask)
                loss = mse = F.mse_loss(pred, B["target"][mask])
                bce = torch.zeros(())
            else:
                ol, vp = model(B)
                bce, mse = sm.losses(ol, vp, B)
                loss = bce + mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            step += 1
            if step % 100 == 0:
                print(f"  step {step}: bce {float(bce):.4f} mse {float(mse):.4f} "
                      f"({(time.time()-t0)/step*1000:.0f} ms/step)", flush=True)
            if step >= args.steps:
                break
    if args.task == "masked":
        res = sm.evaluate_masked(model, test, packer=TPCPacker, nb=NB_T)
    else:
        res = sm.evaluate(model, test, packer=TPCPacker, nb=NB_T)
    res.update(arm=args.arm, task=args.task, modality="tpc", steps=args.steps,
               seed=args.seed, fold=args.fold, pool=args.pool, dtok=args.dtok,
               wiremix=args.wiremix, pw=args.pw,
               ms_per_step=(time.time() - t0) / max(step, 1) * 1000)
    print(json.dumps(res, indent=1))
    with open(os.path.join(sm.HERE, "artifacts", "star_results.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
