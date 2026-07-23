"""THROWAWAY data shim (to be replaced by proper pimm-data integration).

Reuses the existing production on-the-fly TPC pipeline (star_tpc / vit_tpc) to
produce per-band 2D patch token batches, and re-keys the fields to the names the
FM model expects. One event = all 6 planes = one cross-plane token set.

Field contract delivered to the model (all torch, on DEV):
  inp (n_cells, n_slot)  noisy asinh values per slot
  occ (n_cells, n_slot)  occupancy bits
  valid (n_cells, n_slot)
  target (n_rows)        CLEAN asinh value at each active (cell,slot)
  cell (n_rows), slot (n_rows)
  band_id (n_cells), plane_id (n_cells)
  t_phys (n_cells)       physical-time of the patch  (RoPE)
  wire_pos (n_cells)     physical wire position of the patch  (RoPE)
  wirefeat (n_cells, k)  per-patch wire features for FiLM  (here: [wire_pos_norm])
  n_cells, n_slot
"""
import sys
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, "/sdf/group/neutrino/omara/particle-imaging-models/libs/pimm-data/src")
sys.path.insert(0, "/sdf/group/neutrino/omara/helix")
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/research/coeff_foundation_model")
import hdf5plugin  # noqa
import numpy as np
import torch

import star_tpc as stp
import vit_tpc as vtp
import baseline_tpc as bl
from star_model import DEV

N_SLOT = vtp.N_SLOT      # 128
N_BAND = vtp.NB_T        # 4 (A4,D4,D3,D2)
N_PLANE = 6
NW_MAX = 1969


def init_pipeline():
    P = stp._pipeline()
    P["nw"] = np.array([P["geom"][g]["n_wires"] for g in range(6)], np.int64)
    return P


def init_pipeline_cpu():
    """Lightweight worker init: populate ONLY geom/nw (per-plane wire counts) so
    star_tpc._pipeline() short-circuits. The cached assembly path needs nothing
    else, so workers never trigger the GPU forward-build (no CUDA context/contention)."""
    if stp._P.get("nw") is not None:
        return
    import measure_coeffs as M
    geom, _ = M.load_geom()
    stp._P["geom"] = geom
    stp._P["nw"] = np.array([geom[g]["n_wires"] for g in range(6)], np.int64)


def _worker_init(worker_id):
    """DataLoader worker_init_fn (module-level so it pickles under spawn)."""
    init_pipeline_cpu()
    import os as _os                                       # spawn re-imports -> patch globals reset to defaults;
    _pw, _pt = _os.environ.get("FM_PW"), _os.environ.get("FM_PT")   # restore from env set by the trainer
    if _pw and _pt:
        vtp.PW, vtp.PT, vtp.N_SLOT = int(_pw), int(_pt), int(_pw) * int(_pt)


def _to_fm(B):
    wire_pos = B["cell_wire"].float()
    out = dict(
        inp=B["inp"], occ=B["occ"], valid=B["valid"], target=B["target"],
        cell=B["cell"], slot=B["slot"],
        band_id=B["cell_band"], plane_id=B["cell_gid"],
        t_phys=B["cell_t"], wire_pos=wire_pos,
        wirefeat=(wire_pos / NW_MAX)[:, None],
        n_cells=B["n_cells"], n_slot=N_SLOT)
    if "target_charge" in B:
        out["target_charge"] = B["target_charge"]        # deconv-probe target (raw charge/row)
    if "tgt" in B:
        out["inp_clean"] = B["tgt"]                       # dense CLEAN wire values (oracle input)
        out["tgt"] = B["tgt"]                             # dense clean target (n_cells,n_slot) for the fused loss
    return out


def get_event(idx, cap=30000):
    """One event -> FM token batch via the on-the-fly pipeline (no cache)."""
    return _to_fm(bl.event_batch(idx, cap=cap))


def get_cached(path, cap=30000, device=DEV):
    """One event -> FM token batch from a cached sparse-coeff npz (no GPU pipeline).

    device="cpu" => host tensors (assembly is pure numpy; used by DataLoader workers,
    then moved to GPU in the main process). Default DEV preserves old callers.
    """
    d = np.load(path)
    cat = dict(band=d["band"].astype(np.int64), idx=d["idx"].astype(np.int64),
               gid=d["gid"].astype(np.int64), wire=d["wire"].astype(np.int64),
               val=d["val"], val_clean=d["val_clean"])
    cat["unit"] = cat["gid"]                            # unit == gid
    ev = stp.rows_to_struct(cat)                       # cheap numpy
    B = vtp.assemble_tpc_band(ev, list(range(ev["n_chunks"])), device=device)
    return _to_fm(B)                                   # full token count (~25-37k, fits)


def get_cached_charge(wire_path, charge_path, device=DEV):
    """Wire tokens + deconvolution-probe target (true-charge coeff/row) aligned to them."""
    d = np.load(wire_path)
    cat = dict(band=d["band"].astype(np.int64), idx=d["idx"].astype(np.int64),
               gid=d["gid"].astype(np.int64), wire=d["wire"].astype(np.int64),
               val=d["val"], val_clean=d["val_clean"],
               val_charge=np.load(charge_path)["val_charge"])
    cat["unit"] = cat["gid"]
    ev = stp.rows_to_struct(cat)
    B = vtp.assemble_tpc_band(ev, list(range(ev["n_chunks"])), device=device)
    return _to_fm(B)


class CachedTPC(torch.utils.data.Dataset):
    """Map-style dataset: one cached npz -> one FM token batch (CPU tensors).

    All work is numpy + a host-tensor cast, so __getitem__ runs entirely on CPU
    and parallelizes across DataLoader workers, overlapping the ~215ms per-event
    assembly with GPU compute. Use batch_size=None (each item is already one full
    per-event token set) and move to GPU in the training loop.
    """
    def __init__(self, files):
        self.files = list(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        # The cache contains a few corrupt/truncated npz (BadZipFile) and occasional missing
        # files. At 20k events they happen to fall outside the id range; at 40k/80k they don't
        # and a single bad file kills the whole DDP run. Skip forward to the next readable
        # event instead (keeps epoch length fixed; logs each bad file once per worker).
        n = len(self.files)
        for k in range(64):
            p = self.files[(i + k) % n]
            try:
                return get_cached(p, device="cpu")
            except Exception as e:
                if p not in _BAD_FILES:
                    _BAD_FILES.add(p)
                    print(f"[data] skipping unreadable event {p.rsplit('/', 1)[-1]}: "
                          f"{type(e).__name__}", flush=True)
        raise RuntimeError(f"CachedTPC: 64 consecutive unreadable events from index {i}")


_BAD_FILES = set()


class CachedTPCCharge(torch.utils.data.Dataset):
    """Like CachedTPC but yields wire tokens + deconv charge target (for deconv fine-tuning).
    items = list of (wire_npz_path, charge_npz_path). Same async-worker pattern (CPU assembly)."""
    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return get_cached_charge(*self.items[i], device="cpu")
