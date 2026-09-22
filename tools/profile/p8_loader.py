"""P8 — the real loop: does the data path keep the GPU fed, and at what cost?

Reader -> CoeffTokenize -> collate -> H2D -> step, with the production
num_worker, measuring how long the loop BLOCKS on data.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(__file__))
from common import build, emit, peak_mem, to_device

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--workers", type=str, default="0,2,4,8,16")
A = ap.parse_args()
dev = "cuda"
torch.set_float32_matmul_precision("high")

from helix.data import CoeffTPCDataset as DS
from helix.model.tokenize import CoeffTokenize
from helix.paths import root

CORPUS = str(root("HELIX_CORPUS"))
tokz = CoeffTokenize(part="coeff", clean_part="coeff_clean",
                     cfg=dict(cell_t="grid_center"), fm_names=True)


class Wrapped(torch.utils.data.Dataset):
    def __init__(self, keep=None):
        self.ds = DS(data_root=CORPUS, dataset_name="sim_wire",
                     modalities=("coeff", "coeff_clean"), transform=None)
        self.keep = keep

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        s = tokz(self.ds.get_data(i))["coeff"]
        out = {}
        for k, v in s.items():
            if k.startswith("_") or not isinstance(v, np.ndarray):
                continue
            if self.keep is not None and k not in self.keep:
                continue
            out[k] = torch.from_numpy(np.ascontiguousarray(v))
        return out


def collate(batch):                       # pimm's collate at B=1 is a passthrough cat
    return {k: torch.cat([b[k] for b in batch]) if batch[0][k].dim() else batch[0][k]
            for k in batch[0]} if len(batch) > 1 else batch[0]


MODEL_KEYS = ("occ", "inp", "tgt", "valid", "band_id", "plane_id",
              "t_phys", "wire_pos", "wirefeat")

model = build(device=dev)
opt = torch.optim.AdamW(model.param_groups(1.1e-3, weight_decay=0.05), betas=(0.9, 0.95))
R = {"steps": A.steps, "rows": {}}
print("corpus", CORPUS, "len", len(Wrapped()))


def run(nw, keep, pin, tag):
    ds = Wrapped(keep=keep)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=nw,
                    collate_fn=collate, pin_memory=pin,
                    persistent_workers=bool(nw), prefetch_factor=(2 if nw else None))
    it = iter(dl)
    wait = comp = h2d = 0.0
    bytes_ = 0
    torch.cuda.synchronize()
    t_all = time.perf_counter()
    for s in range(A.steps):
        t0 = time.perf_counter()
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl); b = next(it)
        t1 = time.perf_counter()
        bytes_ += sum(v.numel() * v.element_size() for v in b.values() if torch.is_tensor(v))
        B = to_device(b, dev, non_blocking=pin)
        B["n_cells"] = B["plane_id"].shape[0]
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            o = model(B)
        o["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        if s >= 5:                                     # skip warmup / worker spin-up
            wait += t1 - t0; h2d += t2 - t1; comp += t3 - t2
    n = A.steps - 5
    tot = time.perf_counter() - t_all
    row = dict(num_workers=nw, pin=pin, keys=len(keep) if keep else "all",
               wait_ms=wait/n*1e3, h2d_ms=h2d/n*1e3, compute_ms=comp/n*1e3,
               step_ms=(wait+h2d+comp)/n*1e3, MiB_per_event=bytes_/A.steps/2**20,
               ev_per_s=n/(wait+h2d+comp))
    R["rows"][tag] = row
    print(f"  {tag:26s} wait={row['wait_ms']:7.1f} h2d={row['h2d_ms']:6.1f} "
          f"gpu={row['compute_ms']:7.1f} step={row['step_ms']:7.1f} ms  "
          f"{row['MiB_per_event']:5.1f} MiB/ev  {row['ev_per_s']:.2f} ev/s", flush=True)
    del dl, it
    gc.collect()


for nw in [int(x) for x in A.workers.split(",")]:
    run(nw, None, True, f"nw={nw} allkeys pin")
run(4, MODEL_KEYS, True, "nw=4 modelkeys pin")
run(4, MODEL_KEYS, False, "nw=4 modelkeys nopin")
emit("p8_loader", R)
