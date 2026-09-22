"""P25 — the corpus's own distribution of cells and coefficients per event.

Everything else in this study uses 24 events. The figure needs to say where real
data actually sits on the size axis, so this tokenises a spread sample across the
whole run (19,999 events) and records both counts.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import emit

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1200)
A = ap.parse_args()

from helix.data import CoeffTPCDataset
from helix.model.tokenize import CoeffTokenize
from helix.paths import root

ds = CoeffTPCDataset(data_root=str(root("HELIX_CORPUS")), dataset_name="sim_wire",
                     modalities=("coeff", "coeff_clean"), transform=None)
tok = CoeffTokenize(part="coeff", clean_part="coeff_clean",
                    cfg=dict(cell_t="grid_center"), fm_names=True)
N = len(ds)
idx = np.linspace(0, N - 1, min(A.n, N)).astype(int)      # spread over the whole run
print(f"corpus has {N} events; sampling {len(idx)} evenly", flush=True)

cells, coeff, raw = [], [], []
t0 = time.time()
for j, i in enumerate(idx):
    s = ds.get_data(int(i))
    raw.append(int(np.asarray(s["coeff"]["value"]).size))
    s = tok(s)
    cells.append(int(s["coeff"]["plane_id"].shape[0]))
    coeff.append(int(s["coeff"]["cell"].shape[0]))
    if j % 200 == 0:
        print(f"  {j}/{len(idx)}  {time.time()-t0:.0f}s", flush=True)

c = np.array(cells); k = np.array(coeff); r = np.array(raw)
def st(x):
    return dict(n=len(x), mean=float(x.mean()), std=float(x.std()),
                min=int(x.min()), max=int(x.max()),
                **{f"p{q}": float(np.percentile(x, q)) for q in (1, 5, 25, 50, 75, 95, 99)})
R = dict(n_events_sampled=len(idx), corpus_len=N,
         cells=st(c), coeff=st(k), coeff_raw_before_band_drop=st(r),
         occupancy=st(k / c), cells_list=c.tolist(), coeff_list=k.tolist())
for nm, v in (("cells", R["cells"]), ("coefficients", R["coeff"]),
              ("coeff per cell", R["occupancy"])):
    print(f"  {nm:16s} mean {v['mean']:10.1f}  p5 {v['p5']:10.1f}  p50 {v['p50']:10.1f} "
          f" p95 {v['p95']:10.1f}  min {v['min']}  max {v['max']}")
emit("p25_distribution", R)
