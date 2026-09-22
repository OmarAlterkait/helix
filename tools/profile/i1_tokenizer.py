"""I1 — the tokenizer diagnostics the field review says are load-bearing.

Three measurements, no training, no checkpoint:

  1. CELL-SET OVERLAP between unrelated events. If the (plane, band, wire-block,
     tick-block) sets largely coincide, the tokenizer is encoding a fixed
     geometric template, the effective independent-token count is far below the
     4.85e9 nominal, and the "we have plenty of data" reading of the scaling
     regime collapses. docs/REVIEW_FIELD.md §5.
  2. WHICH BAND SUPPLIES THE TOKENS, and whether that band's count is flat
     across events — the malign version of (1).
  3. THE EFFECTIVE MASK RATIO. The sparse-MAE literature defines the ratio over
     OCCUPIED units; helix defines it over cells, and 93% of each cell's slots
     are structural zeros. A masked cell with no active slot is a free
     prediction. docs/REVIEW_FIELD.md §2.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prof"))
from common import emit

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--pairs", type=int, default=2000)
A = ap.parse_args()

from helix.data import CoeffTPCDataset
from helix.model.tokenize import CoeffTokenize, unpack_cell_key
from helix.paths import root

ds = CoeffTPCDataset(data_root=str(root("HELIX_CORPUS")), dataset_name="sim_wire",
                     modalities=("coeff", "coeff_clean"), transform=None)
tok = CoeffTokenize(part="coeff", clean_part="coeff_clean",
                    cfg=dict(cell_t="grid_center"), fm_names=True)
idx = np.linspace(0, len(ds) - 1, min(A.n, len(ds))).astype(int)
print(f"corpus {len(ds)} events; sampling {len(idx)}", flush=True)

keys, bands, occs, valids, ncell = [], [], [], [], []
t0 = time.time()
for j, i in enumerate(idx):
    s = tok(ds.get_data(int(i)))["coeff"]
    keys.append(np.asarray(s["cell_key"]))
    bands.append(np.asarray(s["band_id"]))
    occs.append(np.asarray(s["occ"]))
    valids.append(np.asarray(s["valid"]))
    ncell.append(int(s["plane_id"].shape[0]))
    if j % 50 == 0:
        print(f"  {j}/{len(idx)} {time.time()-t0:.0f}s", flush=True)

R = {"n_events": len(idx)}

# ---- 1. cell-set overlap -----------------------------------------------
rng = np.random.default_rng(0)
sets = [set(k.tolist()) for k in keys]
jac, inter_frac = [], []
for _ in range(A.pairs):
    a, b = rng.choice(len(sets), 2, replace=False)
    A_, B_ = sets[a], sets[b]
    n_i = len(A_ & B_)
    jac.append(n_i / len(A_ | B_))
    inter_frac.append(n_i / min(len(A_), len(B_)))
jac = np.array(jac); inter_frac = np.array(inter_frac)
# a union over many events: how much of one event is "always there"?
universe = set().union(*sets[:100])
always = set(sets[0])
for s_ in sets[1:100]:
    always &= s_
R["cell_overlap"] = dict(
    jaccard_mean=float(jac.mean()), jaccard_p5=float(np.percentile(jac, 5)),
    jaccard_p95=float(np.percentile(jac, 95)),
    shared_frac_of_smaller_mean=float(inter_frac.mean()),
    union_over_100=len(universe), intersection_over_100=len(always),
    mean_cells=float(np.mean(ncell)),
    always_present_frac_of_event=len(always) / float(np.mean(ncell)))
print("\n== cell-set overlap between unrelated events ==")
for k, v in R["cell_overlap"].items():
    print(f"  {k:34s} {v}")
print("  READ: Jaccard > 0.7 => the tokenizer is encoding a fixed template.")

# ---- 2. per-band token supply -------------------------------------------
nb = int(max(b.max() for b in bands)) + 1
per = np.stack([np.bincount(b, minlength=nb) for b in bands])
R["per_band_tokens"] = dict(
    mean=per.mean(0).tolist(), std=per.std(0).tolist(),
    cv=(per.std(0) / np.maximum(per.mean(0), 1)).tolist(),
    share=(per.mean(0) / per.mean(0).sum()).tolist())
print("\n== tokens supplied per band ==")
print(f"  {'band':>5s} {'mean':>9s} {'std':>8s} {'CV':>7s} {'share':>7s}")
for b in range(nb):
    print(f"  {b:5d} {per[:,b].mean():9.1f} {per[:,b].std():8.1f} "
          f"{per[:,b].std()/max(per[:,b].mean(),1):7.3f} "
          f"{per[:,b].mean()/per.mean(0).sum():7.3f}")
print("  READ: a high-share band with CV ~ 0 is an event-independent floor.")

# ---- 3. effective mask ratio --------------------------------------------
ratios = []
for occ, val in zip(occs, valids):
    act_per_cell = (occ.astype(bool) & val).sum(1)
    ratios.append(dict(
        cells=len(act_per_cell),
        empty_cells=float((act_per_cell == 0).mean()),
        slot_occupancy=float((occ.astype(bool) & val).mean()),
        active_per_cell=float(act_per_cell.mean())))
R["mask_units"] = dict(
    empty_cell_frac=float(np.mean([r["empty_cells"] for r in ratios])),
    slot_occupancy=float(np.mean([r["slot_occupancy"] for r in ratios])),
    active_per_cell=float(np.mean([r["active_per_cell"] for r in ratios])))
# what "mask 0.75 of cells" means in occupied-unit terms
R["mask_units"]["masked_cells_at_0.75"] = 0.75
R["mask_units"]["masked_active_slots_at_0.75"] = 0.75      # uniform over cells
R["mask_units"]["free_predictions_frac"] = R["mask_units"]["empty_cell_frac"]
print("\n== mask units ==")
for k, v in R["mask_units"].items():
    print(f"  {k:34s} {v}")
print("  READ: every cell holds >=1 coefficient by construction, so a masked "
      "cell is never a free prediction at the CELL level; the question is "
      "whether the SLOT-level target (128 values, ~8.8 real) is.")

emit("i1_tokenizer", R)
