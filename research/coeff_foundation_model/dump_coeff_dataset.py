#!/usr/bin/env python
"""Dump a clean-support coefficient dataset for M3 (one npz per event).

Each event npz mirrors typical_event_coeffs_doraemon.npz: chunk_id, pmt_id,
label, pe, band_id, idx, value (active coeffs only, nominal sigma=2.6 support)
+ chunk_len, t0_ns. Events spread across files (2 per file by default).

Run from this folder:
  python dump_coeff_dataset.py --events 300 --out artifacts/coeff_dataset
"""
import sys, os, argparse

sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, "/sdf/group/neutrino/omara/helix")
import hdf5plugin  # noqa: F401
import numpy as np

from measure_coeffs_optical import process_chunk, NB
import doraemon_optical as dop

SIGMA_NOM, KAPPA = 2.6, 1.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=300)
    ap.add_argument("--out", default="artifacts/coeff_dataset")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, args.out)
    os.makedirs(out, exist_ok=True)

    n = 0
    for path, ek in dop.iter_events(args.events):
        tag = f"{os.path.basename(path).split('_')[-1].split('.')[0]}_{ek}"
        fp = os.path.join(out, f"{tag}.npz")
        if os.path.exists(fp):
            n += 1
            continue
        ec = dop.read_event_chunks(path, ek)
        rows = {k: [] for k in ("chunk_id", "pmt_id", "label", "pe", "band_id", "idx", "value")}
        for ci, c in enumerate(ec.chunks):
            coeffs, acts, valids, _ = process_chunk(c, SIGMA_NOM, KAPPA)
            for i in range(NB):
                a = acts[i] & valids[i]
                ii = np.nonzero(a)[0]
                rows["chunk_id"].append(np.full(len(ii), ci, np.int32))
                rows["pmt_id"].append(np.full(len(ii), ec.pmt_id[ci], np.int16))
                rows["label"].append(np.full(len(ii), ec.label[ci], np.int16))
                rows["pe"].append(np.full(len(ii), ec.pe[ci], np.int32))
                rows["band_id"].append(np.full(len(ii), i, np.int8))
                rows["idx"].append(ii.astype(np.int32))
                rows["value"].append(coeffs[i][ii].astype(np.float32))
        npz = {k: np.concatenate(v) for k, v in rows.items()}
        npz["chunk_len"] = ec.lengths
        npz["t0_ns"] = ec.t0_ns
        np.savez_compressed(fp, **npz)
        n += 1
        if n % 20 == 0:
            print(f"  dumped {n}/{args.events}", flush=True)
    print(f"done: {n} events -> {out}")


if __name__ == "__main__":
    main()
