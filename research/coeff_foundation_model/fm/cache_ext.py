"""Extend the wire cache with events from ANOTHER source run (for >20k / diversity).

Same pipeline as cache.py, but --split selects a different run and --out_start
offsets the output indices so events land as ev_{out_start+i}.npz in the shared cache.

Run: python cache_ext.py --split run_0027575767 --n 20000 --out_start 20000
"""
import sys, os, time, argparse, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/sdf/group/neutrino/omara/helix/.pylibs",
          "/sdf/group/neutrino/omara/particle-imaging-models/libs/pimm-data/src",
          "/sdf/group/neutrino/omara/helix", os.path.dirname(_HERE), _HERE):
    sys.path.insert(0, p)
import measure_coeffs as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--start", type=int, default=0, help="source event index to start at")
    ap.add_argument("--out_start", type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    M.SPLIT = args.split                              # override BEFORE the pipeline builds its dataset
    import data as D
    import star_tpc as stp
    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.abspath(os.path.join(here, "..", "artifacts", "fm_cache_tpc"))
    os.makedirs(out, exist_ok=True)
    D.init_pipeline()
    n_avail = len(stp._pipeline()["ds"])
    n = min(args.n, n_avail)
    print(f"split={args.split} avail={n_avail} extracting {n} -> ev_{args.out_start:05d}..{args.out_start+n-1:05d}", flush=True)
    t0 = time.time(); done = 0; skipped = 0
    for i in range(args.start, n):
        fp = os.path.join(out, f"ev_{args.out_start + i:05d}.npz")
        if os.path.exists(fp):
            done += 1; continue
        try:
            cat = stp.prep_tpc_rows(i)
            np.savez_compressed(fp,
                band=cat["band"].astype(np.int8), idx=cat["idx"].astype(np.int32),
                gid=cat["gid"].astype(np.int8), wire=cat["wire"].astype(np.int32),
                val=cat["val"].astype(np.float32), val_clean=cat["val_clean"].astype(np.float32))
        except Exception as e:
            skipped += 1; print(f"  SKIP event {i}: {type(e).__name__}: {e}", flush=True); continue
        done += 1
        if done % 200 == 0:
            dt = time.time() - t0
            print(f"  {done}/{n}  ({dt/done*1000:.0f} ms/ev, ETA {dt/done*(n-done)/60:.0f}min)", flush=True)
    print(f"done {done} in {(time.time()-t0)/60:.0f}min", flush=True)


if __name__ == "__main__":
    main()
