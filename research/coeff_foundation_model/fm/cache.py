"""Extract-once cache: run the (expensive GPU) production pipeline + threshold for
N events, save the SPARSE coeff rows (band, idx, val_noisy, val_clean, gid, wire)
per event. ~few MB/event. Token-batch assembly (cheap numpy) is done at load time.
THROWAWAY framework — replaced by pimm-data later.

Run:  python cache.py --events 1000 --out artifacts/fm_cache_tpc
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import data as D            # sets up sys.path + pipeline
import star_tpc as stp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, "..", "artifacts", "fm_cache_tpc")
    out = os.path.abspath(out); os.makedirs(out, exist_ok=True)
    D.init_pipeline()

    t0 = time.time(); done = 0
    for i in range(args.events):
        fp = os.path.join(out, f"ev_{i:05d}.npz")
        if os.path.exists(fp):
            done += 1; continue
        cat = stp.prep_tpc_rows(i)              # GPU pipeline + threshold (cached away)
        # compress + trim dtypes; unit==gid so store gid only
        np.savez_compressed(fp,
            band=cat["band"].astype(np.int8), idx=cat["idx"].astype(np.int32),
            gid=cat["gid"].astype(np.int8), wire=cat["wire"].astype(np.int32),
            val=cat["val"].astype(np.float32), val_clean=cat["val_clean"].astype(np.float32))
        done += 1
        if done % 50 == 0:
            dt = time.time() - t0
            print(f"  cached {done}/{args.events}  ({dt/done*1000:.0f} ms/ev, ETA {dt/done*(args.events-done):.0f}s)",
                  flush=True)
    sz = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out) if f.endswith('.npz'))
    print(f"done: {done} events -> {out}  ({sz/1e9:.2f} GB, {sz/1e6/max(done,1):.1f} MB/ev)")


if __name__ == "__main__":
    main()
