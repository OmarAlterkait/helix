"""``python -m helix.eval RUN_DIR --metrics recon,floor,uprobe,labels --out scores.json`` (see helix.eval)."""

from __future__ import annotations

import argparse
import json
import os


def main(argv=None):
    from helix.eval import METRICS, evaluate

    ap = argparse.ArgumentParser(prog="python -m helix.eval", description=__doc__)
    ap.add_argument("run", help="a pimm run directory (save_path), or one checkpoint inside it")
    ap.add_argument("--metrics", default="recon,floor,uprobe,labels", help=f"comma-separated, of {','.join(METRICS)}")
    ap.add_argument("--out", required=True, help="results JSON (written whole)")
    ap.add_argument("--weights", default="ema", choices=("ema", "last"), help="for a run directory")
    ap.add_argument("--max-events", type=int, default=None, help="only the first N events of the split")
    ap.add_argument("--denoiser", default=None, help="a denoiser run (configs/pimm/denoise.py; its best checkpoint) or one "
                    "checkpoint of it, for --metrics denoise")
    ap.add_argument("--seed", type=int, default=0, help="window placement and probe init")
    ap.add_argument("--boot", type=int, default=200, help="event-bootstrap resamples")
    ap.add_argument("--probe-seeds", type=int, default=2, help="fits per probe arm (seeds seed+k, same rows): "
                    "the mean and half-range over them are reported beside the bootstrap interval")
    ap.add_argument("--presence-q", type=float, default=25.0, help="denoise: the presence label, q > this (e-)")
    a = ap.parse_args(argv)

    res = evaluate(a.run, [m.strip() for m in a.metrics.split(",") if m.strip()], weights=a.weights,
                   max_events=a.max_events, denoiser=a.denoiser, seed=a.seed, n_boot=a.boot,
                   probe_seeds=a.probe_seeds, presence_q=a.presence_q)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=str)
    os.replace(tmp, a.out)
    print(f"[helix.eval] -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
