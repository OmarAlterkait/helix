"""``python -m helix.eval RUN_DIR --metrics recon,floor,uprobe,labels --out scores.json`` (see helix.eval).

The floor's and the denoiser's per-window scores go beside it, ``scores.windows.npz``. Two runs
scored on the same events and seed share their windows (truth and ``[seed, event]`` place them),
so their difference is paired:

    python -m helix.eval compare A.windows.npz B.windows.npz [--metric floor] [--arm trained]
        [--arm-b ARM] [--boot 200] [--seed 0] --out diff.json

bootstraps ``A - B`` of every window number over the same event resamples."""

from __future__ import annotations

import argparse
import json
import os


def compare(argv):
    """``A - B`` of every window number of two runs scored on identical windows, each
    ``[point, lo, hi]`` over shared event resamples: far tighter than two separate intervals."""
    import numpy as np

    from helix.eval.events import bootstrap, window_stat
    from helix.eval.probe import _diff

    ap = argparse.ArgumentParser(prog="python -m helix.eval compare", description=compare.__doc__)
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--metric", default="floor", help="floor or denoise")
    ap.add_argument("--arm", default="trained", help="the score of A (floor: trained, raw, amplitude, presence; "
                    "denoise: charge, presence)")
    ap.add_argument("--arm-b", default=None, help="the score of B (default: --arm)")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    za, zb = np.load(a.a, allow_pickle=False), np.load(a.b, allow_pickle=False)
    pre = f"{a.metric}/"
    keys = ("ev", "kind", "g", "E", "ctx")
    for k in keys:
        x, y = za[f"{pre}meta/{k}"], zb[f"{pre}meta/{k}"]
        if x.shape != y.shape or not np.array_equal(x, y, equal_nan=(x.dtype.kind == "f")):
            raise SystemExit(f"compare: the two runs' windows differ ({k}): score them on the same events and seed")
    meta = [dict(zip(keys, (int(e), str(kd), int(g), float(E), str(c) or None)))
            for e, kd, g, E, c in zip(*(za[f"{pre}meta/{k}"] for k in keys))]
    sa, n, counts = window_stat(za[f"{pre}score/{a.arm}"][:, None], meta)
    sb = window_stat(zb[f"{pre}score/{a.arm_b or a.arm}"][:, None], meta)[0]
    out = dict(a=a.a, b=a.b, metric=a.metric, arm=a.arm, arm_b=a.arm_b or a.arm, n_events=n, n_windows=counts,
               a_minus_b=bootstrap(lambda m: _diff(sa(m), sb(m)), n, a.boot, a.seed))
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
    print(f"[helix.eval compare] -> {a.out}", flush=True)
    return 0


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["compare"]:
        return compare(argv[1:])
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
    ap.add_argument("--fit-windows", default="none", choices=("none", "all", "neg"), help="floor: also fit the probes "
                    "on the fit events' own window cells -- all (iso, bg, near, extra clear) or neg (bg, near) -- "
                    "not only the map rows")
    a = ap.parse_args(argv)

    res = evaluate(a.run, [m.strip() for m in a.metrics.split(",") if m.strip()], weights=a.weights,
                   max_events=a.max_events, denoiser=a.denoiser, seed=a.seed, n_boot=a.boot,
                   probe_seeds=a.probe_seeds, presence_q=a.presence_q, fit_windows=a.fit_windows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    win = {f"{name}/{k}": v for name, r in (res.get("metrics") or {}).items()
           if isinstance(r, dict) and "_windows" in r for k, v in r.pop("_windows").items()}
    if win:
        import numpy as np
        np.savez_compressed(os.path.splitext(a.out)[0] + ".windows.npz", **win)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=str)
    os.replace(tmp, a.out)
    print(f"[helix.eval] -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
