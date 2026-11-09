"""``python -m helix.eval RUN_DIR --metrics recon,floor,uprobe,labels --out scores.json`` (see helix.eval).

The floor's and the denoiser's per-window scores go beside it, ``scores.windows.npz``. Two runs
scored on the same events and seed share their windows (truth and ``[seed, event]`` place them),
so their difference is paired:

    python -m helix.eval compare A.windows.npz B.windows.npz [--metric floor] [--arm trained]
        [--arm-b ARM] [--boot 200] [--seed 0] --out diff.json

bootstraps ``A - B`` of every window number over the same event resamples, averaged over the
probe seeds both carry; ``views`` puts each per-view number's U / V / Y differences side by side,
so a pooled difference carried by one view shows (``same_sign``)."""

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
    # the window identity: these, plus each window's origin and flags wherever EITHER dump has them
    # (both must then: a far window's kind, plane and NaN energy do not say where it is)
    opt = [k for k in ("w0", "t0", "kept", "tok", "shtok") if f"{pre}meta/{k}" in za.files or f"{pre}meta/{k}" in zb.files]
    for k in keys + tuple(opt):
        f = f"{pre}meta/{k}"
        if f not in za.files or f not in zb.files:
            raise SystemExit(f"compare: only one run's windows record {k}: score both with the same helix.eval")
        x, y = za[f], zb[f]
        if x.shape != y.shape or not np.array_equal(x, y, equal_nan=(x.dtype.kind == "f")):
            raise SystemExit(f"compare: the two runs' windows differ ({k}): score them on the same events and seed")
    meta = [dict(zip(keys, (int(e), str(kd), int(g), float(E), str(c) or None)))
            for e, kd, g, E, c in zip(*(za[f"{pre}meta/{k}"] for k in keys))]
    for k in ("kept", "tok", "shtok"):
        if k in opt:
            for m, v in zip(meta, za[f"{pre}meta/{k}"]):
                m[k] = bool(v)
    arm_b = a.arm_b or a.arm
    name = lambda arm, k: arm if k == 0 else f"{arm}.s{k}"                  # noqa: E731 (probe.seed_name)
    seeds = [k for k in range(16) if f"{pre}score/{name(a.arm, k)}" in za.files
             and f"{pre}score/{name(arm_b, k)}" in zb.files]
    if not seeds:
        raise SystemExit(f"compare: no score {a.arm} in A and {arm_b} in B")
    pairs = [(window_stat(za[f"{pre}score/{name(a.arm, k)}"][:, None], meta),
              window_stat(zb[f"{pre}score/{name(arm_b, k)}"][:, None], meta)[0]) for k in seeds]
    n, counts = pairs[0][0][1], pairs[0][0][2]

    def mean_diff(m):
        ds = [_diff(sa(m), sb(m)) for (sa, _, _), sb in pairs]
        return {key: float(np.mean([d[key] for d in ds])) for key in ds[0]}

    diff = bootstrap(mean_diff, n, a.boot, a.seed)
    per_seed = [_diff(sa(np.ones(n, np.int64)), sb(np.ones(n, np.int64))) for (sa, _, _), sb in pairs]
    half = {key: round(float(np.ptp([d[key] for d in per_seed])) / 2, 4) for key in per_seed[0]}
    views = {}
    for key in diff:
        base = key[:-2]
        if key.endswith("_U") and all(base + s in diff for s in ("_V", "_Y")):
            v = [diff[base + s][0] for s in ("_U", "_V", "_Y")]
            views[base] = dict(U=v[0], V=v[1], Y=v[2], mean=round(float(np.mean(v)), 4),
                               same_sign=bool(all(x > 0 for x in v) or all(x < 0 for x in v)))
    out = dict(a=a.a, b=a.b, metric=a.metric, arm=a.arm, arm_b=arm_b, n_events=n, n_windows=counts,
               seeds=seeds, a_minus_b=diff, seed_half_range=half, views=views)
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
    ap.add_argument("--weights", default="last", choices=("last", "ema"), help="for a run directory")
    ap.add_argument("--max-events", type=int, default=None, help="only the first N events of the split")
    ap.add_argument("--denoiser", default=None, help="a denoiser run (configs/pimm/denoise.py; its last checkpoint) or one "
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
