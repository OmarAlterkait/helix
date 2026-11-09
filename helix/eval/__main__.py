"""``python -m helix.eval DENOISER_RUN --out e.json`` (see helix.eval).

The per-window scores go beside it, ``e.windows.npz``: the fine-tuned ``presence`` (log-odds)
and the fit-free ``amplitude``, on the same windows. Two runs scored on the same events and seed
share their windows (truth and ``[seed, event]`` place them), so their difference is paired:

    python -m helix.eval compare --a A.windows.npz [A_s1.windows.npz ...] \\
        [--b B.windows.npz [B_s1.windows.npz ...]] [--arm presence] [--arm-b ARM] \\
        [--boot 200] [--seed 0] --out diff.json

bootstraps ``A - B`` of every window number over the same event resamples. Several files on a
side are the arm's training seeds: they pair seed by seed (one file on a side pairs with every
file on the other), the differences are averaged over the pairs and ``seed_half_range`` is their
(max - min) / 2. Without ``--b`` both scores come from ``--a`` (``--arm presence --arm-b
amplitude``: what the fine-tune adds over the corpus gate). ``views`` puts each per-view
number's U / V / Y differences side by side, so a pooled difference carried by one view shows
(``same_sign``)."""

from __future__ import annotations

import argparse
import json
import os

#: The window metadata that must be present in every dump (the rest of WINDOW_META too, wherever
#: any dump records it).
_CORE = ("ev", "kind", "g", "E", "ctx")


def compare(argv):
    """``A - B`` of every window number of runs scored on identical windows, each ``[point, lo,
    hi]`` over shared event resamples: far tighter than two separate intervals."""
    import numpy as np

    from helix.eval.events import _diff, bootstrap, window_stat

    ap = argparse.ArgumentParser(prog="python -m helix.eval compare", description=compare.__doc__)
    ap.add_argument("--a", nargs="+", required=True, help="A's windows dumps, one per training seed")
    ap.add_argument("--b", nargs="+", default=None, help="B's, one per training seed (default: --a)")
    ap.add_argument("--arm", default="presence", help="the score of A: presence or amplitude")
    ap.add_argument("--arm-b", default=None, help="the score of B (default: --arm)")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fa, fb = list(a.a), list(a.b or a.a)
    arm_b = a.arm_b or a.arm
    if len(fa) != len(fb) and 1 not in (len(fa), len(fb)):
        raise SystemExit(f"compare: {len(fa)} files of A and {len(fb)} of B: they pair seed by seed "
                         "(as many on each side, or one on a side)")
    k = max(len(fa), len(fb))
    pairs = [(fa[i if len(fa) > 1 else 0], fb[i if len(fb) > 1 else 0]) for i in range(k)]
    zs = {f: np.load(f, allow_pickle=False) for f in fa + fb}
    # the window identity: the core keys, plus each window's origin wherever ANY dump has it (all
    # must then: a far window's kind, plane and NaN energy do not say where it is)
    mkeys = sorted({x for z in zs.values() for x in z.files if x.startswith("meta/")} | {f"meta/{c}" for c in _CORE})
    ref = zs[fa[0]]
    for f, z in zs.items():
        for mk in mkeys:
            if mk not in z.files or mk not in ref.files:
                raise SystemExit(f"compare: only some dumps record {mk[5:]}: score all with the same helix.eval")
            x, y = ref[mk], z[mk]
            if x.shape != y.shape or not np.array_equal(x, y, equal_nan=(x.dtype.kind == "f")):
                raise SystemExit(f"compare: the windows of {f} and {fa[0]} differ ({mk[5:]}): "
                                 "score them on the same events and seed")
    for f, arm in [(x, a.arm) for x in fa] + [(x, arm_b) for x in fb]:
        if f"score/{arm}" not in zs[f].files:
            raise SystemExit(f"compare: {f} has no score {arm!r}")
    meta = [dict(zip(_CORE, (int(e), str(kd), int(g), float(E), str(c) or None)))
            for e, kd, g, E, c in zip(*(ref[f"meta/{c}"] for c in _CORE))]
    stats = [(window_stat(zs[x][f"score/{a.arm}"][:, None], meta), window_stat(zs[y][f"score/{arm_b}"][:, None], meta)[0])
             for x, y in pairs]
    n, counts = stats[0][0][1], stats[0][0][2]

    def mean_diff(m):
        ds = [_diff(sa(m), sb(m)) for (sa, _, _), sb in stats]
        return {key: float(np.mean([d[key] for d in ds])) for key in ds[0]}

    diff = bootstrap(mean_diff, n, a.boot, a.seed)
    per_seed = [_diff(sa(np.ones(n, np.int64)), sb(np.ones(n, np.int64))) for (sa, _, _), sb in stats]
    half = {key: round(float(np.ptp([d[key] for d in per_seed])) / 2, 4) for key in per_seed[0]}
    views = {}
    for key in diff:
        base = key[:-2]
        if key.endswith("_U") and all(base + s in diff for s in ("_V", "_Y")):
            v = [diff[base + s][0] for s in ("_U", "_V", "_Y")]
            views[base] = dict(U=v[0], V=v[1], Y=v[2], mean=round(float(np.mean(v)), 4),
                               same_sign=bool(all(x > 0 for x in v) or all(x < 0 for x in v)))
    out = dict(a=fa, b=fb, arm=a.arm, arm_b=arm_b, n_events=n, n_windows=counts, n_pairs=k,
               a_minus_b=diff, seed_half_range=half, views=views)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
    print(f"[helix.eval compare] -> {a.out}", flush=True)
    return 0


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["compare"]:
        return compare(argv[1:])
    import helix.eval as E

    ap = argparse.ArgumentParser(prog="python -m helix.eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a fine-tuned denoiser run (save_path) or one checkpoint inside it; "
                    "with --uprobe, a pretrained FM run")
    ap.add_argument("--uprobe", action="store_true", help="score the frozen along-wire probe of an FM run instead")
    ap.add_argument("--out", required=True, help="results JSON (written whole)")
    ap.add_argument("--weights", default=None, choices=("ema", "last", "best"),
                    help="for a run directory (default: last for a denoiser, ema for --uprobe)")
    ap.add_argument("--corpus", default=None, help="the run's corpus at this site, where its recorded path is "
                    "absent and this site's corpus root does not hold it")
    ap.add_argument("--fm", default=None, help="the denoiser's FM run at this site, where its recorded model.fm "
                    "is absent and does not map onto this site's HELIX_EXP")
    ap.add_argument("--max-events", type=int, default=None, help="only the first N events of the probe split")
    ap.add_argument("--seed", type=int, default=0, help="window placement (and the probe's init)")
    ap.add_argument("--boot", type=int, default=200, help="event-bootstrap resamples")
    ap.add_argument("--presence-q", type=float, default=25.0, help="the presence label, q > this (e-)")
    ap.add_argument("--probe-seeds", type=int, default=2, help="--uprobe: fits per arm (seeds seed+k, same rows)")
    a = ap.parse_args(argv)

    common = dict(corpus=a.corpus, max_events=a.max_events, seed=a.seed, n_boot=a.boot)
    if a.uprobe:
        if a.fm:
            raise SystemExit("--fm relocates a denoiser's FM run; --uprobe reads the FM run itself")
        res = E.evaluate_uprobe(a.run, weights=a.weights or "last", probe_seeds=a.probe_seeds, **common)
    else:
        res = E.evaluate(a.run, weights=a.weights or "last", fm=a.fm, presence_q=a.presence_q, **common)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    win = {}
    for r in (res.get("metrics") or {}).values():
        if isinstance(r, dict) and "_windows" in r:
            win.update(r.pop("_windows"))
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
