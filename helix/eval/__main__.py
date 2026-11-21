"""``python -m helix.eval DENOISER_RUN --out e.json`` (see helix.eval).

The per-window scores go beside it, ``e.windows.npz``: the fine-tuned ``presence`` (log-odds)
and the fit-free ``amplitude``, on the same windows. Two runs scored on the same events and seed
share their windows (truth and ``[seed, event]`` place them), so their difference is paired:

    python -m helix.eval compare --a A.windows.npz [A_s1.windows.npz ...] \\
        [--b B.windows.npz [B_s1.windows.npz ...]] [--arm presence] [--arm-b ARM] \\
        [--boot 200] [--seed 0] [--level 0.95] --out diff.json

bootstraps ``A - B`` of every window number (:func:`helix.eval.events.window_stat`) over the
same resamples. Several files on a side are the arm's training seeds: they pair seed by seed (one
file on a side pairs with every file on the other), the difference is the mean over the pairs,
and each resample draws the pairs with replacement before the events, so the interval carries
the training-seed spread too; ``seed_half_range`` is the pairs' (max - min) / 2. Without ``--b``
both scores come from ``--a`` (``--arm presence --arm-b amplitude``: what the fine-tune adds over
the corpus gate). ``views`` puts each per-view number's U / V / Y differences side by side, so a
pooled difference carried by one view shows (``same_sign``).

    python -m helix.eval rescore A.windows.npz [A_s1.windows.npz ...] --out a.json

re-scores dumps with the current window numbers -- an arm's seeds averaged, seeds then events
resampled -- so a past evaluation reads in today's definitions without a GPU.

    python -m helix.eval subcell --fine FINE_RUN --coarse COARSE_RUN [--coarse-grid 2x16] --out s.json

scores how a fine-grid denoiser shares each bright coarse cell's charge over its sub-cells,
against the even spread of a coarse-grid run (helix.eval.denoise.SubCell)."""

from __future__ import annotations

import argparse
import json
import os

#: The window metadata that must be present in every dump (the rest of WINDOW_META too, wherever
#: any dump records it).
_CORE = ("ev", "kind", "g", "E", "ctx")


def _load_windows(files):
    """``(dumps, meta)`` of windows dumps scored on identical windows (refused otherwise): the
    core keys, plus each window's origin and ``kept`` wherever ANY dump has them (all must then:
    a far window's kind, plane and NaN energy do not say where it is)."""
    import numpy as np
    zs = {f: np.load(f, allow_pickle=False) for f in files}
    mkeys = sorted({x for z in zs.values() for x in z.files if x.startswith("meta/")} | {f"meta/{c}" for c in _CORE})
    ref = zs[files[0]]
    for f, z in zs.items():
        for mk in mkeys:
            if mk not in z.files or mk not in ref.files:
                raise SystemExit(f"only some dumps record {mk[5:]}: score all with the same helix.eval")
            x, y = ref[mk], z[mk]
            if x.shape != y.shape or not np.array_equal(x, y, equal_nan=(x.dtype.kind == "f")):
                raise SystemExit(f"the windows of {f} and {files[0]} differ ({mk[5:]}): "
                                 "score them on the same events and seed")
    cols = {mk[5:]: ref[mk] for mk in mkeys}
    conv = dict(ev=int, kind=str, g=int, E=float, ctx=lambda c: str(c) or None, w0=int, t0=int,
                kept=lambda k: (str(k) == "True") if not isinstance(k, (bool, np.bool_)) else bool(k))
    meta = [{k: conv.get(k, lambda x: x)(cols[k][i]) for k in cols} for i in range(len(cols["ev"]))]
    return zs, meta


def _scores(zs, files, arm):
    for f in files:
        if f"score/{arm}" not in zs[f].files:
            raise SystemExit(f"{f} has no score {arm!r}")
    return [zs[f][f"score/{arm}"][:, None] for f in files]


def compare(argv):
    """``A - B`` of every window number of runs scored on identical windows, each ``[point, lo,
    hi]`` over shared resamples (training seeds, then events): far tighter than two separate
    intervals."""
    import numpy as np

    from helix.eval.events import LEVEL, _diff, bootstrap, window_stat

    ap = argparse.ArgumentParser(prog="python -m helix.eval compare", description=compare.__doc__)
    ap.add_argument("--a", nargs="+", required=True, help="A's windows dumps, one per training seed")
    ap.add_argument("--b", nargs="+", default=None, help="B's, one per training seed (default: --a)")
    ap.add_argument("--arm", default="presence", help="the score of A: presence or amplitude")
    ap.add_argument("--arm-b", default=None, help="the score of B (default: --arm)")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=float, default=LEVEL, help="the interval's coverage")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fa, fb = list(a.a), list(a.b or a.a)
    arm_b = a.arm_b or a.arm
    if len(fa) != len(fb) and 1 not in (len(fa), len(fb)):
        raise SystemExit(f"compare: {len(fa)} files of A and {len(fb)} of B: they pair seed by seed "
                         "(as many on each side, or one on a side)")
    k = max(len(fa), len(fb))
    pairs = [(fa[i if len(fa) > 1 else 0], fb[i if len(fb) > 1 else 0]) for i in range(k)]
    try:
        zs, meta = _load_windows(list(dict.fromkeys(fa + fb)))
        sa_, sb_ = _scores(zs, fa, a.arm), _scores(zs, fb, arm_b)
    except SystemExit as e:
        raise SystemExit(f"compare: {e}") from None
    by_a, by_b = dict(zip(fa, sa_)), dict(zip(fb, sb_))
    stats = [(window_stat(by_a[x], meta), window_stat(by_b[y], meta)[0]) for x, y in pairs]
    n, counts = stats[0][0][1], stats[0][0][2]
    diffs = lambda m: [_diff(sa(m), sb(m)) for (sa, _, _), sb in stats]          # noqa: E731

    def mean_diff(m, s=None):
        ds = diffs(m)
        w = np.ones(len(ds)) if s is None else np.asarray(s, np.float64)
        return {key: float(np.average([d[key] for d in ds], weights=w)) for key in ds[0]}

    diff = bootstrap(mean_diff, n, a.boot, a.seed, n_seeds=k if k > 1 else None, level=a.level)
    per_seed = diffs(np.ones(n, np.int64))
    half = {key: round(float(np.ptp([d[key] for d in per_seed])) / 2, 4) for key in per_seed[0]}
    views = {}
    for key in diff:
        base = key[:-2]
        if key.endswith("_U") and all(base + s in diff for s in ("_V", "_Y")):
            v = [diff[base + s][0] for s in ("_U", "_V", "_Y")]
            views[base] = dict(U=v[0], V=v[1], Y=v[2], mean=round(float(np.mean(v)), 4),
                               same_sign=bool(all(x > 0 for x in v) or all(x < 0 for x in v)))
    out = dict(a=fa, b=fb, arm=a.arm, arm_b=arm_b, n_events=n, n_windows=counts, n_pairs=k, level=a.level,
               resampled="training-seed pairs, then events" if k > 1 else "events",
               a_minus_b=diff, seed_half_range=half, views=views)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
    print(f"[helix.eval compare] -> {a.out}", flush=True)
    return 0


def rescore(argv):
    """Every window number of one arm's windows dumps (one per training seed, averaged), each
    ``[point, lo, hi]`` over seeds-then-events resamples, for every score the dumps hold."""
    import numpy as np

    from helix.eval.events import HEADLINE, LEVEL, SECONDARY, bootstrap, window_stat

    ap = argparse.ArgumentParser(prog="python -m helix.eval rescore", description=rescore.__doc__)
    ap.add_argument("dumps", nargs="+", help="the arm's windows dumps, one per training seed")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=float, default=LEVEL)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        zs, meta = _load_windows(list(a.dumps))
    except SystemExit as e:
        raise SystemExit(f"rescore: {e}") from None
    names = sorted({x[6:] for x in zs[a.dumps[0]].files if x.startswith("score/")})
    k = len(a.dumps)
    res = {}
    for name in names:
        stats = [window_stat(s, meta) for s in _scores(zs, a.dumps, name)]
        n, counts = stats[0][1], stats[0][2]

        def mean_stat(m, s=None, stats=stats):
            vs = [st(m) for st, _, _ in stats]
            w = np.ones(len(vs)) if s is None else np.asarray(s, np.float64)
            return {key: float(np.average([v[key] for v in vs], weights=w)) for key in vs[0]}
        r = bootstrap(mean_stat, n, a.boot, a.seed, n_seeds=k if k > 1 else None, level=a.level)
        r["n_windows"] = counts
        res[name] = r
    out = dict(dumps=list(a.dumps), n_events=n, level=a.level, n_seeds=k,
               resampled="training seeds, then events" if k > 1 else "events",
               headline={nm: {h: res[nm][h] for h in HEADLINE + SECONDARY} for nm in names}, scores=res)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
    print(f"[helix.eval rescore] -> {a.out}", flush=True)
    return 0


def _write(res, out):
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=str)
    os.replace(tmp, out)


def subcell(argv):
    """The sub-cell charge score (helix.eval.denoise.SubCell) of a fine-grid denoiser run against
    the even spread of a coarse-grid one, on the fine run's probe events and fine truth."""
    import helix.eval as E

    ap = argparse.ArgumentParser(prog="python -m helix.eval subcell", description=subcell.__doc__)
    ap.add_argument("--fine", required=True, help="the fine-grid denoiser run (its corpus holds the fine truth)")
    ap.add_argument("--coarse", required=True, help="the coarse-grid denoiser run")
    ap.add_argument("--coarse-grid", default="2x16", help="the grid --coarse was trained on")
    ap.add_argument("--weights", default="last", choices=("last", "best"))
    ap.add_argument("--corpus", default=None, help="the fine run's corpus root at this site")
    ap.add_argument("--fm", default=None, help="the runs' FM run at this site")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--bright-q", type=float, default=None, help="a bright parent's truth charge, e- (default BRIGHT_Q)")
    ap.add_argument("--presence-q", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = E.evaluate_subcell(a.fine, a.coarse, coarse_grid=a.coarse_grid, weights=a.weights, corpus=a.corpus,
                             fm=a.fm, max_events=a.max_events, seed=a.seed, n_boot=a.boot,
                             presence_q=a.presence_q, bright_q=a.bright_q, level=a.level)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    _write(res, a.out)
    print(f"[helix.eval subcell] -> {a.out}", flush=True)
    return 0


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["compare"]:
        return compare(argv[1:])
    if argv[:1] == ["rescore"]:
        return rescore(argv[1:])
    if argv[:1] == ["subcell"]:
        return subcell(argv[1:])
    import helix.eval as E

    ap = argparse.ArgumentParser(prog="python -m helix.eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a fine-tuned denoiser run (save_path) or one checkpoint inside it; "
                    "with --uprobe, a pretrained FM run")
    ap.add_argument("--uprobe", action="store_true", help="score the frozen along-wire probe of an FM run instead")
    ap.add_argument("--out", required=True, help="results JSON (written whole)")
    ap.add_argument("--weights", default=None, choices=("ema", "last", "best"),
                    help="for a run directory (default: last; ema and best read runs from before 2026-11-09)")
    ap.add_argument("--corpus", default=None, help="the run's corpus at this site, where its recorded path is "
                    "absent and this site's corpus root does not hold it")
    ap.add_argument("--fm", default=None, help="the denoiser's FM run at this site, where its recorded model.fm "
                    "is absent and does not map onto this site's HELIX_EXP")
    ap.add_argument("--max-events", type=int, default=None, help="only the first N events of the probe split")
    ap.add_argument("--seed", type=int, default=0, help="window placement (and the probe's init)")
    ap.add_argument("--boot", type=int, default=200, help="event-bootstrap resamples")
    ap.add_argument("--level", type=float, default=0.95, help="the bootstrap interval's coverage")
    ap.add_argument("--presence-q", type=float, default=25.0, help="the presence label, q > this (e-)")
    ap.add_argument("--empty-frac", type=float, default=0.05,
                    help="the share of the uncharged covered cells sampled per event for the per-cell numbers")
    ap.add_argument("--dump-cells", default=None, metavar="DIR",
                    help="write each event's per-cell outputs (charged, window and sampled cells, with "
                    "weights) to DIR/ev<index>.npz")
    ap.add_argument("--model-grid", default=None, metavar="WxT",
                    help="the grid the denoiser was trained on, when the evaluated truth (--corpus) is coarser: "
                    "each truth cell is scored on its sub-cells' aggregated predictions")
    ap.add_argument("--probe-seeds", type=int, default=2, help="--uprobe: fits per arm (seeds seed+k, same rows)")
    a = ap.parse_args(argv)

    common = dict(corpus=a.corpus, max_events=a.max_events, seed=a.seed, n_boot=a.boot)
    if a.uprobe:
        if a.fm or a.model_grid:
            raise SystemExit("--fm and --model-grid are a denoiser's; --uprobe reads an FM run")
        res = E.evaluate_uprobe(a.run, weights=a.weights or "last", probe_seeds=a.probe_seeds, **common)
    else:
        res = E.evaluate(a.run, weights=a.weights or "last", fm=a.fm, presence_q=a.presence_q,
                         empty_frac=a.empty_frac, level=a.level, dump_cells=a.dump_cells,
                         model_grid=a.model_grid, **common)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    win = {}
    for r in (res.get("metrics") or {}).values():
        if isinstance(r, dict) and "_windows" in r:
            win.update(r.pop("_windows"))
    if win:
        import numpy as np
        np.savez_compressed(os.path.splitext(a.out)[0] + ".windows.npz", **win)
    _write(res, a.out)
    print(f"[helix.eval] -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
