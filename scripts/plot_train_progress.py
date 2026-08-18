"""Training curves for one or more pimm runs, parsed from ``train.log``.

The tfevents files are per-launch (a preempted-and-resumed run writes one per
link), so stitching them means reconciling several event files with overlapping
wall-clock. ``train.log`` is appended by every link of the same run and already
carries the epoch/iteration counter the trainer restored, so it is the single
monotone record of what happened.

Two line kinds are read::

    Train: [ep/EP][it/ITERS] ... loss: L bce: B val: V masked_frac: F Lr: R
    [coeff-eval] batches=N bce=B loss=L masked_frac=F val=V

The eval line carries no step of its own; it is stamped with the step of the
Train line immediately above it, which is where the evaluator hook fired.

Usage::

    python3 scripts/plot_train_progress.py \\
        k30=/sdf/data/neutrino/omara/exp/helix/coeff-fm-train \\
        r1=/sdf/data/neutrino/omara/exp/helix/coeff-fm-train-r1 \\
        --out /sdf/data/neutrino/omara/exp/figs
"""

import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_RE = re.compile(
    r"Train: \[(\d+)/(\d+)\]\[(\d+)/(\d+)\].*?"
    r"loss: ([\d.eE+-]+) bce: ([\d.eE+-]+) val: ([\d.eE+-]+) "
    r"masked_frac: ([\d.eE+-]+) Lr: ([\d.eE+-]+)")
EVAL_RE = re.compile(
    r"\[coeff-eval\] batches=(\d+) bce=([\d.eE+-]+) loss=([\d.eE+-]+) "
    r"masked_frac=([\d.eE+-]+) val=([\d.eE+-]+)")


def parse(run_dir):
    """-> (train dict of arrays, eval dict of arrays). Steps are GLOBAL."""
    path = os.path.join(run_dir, "train.log")
    tr = dict(step=[], loss=[], bce=[], val=[], mask=[], lr=[], epoch=[])
    ev = dict(step=[], loss=[], bce=[], val=[], mask=[])
    last_step = None
    with open(path) as fh:
        for line in fh:
            m = TRAIN_RE.search(line)
            if m:
                ep, _, it, iters = (int(m.group(i)) for i in (1, 2, 3, 4))
                # pimm's counter is 1-based in both fields, and `it` counts to
                # `iters` inclusive — so the last iteration of epoch e and the
                # zeroth of e+1 must not collide.
                step = (ep - 1) * iters + it
                tr["step"].append(step)
                tr["epoch"].append(ep)
                for k, g in (("loss", 5), ("bce", 6), ("val", 7),
                             ("mask", 8), ("lr", 9)):
                    tr[k].append(float(m.group(g)))
                last_step = step
                continue
            m = EVAL_RE.search(line)
            if m and last_step is not None:
                ev["step"].append(last_step)
                for k, g in (("bce", 2), ("loss", 3), ("mask", 4), ("val", 5)):
                    ev[k].append(float(m.group(g)))
    tr = {k: np.asarray(v, float) for k, v in tr.items()}
    ev = {k: np.asarray(v, float) for k, v in ev.items()}
    # A resume that WARM-STARTED instead of resuming resets the counter, and the
    # curve would then silently fold back on itself. Say so rather than plot it.
    if len(tr["step"]) > 1:
        back = np.flatnonzero(np.diff(tr["step"]) < 0)
        if len(back):
            print(f"  WARNING {os.path.basename(run_dir)}: step decreases at "
                  f"{len(back)} place(s) (first at index {back[0]}, "
                  f"{tr['step'][back[0]]:.0f} -> {tr['step'][back[0]+1]:.0f}) — "
                  f"a link restarted the counter", file=sys.stderr)
    return tr, ev


def _smooth(y, k):
    """Centred box mean, edge-shrinking so the ends are not pulled inward."""
    if k <= 1 or len(y) < 3:
        return y
    k = min(k, len(y))
    c = np.cumsum(np.insert(y, 0, 0.0))
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        lo, hi = max(0, i - k // 2), min(len(y), i + k // 2 + 1)
        out[i] = (c[hi] - c[lo]) / (hi - lo)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=/path/to/run_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smooth", type=int, default=9,
                    help="box width over the 200-step train points")
    ap.add_argument("--tail-frac", type=float, default=0.5,
                    help="fraction of the run the zoom panels cover")
    ap.add_argument("--nulls", nargs="*", default=[], metavar="LABEL=HVAL,HOCC",
                    help="per-run reference levels, in nats: HVAL is the "
                         "MARGINAL bin entropy of the value target and HOCC the "
                         "Bernoulli entropy of the occupancy rate over VALID "
                         "slots. Both are corpus-dependent, so each run needs "
                         "its own; measured with _diag/ref_eval/nulls.py over "
                         "the full 577-event val split:\n"
                         "  k30 = 4.2430,0.2505   r1 = 4.2902,0.2479\n"
                         "Runs with no entry get no reference line. Do NOT use "
                         "ln(K): uniform-over-128-bins is not a predictor "
                         "anything could achieve, and drawing it overstated the "
                         "value head's gain by 1.42x. Research never used an "
                         "entropy null at all — it reported var_expl against a "
                         "measured per-band target variance.")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    runs = []
    for spec in a.runs:
        label, _, path = spec.partition("=")
        if not path:
            label, path = os.path.basename(spec.rstrip("/")), spec
        print(f"parsing {label}: {path}")
        tr, ev = parse(path)
        print(f"  {len(tr['step'])} train points, {len(ev['step'])} evals, "
              f"final step {tr['step'][-1]:.0f}")
        runs.append((label, tr, ev))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Both heads are cross-entropies in nats, which is not a scale anyone reads.
    # Put each on its own interpretable axis, against the score of the best
    # CONSTANT predictor — that is what says whether the model learned anything.
    #
    #   value head    exp(CE) = the effective number of bins a coefficient is
    #                 still spread over. The reference is exp(H_marginal), NOT K:
    #                 a uniform-over-128 predictor is not achievable by anything,
    #                 and drawing it at 128 overstated the gain by 1.42x. The
    #                 measured marginals are ~4.24-4.29 nats, i.e. ~70-73 bins.
    #   occupancy     BCE against the Bernoulli entropy of the occupancy rate
    #                 over VALID slots (~0.068, giving ~0.25 nats — not the 0.047
    #                 rate over live-cell slots, which gave 0.19 and understated
    #                 the learning by 13 points).
    #
    # Both nulls are CORPUS-dependent, so one shared line cannot serve two runs;
    # they come in per-run via --nulls and are drawn in each run's own colour.
    # Research reported neither — it used var_expl against a measured per-band
    # target variance, which is grid-free and is the better thing to port.
    #
    # The learning-rate panel is gone: WSD's stable phase is flat by
    # construction (see the scheduler note in coeff_fm_train.py), so it can only
    # ever confirm the config. Total loss is gone too — it is bce + val exactly,
    # so it adds no axis the two component panels do not already carry.
    nulls = {}
    for spec in a.nulls:
        lab, _, vals = spec.partition("=")
        hv, _, ho = vals.partition(",")
        nulls[lab] = (float(hv), float(ho))
    if nulls:
        print("\nreference levels (measured, per run):")
        for lab, (hv, ho) in nulls.items():
            print(f"  {lab}: value H={hv:.4f} nats -> {np.exp(hv):.1f} effective "
                  f"bins;  occupancy H={ho:.4f} nats")
    else:
        print("\nno --nulls given; drawing no reference lines (see --help)")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    for i, (label, tr, ev) in enumerate(runs):
        c = colors[i % len(colors)]
        ax.plot(tr["step"], np.exp(_smooth(tr["val"], a.smooth)), color=c,
                lw=1.0, alpha=0.35)
        ax.plot(ev["step"], np.exp(ev["val"]), color=c, lw=1.8, label=label)
        if len(ev["step"]):
            ax.annotate(f"{np.exp(ev['val'][-1]):.1f}",
                        (ev["step"][-1], np.exp(ev["val"][-1])),
                        textcoords="offset points", xytext=(-8, 6), fontsize=9,
                        color=c, ha="right")
    for i, (label, _, _) in enumerate(runs):
        if label in nulls:
            lv = np.exp(nulls[label][0])
            ax.axhline(lv, color=colors[i % len(colors)], ls=":", lw=1.2)
            ax.annotate(f"{label} marginal = {lv:.0f} bins", (0.02, lv),
                        xycoords=("axes fraction", "data"), fontsize=8,
                        va="bottom", color=colors[i % len(colors)])
    ax.set_yscale("log")
    ax.set_title("value head — effective bins in play,  exp(CE)", fontsize=10)
    ax.set_xlabel("step"); ax.set_ylabel("effective bins")
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=9)

    ax = axes[1]
    for i, (label, tr, ev) in enumerate(runs):
        c = colors[i % len(colors)]
        ax.plot(tr["step"], _smooth(tr["bce"], a.smooth), color=c, lw=1.0, alpha=0.35)
        ax.plot(ev["step"], ev["bce"], color=c, lw=1.8, label=label)
        if len(ev["step"]):
            ax.annotate(f"{ev['bce'][-1]:.4f}", (ev["step"][-1], ev["bce"][-1]),
                        textcoords="offset points", xytext=(-8, 6), fontsize=9,
                        color=c, ha="right")
    for i, (label, _, _) in enumerate(runs):
        if label in nulls:
            lo = nulls[label][1]
            ax.axhline(lo, color=colors[i % len(colors)], ls=":", lw=1.2)
            ax.annotate(f"{label} base-rate = {lo:.3f}", (0.02, lo),
                        xycoords=("axes fraction", "data"), fontsize=8,
                        va="bottom", color=colors[i % len(colors)])
    ax.set_title("occupancy head — BCE vs the constant-rate predictor", fontsize=10)
    ax.set_xlabel("step"); ax.set_ylabel("nats")
    ax.grid(alpha=0.25); ax.legend(fontsize=9)

    ax = axes[2]
    if len(runs) == 2:
        (la, ea), (lb, eb) = (runs[0][0], runs[0][2]), (runs[1][0], runs[1][2])
        n = min(len(ea["step"]), len(eb["step"]))
        ax.plot(ea["step"][:n],
                100 * (np.exp(ea["val"][:n]) / np.exp(eb["val"][:n]) - 1),
                lw=1.6, label="value (effective bins)")
        ax.plot(ea["step"][:n], 100 * (ea["bce"][:n] / eb["bce"][:n] - 1),
                lw=1.6, ls="--", label="occupancy (BCE)")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{la} relative to {lb}  (>0 = {la} worse)", fontsize=10)
        ax.set_xlabel("step"); ax.set_ylabel("% difference")
        ax.grid(alpha=0.25); ax.legend(fontsize=9)
    else:
        ax.set_axis_off()
    fig.suptitle("Held-out progression (faint = train, solid = eval)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p1 = os.path.join(a.out, "train_progress.png")
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    print("wrote", p1)

    # ---- a text summary, so the numbers exist outside the picture ----------
    print("\n%-8s %8s %10s %10s %10s %10s %9s" %
          ("run", "steps", "eval val", "eff bins", "eval bce", "vs null", "mask frac"))
    for label, tr, ev in runs:
        ho = nulls.get(label, (None, None))[1]
        rel = "  n/a" if ho is None else "%4.0f%%" % (100 * (ev["bce"][-1] / ho - 1))
        print("%-8s %8.0f %10.4f %10.1f %10.4f %10s %9.4f" %
              (label, tr["step"][-1], ev["val"][-1], np.exp(ev["val"][-1]),
               ev["bce"][-1], rel, tr["mask"].mean()))


if __name__ == "__main__":
    main()
