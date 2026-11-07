"""Evaluating a trained run: one library, one entry point.

    python -m helix.eval RUN_DIR --metrics recon,floor,uprobe,labels --out scores.json
    python -m helix.eval RUN_DIR --metrics denoise --denoiser DENOISE_RUN_DIR --out d.json

``RUN_DIR`` is a pimm run (``save_path``) or one checkpoint inside it; the model, its tokenizer
and its corpus come from the run itself (:func:`helix.model.checkpoint.load_run`). Events are the
run's corpus restricted to the ``probe`` role of its holdout, read once with their cells truth
(``scripts/build_truth.py``) and shared by every metric (:mod:`helix.eval.events`).

Metrics:

  recon    masked reconstruction (:mod:`helix.eval.recon`, the training run's own validation
           metric, CoeffFMEvaluator): loss, bce, val, var_expl, charge_closure, charge_resid
  floor    the detection floor (:class:`helix.eval.probe.Floor`): a probe on frozen features predicts
           per-cell charge; efficiency at a 1% false-positive rate per view, by energy; the
           trained arm against the fit-free coefficient amplitude
  uprobe   the along-wire probe (:class:`helix.eval.probe.UProbe`): within-group R^2 of ``u`` per
           patch, per view; trained against the geometry columns alone
  labels   the particle probe (:class:`helix.eval.probe.Labels`): each signal cell's motif and pid,
           balanced accuracy; trained against the token layout alone
  denoise  a denoiser's per-cell charge on the floor's windows (:mod:`helix.eval.denoise`)

The probes (floor, uprobe, labels) fit on the first two thirds of the events (at most
``helix.eval.probe.FIT_ROWS`` rows each, sampled evenly over them) and score every row of the
rest as it arrives; denoise scores the same rest. Every probe arm is compared with a reference
that is not a probe of another encoder: a randomly initialised encoder under the same probe
measured below the fit-free amplitude on the floor and below geometry along the wire, so it
bounds the probe's fitting, not the representation. Intervals are [16, 84]% over an event
bootstrap of the scored events.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

METRICS = ("recon", "floor", "uprobe", "labels", "denoise")

__all__ = ["METRICS", "Context", "evaluate"]


@dataclass
class Context:
    """What every metric is built from: the run, the encoder it probes, how many leading events
    the probes fit on, and the seeds."""
    run: object
    model: object
    n_train: int
    seed: int = 0
    n_boot: int = 200
    denoiser: object = None          # predict(event, keys) -> (y, presence), for denoise


def evaluate(path, metrics=("recon", "floor", "uprobe", "labels"), *, weights="ema", max_events=None,
             denoiser=None, seed=0, n_boot=200, device=None, log=print):
    """Score the run at ``path`` on ``metrics``; returns the results as one JSON-able dict."""
    from helix.eval.denoise import Denoise, load_denoiser
    from helix.eval.events import dataset, events
    from helix.eval.probe import Floor, Labels, UProbe, split_events
    from helix.eval.recon import Recon
    from helix.integrations._bootstrap import provenance
    from helix.model.checkpoint import load_run

    metrics = tuple(metrics)
    unknown = sorted(set(metrics) - set(METRICS))
    if unknown:
        raise ValueError(f"unknown metric(s) {unknown}; known: {METRICS}")
    if ("denoise" in metrics) != (denoiser is not None):
        raise ValueError("the denoise metric needs a denoiser run (denoiser=...), and only it does")
    run = load_run(path, weights, device=device)
    dev = next(run.model.parameters()).device
    ds = dataset(run)
    n = len(ds) if not max_events else min(int(max_events), len(ds))
    if not n:
        raise ValueError(f"{path}: the probe split of the run's corpus is empty")
    predict = None
    if denoiser is not None:
        predict, cfg = load_denoiser(denoiser, dev)
        if cfg != run.tokenizer:
            raise ValueError(f"the denoiser tokenizes with {cfg}, the run with {run.tokenizer}")
    probes = {"floor", "uprobe", "labels"} & set(metrics)
    ctx = Context(run, run.model, split_events(n) if probes or predict else n, seed, n_boot, predict)
    log(f"[helix.eval] {run.provenance['run']} weights={run.provenance['weights']} "
        f"step={run.provenance['step']} events={n} (probes fit on {ctx.n_train}) metrics={','.join(metrics)}")

    ms = [dict(recon=Recon, floor=Floor, uprobe=UProbe, labels=Labels, denoise=Denoise)[m](ctx) for m in metrics]
    for ev in events(ds, run.tokenizer, dev, n):
        for m in ms:
            m.add(ev)
        if ev.index % 100 == 0 or ev.index + 1 == n:
            log(f"  event {ev.index + 1}/{n}")

    out = dict(run=run.provenance, tokenizer=asdict(run.tokenizer), n_events=n, seed=seed,
               n_boot=n_boot, denoiser=None if denoiser is None else str(denoiser), code=provenance(),
               metrics={})
    for m in ms:
        log(f"[helix.eval] {m.name}")
        out["metrics"][m.name] = m.result()
    return out
