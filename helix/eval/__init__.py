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
           under the model's random mask, and the same under a 75% region mask (``region``)
  floor    the detection floor (:class:`helix.eval.probe.Floor`): a probe on frozen features predicts
           per-cell charge; efficiency at a 1% false-positive rate per view, by energy, with the
           threshold on the far and on the near background, and the AUC; the trained arm
           against the raw tokenizer inputs under the same probe and the fit-free amplitude
  uprobe   the along-wire probe (:class:`helix.eval.probe.UProbe`): within-group R^2 of ``u`` per
           patch, per view; trained against the raw inputs and the geometry columns alone
  labels   the particle probe (:class:`helix.eval.probe.Labels`): each signal cell's motif and pid,
           balanced accuracy; trained against the raw inputs and the token layout alone
  denoise  a denoiser's per-cell charge on the floor's windows, and its charge resolution,
           linearity, closure and presence calibration on every charged cell
           (:mod:`helix.eval.denoise`)

The probes (floor, uprobe, labels) fit on the first two thirds of the events (at most
``helix.eval.probe.FIT_ROWS`` rows each, sampled evenly over them), ``probe_seeds`` times each,
and score every row of the rest as it arrives; denoise scores the same rest. Every probe arm is
compared with a reference that is not a probe of another encoder: a randomly initialised
encoder under the same probe measured below the fit-free amplitude on the floor and below
geometry along the wire, so it bounds the probe's fitting, not the representation. Intervals
are [16, 84]% over an event bootstrap of the scored events.

The JSON records what was evaluated: ``run`` (the run's provenance, its ``weights`` included),
``corpus`` (the evaluated shards' ``basis_digest``), ``truth`` (the per-sample charge cut the
cells truth carries) and ``weights`` (per metric, the weight set it read: the run's for recon
and the probes, the denoiser's own checkpoint for denoise).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

METRICS = ("recon", "floor", "uprobe", "labels", "denoise")

__all__ = ["METRICS", "Context", "evaluate"]


@dataclass
class Context:
    """What every metric is built from: the run, the encoder it probes, how many leading events
    the probes fit on, the seeds, and how many fits per probe arm."""
    run: object
    model: object
    n_train: int
    seed: int = 0
    n_boot: int = 200
    denoiser: object = None          # predict(event, keys) -> (y, presence), for denoise
    probe_seeds: int = 2             # fits per probe arm, seeds seed + k, on the same rows
    presence_q: float = 25.0         # denoise: the presence label, q > presence_q (e-)


def evaluate(path, metrics=("recon", "floor", "uprobe", "labels"), *, weights="ema", max_events=None,
             denoiser=None, seed=0, n_boot=200, probe_seeds=2, presence_q=25.0, device=None, log=print):
    """Score the run at ``path`` on ``metrics``; returns the results as one JSON-able dict."""
    from helix.eval.denoise import Denoise, load_denoiser
    from helix.eval.events import dataset, events, identity
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
    if int(probe_seeds) < 1:
        raise ValueError(f"probe_seeds={probe_seeds}: a probe arm needs at least one fit")
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
    ctx = Context(run, run.model, split_events(n) if probes or predict else n, seed, n_boot, predict,
                  int(probe_seeds), float(presence_q))
    log(f"[helix.eval] {run.provenance['run']} weights={run.provenance['weights']} "
        f"step={run.provenance['step']} events={n} (probes fit on {ctx.n_train}) metrics={','.join(metrics)}")

    ms = [dict(recon=Recon, floor=Floor, uprobe=UProbe, labels=Labels, denoise=Denoise)[m](ctx) for m in metrics]
    for ev in events(ds, run.tokenizer, dev, n):
        for m in ms:
            m.add(ev)
        if ev.index % 100 == 0 or ev.index + 1 == n:
            log(f"  event {ev.index + 1}/{n}")

    ident = identity(ds)
    dn = getattr(predict, "provenance", None) or {}
    out = dict(run=run.provenance, tokenizer=asdict(run.tokenizer), n_events=n, seed=seed,
               n_boot=n_boot, probe_seeds=int(probe_seeds),
               denoiser=None if denoiser is None else str(denoiser), code=provenance(),
               corpus=dict(basis_digest=ident["basis_digest"]),
               truth=dict(q_cut=ident["truth_q_cut"], q_cut_source=ident["truth_q_cut_source"]),
               weights={m: {k: (dn if m == "denoise" else run.provenance).get(k)
                            for k in ("run", "weights", "checkpoint", "step")} for m in metrics},
               metrics={})
    for m in ms:
        log(f"[helix.eval] {m.name}")
        out["metrics"][m.name] = m.result()
    return out
