"""Evaluating a trained run: one library, one entry point.

    python -m helix.eval RUN_DIR --metrics recon,floor,uprobe --out scores.json
    python -m helix.eval RUN_DIR --metrics denoise --denoiser DENOISE_RUN_DIR --out d.json

``RUN_DIR`` is a pimm run (``save_path``) or one checkpoint inside it; the model, its tokenizer
and its corpus come from the run itself (:func:`helix.model.checkpoint.load_run`). Events are the
run's corpus restricted to the ``probe`` role of its holdout, read once with their cells truth
(``scripts/build_truth.py``) and shared by every metric (:mod:`helix.eval.events`).

Metrics:

  recon    masked reconstruction (:mod:`helix.eval.recon`, the training run's own validation
           metric, CoeffFMEvaluator): loss, bce, val, var_expl, charge_closure, charge_resid
  floor    the detection floor (:mod:`helix.eval.floor`): a probe on frozen features predicts
           per-cell charge; map/presence/faint AUC, floor efficiency at 1% FPR by energy,
           localisation, pair separation; trained and random-init arms; rankme
  uprobe   the along-wire probe (:mod:`helix.eval.uprobe`): fisher_r of ``u`` per patch;
           geo, trained and random-init arms
  denoise  a denoiser's per-cell charge on the floor's cells and windows (:mod:`helix.eval.denoise`)

The probes (floor, uprobe) fit on the first two thirds of the events and score the rest; denoise
scores the same rest. Intervals are [16, 84]% over an event bootstrap of the scored events.
"""

from __future__ import annotations

METRICS = ("recon", "floor", "uprobe", "denoise")

__all__ = ["METRICS", "evaluate"]


def evaluate(path, metrics=("recon", "floor", "uprobe"), *, weights="ema", split="probe", max_events=None,
             denoiser=None, seed=0, n_boot=200, device=None, log=print):
    """Score the run at ``path`` on ``metrics``; returns the results as one JSON-able dict."""
    import torch
    from helix.eval.events import dataset, events
    from helix.model.checkpoint import load_run

    metrics = tuple(metrics)
    unknown = sorted(set(metrics) - set(METRICS))
    if unknown:
        raise ValueError(f"unknown metric(s) {unknown}; known: {METRICS}")
    if "denoise" in metrics and denoiser is None:
        raise ValueError("the denoise metric needs a denoiser run (denoiser=...)")
    run = load_run(path, weights, device=device)
    dev = next(run.model.parameters()).device
    ds = dataset(run, split)
    n = len(ds) if not max_events else min(int(max_events), len(ds))
    if not n:
        raise ValueError(f"{path}: the {split!r} split of the run's corpus is empty")
    log(f"[helix.eval] {run.provenance['run']} weights={run.provenance['weights']} "
        f"step={run.provenance['step']} split={split} events={n} metrics={','.join(metrics)}")

    ms = []
    for m in metrics:
        if m == "recon":
            from helix.eval.recon import Recon
            ms.append(Recon(run))
        elif m == "floor":
            from helix.eval.floor import Floor
            ms.append(Floor(run, n, seed=seed, n_boot=n_boot))
        elif m == "uprobe":
            from helix.eval.uprobe import UProbe
            ms.append(UProbe(run, n, seed=seed, n_boot=n_boot))
        else:
            from helix.eval.denoise import Denoise, load_denoiser
            predict, cfg = load_denoiser(denoiser, dev)
            if cfg != run.tokenizer:
                raise ValueError(f"the denoiser tokenizes with {cfg}, the run with {run.tokenizer}")
            ms.append(Denoise(predict, n, run.tokenizer, seed=seed, n_boot=n_boot))

    for ev in events(ds, run.tokenizer, dev, n):
        for m in ms:
            m.add(ev)
        if ev.index % 20 == 0 or ev.index + 1 == n:
            log(f"  event {ev.index + 1}/{n}")
        del ev
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    from dataclasses import asdict
    from helix.integrations._bootstrap import provenance
    out = dict(run=run.provenance, tokenizer=asdict(run.tokenizer), split=split, n_events=n, seed=seed,
               n_boot=n_boot, denoiser=None if denoiser is None else str(denoiser), code=provenance(),
               metrics={})
    for m in ms:
        log(f"[helix.eval] {m.name}")
        out["metrics"][m.name] = m.result()
    return out
