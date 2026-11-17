"""Evaluating a trained run: one library, one entry point.

    python -m helix.eval DENOISER_RUN --out e.json          # the evaluation
    python -m helix.eval --uprobe FM_RUN --out u.json       # the frozen along-wire probe
    python -m helix.eval compare --a A.windows.npz [A2 ...] [--b B.windows.npz ...] --out d.json
    python -m helix.eval FINE_RUN --corpus COARSE_ROOT --model-grid 1x4 --out e.json   # aggregated
    python -m helix.eval subcell --fine FINE_RUN --coarse COARSE_RUN --out s.json      # sub-cell charge

A denoiser trained on a finer truth than the one evaluated (``--model-grid``) is scored on each
truth cell's aggregate of its sub-cells' predictions (:func:`helix.eval.denoise.coarsen`), so
it compares with a coarse-grid run on the same windows. ``subcell``
(:class:`helix.eval.denoise.SubCell`) scores how such a run shares a bright coarse cell's charge
over its sub-cells, against spreading a coarse run's charge evenly.

The evaluation of a pretrained encoder is a fixed-budget FINE-TUNE: the denoiser trained on
1,024 labelled events (``configs/pimm/ft_eval.py``; a study is ``configs/sweeps/ft_eval.yaml``),
scored by :mod:`helix.eval.denoise` -- detection on the windows (the presence log-odds against
the fit-free amplitude, on identical windows; the headline is the LAUC, the efficiency averaged
over log-FPR in [1e-3, 1e-1]) and per cell, charge accuracy in bins of the predicted charge, and
the presence calibration, on the natural covered cells. The same protocol with
``model.freeze_encoder=True`` is the frozen diagnostic: one scoring path.

``DENOISER_RUN`` is a pimm run directory (its ``model/last``, the end of the budget, unless
``weights`` says otherwise) or one checkpoint inside it. Everything else comes from the run: the
events are its config's corpus restricted to the ``probe`` role of its holdout -- every probe
event is scored, since nothing is fitted here, except one the dataset rejects (its truth off its
planes; recorded under ``rejected``) -- tokenized as its FM run tokenizes
(:mod:`helix.eval.events`).

``uprobe`` (:mod:`helix.eval.probe`) is the one frozen-feature key kept: the along-wire
coordinate from a pretrained FM run's encoder, the 3D information a single plane cannot give. It
fits on the first two thirds of the probe events and scores the rest.

A run records absolute paths -- its corpus, a denoiser's FM run -- and a run trained at another
site names paths this one does not have. Both are found here (:func:`helix.data.verify.
resolve_corpus`, :func:`resolve_exp`) or given (``corpus``, ``fm``), and the corpus is held to
the run's recorded identity (a different ``basis_digest`` is refused). The JSON records what was
evaluated: ``run`` (the run's provenance, the weights it read included), ``corpus`` (recorded and
resolved path, the identity check, the evaluated shards' ``basis_digest``), ``fm`` (likewise),
``truth`` (the per-sample charge cut the cells truth carries) and ``code``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

__all__ = ["Context", "evaluate", "evaluate_subcell", "evaluate_uprobe", "resolve_exp"]


@dataclass
class Context:
    """What the denoiser's metric is built from: the run (its tokenizer), the predictor, the
    window seed, the bootstrap and the per-cell sampling (:mod:`helix.eval.denoise`)."""
    run: object
    denoiser: object                 # predict(event, keys) -> (y, logit[, q_mean[, y_sd]])
    seed: int = 0
    n_boot: int = 200
    presence_q: float = 25.0         # the presence label, q > presence_q (e-)
    empty_frac: float = 0.05         # the share of the uncharged covered cells sampled per event
    level: float = 0.95              # the bootstrap interval's coverage
    dump_cells: str = None           # a directory for each event's per-cell outputs, or None


def resolve_exp(recorded, *, override=None):
    """``(path, note)``: a run path another run recorded (a denoiser's ``model.fm``), at THIS
    site. In order: ``override``; the recorded path, if it exists; the same path relative to
    another site's ``HELIX_EXP`` (``helix/sites/*.yaml``) under this site's -- an S3DF
    ``/sdf/.../exp/helix/fm_cooldown`` is NERSC's ``$HELIX_EXP/fm_cooldown``."""
    from helix import paths

    if override:
        if not os.path.exists(str(override)):
            raise FileNotFoundError(f"--fm {override}: no such run or checkpoint")
        return str(override), "given"
    recorded = str(recorded)
    if os.path.exists(recorded):
        return recorded, "as recorded"
    try:
        here, _ = paths.resolve("HELIX_EXP")
    except paths.SiteError:
        here = None
    for site in paths.available_sites():
        base = os.path.expandvars(str((paths._read_profile(site).get("roots") or {}).get("HELIX_EXP") or ""))
        if not base or "$" in base or here is None:
            continue
        rel = os.path.relpath(recorded, base)
        if not rel.startswith("..") and (here / rel).exists():
            return str(here / rel), f"relocated from {site}'s HELIX_EXP"
    raise FileNotFoundError(f"{recorded}: not at this site, and not under another site's HELIX_EXP that "
                            f"maps onto this one's ({here}); pass its location (--fm)")


def _corpus(run, override):
    """The run's corpus at this site and the identity check: ``(dataset, info)``."""
    from helix.data.verify import resolve_corpus
    from helix.eval.events import dataset

    d = run.config["data"]["train"]
    root, check = resolve_corpus(d["data_root"], override=override,
                                 identity=(run.provenance.get("corpus") or {}).get("identity"),
                                 dataset_name=d.get("dataset_name", "sim_wire"), split=d.get("split"))
    return dataset(run, root), dict(recorded=d["data_root"], resolved=root, check=check)


def _score(run, ds, metric, max_events, log):
    """Feed the probe events at the first ``max_events`` positions to ``metric``, a position the
    dataset rejects skipped (:func:`helix.eval.events.events`); ``(n_scored, rejected positions,
    identity)``."""
    from helix.eval.events import events, identity

    n = len(ds) if not max_events else min(int(max_events), len(ds))
    if not n:
        raise ValueError(f"{run.provenance['run']}: the probe split of the run's corpus is empty")
    dev = next(run.model.parameters()).device
    rejected, k = [], 0
    for ev in events(ds, run.tokenizer, dev, n, rejected=rejected):
        metric.add(ev)
        k += 1
        if ev.index % 100 == 0 or ev.index + 1 == n:
            log(f"  event {ev.index + 1}/{n}")
    if rejected:
        log(f"  rejected (truth off its planes, not scored): positions {rejected}")
    return k, rejected, identity(ds)


def _record(run, n, rejected, ident, corpus, seed, n_boot):
    from helix.integrations._bootstrap import provenance
    return dict(run=run.provenance, tokenizer=asdict(run.tokenizer), n_events=n, rejected=rejected, seed=seed,
                n_boot=n_boot, code=provenance(), corpus=dict(corpus, basis_digest=ident["basis_digest"]),
                truth=dict(q_cut=ident["truth_q_cut"], q_cut_source=ident["truth_q_cut_source"]))


def _load(path, weights, fm, device):
    """``(predict, fm record)``: :func:`helix.eval.denoise.load_denoiser` on the denoiser run at
    ``path``, its FM run found at this site (:func:`resolve_exp`)."""
    import json

    from helix.eval.denoise import load_denoiser
    from helix.model.checkpoint import run_dir

    recorded = (json.load(open(os.path.join(run_dir(path), "resolved_config.json"))).get("model") or {}).get("fm")
    fm_path, fm_note = resolve_exp(recorded, override=fm) if recorded or fm else (None, "none recorded")
    predict, _ = load_denoiser(path, device, weights=weights, fm=fm_path)
    return predict, dict(recorded=recorded, resolved=fm_path, note=fm_note)


def evaluate(path, *, weights="last", corpus=None, fm=None, max_events=None, seed=0, n_boot=200,
             presence_q=25.0, empty_frac=0.05, level=0.95, dump_cells=None, model_grid=None, device=None,
             log=print):
    """Score the fine-tuned denoiser at ``path`` on its own probe events; returns the results as
    one JSON-able dict, the window scores under ``metrics.denoise._windows``; ``dump_cells`` a
    directory for each event's per-cell outputs (:meth:`helix.eval.denoise.Denoise.dump`).

    ``model_grid`` (``"1x4"``): the grid the denoiser was trained on, when the evaluated truth's
    (``corpus``) is coarser -- each truth cell is then scored on the aggregate of its sub-cells'
    predictions (:func:`helix.eval.denoise.coarsen`). Default: the truth's own grid. Nothing
    checks it against the run's training truth: a run does not record its grid."""
    from helix.data.cells import parse_grid
    from helix.eval.denoise import Denoise, coarsen

    predict, fmrec = _load(path, weights, fm, device)
    run = predict.run
    if model_grid:
        predict = coarsen(predict, parse_grid(model_grid))
    ds, cinfo = _corpus(run, corpus)
    log(f"[helix.eval] {run.provenance['run']} weights={run.provenance['weights']} step={run.provenance['step']}"
        f" corpus={cinfo['resolved']} ({cinfo['check']}) fm={fmrec['resolved']} ({fmrec['note']})"
        + (f" model_grid={model_grid} (aggregated)" if model_grid else ""))
    m = Denoise(Context(run, predict, seed, n_boot, float(presence_q), float(empty_frac), float(level), dump_cells))
    n, rejected, ident = _score(run, ds, m, max_events, log)
    out = _record(run, n, rejected, ident, cinfo, seed, n_boot)
    out.update(fm=fmrec, presence_q=float(presence_q), level=float(level), dump_cells=dump_cells,
               model_grid=str(parse_grid(model_grid)) if model_grid else None, metrics={"denoise": m.result()})
    return out


def evaluate_subcell(fine, coarse, *, coarse_grid="2x16", weights="last", corpus=None, fm=None, max_events=None,
                     seed=0, n_boot=200, presence_q=25.0, bright_q=None, level=0.95, device=None, log=print):
    """The sub-cell charge score (:class:`helix.eval.denoise.SubCell`) of the denoiser run
    ``fine`` against the even spread of the run ``coarse`` (trained on ``coarse_grid``), on the
    probe events of ``fine``'s corpus (its fine truth). Both must tokenize alike: one event's
    tokens serve both encoders."""
    from helix.data.cells import parse_grid
    from helix.eval.denoise import BRIGHT_Q, SubCell

    pf, fmf = _load(fine, weights, fm, device)
    pc, fmc = _load(coarse, weights, fm, device)
    run = pf.run
    if run.tokenizer != pc.run.tokenizer:
        raise ValueError(f"{fine} and {coarse} tokenize differently; the sub-cell score encodes one event for both")
    ds, cinfo = _corpus(run, corpus)
    log(f"[helix.eval subcell] fine={run.provenance['run']} coarse={pc.run.provenance['run']} ({coarse_grid}) "
        f"weights={weights} corpus={cinfo['resolved']} ({cinfo['check']})")
    m = SubCell(pf, pc, parse_grid(coarse_grid), bright_q=BRIGHT_Q if bright_q is None else bright_q,
                presence_q=presence_q, n_boot=n_boot, seed=seed, level=level)
    n, rejected, ident = _score(run, ds, m, max_events, log)
    out = _record(run, n, rejected, ident, cinfo, seed, n_boot)
    out.update(fm=fmf, coarse=dict(run=pc.run.provenance, fm=fmc), metrics={"subcell": m.result()})
    return out


def evaluate_uprobe(path, *, weights="last", corpus=None, max_events=None, seed=0, n_boot=200, probe_seeds=2,
                    device=None, log=print):
    """The along-wire probe (:class:`helix.eval.probe.UProbe`) of the FM run at ``path``."""
    from helix.eval.probe import ProbeContext, UProbe, split_events
    from helix.model.checkpoint import load_run

    if int(probe_seeds) < 1:
        raise ValueError(f"probe_seeds={probe_seeds}: a probe arm needs at least one fit")
    run = load_run(path, weights, device=device)
    if run.config["model"].get("type") == "Coeff-Denoise":
        raise ValueError(f"{path}: a denoiser run; the along-wire probe reads a pretrained FM's encoder")
    ds, cinfo = _corpus(run, corpus)
    n = len(ds) if not max_events else min(int(max_events), len(ds))
    ctx = ProbeContext(run, run.model, split_events(n), seed, n_boot, int(probe_seeds))
    log(f"[helix.eval] uprobe {run.provenance['run']} weights={run.provenance['weights']} "
        f"step={run.provenance['step']} events={n} (fit on {ctx.n_train}) corpus={cinfo['resolved']} ({cinfo['check']})")
    m = UProbe(ctx)
    n, rejected, ident = _score(run, ds, m, n, log)
    out = _record(run, n, rejected, ident, cinfo, seed, n_boot)
    out.update(probe_seeds=int(probe_seeds), metrics={"uprobe": m.result()})
    return out
