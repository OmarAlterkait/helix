"""The evaluated events: a run's held-out split, tokenized once, with its cells truth.

Every metric reads the same :class:`Event` objects, so an event is read, tokenized and moved to
the device once, and the encoder's frozen features and the event's map rows and windows are
computed once per event however many metrics use them.

WINDOWS
=======
The evaluation windows and the detection floor, on the fixed fine grid.

Every model is scored on the same 2-wire x 16-tick cells (:mod:`helix.data.cells`) whatever its
patch size, so a bigger patch cannot lose sub-patch detail without the score moving. A cell no
token covers is scored as predicted 0, never dropped: which cells are scored never depends on the
tokenizer.

Truth (per event, from the cells record; no tokenizer, no checkpoint):

  map rows   fine cells a probe is FITTED on: every cell with truth charge (capped), cells near
             signal, cells at random kept coefficients (noise-dominated). Not scored.
  windows    32 wires x 128 ticks (16 x 8 fine cells), all cells kept:
               iso   one isolated, compact particle (>=90% of its plane charge inside, <5% of
                     the window from anything else), placed at a RANDOM offset around it (a
                     window centred on truth would put a blurred prediction's centre on the
                     answer)
               bg    kept coefficients but no truth charge within +-16 w / +-128 t; up to
                     :data:`N_BG_PLANE` per plane, so every view has its own background
                     (the FAR background)
               near  the NEAR background: no truth charge in the window or within one cell
                     of it, but some within +-16 w / +-128 t -- exactly the surroundings
                     ``bg`` must keep empty (:func:`helix.data.denoise.near_windows`, the
                     denoiser's own training windows) -- and at least one kept coefficient
                     inside, as every ``bg`` window has; up to :data:`N_BG_PLANE` per plane.
                     Drawn after the others, so iso and bg are the same with or without them

The number (:func:`score_windows`): a window's score is its largest cell score; an iso window
is detected when it scores above the 99th percentile of the background windows OF ITS VIEW (U,
V or Y, both volumes) -- a 1% false-positive rate per view, since the views differ in noise
and one pooled threshold is set by the views with most background (measured: Y ran at 2.4-3.2%
under a pooled 1%). Reported as the detected fraction by the particle's deposited energy (its
whole track's step energy; it tracks the charge in the window, Spearman 0.94-0.96 on every
view at ~20k e/MeV), with a [16, 84]% interval over an event bootstrap.

The far background alone sets that threshold, while an iso window caps foreign charge only
INSIDE itself (5%), so a prediction that hazes charge around tracks gains iso efficiency at no
cost on it. The near background prices the haze: the same efficiency with each view's
threshold set on the near windows instead, and the rate at which the far threshold fires on
them (1% if near and far look alike to the prediction). Beside them, threshold-free, the AUC
of iso against far background (seed pairs agree to ~0.005 on it, ~0.04 on the efficiency at
1%); and every number per view as well as pooled over views.

Also here: the within-group R^2 the along-wire probe reports (:func:`group_r2`).

numpy at import; torch and pimm-data only when events are read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from helix.data.cells import FT, FW, fkey, unkey

@dataclass
class Event:
    index: int                 # position in the evaluated sequence, 0..n-1
    cells: dict                # helix.data.cells record
    coeff_cells: np.ndarray    # fine cells of the kept coefficients (helix.data.cells.coeff_cells)
    coeff_amp: tuple           # (those cells, their largest |coefficient| / sigma): coeff_amplitude
    band_lengths: np.ndarray
    cell_key: np.ndarray       # per token, the tokenizer's cell key (numpy)
    batch: dict                # the FM's batch (torch, on the device)
    _feats: object = field(default=None, repr=False)
    _raw: object = field(default=None, repr=False)
    _truth: tuple = field(default=None, repr=False)         # (seed, truth)

    def raw_inputs(self):
        """Per token, the tokenizer's own input to the encoder -- ``inp`` and ``occ``
        (``n_slot`` each), concatenated, float32, cached: what the probes' ``raw`` arm reads in
        place of encoder features."""
        if self._raw is None:
            import torch
            self._raw = torch.cat([self.batch["inp"].float(), self.batch["occ"].float()], 1)
        return self._raw

    def features(self, model):
        """``model.encode(batch)`` (all tokens visible), float32, cached: an evaluation probes
        one encoder."""
        if self._feats is None:
            import torch
            from helix.eval.recon import autocast
            with torch.no_grad(), autocast(self.batch["plane_id"].device):
                self._feats = model.encode(self.batch).float()
        return self._feats

    def truth(self, seed, windows):
        """:func:`event_truth` of the event (its windows only if ``windows``),
        drawn with ``[seed, index]``, cached."""
        s, t = self._truth or (None, None)
        if s != seed or (windows and "wkey" not in t):
            t = event_truth(self.cells, self.coeff_cells, np.random.default_rng([seed, self.index]), windows)
            self._truth = (seed, t)
        return t


def dataset(run):
    """The run's corpus (``data.train`` of its config) restricted to the ``probe`` role of its
    holdout, with the coefficients and their cells truth."""
    from helix.data import CoeffTPCDataset
    d = (run.config.get("data") or {}).get("train") or {}
    return CoeffTPCDataset(data_root=d["data_root"], split=d.get("split", ""),
                           dataset_name=d.get("dataset_name", "sim_wire"), modalities=("coeff", "cells"),
                           holdout=d.get("holdout"), split_role="probe")


#: The simulation's per-sample charge cut (e-) of the d1m campaign's hits (JAXTPC f0d2ed39,
#: ``threshold: 25.0``), recorded when a cells shard does not carry its own.
D1M_TRUTH_Q_CUT = 25.0


def identity(ds):
    """What the evaluated shards say they are: ``basis_digest`` (the corpus's DSP identity, from
    the coefficient shards' ``/config``; the dataset refuses shards that disagree on it, so one
    value) and ``truth_q_cut`` (the per-sample charge cut, e-, the cells truth was built under:
    its shards' ``label_json`` ``hits_threshold``, written by :func:`helix.data.cells.build_shard`)
    with where that came from. ``None`` where the dataset cannot say (a plain list in tests)."""
    out = dict(basis_digest=None, truth_q_cut=None, truth_q_cut_source=None)
    r = getattr(ds, "coeff_reader", None)
    if r is not None:
        meta = getattr(r, "_meta_by_path", {})
        dig = sorted({_text(m.get("basis_digest")) for m in meta.values()} - {None})
        out["basis_digest"] = dig[0] if len(dig) == 1 else (dig or None)
    c = getattr(ds, "cells_reader", None)
    if c is not None:
        import json

        import h5py
        cuts = set()
        for p in c.h5_files:
            with h5py.File(p, "r") as f:
                cuts.add(json.loads(_text(f["config"].attrs.get("label_json")) or "{}").get("hits_threshold"))
        cuts.discard(None)
        if cuts:
            cuts = sorted(cuts)
            out.update(truth_q_cut=cuts[0] if len(cuts) == 1 else cuts,
                       truth_q_cut_source="cells /config label_json hits_threshold")
        else:   # TODO: a cells shard that records no cut; every d1m shard so far was built under 25 e-
            out.update(truth_q_cut=D1M_TRUTH_Q_CUT, truth_q_cut_source="not recorded by the shards: the d1m default")
    return out


def _text(x):
    if x is None:
        return None
    return x.decode() if isinstance(x, bytes) else str(x)


def events(ds, tokenizer, device, max_events=None):
    """:class:`Event` for each of the first ``max_events`` events of ``ds``, in its order."""
    import torch
    from helix.data.cells import coeff_amplitude
    from helix.model.tokenize import CoeffTokenize

    tokenize = CoeffTokenize(part="coeff", cfg=tokenizer)          # the training transform itself
    n = len(ds) if not max_events else min(int(max_events), len(ds))
    for i in range(n):
        s = ds.get_data(i)
        c = dict(s["coeff"])
        meta = c["_meta"]
        fm = tokenize(dict(s))["coeff"]
        B = {k: torch.as_tensor(v).to(device) for k, v in fm.items() if isinstance(v, np.ndarray)}
        B["n_cells"] = B["plane_id"].shape[0]
        amp = coeff_amplitude(c, tokenizer)
        yield Event(index=i, cells=s["cells"], coeff_cells=amp[0], coeff_amp=amp,
                    band_lengths=np.asarray(meta["band_lengths"]), cell_key=np.asarray(fm["cell_key"]), batch=B)


# ---- windows ---------------------------------------------------------------------------------

WW, WT = 32, 128                     # window, wires x raw ticks
ENERGY_BINS = (0.0, 0.1, 0.2, 0.5, 1e9)                  # MeV
#: per event: signal map rows at most, iso windows at most, bg windows per plane at most
CAP_SIGNAL, N_ISO, N_BG_PLANE = 3000, 60, 10
KINDS = ("iso", "bg", "near", "isox")
VIEWS = ("U", "V", "Y")
ENERGY_NAMES = tuple(f"{ENERGY_BINS[i]:g}-{ENERGY_BINS[i + 1]:g}" for i in range(len(ENERGY_BINS) - 1))


def window_cells(g, w0, t0):
    """The 16 x 8 fine-cell keys of the window at pixel origin (w0, t0), wire-major."""
    fw = w0 // FW + np.arange(WW // FW)
    ft = t0 // FT + np.arange(WT // FT)
    W, T = np.meshgrid(fw, ft, indexing="ij")
    return fkey(np.full(W.size, g), W.ravel(), T.ravel())


def event_truth(cells, coeff_cells, rng, windows=True):
    """The map rows and windows of one event, from its cells record.

    ``cells``: a :mod:`helix.data.cells` record (tracks numbered by row, energies in
    ``trk_energy``). ``coeff_cells``: fine-cell keys of the event's kept coefficients (any band,
    :func:`helix.data.cells.coeff_cells`). Returns ``mrows``/``mq`` (map rows and their truth
    charge) and, with ``windows``, ``wkey`` (window cells in window order, :func:`window_cells`'
    order inside each) and ``meta`` (one dict per window: ``kind``, ``g``, ``w0``, ``t0``, ``E``
    -- the iso particle's energy, NaN for a background window, ``bg`` or ``near``; ``ctx`` -- for
    an iso window ``clear`` / ``crowded`` / ``mixed`` by other particles' charge around it, as in
    the code below, None otherwise). The map
    rows are drawn first, so they are the same with or without the windows.
    """
    E = dict(enumerate(np.asarray(cells["trk_energy"], np.float64).tolist()))
    uk = np.asarray(cells["key"], np.int64)
    fq = np.asarray(cells["q"], np.float64)
    ck = np.unique(np.asarray(coeff_cells, np.int64))

    # map rows: signal, near signal, random kept coefficients
    sig = uk[fq > 0]
    if len(sig) > CAP_SIGNAL:
        sig = rng.choice(sig, CAP_SIGNAL, replace=False)
    sg, sw, st = unkey(sig)
    near = fkey(sg, np.maximum(sw + rng.integers(-8, 9, len(sig)), 0), st + rng.integers(-8, 9, len(sig)))
    near = near[np.isin(near, ck)]
    coeff = rng.choice(ck, min(len(sig) // 2 + 1, len(ck)), replace=False) if len(ck) else ck
    mrows = np.unique(np.concatenate([sig, near, coeff]))
    o = np.argsort(uk)
    uk_s, fq_s = uk[o], fq[o]
    if len(uk_s):
        pos = np.clip(np.searchsorted(uk_s, mrows), 0, len(uk_s) - 1)
        mq = np.where(uk_s[pos] == mrows, fq_s[pos], 0.0)
    else:
        mq = np.zeros(len(mrows))
    if not windows:
        return dict(mrows=mrows, mq=mq)

    # One row per (cell, particle), placed at the cell's origin. Every box below is grid-aligned
    # (origins are multiples of FW / FT, paddings too), so a row is inside a box exactly when its
    # pixels are.
    G, cw_, ct_ = unkey(uk[np.asarray(cells["ct_cell"], np.int64)])
    W, T = cw_ * FW, ct_ * FT
    Q, TRK = np.asarray(cells["ct_q"], np.float64), np.asarray(cells["ct_track"], np.int64)

    # per (plane, particle): charge, and its pixel charge centroid (stored: finer than a cell)
    pu, pinv = np.unique((G.astype(np.int64) << 40) | TRK, return_inverse=True)
    pinv = pinv.ravel()
    pq = np.bincount(pinv, weights=Q)
    pcw, pct = np.asarray(cells["pt_wire"], np.float64), np.asarray(cells["pt_tick"], np.float64)

    # window queries are searchsorted slices returning row INDICES; an event-sized mask per
    # query saturates memory bandwidth across workers
    o_gw = np.lexsort((W, G))
    GW = G[o_gw].astype(np.int64) * (1 << 24) + W[o_gw]

    def contents(g, w0, t0, pad_w=0, pad_t=0):
        lo = np.searchsorted(GW, g * (1 << 24) + (w0 - pad_w))
        hi = np.searchsorted(GW, g * (1 << 24) + (w0 + WW + pad_w))
        idx = o_gw[lo:hi]
        return idx[(T[idx] >= t0 - pad_t) & (T[idx] < t0 + WT + pad_t)]

    def origin(cw, ct):            # clamped at 0: a negative wire corrupts the packed key
        return max(0, int(round((cw - WW / 2) / FW)) * FW), max(0, int(round((ct - WT / 2) / FT)) * FT)

    def iso_window(j):
        """Particle ``j``'s window at a jittered origin if it passes the iso criteria (>= 90% of
        its plane charge inside, < 5% of the window foreign), else None; with ``ctx``, where it
        sits relative to OTHER particles' charge, for the matched comparison: "clear" none in the
        padding a far (bg) window must keep empty (its own particle may reach there, which a far
        window never has); "crowded" none within the window and a one-cell margin but some in
        the padding -- the near windows' own rule; "mixed" other charge in or right beside the
        window (an iso window allows 5% inside), in neither."""
        g = int(pu[j] >> 40)
        w0, t0 = origin(pcw[j] + rng.uniform(-6, 6), pct[j] + rng.uniform(-32, 32))
        idx = contents(g, w0, t0)
        if not len(idx):
            return None
        mine = pinv[idx] == j
        qin, qall = Q[idx[mine]].sum(), Q[idx].sum()
        if qin < 0.9 * pq[j] or qall - qin > 0.05 * qall:
            return None
        other = lambda pw, pt: bool((pinv[contents(g, w0, t0, pw, pt)] != j).any())   # noqa: E731
        ctx = "clear" if not other(16, 128) else "mixed" if other(FW, FT) else "crowded"
        return g, w0, t0, E.get(int(TRK[idx[mine][0]]), 0.0), ctx

    wins, has_clear = [], set()
    good = np.nonzero(pq > 0)[0]
    rng.shuffle(good)
    for j in good[:4000]:
        if len(wins) >= N_ISO:
            break
        w = iso_window(j)
        if w is not None:
            wins.append(("iso",) + w)
            if w[4] == "clear":
                has_clear.add(int(j))

    if len(ck):
        cg = unkey(ck)[0]
        for g in np.unique(cg).tolist():
            n_b = 0
            for x in rng.permutation(ck[cg == g])[:500].tolist():
                if n_b >= N_BG_PLANE:
                    break
                _, fw, ft = (int(v) for v in unkey(x))
                w0, t0 = max(0, fw * FW - WW // 2), max(0, ft * FT - WT // 2)
                if len(contents(g, w0, t0, 16, 128)):
                    continue
                wins.append(("bg", g, w0, t0, np.nan, None))
                n_b += 1

    # near background: drawn last, so every window above is the same with or without it.
    # Oversampled, then kept only with a kept coefficient inside, as every bg window has: the two
    # backgrounds then differ in their distance to charge alone.
    from helix.data.denoise import near_windows
    planes = np.unique(unkey(uk[fq > 0])[0])
    n_near = dict.fromkeys(planes.tolist(), 0)
    for g, w0, t0 in (near_windows(uk, fq, rng, n=4 * N_BG_PLANE * len(planes)) if len(planes) and len(ck) else []):
        if n_near[g] >= N_BG_PLANE or not np.isin(window_cells(g, w0, t0), ck).any():
            continue
        wins.append(("near", g, w0, t0, np.nan, None))
        n_near[g] += 1

    # extra CLEAR iso windows ("isox"): drawn last, so every window above is the same with or without
    # them, and kept out of the iso numbers. Clear windows are ~6% of iso ones, too few for the
    # matched comparison below 0.1 MeV; one more try per particle without a clear window, at most
    # N_ISO per event.
    n_x = 0
    for j in good.tolist():
        if n_x >= N_ISO:
            break
        if j in has_clear:
            continue
        w = iso_window(j)
        if w is not None and w[4] == "clear":
            wins.append(("isox",) + w)
            n_x += 1

    wkey = np.concatenate([window_cells(g, w0, t0) for _, g, w0, t0, _, _ in wins]) if wins \
        else np.zeros(0, np.int64)
    meta = [dict(kind=k, g=g, w0=w0, t0=t0, E=e, ctx=c) for k, g, w0, t0, e, c in wins]
    return dict(mrows=mrows, mq=mq, wkey=wkey, meta=meta)


def _ratio(a, b):
    return float(a / b) if b else float("nan")


def _frac(w, hit, sel):
    return _ratio((w * hit)[sel].sum(), w[sel].sum())


def window_stat(score, meta):
    """``(stat, n_events, n_windows)`` of one per-cell ``score`` (flat over the windows' cells in
    window order; larger = more charge): ``stat(m)`` the floor's numbers under per-event weights
    ``m`` (:func:`bootstrap`), keyed by ``<E bin>`` and, where per view, ``_<view>``:

      floor_eff1pct_<E>[_<V>]       iso efficiency, threshold each view's 99th percentile of its
                                    FAR (``bg``) windows -- the floor itself
      floor_eff1pct_<E>_ev          the same, each event's iso windows of the bin weighted
                                    equally (pooled: an event with more windows weighs more)
      floor_eff1pct_near_<E>[_<V>]  threshold set on the NEAR windows instead; a view without
                                    near windows is left out
      near_fpr_at_far[_<V>]         the fraction of near windows above the far threshold
      auc_<E>[_<V>]                 P(iso scores above a far window of its view), ties 1/2
      floor_eff1pct_clear_<E>       MATCHED: iso windows with no other charge in the padding
                                    (``ctx`` clear) at the far threshold
      floor_eff1pct_crowded_near_<E>  MATCHED: iso windows by the near windows' own rule (other
                                    charge in the padding, none within a cell) at the near threshold
      floor_eff1pct_crowded_<E>     those crowded iso windows at the far threshold, for contrast
      floor_eff1pct_mixed_<E>       the mixed iso windows (other charge in or beside the window,
                                    which a detection may be crediting) at the far threshold:
                                    all-iso is the count-weighted sum of clear, crowded and mixed
                                    (clear among the iso windows, ``n_windows`` ``iso_clear_*``;
                                    the clear efficiency itself also scores the extra clear draws,
                                    ``isox``, which no other number includes)
      auc_clear_<E>, auc_crowded_near_<E>  the matched pairs threshold-free: clear iso against far,
                                    crowded iso against near windows, within each view
      (the two matched efficiencies also per view, ``_<V>``; ``n_windows`` counts each context
      per view and per energy bin)

    ``meta``: one dict per window, its event as ``ev``."""
    n = len(meta)
    s = np.asarray(score, np.float64).reshape(n, -1).max(1) if n else np.zeros(0)
    ev = np.array([m["ev"] for m in meta], np.int64)
    kind = np.array([KINDS.index(m["kind"]) for m in meta], np.int64)
    view = np.array([m["g"] % 3 for m in meta], np.int64)
    eb = np.array([np.searchsorted(ENERGY_BINS, m["E"], side="right") - 1 if m["kind"] in ("iso", "isox") else -1
                   for m in meta], np.int64)
    evs, we = np.unique(ev, return_inverse=True)
    iso, bg, near, isox = kind == 0, kind == 1, kind == 2, kind == 3
    ctx = np.array([m.get("ctx") or "" for m in meta])
    # clear: the iso windows that are, plus the extra clear draws (isox); crowded and mixed are iso only
    clear = ((ctx == "clear") & iso) | isox
    crowded, mixed = (ctx == "crowded") & iso, (ctx == "mixed") & iso
    # per-event equal weights of the iso windows: 1 / (the event's iso windows in that energy bin)
    cell = we * len(ENERGY_NAMES) + np.maximum(eb, 0)
    cnt = np.bincount(cell[iso], minlength=len(evs) * len(ENERGY_NAMES))
    inv_n = np.where(iso, 1.0 / np.maximum(cnt[cell], 1), 0.0)
    # AUC: per view, the far windows in score order and where each iso score falls among them
    # (fixed); a resample only re-weighs, so a statistic is a cumulative sum, not a sort
    def pairs(ref, sel):
        """Per view: ``ref``'s windows in score order and where each ``sel`` score falls among them."""
        out = []
        for v in range(3):
            b = np.nonzero(ref & (view == v))[0]
            b = b[np.argsort(s[b], kind="stable")]
            i = np.nonzero(sel & (view == v))[0]
            out.append((b, i, np.searchsorted(s[b], s[i], "left"), np.searchsorted(s[b], s[i], "right")))
        return out
    auc = pairs(bg, iso)
    auc_ctx = {"clear": pairs(bg, clear), "crowded_near": pairs(near, crowded)}

    def thresholds(w, ref):
        thr = np.full(3, np.nan)
        for v in range(3):
            b = np.repeat(s[ref & (view == v)], w[ref & (view == v)])
            if len(b):
                thr[v] = np.quantile(b, 0.99)
        return thr

    def stat(m):
        w = m[we]
        far = thresholds(w, bg)
        det = s > np.where(np.isnan(far), np.inf, far)[view]          # a view without bg: none detected
        thr_n = thresholds(w, near)
        has_n = ~np.isnan(thr_n)[view]
        det_n = s > np.where(np.isnan(thr_n), np.inf, thr_n)[view]
        # per view and iso window: the weight of far windows it beats (+1/2 the ties), and of all
        def beats(ps):
            bt, tt = np.zeros(n), np.zeros(n)
            for b, i, lo, hi in ps:
                cw = np.concatenate([[0.0], np.cumsum(w[b])])
                bt[i], tt[i] = (cw[lo] + cw[hi]) / 2, cw[-1]
            return bt, tt
        beat, tot = beats(auc)
        ctx_beats = {k: beats(ps) for k, ps in auc_ctx.items()}
        out = {}
        for k, nm in enumerate(ENERGY_NAMES):
            sel = iso & (eb == k)
            out[f"floor_eff1pct_{nm}"] = _frac(w, det, sel)
            out[f"floor_eff1pct_{nm}_ev"] = _frac(w * inv_n, det, sel)
            out[f"floor_eff1pct_near_{nm}"] = _frac(w, det_n, sel & has_n)
            selc = (iso | isox) & (eb == k)                            # the matched sets' energy bin
            out[f"floor_eff1pct_clear_{nm}"] = _frac(w, det, selc & clear)
            out[f"floor_eff1pct_crowded_near_{nm}"] = _frac(w, det_n, sel & crowded & has_n)
            out[f"floor_eff1pct_crowded_{nm}"] = _frac(w, det, sel & crowded)
            out[f"floor_eff1pct_mixed_{nm}"] = _frac(w, det, sel & mixed)
            for key, m in (("clear", clear), ("crowded_near", crowded)):
                bt, tt = ctx_beats[key]
                out[f"auc_{key}_{nm}"] = _ratio((w * bt)[selc & m].sum(), (w * tt)[selc & m].sum())
            for v in range(3):
                sv = sel & (view == v)
                out[f"floor_eff1pct_clear_{nm}_{VIEWS[v]}"] = _frac(w, det, selc & (view == v) & clear)
                out[f"floor_eff1pct_crowded_near_{nm}_{VIEWS[v]}"] = _frac(w, det_n, sv & crowded & has_n)
            for v in range(3):
                sv = sel & (view == v)
                out[f"floor_eff1pct_{nm}_{VIEWS[v]}"] = _frac(w, det, sv)
                out[f"floor_eff1pct_near_{nm}_{VIEWS[v]}"] = _frac(w, det_n, sv & has_n)
                out[f"auc_{nm}_{VIEWS[v]}"] = _ratio((w * beat)[sv].sum(), (w * tot)[sv].sum())
            out[f"auc_{nm}"] = _ratio((w * beat)[sel].sum(), (w * tot)[sel].sum())
        out["near_fpr_at_far"] = _frac(w, det, near)
        for v in range(3):
            out[f"near_fpr_at_far_{VIEWS[v]}"] = _frac(w, det, near & (view == v))
        return out

    counts = {f"{k}_{VIEWS[v]}": int(((kind == i) & (view == v)).sum())
              for i, k in enumerate(KINDS) for v in range(3)}
    ctxs = (("clear", clear & iso), ("crowded", crowded), ("mixed", mixed))   # iso windows; isox apart
    counts.update({f"iso_{c}_{VIEWS[v]}": int((m & (view == v)).sum()) for c, m in ctxs for v in range(3)})
    counts.update({f"iso_{c}_{nm}": int((m & (eb == k)).sum()) for c, m in ctxs for k, nm in enumerate(ENERGY_NAMES)})
    counts["isox"] = int(isox.sum())
    return stat, len(evs), counts


def score_windows(score, meta, n_boot=200, seed=0):
    """The floor of one per-cell ``score``: :func:`window_stat`'s numbers (each ``[point, lo,
    hi]``) and the window counts ``n_windows``."""
    stat, n_ev, n_win = window_stat(score, meta)
    out = bootstrap(stat, n_ev, n_boot, seed)
    out["n_windows"] = n_win
    return out


def seed_spread(points):
    """Over the probe seeds: ``{"seed_mean": {k: mean}, "seed_half_range": {k: (max - min) / 2}}``
    of each key's point value, one dict per seed -- the fit's own variance, which the event
    bootstrap (one fitted probe, resampled events) does not contain."""
    out = {"seed_mean": {}, "seed_half_range": {}}
    for k in points[0]:
        v = np.array([p[k] for p in points], np.float64)
        ok = v[np.isfinite(v)]
        out["seed_mean"][k] = round(float(ok.mean()), 4) if len(ok) else float("nan")
        out["seed_half_range"][k] = round(float(ok.max() - ok.min()) / 2, 4) if len(ok) else float("nan")
    return out


def plane_group(e, g):
    """One id per (event, plane gid); :func:`group_event` inverts it."""
    return np.asarray(e, np.int64) * 16 + np.asarray(g, np.int64)


def group_event(k):
    return np.asarray(k, np.int64) // 16


def group_r2(p, y, group):
    """Per group: ``(keys, sse, sst)`` of prediction ``p`` against truth ``y`` once both are
    centred within the group, so a per-group offset costs nothing. The pooled within-group
    ``R^2 = 1 - sum(sse) / sum(sst)``; unlike a correlation it also charges a prediction of the
    wrong scale."""
    keys, inv = np.unique(np.asarray(group), return_inverse=True)
    inv, G = inv.ravel(), len(keys)
    n = np.bincount(inv, minlength=G)
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    dp = p - (np.bincount(inv, p, G) / n)[inv]
    dy = y - (np.bincount(inv, y, G) / n)[inv]
    return keys, np.bincount(inv, (dy - dp) ** 2, G), np.bincount(inv, dy * dy, G)


def bootstrap(stat, n_events, n_boot=200, seed=0):
    """``{key: [point, lo, hi]}``: ``stat(m) -> {key: value}`` of per-event weights ``m`` (the
    point: every event once), and its [16, 84]% interval over ``n_boot`` resamples of the
    events with replacement (``m``: how often each was drawn)."""
    point = stat(np.ones(n_events, np.int64))
    rng = np.random.default_rng(seed)
    boots = [stat(np.bincount(rng.integers(0, n_events, n_events), minlength=n_events)) for _ in range(n_boot)]
    out = {}
    for k, v in point.items():
        bv = np.array([b.get(k, np.nan) for b in boots], float)
        lo, hi = np.nanpercentile(bv, [16, 84]) if np.isfinite(bv).any() else (np.nan, np.nan)
        out[k] = [round(float(v), 4), round(float(lo), 4), round(float(hi), 4)]
    return out
