"""The evaluated events: a run's held-out split, tokenized once, with its cells truth.

Every metric reads the same :class:`Event` objects, so an event is read, tokenized and moved to
the device once, and its windows (and, for the along-wire probe, the encoder's frozen features)
are computed once per event however many metrics use them.

WINDOWS
=======
The evaluation windows and the detection floor, on the fixed fine grid.

Every model is scored on the same 2-wire x 16-tick cells (:mod:`helix.data.cells`) whatever its
patch size, so a bigger patch cannot lose sub-patch detail without the score moving. A cell no
token covers is scored as its predictor's floor (0, or a large negative log-odds), never dropped:
which cells are scored never depends on the tokenizer.

Windows, per event, from the cells record (no tokenizer, no checkpoint): 32 wires x 128 ticks
(16 x 8 fine cells), all cells kept:

  iso   one isolated, compact particle (>=90% of its plane charge inside, <5% of the window from
        anything else), placed at a RANDOM offset around it (a window centred on truth would put
        a blurred prediction's centre on the answer); ``ctx`` says where it sits relative to
        OTHER particles' charge: ``clear`` (none in the padding a far window keeps empty),
        ``crowded`` (some in that padding, none within a cell: the near windows' own rule) or
        ``mixed`` (other charge in or right beside the window; not scored)
  bg    the FAR background: origin uniform over the plane's kept-coefficient extent, no truth
        charge within +-16 w / +-128 t and at least one kept coefficient inside; up to
        :data:`N_BG_PLANE` per plane, so every view has its own background
  near  the NEAR background: no truth charge in the window or within one cell of it, but some
        within +-16 w / +-128 t -- exactly the surroundings ``bg`` must keep empty
        (:func:`helix.data.denoise.near_windows`, the denoiser's own training windows) -- and at
        least one kept coefficient inside, as every ``bg`` window has (both uniform in area: they
        differ in distance to charge alone); up to :data:`N_BG_PLANE` per plane
  isox  extra CLEAR iso windows (clear windows are ~6% of iso ones, too few below 0.1 MeV): they
        count in the clear efficiency and nowhere else

Each kind is drawn after the ones above it, so earlier windows do not depend on later kinds.

The numbers (:func:`window_stat`, defined there): a window's score is its largest cell score;
the MATCHED pairs are iso against background at the same distance from charge -- clear iso
against the FAR windows, crowded iso against the NEAR windows (the far background alone lets a
prediction that hazes charge around tracks gain efficiency at no cost; the near background
prices the haze) -- each view against its OWN background (the views differ in noise: a pooled
threshold is set by the views with most background; Y ran at 2.4-3.2% under a pooled 1%).

The headline is the LAUC: the efficiency averaged over log10(false-positive rate) in [1e-3,
1e-1], view-mean, of the two pairs at 0.1-0.2 MeV (:data:`HEADLINE`). Beside it, always: the
per-view curve at 1e-3 ... 1e-1, the efficiency at 1% (the old headline, :data:`SECONDARY`), the
energy at 50% efficiency at 1% (E50) and the share of positive windows holding a kept
coefficient. Intervals are central 95% over an event bootstrap (:func:`bootstrap`; with several
training seeds, seeds then events).

The area under the whole ROC is NOT reported: every background window holds a kept coefficient by
construction while many low-energy signal windows hold none (d1m, 600 events: 28-38% of the clear
windows below 0.1 MeV, 9-14% at 0.1-0.2 MeV; ``frac_kept``), so for the amplitude those positives
sit below every background window whatever their charge, and the full-range area measures that.

numpy at import; torch and pimm-data only when events are read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from helix.data.cells import FT, FW, fkey, unkey

@dataclass
class Event:
    index: int                 # its position in the dataset (a rejected position is skipped, not refilled)
    cells: dict                # helix.data.cells record
    coeff_cells: np.ndarray    # fine cells of the kept coefficients (helix.model.cells_join.coeff_cells)
    coeff_amp: tuple           # (those cells, their largest |coefficient| / sigma): coeff_amplitude
    band_lengths: np.ndarray
    cell_key: np.ndarray       # per token, the tokenizer's cell key (numpy)
    batch: dict                # the FM's batch (torch, on the device)
    _feats: object = field(default=None, repr=False)
    _raw: object = field(default=None, repr=False)
    _truth: tuple = field(default=None, repr=False)         # (seed, truth)

    def raw_inputs(self):
        """Per token, the tokenizer's own input to the encoder -- ``inp`` and ``occ``
        (``n_slot`` each), concatenated, float32, cached: what the along-wire probe's ``raw``
        arm reads in place of encoder features."""
        if self._raw is None:
            import torch
            self._raw = torch.cat([self.batch["inp"].float(), self.batch["occ"].float()], 1)
        return self._raw

    def features(self, model):
        """``model.encode(batch)`` (all tokens visible), float32, cached for THAT model: asked
        for another model's features, the event encodes again rather than return the first's."""
        if self._feats is None or self._feats[0] is not model:
            import torch
            from helix.eval.recon import autocast
            with torch.no_grad(), autocast(self.batch["plane_id"].device):
                self._feats = (model, model.encode(self.batch).float())
        return self._feats[1]

    def truth(self, seed):
        """:func:`event_truth` of the event, drawn with ``[seed, index]``, cached."""
        s, t = self._truth or (None, None)
        if s != seed:
            t = event_truth(self.cells, self.coeff_cells, np.random.default_rng([seed, self.index]))
            self._truth = (seed, t)
        return t


def dataset(run, data_root=None):
    """The run's corpus (``data.train`` of its config) restricted to the ``probe`` role of its
    holdout, with the coefficients and their cells truth. ``data_root``: where that corpus is
    at this site (:func:`helix.data.verify.resolve_corpus`); default the run's recorded path."""
    from helix.data import CoeffTPCDataset
    d = (run.config.get("data") or {}).get("train") or {}
    return CoeffTPCDataset(data_root=str(data_root or d["data_root"]), split=d.get("split", ""),
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
        meta = getattr(r, "_sig_by_path", {})            # path -> helix.data.verify.shard_signature
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


def events(ds, tokenizer, device, max_events=None, rejected=None):
    """:class:`Event` for each of the first ``max_events`` positions of ``ds``, in its order,
    ``Event.index`` the position.

    A position the dataset REJECTS (:meth:`helix.data.CoeffTPCDataset._get_data` gives None: its
    truth is off its planes) is skipped and its position appended to ``rejected``. ``get_data``
    is not used: it replaces a rejected event by the next one -- right for a DataLoader, which
    must never fail an index, and wrong here, where that next event would be scored twice."""
    import torch
    from helix.model.cells_join import coeff_amplitude
    from helix.model.tokenize import CoeffTokenize

    tokenize = CoeffTokenize(part="coeff", cfg=tokenizer)          # the training transform itself
    read = getattr(ds, "_get_data", None) or ds.get_data
    n = len(ds) if not max_events else min(int(max_events), len(ds))
    for i in range(n):
        s = read(i)
        if s is None:
            if rejected is not None:
                rejected.append(i)
            continue
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
#: per event: iso windows at most, bg (and near) windows per plane at most
N_ISO, N_BG_PLANE = 60, 10
N_BG_TRIES = 4000                    # far-background draws per plane
KINDS = ("iso", "bg", "near", "isox")
VIEWS = ("U", "V", "Y")
ENERGY_NAMES = tuple(f"{ENERGY_BINS[i]:g}-{ENERGY_BINS[i + 1]:g}" for i in range(len(ENERGY_BINS) - 1))


def window_cells(g, w0, t0):
    """The 16 x 8 fine-cell keys of the window at pixel origin (w0, t0), wire-major."""
    fw = w0 // FW + np.arange(WW // FW)
    ft = t0 // FT + np.arange(WT // FT)
    W, T = np.meshgrid(fw, ft, indexing="ij")
    return fkey(np.full(W.size, g), W.ravel(), T.ravel())


def cell_charge(cells, keys):
    """The truth charge (e-) of fine cells ``keys`` from a cells record, 0 where it has none."""
    uk, fq = np.asarray(cells["key"], np.int64), np.asarray(cells["q"], np.float64)
    keys = np.asarray(keys, np.int64)
    if not len(uk):
        return np.zeros(len(keys))
    o = np.argsort(uk)
    uk_s, fq_s = uk[o], fq[o]
    pos = np.clip(np.searchsorted(uk_s, keys), 0, len(uk_s) - 1)
    return np.where(uk_s[pos] == keys, fq_s[pos], 0.0)


def event_truth(cells, coeff_cells, rng):
    """The windows of one event, from its cells record.

    ``cells``: a :mod:`helix.data.cells` record (tracks numbered by row, energies in
    ``trk_energy``). ``coeff_cells``: fine-cell keys of the event's kept coefficients (any band,
    :func:`helix.model.cells_join.coeff_cells`). Returns ``wkey`` (window cells in window order,
    :func:`window_cells`' order inside each) and ``meta`` (one dict per window: ``kind``, ``g``,
    ``w0``, ``t0``, ``E`` -- the iso particle's energy, NaN for a background window; ``ctx`` --
    for an iso window ``clear`` / ``crowded`` / ``mixed`` by other particles' charge around it,
    None otherwise).
    """
    E = dict(enumerate(np.asarray(cells["trk_energy"], np.float64).tolist()))
    uk = np.asarray(cells["key"], np.int64)
    fq = np.asarray(cells["q"], np.float64)
    ck = np.unique(np.asarray(coeff_cells, np.int64))

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

    # far background: origins UNIFORM over the plane's kept-coefficient extent (on the cell grid),
    # kept with no truth charge within +-16 w / +-128 t and >= 1 kept coefficient inside -- the
    # near windows' rule (uniform in area, then >= 1 kept), so the two backgrounds differ in their
    # distance to charge alone. Centring on a random kept cell instead weights a window by its
    # kept count, so the far tail leans to dense kept clusters.
    if len(ck):
        cg, cw, ct = unkey(ck)
        for g in np.unique(cg).tolist():
            m = cg == g
            wl, wh = int(cw[m].min()) - WW // FW + 1, int(cw[m].max())
            tl, th = int(ct[m].min()) - WT // FT + 1, int(ct[m].max())
            n_b = 0
            for _ in range(N_BG_TRIES):
                if n_b >= N_BG_PLANE:
                    break
                w0, t0 = int(rng.integers(wl, wh + 1)) * FW, int(rng.integers(tl, th + 1)) * FT
                if w0 < 0 or t0 < 0 or len(contents(g, w0, t0, 16, 128)) \
                        or not np.isin(window_cells(g, w0, t0), ck).any():
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
    # kept: the window holds at least one kept coefficient (every bg and near window does, by
    # construction; an iso window without one is beyond the amplitude's reach)
    kept = np.isin(wkey, ck).reshape(len(wins), -1).any(1) if wins else np.zeros(0, bool)
    meta = [dict(kind=k, g=g, w0=w0, t0=t0, E=e, ctx=c, kept=bool(kp))
            for (k, g, w0, t0, e, c), kp in zip(wins, kept)]
    return dict(wkey=wkey, meta=meta)


def _diff(a, b):
    return {k: a[k] - b[k] for k in a}


#: The headline's false-positive range: efficiency averaged over log10(FPR) in [1e-3, 1e-1].
LAUC_RANGE = (1e-3, 1e-1)
#: Where the LAUC integrand is evaluated: 41 points uniform in log10(FPR) over LAUC_RANGE.
LAUC_GRID = np.logspace(np.log10(LAUC_RANGE[0]), np.log10(LAUC_RANGE[1]), 41)
#: The operating points of the per-view curves (each view at its own threshold).
CURVE_FPRS = (1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1)
#: The fine energy bins (MeV) of the turn-on, E50: 15 bins uniform in log E over 0.01-10 MeV.
E50_BINS = np.logspace(-2, 1, 16)
#: The false-positive rate E50 is quoted at, and the (weighted) windows a fine bin needs to count.
E50_FPR, E50_MIN = 1e-2, 20
#: The bootstrap interval's coverage: [2.5, 97.5] percentiles.
LEVEL = 0.95
#: The matched pairs: (name, positives, background kind). See :func:`window_stat`.
POPS = (("clear", "far"), ("crowded_near", "near"))
_FPRS = np.concatenate([LAUC_GRID, CURVE_FPRS])           # every threshold one resample needs
_DLOG = np.diff(np.log10(LAUC_GRID))


def _fpr_name(f):
    return f"{f:g}"


def wquantile(xs, cw, q):
    """The ``q`` quantiles of sorted values ``xs`` carrying integer weights whose cumulative sum
    is ``cw``: exactly ``np.quantile(np.repeat(xs, w), q)`` (linear interpolation), without the
    repeat. A bootstrap resample weighs a window by its event's draw count."""
    n = cw[-1]
    h = (n - 1) * np.asarray(q, np.float64)
    lo = np.floor(h)
    frac = h - lo
    i = np.searchsorted(cw, lo, side="right")
    j = np.searchsorted(cw, np.minimum(lo + 1, n - 1), side="right")
    return xs[i] + frac * (xs[j] - xs[i])


def _above(ps, cw, t):
    """Of sorted scores ``ps`` with cumulative weights ``cw`` (0 prepended): the weight strictly
    above each threshold ``t``."""
    return cw[-1] - cw[np.searchsorted(ps, t, side="right")]


def e50(eff, centers):
    """The energy at which the turn-on ``eff`` (per fine bin, NaN where a bin has too few windows)
    first reaches 50%, interpolated linearly in log E between that bin's centre and the populated
    bin below it. NaN when no bin reaches 50% or the lowest populated bin already does (the
    crossing is outside the bins)."""
    ok = np.isfinite(eff)
    e, c = eff[ok], centers[ok]
    hit = np.nonzero(e >= 0.5)[0]
    if not len(hit) or hit[0] == 0:
        return float("nan")
    i = hit[0]
    x = np.interp(0.5, [e[i - 1], e[i]], np.log([c[i - 1], c[i]]))
    return float(np.exp(x))


def window_stat(score, meta, events=None):
    """``(stat, n_events, n_windows)`` of one per-cell ``score`` (flat over the windows' cells in
    window order; larger = more charge): ``stat(m)`` the numbers under per-event weights ``m``
    (:func:`bootstrap`; one weight per event of ``events``, default every event a window names,
    in sorted order). A window's score is its largest cell score.

    Two matched pairs (:data:`POPS`), each scored per view (U, V, Y; both volumes) against that
    view's OWN background, since the views differ in noise:

      clear          clear iso windows (and the extra clear draws, ``isox``) against the FAR
                     windows (``bg``)
      crowded_near   crowded iso windows against the NEAR windows; a view without near windows
                     is left out

    At a false-positive rate ``f`` the threshold is the ``1 - f`` quantile of the view's
    background window scores (``np.quantile``, linear), and the efficiency the share of the
    positives scoring strictly above it. Keys, each with a view suffix -- ``_U`` / ``_V`` /
    ``_Y``, ``_mean`` (the mean of the three views) or none (pooled over views, each at its own
    threshold) -- and per energy bin ``<E>`` of the iso particle's deposited energy:

      lauc_<pop>_<E>             THE HEADLINE (``lauc_clear_0.1-0.2_mean`` and
                                 ``lauc_crowded_near_0.1-0.2_mean``): the efficiency averaged
                                 over log10(FPR) in [1e-3, 1e-1] (:data:`LAUC_RANGE`) -- the
                                 trapezoid integral over the 41 points of :data:`LAUC_GRID`,
                                 divided by the two decades. One operating point ranks similar
                                 arms differently from the next; the average over the range
                                 where the pretraining gain sits does not depend on one choice
      eff<f>_<pop>_<E>           the efficiency at each FPR ``f`` of :data:`CURVE_FPRS`: the
                                 per-view curve (1e-3 rests on ~0.001 x the view's background
                                 windows: ~80 at 4113 events, ~12 at 600)
      floor_eff1pct_<pop>_<E>    the efficiency at 1% FPR: the pre-2026-11-16 headline, kept so
                                 past runs compare (equal to ``eff0.01_<pop>_<E>``)
      e50_<pop>                  the deposited energy (MeV) at which the efficiency at 1% FPR
                                 reaches 50% (:func:`e50` over :data:`E50_BINS`)
      frac_kept_<pop>_<E>        the share of the positive windows holding at least one kept
                                 coefficient (only when the windows record ``kept``): the
                                 amplitude's ceiling. Every background window holds one by
                                 construction, which is why the area under the whole ROC is not
                                 reported: a positive with no kept coefficient sits below every
                                 background window for the amplitude
      near_fpr_at_far_<V>        the share of near windows above the view's far 1% threshold
                                 (1% if near and far look alike to the prediction)

    ``n_windows``: each kind per view, and the clear and crowded windows per energy bin and view.
    ``meta``: one dict per window, its event as ``ev``."""
    n = len(meta)
    s = np.asarray(score, np.float64).reshape(n, -1).max(1) if n else np.zeros(0)
    ev = np.array([m["ev"] for m in meta], np.int64)
    kind = np.array([m["kind"] for m in meta], dtype=object)
    view = np.array([m["g"] % 3 for m in meta], np.int64)
    E = np.array([m["E"] if m["kind"] in ("iso", "isox") else np.nan for m in meta], np.float64)
    eb = np.where(np.isfinite(E), np.searchsorted(ENERGY_BINS, np.nan_to_num(E), side="right") - 1, -1)
    fb = np.where(np.isfinite(E), np.searchsorted(E50_BINS, np.nan_to_num(E), side="right") - 1, -1)
    ctx = np.array([m.get("ctx") or "" for m in meta], dtype=object)
    has_kept = n > 0 and all(m.get("kept") is not None and m.get("kept") != "" for m in meta)
    kept = np.array([bool(m["kept"]) for m in meta]) if has_kept else None
    evs = np.unique(ev) if events is None else np.unique(np.asarray(events, np.int64))
    we = np.searchsorted(evs, ev)
    if n and (we >= len(evs)).any() or n and (evs[np.minimum(we, len(evs) - 1)] != ev).any():
        raise ValueError("window_stat: a window's event is not among `events`")
    clear = ((ctx == "clear") & (kind == "iso")) | (kind == "isox")
    crowded = (ctx == "crowded") & (kind == "iso")
    pos_of = {"clear": clear, "crowded_near": crowded}
    neg_of = {"far": kind == "bg", "near": kind == "near"}
    i_far = len(LAUC_GRID) + CURVE_FPRS.index(E50_FPR)       # the 1% threshold (E50, floor_eff1pct, near_fpr)
    e_cent = np.sqrt(E50_BINS[1:] * E50_BINS[:-1])

    def sort(idx):
        o = idx[np.argsort(s[idx], kind="stable")]
        return s[o], we[o], o

    # per (pop, view): the sorted background; per (pop, view, E bin): the sorted positives
    plan = {}
    for pop, bgk in POPS:
        for v in range(3):
            neg = sort(np.nonzero(neg_of[bgk] & (view == v))[0])
            pos = {k: sort(np.nonzero(pos_of[pop] & (view == v) & (eb == k))[0]) for k in range(len(ENERGY_NAMES))}
            allp = np.nonzero(pos_of[pop] & (view == v) & (fb >= 0) & (fb < len(e_cent)))[0]
            plan[pop, v] = (neg, pos, allp)
    near_v = [np.nonzero((kind == "near") & (view == v))[0] for v in range(3)]

    def stat(m):
        mw = np.asarray(m, np.float64)
        w = mw[we]
        out = {}
        far_thr = {}
        for pop, bgk in POPS:
            e1 = {}                                          # E bin -> (n_pos[3], n_above[3, n_fpr])
            kp = {}
            e50v = []
            pooled_turn = np.zeros((2, len(e_cent)))
            for v in range(3):
                (ns, nwe, _), pos, allp = plan[pop, v]
                cw = np.cumsum(mw[nwe])
                if len(ns) and cw[-1] > 0:
                    thr = wquantile(ns, cw, 1.0 - _FPRS)
                elif bgk == "far":
                    thr = np.full(len(_FPRS), np.inf)        # a view without far windows: none detected
                else:
                    thr = None                               # a view without near windows: left out
                if bgk == "far":
                    far_thr[v] = thr
                for k in range(len(ENERGY_NAMES)):
                    ps, pwe, o = pos[k]
                    pw = mw[pwe] if thr is not None else np.zeros(len(ps))     # left out: no positive
                    cwp = np.concatenate([[0.0], np.cumsum(pw)])
                    n_pos = cwp[-1]
                    above = _above(ps, cwp, thr) if thr is not None else np.zeros(len(_FPRS))
                    e1.setdefault(k, ([], []))
                    e1[k][0].append(n_pos)
                    e1[k][1].append(above)
                    if kept is not None:
                        kp.setdefault(k, []).append(((mw[pwe] * kept[o]).sum(), mw[pwe].sum()))
                # the turn-on at E50_FPR over the fine energy bins
                if thr is not None and len(allp):
                    pw = w[allp]
                    det = s[allp] > thr[i_far]
                    num = np.bincount(fb[allp], pw * det, len(e_cent))
                    den = np.bincount(fb[allp], pw, len(e_cent))
                    pooled_turn += [num, den]
                    e50v.append(e50(np.where(den >= E50_MIN, num / np.maximum(den, 1e-12), np.nan), e_cent))
                else:
                    e50v.append(float("nan"))
            for k, nm in enumerate(ENERGY_NAMES):
                n_pos, above = np.array(e1[k][0]), np.array(e1[k][1])            # (3,), (3, n_fpr)
                with np.errstate(invalid="ignore", divide="ignore"):
                    eff_v = above / n_pos[:, None]                                # NaN where no positive
                    eff_p = above.sum(0) / n_pos.sum()
                for tag, curve in [("", eff_p)] + [(f"_{VIEWS[v]}", eff_v[v]) for v in range(3)] + [("_mean", eff_v.mean(0))]:
                    g = curve[:len(LAUC_GRID)]
                    out[f"lauc_{pop}_{nm}{tag}"] = float(((g[1:] + g[:-1]) / 2 * _DLOG).sum() / _DLOG.sum())
                    for j, f in enumerate(CURVE_FPRS):
                        out[f"eff{_fpr_name(f)}_{pop}_{nm}{tag}"] = float(curve[len(LAUC_GRID) + j])
                    out[f"floor_eff1pct_{pop}_{nm}{tag}"] = float(curve[i_far])
                if kept is not None:
                    a = np.array(kp[k])                                            # (3, 2)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        fv = a[:, 0] / a[:, 1]
                    out[f"frac_kept_{pop}_{nm}"] = _ratio(a[:, 0].sum(), a[:, 1].sum())
                    out[f"frac_kept_{pop}_{nm}_mean"] = float(fv.mean())
                    out.update({f"frac_kept_{pop}_{nm}_{VIEWS[v]}": float(fv[v]) for v in range(3)})
            num, den = pooled_turn
            out[f"e50_{pop}"] = e50(np.where(den >= E50_MIN, num / np.maximum(den, 1e-12), np.nan), e_cent)
            out[f"e50_{pop}_mean"] = float(np.mean(e50v))
            out.update({f"e50_{pop}_{VIEWS[v]}": e50v[v] for v in range(3)})
        for v in range(3):
            nv = near_v[v]
            out[f"near_fpr_at_far_{VIEWS[v]}"] = _frac(w[nv], s[nv] > far_thr[v][i_far], slice(None))
        return out

    counts = {f"{k}_{VIEWS[v]}": int(((kind == k) & (view == v)).sum()) for k in KINDS for v in range(3)}
    counts.update({f"{c}_{nm}_{VIEWS[v]}": int((sel & (eb == k) & (view == v)).sum())
                   for c, sel in (("clear", clear), ("crowded", crowded))
                   for k, nm in enumerate(ENERGY_NAMES) for v in range(3)})
    return stat, len(evs), counts


def _ratio(a, b):
    return float(a / b) if b else float("nan")


def _frac(w, hit, sel):
    return _ratio((w * hit)[sel].sum(), w[sel].sum())


#: The window numbers every evaluation leads with: the LAUC of the two matched pairs at
#: 0.1-0.2 MeV, view-mean; and the pre-2026-11-16 headline (efficiency at 1% FPR), secondary.
HEADLINE = ("lauc_clear_0.1-0.2_mean", "lauc_crowded_near_0.1-0.2_mean")
SECONDARY = ("floor_eff1pct_clear_0.1-0.2_mean", "floor_eff1pct_crowded_near_0.1-0.2_mean")


def score_windows(score, meta, n_boot=200, seed=0, events=None, level=LEVEL):
    """The numbers of one per-cell ``score``: :func:`window_stat`'s (each ``[point, lo, hi]``,
    the ``level`` event-bootstrap interval) and the window counts ``n_windows``."""
    stat, n_ev, n_win = window_stat(score, meta, events)
    out = bootstrap(stat, n_ev, n_boot, seed, level=level)
    out["n_windows"] = n_win
    return out


def amplitude(ev, keys):
    """The amplitude baseline's score of fine cells ``keys``: the largest kept coefficient's
    |value| / sigma in the cell, 0 where none -- fit-free, what the corpus gate alone gives."""
    uk, amp = ev.coeff_amp
    if not len(uk):
        return np.zeros(len(keys))
    p = np.clip(np.searchsorted(uk, keys), 0, len(uk) - 1)
    return np.where(uk[p] == keys, amp[p], 0.0)


#: The per-window metadata a windows dump carries: what places a window (a far window's kind,
#: plane and NaN energy alone do not), and whether it holds a kept coefficient.
WINDOW_META = ("ev", "kind", "g", "E", "ctx", "w0", "t0", "kept")


class Scored:
    """The scored events' windows and named per-cell scores on them, scored by
    :func:`score_windows`. ``events``: every scored event's index, windows or not, so the
    bootstrap resamples the same events as the evaluation's other numbers."""

    def __init__(self):
        self.meta, self.s, self.events = [], {}, []

    def add(self, ev, tr, scores):
        """``tr``: the event's truth; ``scores``: ``{name: score per window cell}``."""
        self.events.append(int(ev.index))
        self.meta += [dict(m, ev=ev.index) for m in tr["meta"]]
        for k, s in scores.items():
            self.s.setdefault(k, []).append(np.asarray(s, np.float32))

    def result(self, n_boot, seed, names=None, level=LEVEL):
        if not self.meta:
            raise ValueError("no scored window")
        return {k: score_windows(np.concatenate(self.s[k]), self.meta, n_boot, seed, self.events, level)
                for k in (names or self.s)}

    def windows(self):
        """Every scored window once: its metadata as arrays and each named score's window score
        (the max over its cells, what every number thresholds) -- written beside the results so
        two runs scored on the SAME windows compare paired (``python -m helix.eval compare``)."""
        n = len(self.meta)
        out = {f"meta/{k}": np.array([m.get(k) if m.get(k) is not None else "" for m in self.meta])
               for k in WINDOW_META}
        out.update({f"score/{k}": np.concatenate(v).reshape(n, -1).max(1) for k, v in self.s.items()})
        return out


def bootstrap(stat, n_events, n_boot=200, seed=0, n_seeds=None, level=LEVEL):
    """``{key: [point, lo, hi]}``: the point is ``stat`` with every event (and seed) once; ``[lo,
    hi]`` the central ``level`` interval (percentiles, NaNs ignored) over ``n_boot`` resamples.

    ``stat(m) -> {key: value}`` of per-event weights ``m`` (how often each event was drawn, with
    replacement). With ``n_seeds`` -- the training seeds of an arm, scored on the same events --
    ``stat(m, s)`` also takes per-seed weights: each resample first draws the seeds with
    replacement, then the events, so the interval carries the training-seed spread as well as the
    events'."""
    if n_seeds:
        point = stat(np.ones(n_events, np.int64), np.ones(n_seeds, np.int64))
    else:
        point = stat(np.ones(n_events, np.int64))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        if n_seeds:
            sw = np.bincount(rng.integers(0, n_seeds, n_seeds), minlength=n_seeds)
            boots.append(stat(np.bincount(rng.integers(0, n_events, n_events), minlength=n_events), sw))
        else:
            boots.append(stat(np.bincount(rng.integers(0, n_events, n_events), minlength=n_events)))
    a = 100 * (1 - level) / 2
    out = {}
    for k, v in point.items():
        bv = np.array([b.get(k, np.nan) for b in boots], float)
        lo, hi = np.nanpercentile(bv, [a, 100 - a]) if np.isfinite(bv).any() else (np.nan, np.nan)
        out[k] = [round(float(v), 4), round(float(lo), 4), round(float(hi), 4)]
    return out
