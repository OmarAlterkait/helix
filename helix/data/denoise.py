"""The denoiser's labels: query cells of one event and their truth, as a transform.

``DenoiseRows`` runs on a ``("coeff", "cells")`` sample after ``CoeffTokenize`` and before
``CoeffCollect``: from the event's tokens (part ``coeff``) and its cells truth
(:mod:`helix.data.cells`) it draws the query cells and adds, beside the tokens, what
:class:`helix.model.denoise.DenoiseModel` consumes::

    q_idx      (n, n_bands) int64       per band, the token covering the cell, -1 if none
    q_aux      (n, 13 n_bands) float32  the cell's offsets inside those tokens
                                        (:func:`helix.data.cells.cell_inputs`)
    q_nbr      (n, n_bands, 9) int64    the 3 x 3 token neighbourhood per band
                                        (:func:`helix.data.cells.cell_neighbors`)
    q_y        (n,) float32             ``log1p(q / Q0)``, the cell's full charge
    q_present  (n,) float32             1 where the charge is above ``support_min_q``

Query cells follow the evaluation's row recipe, plus what the floor's windows score:

* every charged cell (``q > support_min_q``), up to ``sig_cap``;
* as many cells within +-8 cells of them under a kept coefficient, and as many random cells
  under kept coefficients (the noise-only cells a model must keep at zero);
* ``cov_per_sig`` per charged cell anywhere inside a kept coefficient's token -- the
  coefficient-free cells the floor's noise windows are mostly made of;
* every cell of ``win_per_event`` charge-free windows (the floor's noise windows);
* ``near_any_per_sig`` per charged cell within +-8 cells of charge, coefficient or not, and every
  cell of ``win_near_per_event`` charge-free windows NEAR activity
  (:func:`near_windows`): the surroundings of activity, where a regression
  left to extrapolate predicts a haze.

Cells no token covers are not queried: the model predicts zero there by construction, as the
evaluation does. ``seed=None`` draws afresh at every access (training); an int fixes each event's
draw by its name (validation). The defaults are the M6d / M6dc recipe (DENOISE, at git tag
archive-2026-11), except ``support_min_q``: 25 e-, the cells-3 truth's own cut.

numpy only (no pimm-data): ``helix.integrations.pimm`` registers it by name, as it does
``CoeffTokenize``.
"""
from __future__ import annotations

import numpy as np

from helix.data.cells import FT, FW, Q0, cell_inputs, cell_neighbors, coeff_cells, fkey, unkey

#: The keys this transform adds to the token part.
KEYS = ("q_idx", "q_aux", "q_nbr", "q_y", "q_present")


def kept_coefficients(tok, cfg):
    """``(band, plane_gid, wire, tau)`` of the coefficients in the tokens (``CoeffTokenize``'s
    output: per kept row its token ``cell`` and ``slot``), in the tokenizer's row order."""
    cell, slot = np.asarray(tok["cell"], np.int64), np.asarray(tok["slot"], np.int64)
    wire = np.asarray(tok["cell_wb"], np.int64)[cell] * cfg.pw + slot // cfg.pt
    tau = np.asarray(tok["cell_tb"], np.int64)[cell] * cfg.pt + slot % cfg.pt
    return (np.asarray(tok["band"], np.int64), np.asarray(tok["plane_id"], np.int64)[cell], wire, tau)


class DenoiseRows:
    """Query cells and their labels for one event (see the module docstring)."""

    scope = "sample"

    def __init__(self, part="coeff", *, sig_cap=20000, cov_per_sig=2.0, win_per_event=48,
                 near_any_per_sig=1.0, win_near_per_event=24, support_min_q=25.0, seed=None):
        self.part = part
        self.sig_cap = int(sig_cap)
        self.cov_per_sig, self.win_per_event = float(cov_per_sig), int(win_per_event)
        self.near_any_per_sig, self.win_near_per_event = float(near_any_per_sig), int(win_near_per_event)
        # A cell counts as 'charged' -- for sampling (charged cells, cells near them, the
        # charge-free windows) and for the presence label -- only above this charge (e-).
        # The complete truth gives sub-electron diffusion tails to most cells near activity;
        # with 0 they all become 'charged' (DENOISE at tag archive-2026-11, "Complete-truth label"). The
        # cells-3 q carries the simulation's own 25 e- cut (d1m), so the default is that cut: the
        # presence label is the evaluation's (helix.eval.denoise.PRESENCE_Q) and cat's bin 0
        # (helix.model.denoise). The regression target is unchanged: the cell's full charge.
        self.support_min_q = float(support_min_q)
        self.seed = seed

    def __call__(self, data):
        from helix.model.tokenize import _name_hash
        tok = data[self.part]
        meta = tok.get("_meta") or {}
        if "cfg" not in meta or "cell_key" not in tok:
            raise KeyError(f"DenoiseRows: part {self.part!r} is not tokenized -- it runs after CoeffTokenize")
        if "cells" not in data:
            raise KeyError("DenoiseRows: no 'cells' in the sample -- the dataset needs modalities=('coeff', 'cells')")
        cfg, bl = meta["cfg"], np.asarray(meta["band_lengths"])
        rng = np.random.default_rng(None if self.seed is None else [int(self.seed), _name_hash(str(data.get("name", "")))])
        rows, q = self.rows(kept_coefficients(tok, cfg), data["cells"], cfg, rng)
        idx, aux = cell_inputs(rows, tok["cell_key"], bl, cfg)
        ok = (idx >= 0).any(1)                                  # uncovered cells are not queried
        y = np.log1p(q[ok] / Q0).astype(np.float32)
        tok.update(q_idx=idx[ok], q_aux=aux[ok], q_nbr=cell_neighbors(rows[ok], tok["cell_key"], bl, cfg),
                   q_y=y, q_present=(y > np.log1p(self.support_min_q / Q0)).astype(np.float32))
        return data

    def rows(self, coeffs, cells, cfg, rng):
        """``(keys, q)``: the query cells (sorted, unique) and their truth charge (e-)."""
        from helix.model.tokenize import tick_of_tau
        band, gid, wire, tau = coeffs
        tkeys, tq = np.asarray(cells["key"], np.int64), np.asarray(cells["q"], np.float64)
        ck = coeff_cells(band, gid, wire, tau, cfg)              # cells under a kept coefficient
        sup = tq > self.support_min_q
        sig = tkeys[sup]
        if len(sig) > self.sig_cap:
            sig = rng.choice(sig, self.sig_cap, replace=False)
        parts = [sig]
        if len(sig):                                            # as many cells near them, under a coefficient
            n = len(sig)
            g, w, t = unkey(rng.choice(sig, n, replace=True))
            near = fkey(g, np.maximum(w + rng.integers(-8, 9, n), 0), np.maximum(t + rng.integers(-8, 9, n), 0))
            parts.append(near[np.isin(near, ck)])
        if len(ck):                                             # as many random cells under a coefficient
            parts.append(rng.choice(ck, min(len(sig) + 1, len(ck)), replace=False))
        if self.cov_per_sig > 0 and len(band):
            # Cells anywhere inside a token's footprint, not only under a kept coefficient: a
            # noise token covers up to 16 wires x 128 ticks, most of it coefficient-free, and
            # the floor's noise windows score the MAX over such cells. Pick kept coefficients,
            # then a uniform point in their token.
            n = int(len(sig) * self.cov_per_sig) + 1
            j = rng.integers(0, len(band), n)
            b, g = band[j], gid[j]
            w = (wire[j] // cfg.pw) * cfg.pw + rng.integers(0, cfg.pw, n)
            ta = (tau[j] // cfg.pt) * cfg.pt + rng.integers(0, cfg.pt, n)
            tick = tick_of_tau(ta, g, b, cfg) + rng.random(n) * (1 << np.asarray(cfg.lev))[b]
            parts.append(fkey(g, np.maximum(w, 0) // FW, np.floor(np.maximum(tick, 0) / FT).astype(np.int64)))
        if self.win_per_event > 0 and len(ck):
            parts.append(self._noise_windows(ck, tkeys[sup], rng))
        if len(sig) and self.near_any_per_sig > 0:
            n = int(len(sig) * self.near_any_per_sig)
            g, w, t = unkey(rng.choice(sig, n, replace=True))
            parts.append(fkey(g, np.maximum(w + rng.integers(-8, 9, n), 0), np.maximum(t + rng.integers(-8, 9, n), 0)))
        if self.win_near_per_event > 0:
            from helix.eval.events import window_cells
            for g, w0, t0 in near_windows(tkeys, np.where(sup, tq, 0.0), rng, n=self.win_near_per_event):
                parts.append(window_cells(g, w0, t0))
        keys = np.unique(np.concatenate(parts).astype(np.int64))
        q = np.zeros(len(keys))
        if len(tkeys):
            pos = np.clip(np.searchsorted(tkeys, keys), 0, len(tkeys) - 1)
            hit = tkeys[pos] == keys
            q[hit] = tq[pos[hit]]
        return keys, q

    def _noise_windows(self, ck, charged, rng):
        """Every cell of up to ``win_per_event`` charge-free windows (WW x WT), each centred on a
        kept coefficient with no charged cell within the padding the evaluation's noise windows
        keep empty (16 wires, 128 ticks). The floor scores the MAX over such a window."""
        from helix.eval.events import WT, WW
        tg, tw, tt = unkey(charged)
        by_plane = {int(g): (tw[tg == g], tt[tg == g]) for g in np.unique(tg)}
        pw_, pt_ = 16 // FW, 128 // FT
        out = []
        for x in rng.permutation(ck)[:20 * self.win_per_event]:
            g, fw, ft = (int(v[0]) for v in unkey(np.array([x])))
            w0, t0 = max(0, fw - WW // FW // 2), max(0, ft - WT // FT // 2)        # in fine-cell units
            cw, ct = by_plane.get(g, (np.zeros(0, int), np.zeros(0, int)))
            if ((cw >= w0 - pw_) & (cw < w0 + WW // FW + pw_) & (ct >= t0 - pt_) & (ct < t0 + WT // FT + pt_)).any():
                continue
            ww, tt_ = np.meshgrid(np.arange(w0, w0 + WW // FW), np.arange(t0, t0 + WT // FT), indexing="ij")
            out.append(fkey(np.full(ww.size, g), ww.ravel(), tt_.ravel()))
            if len(out) >= self.win_per_event:
                break
        return np.concatenate(out) if out else np.zeros(0, np.int64)


def near_windows(keys, q, rng, n=60, margin=1, pad=(8, 8)):
    """Charge-free windows NEAR activity, as pixel origins ``[(g, w0, t0)]``.

    No charge cell within ``margin`` cells of the window, but at least one within ``pad`` (wire,
    tick) cells -- 16 wires / 128 ticks at the default, exactly the surroundings the floor's own
    noise windows must keep EMPTY. ``keys``/``q``: the event's fine cells and charge. Up to ``n``
    windows, shared equally over the planes that carry charge.
    """
    from helix.eval.events import WT, WW
    keys = np.asarray(keys, np.int64)[np.asarray(q) > 0]
    if not len(keys):
        return []
    G, CW, CT = unkey(keys)
    ww, wt = WW // FW, WT // FT
    planes = np.unique(G)
    per = -(-n // len(planes))
    out = []
    for g in planes.tolist():
        m = G == g
        o = max(pad) + margin + 1                                 # grid offset: origins may sit left of 0
        occ = np.zeros((int(CW[m].max()) + ww + 2 * o + 1, int(CT[m].max()) + wt + 2 * o + 1), np.int32)
        occ[CW[m] + o, CT[m] + o] = 1
        S = np.pad(occ.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

        def box(i0, i1, j0, j1):                                  # charge cells in [i0,i1) x [j0,j1), grid coords
            i0, i1 = np.clip(i0, 0, occ.shape[0]), np.clip(i1, 0, occ.shape[0])
            j0, j1 = np.clip(j0, 0, occ.shape[1]), np.clip(j1, 0, occ.shape[1])
            return S[i1, j1] - S[i0, j1] - S[i1, j0] + S[i0, j0]

        I, J = np.meshgrid(np.arange(o, occ.shape[0] - ww), np.arange(o, occ.shape[1] - wt), indexing="ij")
        I, J = I.ravel(), J.ravel()
        empty = box(I - margin, I + ww + margin, J - margin, J + wt + margin) == 0
        near = box(I - pad[0], I + ww + pad[0], J - pad[1], J + wt + pad[1]) > 0
        cand = np.nonzero(empty & near)[0]
        for c in rng.permutation(cand)[:per].tolist():
            out.append((int(g), int(I[c] - o) * FW, int(J[c] - o) * FT))
    return out[:n] if len(out) > n else out
