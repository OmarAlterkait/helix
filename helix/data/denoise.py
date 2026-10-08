"""Events for supervised denoising: noisy tokens + pre-response charge per cell.

Each item is one corpus event: its tokens exactly as the FM sees them (noisy
input, no clean modality) plus a set of query cells with their truth
``y = log1p(q / Q0)`` (scripts/build_denoise_truth.py) and their per-band
covering tokens (:func:`helix.probe.resolution.cell_inputs`).

Query cells: every charge-carrying cell (capped), cells near them, random cells
under kept coefficients (the evaluation's row recipe), and random cells anywhere
inside a token's footprint -- the noise-only cells a model must keep at zero,
including the coefficient-free ones the floor's noise windows are mostly made of. Cells no token covers are not queried: the
model predicts zero there by construction, as the evaluation does.

Splits come from each run's ``holdout.json``: ``probe`` and ``val`` are listed,
``train`` is the rest. ``n_events`` draws a fixed random subset of the split
(``subset_seed``), so a label-efficiency sweep trains on nested subsets.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

Q0_DEFAULT = 2043.2578108909702      # median positive cell charge, truth_v2 probe-train events
_NOISY_SHARD = re.compile(r"sim_wire_coeff_\d+\.h5")   # not the *_clean_* shards


def split_events(corpus_root, runs, split):
    """[(run, shard_path, position, event)] of ``split`` in corpus order."""
    import h5py
    out = []
    for run in runs:
        rd = os.path.join(corpus_root, run)
        hold = json.load(open(os.path.join(rd, "holdout.json")))
        held = {(e["source_file"], int(e["event"])): s for s in ("probe", "val") for e in hold[s]}
        for shard in sorted(f for f in os.listdir(rd) if _NOISY_SHARD.fullmatch(f)):
            path = os.path.join(rd, shard)
            with h5py.File(path, "r") as f:
                ev = f["ident"]["event"][:].astype(np.int64)
                sf = [x.decode() if isinstance(x, bytes) else str(x) for x in f["ident"]["source_file"][:]]
            for pos, (e, s) in enumerate(zip(ev.tolist(), sf)):
                if held.get((s, e), "train") == split:
                    out.append((run, path, pos, e))
    return out


class DenoiseEvents:
    def __init__(self, corpus_root, runs, truth_root, split="train", *, n_events=None, subset_seed=0,
                 cfg=None, q0=Q0_DEFAULT, sig_cap=20000, neg_per_sig=1.0, near_per_sig=1.0, cov_per_sig=1.0, win_per_event=0,
                 near_any_per_sig=0.0, win_near_per_event=0,
                 sample_seed=None, items=None):
        from helix.model.tokenize import PatchConfig
        self.cfg = cfg or PatchConfig(cell_t="grid_center")
        self.truth_root, self.q0 = truth_root, float(q0)
        self.sig_cap, self.neg_per_sig, self.near_per_sig = sig_cap, neg_per_sig, near_per_sig
        self.cov_per_sig, self.win_per_event = cov_per_sig, win_per_event
        self.near_any_per_sig, self.win_near_per_event = near_any_per_sig, win_near_per_event
        self.sample_seed = sample_seed                     # None: a fresh draw per access (training)
        self.items = items if items is not None else split_events(corpus_root, runs, split)
        if n_events is not None and n_events < len(self.items):
            pick = np.random.default_rng(subset_seed).permutation(len(self.items))[:n_events]
            self.items = [self.items[i] for i in sorted(pick)]
        self._cfg_cache, self._truth_cache = {}, {}

    def __len__(self):
        return len(self.items)

    def _shard_cfg(self, path):
        if path not in self._cfg_cache:
            import h5py
            with h5py.File(path, "r") as f:
                self._cfg_cache[path] = {k: f["config"][k][:] for k in ("gids", "n_wires", "band_lengths", "norm_sigma")}
        return self._cfg_cache[path]

    def _truth(self, run, shard_path, pos):
        tag = os.path.basename(shard_path).replace("sim_wire_coeff_", "").replace(".h5", "")
        key = (run, tag)
        if key not in self._truth_cache:
            if len(self._truth_cache) > 8:
                self._truth_cache.clear()
            self._truth_cache[key] = dict(np.load(os.path.join(self.truth_root, run, f"truth_{tag}.npz")))
        t = self._truth_cache[key]
        a, b = t["offsets"][pos], t["offsets"][pos + 1]
        return t["keys"][a:b], t["q"][a:b], int(t["event"][pos])

    def tokens(self, shard_path, pos):
        """The event's tokens as the FM sees them (numpy dict, FM names)."""
        from helix.core.coeff_io import read_coeff_event
        from helix.model.tokenize import assemble, to_fm
        c = self._shard_cfg(shard_path)
        ce = read_coeff_event(shard_path, pos)
        tok = assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, ce.value, gids=c["gids"], n_wires=c["n_wires"],
                       band_lengths=c["band_lengths"], norm_sigma=c["norm_sigma"], cfg=self.cfg)
        return ce, to_fm(tok), c["band_lengths"]

    def rows(self, ce, tkeys, tq, rng):
        """Query cells and their truth, after the evaluation's recipe."""
        from helix.model.tokenize import tick_of_tau
        from helix.probe.resolution import FT, FW, fkey, unkey
        k = ce.band < self.cfg.n_bands
        ct = tick_of_tau(ce.tau[k], ce.plane_gid[k], ce.band[k], self.cfg)
        ck = np.unique(fkey(ce.plane_gid[k], ce.wire[k] // FW, np.floor(ct / FT).astype(np.int64)))
        sig = tkeys[tq > 0]
        if len(sig) > self.sig_cap:
            sig = rng.choice(sig, self.sig_cap, replace=False)
        parts = [sig]
        if len(sig) and self.near_per_sig > 0:
            n = int(len(sig) * self.near_per_sig)
            base = rng.choice(sig, n, replace=True)
            g, w, t = unkey(base)
            near = fkey(g, np.maximum(w + rng.integers(-8, 9, n), 0), np.maximum(t + rng.integers(-8, 9, n), 0))
            parts.append(near[np.isin(near, ck)])
        if len(ck):
            parts.append(rng.choice(ck, min(int(len(sig) * self.neg_per_sig) + 1, len(ck)), replace=False))
        if self.cov_per_sig > 0 and k.any():
            # Cells anywhere inside a token's footprint, not only under a kept
            # coefficient: a noise token covers up to 16 wires x 128 ticks, most of
            # it coefficient-free, and the floor's noise windows score the MAX over
            # such cells -- a cell type the model must be shown, not left to
            # extrapolate. Pick kept coefficients, then a uniform point in their token.
            n = int(len(sig) * self.cov_per_sig) + 1
            j = rng.integers(0, int(k.sum()), n)
            b, g = ce.band[k][j], ce.plane_gid[k][j]
            wire = (ce.wire[k][j] // self.cfg.pw) * self.cfg.pw + rng.integers(0, self.cfg.pw, n)
            tau = (ce.tau[k][j] // self.cfg.pt) * self.cfg.pt + rng.integers(0, self.cfg.pt, n)
            dec = (1 << np.asarray(self.cfg.lev))[b]
            tick = tick_of_tau(tau, g, b, self.cfg) + rng.random(n) * dec
            parts.append(fkey(g, np.maximum(wire, 0) // FW, np.floor(np.maximum(tick, 0) / FT).astype(np.int64)))
        if self.win_per_event > 0 and len(ck):
            parts.append(self._noise_windows(ck, tkeys, rng))
        if len(sig) and self.near_any_per_sig > 0:
            # cells near charge whether or not a coefficient survived there: the
            # empty, coefficient-free surroundings of activity, where a regression
            # left to extrapolate predicts a haze (the floor's iso deposits sit there)
            n = int(len(sig) * self.near_any_per_sig)
            g, w, t = unkey(rng.choice(sig, n, replace=True))
            parts.append(fkey(g, np.maximum(w + rng.integers(-8, 9, n), 0), np.maximum(t + rng.integers(-8, 9, n), 0)))
        if self.win_near_per_event > 0:
            from helix.probe.resolution import near_windows, window_cells
            for g, w0, t0 in near_windows(tkeys, tq, rng, n=self.win_near_per_event):
                parts.append(window_cells(g, w0, t0))
        rows = np.unique(np.concatenate(parts))
        q = np.zeros(len(rows))
        if len(tkeys):
            pos = np.clip(np.searchsorted(tkeys, rows), 0, len(tkeys) - 1)
            hit = tkeys[pos] == rows
            q[hit] = tq[pos[hit]]
        return rows, np.log1p(q / self.q0).astype(np.float32)

    def _noise_windows(self, ck, tkeys, rng):
        """Every cell of up to ``win_per_event`` charge-free windows (WW x WT),
        each centred on a kept coefficient with no truth charge within the same
        padding the evaluation's noise windows use (16 wires, 128 ticks). The
        floor scores the MAX over such a window, so training sees all of it."""
        from helix.probe.resolution import FT, FW, WT, WW, fkey, unkey
        tg, tw, tt = unkey(tkeys)
        by_plane = {int(g): (tw[tg == g], tt[tg == g]) for g in np.unique(tg)}
        out, n = [], 0
        for x in rng.permutation(ck)[:20 * self.win_per_event]:
            g, fw, ft = (int(v) for v in unkey(np.array([x])))
            w0, t0 = max(0, fw - WW // FW // 2), max(0, ft - WT // FT // 2)        # in fine-cell units
            pw_, pt_ = 16 // FW, 128 // FT
            cw, ct = by_plane.get(g, (np.zeros(0, int), np.zeros(0, int)))
            if ((cw >= w0 - pw_) & (cw < w0 + WW // FW + pw_) & (ct >= t0 - pt_) & (ct < t0 + WT // FT + pt_)).any():
                continue
            ww, tt_ = np.meshgrid(np.arange(w0, w0 + WW // FW), np.arange(t0, t0 + WT // FT), indexing="ij")
            out.append(fkey(np.full(ww.size, g), ww.ravel(), tt_.ravel()))
            n += 1
            if n >= self.win_per_event:
                break
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    def __getitem__(self, i):
        from helix.probe.resolution import cell_inputs
        run, path, pos, event = self.items[i]
        ce, B, bl = self.tokens(path, pos)
        tkeys, tq, tev = self._truth(run, path, pos)
        if tev != event:
            raise ValueError(f"{path}#{pos}: truth is for event {tev}, corpus has {event}")
        rng = np.random.default_rng(None if self.sample_seed is None else (self.sample_seed, i))
        rows, y = self.rows(ce, tkeys, tq, rng)
        idx, aux = cell_inputs(rows, B["cell_key"], bl, self.cfg)
        ok = (idx >= 0).any(1)                               # uncovered cells are not queried
        return dict(B=B, idx=idx[ok], aux=aux[ok], y=y[ok], name=f"{run}/{os.path.basename(path)}#{pos}")
