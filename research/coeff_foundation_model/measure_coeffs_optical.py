#!/usr/bin/env python
"""Optical twin of measure_coeffs.py — dyadic-tree statistics for the unified design.

Production-faithful optical path (helix.optical): per-chunk coif3 L10
periodization DWT, per-chunk noise sigma from the UNPADDED chunk (db1 finest
MAD), per-band VisuShrink hard threshold t_j = kappa*sigma*sqrt(2 ln N_j)
(kappa=1.2 production), A10 kept untouched. Each chunk is padded to its own
multiple of 2^10 (vs production's per-event common pad) so band lengths are
exactly dyadic and the tree index map is parent(l, tau) = (l+1, tau>>1);
threshold N_j differs from production by a few % in ln(N) for short chunks.

Measures (signal chunks = |x|max > 50 ADC, per helix.optical.metrics):
  1. per-band valid/active counts + survival, signal vs noise chunks
     (droppable-level determination = bands whose signal counts ~= noise counts)
  2. tree lift across the full depth: P(child|parent)/P(child) for every
     adjacent detail pair D_{j+1}->D_j, plus Delta=2,3 ancestor conditionals,
     plus the strict Markov test P(child | parent inactive, grandparent active)
     vs P(child | parent inactive)
  3. alignment shift sweep: P(child tau | parent (tau+s)>>1) for s in -3..3
  4. active-|c| percentiles per band + >10x adjacent-pair fraction
  5. coarse-grid (level-10 cell = 1024 ticks) column stats: detail-union
     occupancy, fan-out per active cell, bands-per-active-cell (cone depth);
     production token analog (A10 kept => all valid cells occupied)
  6. per-event totals; dumps the median-total event as
     artifacts/typical_event_coeffs_optical.npz (chunk_id, side, pmt, band_id,
     idx, value) and stats to artifacts/optical_tree_stats.json

Run from this folder:  python measure_coeffs_optical.py --events 100
"""
import sys, os, json, argparse

sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, "/sdf/group/neutrino/omara/helix")
import hdf5plugin  # noqa: F401  (must precede h5py)
import numpy as np
import pywt
import warnings

warnings.filterwarnings("ignore", message="Level value of .* is too high")

PATH = "/sdf/home/y/youngsam/sw/dune/sim/goop/data/light_output.h5"
LEVEL, WAVELET, MODE = 10, "coif3", "periodization"
KAPPA = 1.2
SIGNAL_PEAK = 50.0
HERE = os.path.dirname(os.path.abspath(__file__))

BANDS = ["A%d" % LEVEL] + ["D%d" % j for j in range(LEVEL, 0, -1)]  # list idx 0..10
NB = len(BANDS)


def band_level(i):  # list idx -> level
    return LEVEL if i == 0 else LEVEL - i + 1


def process_chunk(c, sigma, kappa):
    """DWT one chunk; return (coeffs, act_masks, valid_masks, actA_thresholded)."""
    L = len(c)
    step = 1 << LEVEL
    Lp = int(np.ceil(L / step) * step)
    x = np.zeros(Lp, np.float32)
    x[:L] = c
    coeffs = pywt.wavedec(x, WAVELET, level=LEVEL, mode=MODE)
    acts, valids = [], []
    actA = None
    for i, cb in enumerate(coeffs):
        j = band_level(i)
        n = cb.shape[-1]
        t = kappa * sigma * np.sqrt(2.0 * np.log(max(n, 2)))
        thr = np.abs(cb) >= t
        valid = (np.arange(n) << j) < L          # coeff anchored inside unpadded chunk
        if i == 0:
            actA = thr                           # would-be-active A10 (diagnostic only)
            acts.append(valid.copy())            # production: A10 kept entirely
        else:
            acts.append(thr)
        valids.append(valid)
    return coeffs, acts, valids, actA


class Acc:
    """Pooled accumulators over signal (or noise) chunks."""

    def __init__(self):
        self.valid = np.zeros(NB, np.int64)
        self.act = np.zeros(NB, np.int64)
        self.vals = [[] for _ in range(NB)]                 # active |c|
        self.adj_n = np.zeros(NB, np.int64)                 # adjacent active pairs
        self.adj_big = np.zeros(NB, np.int64)               # ... with ratio > 10
        # tree: child list idx i (2..10), Delta in {1,2,3}: [n_cond, k_cond]
        self.tree = np.zeros((NB, 4, 2), np.int64)
        # strict Markov: child i>=3: [n(~p1), k(~p1), n(~p1&p2), k(~p1&p2)]
        self.markov = np.zeros((NB, 4), np.int64)
        # shift sweep child i in 2..10, s in -3..3 -> [n, k]
        self.shift = np.zeros((NB, 7, 2), np.int64)
        # A10*(thresholded) -> D10 lateral: [nA*, kD10|A*]
        self.lateral = np.zeros(2, np.int64)
        # coarse cells (signal chunks): occupancy + fanout + cone depth
        self.cells_valid = 0
        self.cells_union = 0                                # >=1 active detail D10..D2
        self.fanout = []                                    # active coeffs per active cell
        self.cone = []                                      # distinct active detail bands per active cell

    def add(self, coeffs, acts, valids, actA):
        for i in range(NB):
            v, a = valids[i], acts[i] & valids[i]
            self.valid[i] += int(v.sum())
            self.act[i] += int(a.sum())
            if i > 0 and a.any() and self.vals[i] is not None:
                self.vals[i].append(np.abs(coeffs[i][a]).astype(np.float32))
                both = a[:-1] & a[1:]
                if both.any():
                    x0 = np.abs(coeffs[i][:-1][both])
                    x1 = np.abs(coeffs[i][1:][both])
                    r = np.maximum(x0, x1) / np.maximum(np.minimum(x0, x1), 1e-12)
                    self.adj_n[i] += int(both.sum())
                    self.adj_big[i] += int((r > 10.0).sum())
        if actA is not None and self.vals[0] is not None:
            aA = actA & valids[0]
            if aA.any():
                self.vals[0].append(np.abs(coeffs[0][aA]).astype(np.float32))
        # tree conditionals (detail children i=2..10; parents are details)
        for i in range(2, NB):
            n = len(acts[i])
            idx = np.arange(n)
            vc = valids[i]
            ac = acts[i]
            for d in (1, 2, 3):
                if i - d < 1:
                    continue
                pa = acts[i - d][idx >> d] & vc
                self.tree[i, d, 0] += int(pa.sum())
                self.tree[i, d, 1] += int((pa & ac).sum())
            if i >= 3:
                p1 = acts[i - 1][idx >> 1]
                p2 = acts[i - 2][idx >> 2]
                m0 = (~p1) & vc
                m1 = (~p1) & p2 & vc
                self.markov[i] += [int(m0.sum()), int((m0 & ac).sum()),
                                   int(m1.sum()), int((m1 & ac).sum())]
            par = acts[i - 1]
            for si, s in enumerate(range(-3, 4)):
                a_idx = (idx + s) >> 1
                ok = (a_idx >= 0) & (a_idx < len(par)) & vc
                pa = np.zeros(n, bool)
                pa[ok] = par[np.clip(a_idx, 0, len(par) - 1)[ok]]
                self.shift[i, si, 0] += int(pa.sum())
                self.shift[i, si, 1] += int((pa & acts[i]).sum())
        # A10* -> D10 lateral (same grid)
        if actA is not None:
            m = actA & valids[0]
            self.lateral[0] += int(m.sum())
            self.lateral[1] += int((m & acts[1] & valids[1]).sum())
        # coarse cells: union over details D10..D2 (i=1..9; D1 excluded as noise floor)
        ncell = len(acts[0])
        nvalid_cells = int(valids[0].sum())
        cnt = np.zeros(ncell, np.int64)
        nbands = np.zeros(ncell, np.int64)
        for i in range(1, NB - 1):
            a = acts[i] & valids[i]
            if not a.any():
                continue
            cells = np.arange(len(a))[a] >> (LEVEL - band_level(i))
            c = np.bincount(cells, minlength=ncell)
            cnt += c
            nbands += (c > 0)
        occ = cnt > 0
        self.cells_valid += nvalid_cells
        self.cells_union += int(occ.sum())
        if occ.any():
            self.fanout.append(cnt[occ])
            self.cone.append(nbands[occ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=100)
    ap.add_argument("--kappa", type=float, default=KAPPA)
    ap.add_argument("--path", default=PATH)
    args = ap.parse_args()

    from helix.optical import io as oio
    cfg = oio.config_from_file(args.path)
    events = oio.list_events(args.path)[: args.events]

    sig_acc, noi_acc = Acc(), Acc()
    noi_acc.vals = [None] * NB  # don't accumulate values for noise chunks
    ev_total, ev_nsig, ev_nchunk = [], [], []
    per_chunk_sig = np.zeros(NB, np.float64)  # mean active per signal chunk
    nsig_chunks = nnoi_chunks = 0

    for ek in events:
        ec = oio.read_event_chunks(args.path, ek, cfg)
        sigmas = oio.chunk_noise_sigma(ec.chunks)
        tot = 0
        ns = 0
        for c, s in zip(ec.chunks, sigmas):
            coeffs, acts, valids, actA = process_chunk(c, float(s), args.kappa)
            is_sig = np.abs(c).max() > SIGNAL_PEAK
            acc = sig_acc if is_sig else noi_acc
            acc.add(coeffs, acts, valids, actA)
            n_act = sum(int((a & v).sum()) for a, v in zip(acts, valids))
            tot += n_act
            if is_sig:
                ns += 1
                nsig_chunks += 1
                per_chunk_sig += [int((a & v).sum()) for a, v in zip(acts, valids)]
            else:
                nnoi_chunks += 1
        ev_total.append(tot)
        ev_nsig.append(ns)
        ev_nchunk.append(len(ec.chunks))
        if (len(ev_total)) % 20 == 0:
            print(f"  scanned {len(ev_total)}/{len(events)} events", flush=True)

    ev_total = np.array(ev_total)
    out = {"kappa": args.kappa, "n_events": len(events),
           "n_signal_chunks": nsig_chunks, "n_noise_chunks": nnoi_chunks,
           "signal_chunks_per_event": float(np.mean(ev_nsig)),
           "chunks_per_event": float(np.mean(ev_nchunk))}

    # ---- 1. per-band counts ----
    print("\n=== 1. Per-band counts (kappa=%.2f; A10 kept untouched in production) ===" % args.kappa)
    print(f"{'band':>5} {'sig surv%':>10} {'noi surv%':>10} {'act/sigchunk':>13} "
          f"{'act/noichunk':>13} {'sig:noi':>8}")
    band_rows = []
    for i in range(NB):
        ss = sig_acc.act[i] / max(sig_acc.valid[i], 1)
        sn = noi_acc.act[i] / max(noi_acc.valid[i], 1)
        ps = sig_acc.act[i] / max(nsig_chunks, 1)
        pn = noi_acc.act[i] / max(nnoi_chunks, 1)
        ratio = ps / max(pn, 1e-9)
        print(f"{BANDS[i]:>5} {100*ss:>9.3f}% {100*sn:>9.3f}% {ps:>13.1f} {pn:>13.1f} {ratio:>8.2f}")
        band_rows.append(dict(band=BANDS[i], sig_survival=ss, noi_survival=sn,
                              act_per_sig_chunk=ps, act_per_noi_chunk=pn, sig_noi_ratio=ratio))
    out["bands"] = band_rows

    # ---- 2. tree lift ----
    print("\n=== 2. Tree lift (signal chunks; P(child act | ancestor act) / P(child act)) ===")
    print(f"{'child':>5} {'P(act)':>9} | {'D=1 cond':>9} {'lift':>7} | {'D=2 cond':>9} "
          f"{'lift':>7} | {'D=3 cond':>9} {'lift':>7}")
    tree_rows = []
    for i in range(2, NB):
        marg = sig_acc.act[i] / max(sig_acc.valid[i], 1)
        row = dict(child=BANDS[i], parent=BANDS[i - 1], marginal=marg)
        line = f"{BANDS[i]:>5} {marg:>9.5f} |"
        for d in (1, 2, 3):
            n, k = sig_acc.tree[i, d]
            if n > 0 and i - d >= 1:
                cond = k / n
                row[f"cond_d{d}"], row[f"lift_d{d}"] = cond, cond / max(marg, 1e-12)
                line += f" {cond:>9.5f} {cond/max(marg,1e-12):>6.1f}x |"
            else:
                line += f" {'-':>9} {'-':>7} |"
        print(line)
        tree_rows.append(row)
    out["tree"] = tree_rows

    print("\n--- strict Markov test: P(child | ~parent, grandparent) vs P(child | ~parent) ---")
    mk_rows = []
    for i in range(3, NB):
        n0, k0, n1, k1 = sig_acc.markov[i]
        if n0 and n1:
            p_np = k0 / n0
            p_npg = k1 / n1
            print(f"{BANDS[i]:>5}: P(|~par)={p_np:.5f}  P(|~par,grand)={p_npg:.5f}  "
                  f"residual lift={p_npg/max(p_np,1e-12):.2f}x")
            mk_rows.append(dict(child=BANDS[i], p_noparent=p_np, p_noparent_grand=p_npg))
    out["markov"] = mk_rows

    # ---- 3. shift sweep ----
    print("\n=== 3. Alignment shift sweep: P(child tau | parent (tau+s)>>1), s=-3..3 ===")
    shift_rows = []
    hdr = "  ".join(f"s={s:+d}" for s in range(-3, 4))
    print(f"{'child':>5}  {hdr}")
    for i in range(2, NB):
        conds = []
        for si in range(7):
            n, k = sig_acc.shift[i, si]
            conds.append(k / n if n else float("nan"))
        print(f"{BANDS[i]:>5}  " + "  ".join(f"{c:.4f}" for c in conds))
        shift_rows.append(dict(child=BANDS[i], shifts=list(range(-3, 4)), cond=conds))
    out["shift_sweep"] = shift_rows

    nA, kA = sig_acc.lateral
    margD10 = sig_acc.act[1] / max(sig_acc.valid[1], 1)
    if nA:
        out["lateral_A10star_D10"] = dict(cond=kA / nA, marginal_D10=margD10,
                                          lift=(kA / nA) / max(margD10, 1e-12))
        print(f"\nA10*(thresholded) -> D10 lateral: cond={kA/nA:.4f} "
              f"marg={margD10:.4f} lift={(kA/nA)/max(margD10,1e-12):.1f}x")

    # ---- 4. value stats ----
    print("\n=== 4. Active-|c| percentiles (signal chunks) + >10x adjacency ===")
    print(f"{'band':>5} {'p1':>8} {'p50':>8} {'p99':>9} {'adj>10x':>9}")
    val_rows = []
    for i in range(NB):
        if not sig_acc.vals[i]:
            continue
        v = np.concatenate(sig_acc.vals[i])
        p1, p50, p99 = np.percentile(v, [1, 50, 99])
        frac = sig_acc.adj_big[i] / max(sig_acc.adj_n[i], 1) if i > 0 else float("nan")
        tag = " (A10* thresholded-active)" if i == 0 else ""
        print(f"{BANDS[i]:>5} {p1:>8.2f} {p50:>8.2f} {p99:>9.1f} {100*frac:>8.2f}%{tag}")
        val_rows.append(dict(band=BANDS[i], p1=p1, p50=p50, p99=p99,
                             adj_gt10x=None if i == 0 else frac, n_active=int(v.size)))
    out["values"] = val_rows

    # ---- 5. coarse columns ----
    fan = np.concatenate(sig_acc.fanout) if sig_acc.fanout else np.zeros(1)
    cone = np.concatenate(sig_acc.cone) if sig_acc.cone else np.zeros(1)
    occ = sig_acc.cells_union / max(sig_acc.cells_valid, 1)
    occ_n = noi_acc.cells_union / max(noi_acc.cells_valid, 1)
    print("\n=== 5. Coarse columns (level-10 cell = 1024 ticks; detail-union D10..D2) ===")
    print(f"signal-chunk cell occupancy: {100*occ:.1f}%   (noise chunks: {100*occ_n:.1f}%)")
    print(f"fan-out per active cell: mean {fan.mean():.1f}  p95 {np.percentile(fan,95):.0f}  max {fan.max():.0f}")
    print(f"active detail bands per active cell (cone depth, of 9): mean {cone.mean():.2f}  "
          f"p95 {np.percentile(cone,95):.0f}  max {cone.max():.0f}")
    print("production token analog: A10 kept => 100% of valid cells occupied; "
          f"valid cells/event = {sig_acc.cells_valid/len(events) + noi_acc.cells_valid/len(events):.0f}")
    out["cells"] = dict(occupancy_signal=occ, occupancy_noise=occ_n,
                        fanout_mean=float(fan.mean()), fanout_p95=float(np.percentile(fan, 95)),
                        fanout_max=int(fan.max()), cone_mean=float(cone.mean()),
                        cone_p95=float(np.percentile(cone, 95)),
                        cells_per_event=float((sig_acc.cells_valid + noi_acc.cells_valid) / len(events)))

    # ---- 6. per-event totals + typical dump ----
    pct = np.percentile(ev_total, [5, 50, 95])
    print(f"\n=== 6. Per-event total survivors (A10 kept + active details) ===")
    print(f"min {ev_total.min()}  p5 {pct[0]:.0f}  p50 {pct[1]:.0f}  p95 {pct[2]:.0f}  max {ev_total.max()}")
    print(f"signal chunks/event: {np.mean(ev_nsig):.1f} of {np.mean(ev_nchunk):.1f}")
    out["event_totals"] = dict(min=int(ev_total.min()), p5=float(pct[0]), p50=float(pct[1]),
                               p95=float(pct[2]), max=int(ev_total.max()))

    med_ev = int(np.argsort(ev_total)[len(ev_total) // 2])
    ek = events[med_ev]
    ec = oio.read_event_chunks(args.path, ek, cfg)
    sigmas = oio.chunk_noise_sigma(ec.chunks)
    rows = {k: [] for k in ("chunk_id", "side", "pmt_id", "band_id", "idx", "value")}
    for ci, (c, s) in enumerate(zip(ec.chunks, sigmas)):
        coeffs, acts, valids, _ = process_chunk(c, float(s), args.kappa)
        for i in range(NB):
            a = acts[i] & valids[i]
            idx = np.nonzero(a)[0]
            rows["chunk_id"].append(np.full(len(idx), ci, np.int32))
            rows["side"].append(np.full(len(idx), 0 if ec.side[ci] == "east" else 1, np.int8))
            rows["pmt_id"].append(np.full(len(idx), ec.pmt_id[ci], np.int16))
            rows["band_id"].append(np.full(len(idx), i, np.int8))
            rows["idx"].append(idx.astype(np.int32))
            rows["value"].append(coeffs[i][idx].astype(np.float32))
    npz = {k: np.concatenate(v) for k, v in rows.items()}
    npz["chunk_len"] = ec.lengths.astype(np.int32)
    npz["t0_ns"] = ec.t0_ns
    dump = os.path.join(HERE, "artifacts", "typical_event_coeffs_optical.npz")
    np.savez_compressed(dump, **npz)
    print(f"\ntypical event = {ek} (total {ev_total[med_ev]}), dumped {len(npz['value'])} rows -> {dump}")
    out["typical_event"] = dict(key=ek, total=int(ev_total[med_ev]))

    jpath = os.path.join(HERE, "artifacts", "optical_tree_stats.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"stats -> {jpath}")


if __name__ == "__main__":
    main()
