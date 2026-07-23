#!/usr/bin/env python
"""Clean-support tree statistics on the doraemon optical set (M0, T0.2/T0.3).

Convention (EXECUTION_PLAN.md, settled): doraemon is NOISE-FREE; support =
production threshold with NOMINAL sigma = 2.6 ADC (t_j = kappa*2.6*sqrt(2 ln
N_j), kappa=1.2, A10 kept) — the signal coefficients that would survive
production thresholding. Chunk classes come from TRUTH (pe>0), not the
contaminated |x|max>50 heuristic (skeptic S13).

Reuses Acc/process_chunk/BANDS from measure_coeffs_optical.py (same folder).
Outputs artifacts/optical_tree_stats_doraemon.json + a typical-event dump
artifacts/typical_event_coeffs_doraemon.npz.

Run from this folder:  python measure_coeffs_doraemon.py --events 200
"""
import sys, os, json, argparse

sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, "/sdf/group/neutrino/omara/helix")
import hdf5plugin  # noqa: F401  (must precede h5py)
import numpy as np

from measure_coeffs_optical import Acc, process_chunk, BANDS, NB
import doraemon_optical as dop

SIGMA_NOM = 2.6
KAPPA = 1.2
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=200)
    ap.add_argument("--kappa", type=float, default=KAPPA)
    ap.add_argument("--sigma", type=float, default=SIGMA_NOM)
    args = ap.parse_args()

    sig_acc, noi_acc = Acc(), Acc()
    noi_acc.vals = [None] * NB
    ev_total, ev_nchunk, ev_keys = [], [], []
    nsig = nnoi = 0
    pe_zero_chunks = 0

    events = list(dop.iter_events(args.events))
    for i, (path, ek) in enumerate(events):
        ec = dop.read_event_chunks(path, ek)
        tot = 0
        for c, pe in zip(ec.chunks, ec.pe):
            coeffs, acts, valids, actA = process_chunk(c, args.sigma, args.kappa)
            is_sig = pe > 0                      # truth-based class
            if pe == 0:
                pe_zero_chunks += 1
            acc = sig_acc if is_sig else noi_acc
            acc.add(coeffs, acts, valids, actA)
            tot += sum(int((a & v).sum()) for a, v in zip(acts, valids))
            if is_sig:
                nsig += 1
            else:
                nnoi += 1
        ev_total.append(tot)
        ev_nchunk.append(len(ec.chunks))
        ev_keys.append((path, ek))
        if (i + 1) % 20 == 0:
            print(f"  scanned {i+1}/{len(events)} events", flush=True)

    ev_total = np.array(ev_total)
    out = {"convention": "clean-support (noise-free data, nominal sigma)",
           "sigma_nominal": args.sigma, "kappa": args.kappa,
           "n_events": len(events), "n_pe_chunks": nsig,
           "n_zero_pe_chunks": nnoi,
           "chunks_per_event": float(np.mean(ev_nchunk))}

    print(f"\n=== T0.3 truth-anchored chunk classes ===")
    print(f"chunks with PE>0: {nsig}   zero-PE chunks: {pe_zero_chunks} "
          f"({100*pe_zero_chunks/max(nsig+nnoi,1):.2f}%)")

    print(f"\n=== Per-band clean-support counts (sigma_nom={args.sigma}, kappa={args.kappa}) ===")
    print(f"{'band':>5} {'surv%':>9} {'act/chunk':>10}")
    rows = []
    for i in range(NB):
        ss = sig_acc.act[i] / max(sig_acc.valid[i], 1)
        ps = sig_acc.act[i] / max(nsig, 1)
        print(f"{BANDS[i]:>5} {100*ss:>8.3f}% {ps:>10.1f}")
        rows.append(dict(band=BANDS[i], survival=ss, act_per_chunk=ps))
    out["bands"] = rows

    print("\n=== Tree lift (PE>0 chunks) ===")
    print(f"{'child':>5} {'P(act)':>9} | {'D=1 cond':>9} {'lift':>7} | {'D=2 lift':>9} | {'D=3 lift':>9}")
    tree_rows = []
    for i in range(2, NB):
        marg = sig_acc.act[i] / max(sig_acc.valid[i], 1)
        row = dict(child=BANDS[i], marginal=marg)
        line = f"{BANDS[i]:>5} {marg:>9.5f} |"
        for d in (1, 2, 3):
            n, k = sig_acc.tree[i, d]
            if n > 0 and i - d >= 1:
                cond = k / n
                row[f"cond_d{d}"], row[f"lift_d{d}"] = cond, cond / max(marg, 1e-12)
                line += (f" {cond:>9.5f} {cond/max(marg,1e-12):>6.1f}x |" if d == 1
                         else f" {cond/max(marg,1e-12):>8.1f}x |")
            else:
                line += f" {'-':>9} |" if d > 1 else f" {'-':>9} {'-':>7} |"
        print(line)
        tree_rows.append(row)
    out["tree"] = tree_rows

    mk_rows = []
    print("\n--- strict Markov residual ---")
    for i in range(3, NB):
        n0, k0, n1, k1 = sig_acc.markov[i]
        if n0 and n1:
            print(f"{BANDS[i]:>5}: P(|~par)={k0/n0:.5f}  P(|~par,grand)={k1/n1:.5f}  "
                  f"residual={k1/n1/max(k0/n0,1e-12):.2f}x")
            mk_rows.append(dict(child=BANDS[i], p_np=k0 / n0, p_npg=k1 / n1))
    out["markov"] = mk_rows

    print("\n=== Values (active |c|, PE>0 chunks) ===")
    val_rows = []
    for i in range(NB):
        if not sig_acc.vals[i]:
            continue
        v = np.concatenate(sig_acc.vals[i])
        p1, p50, p99 = np.percentile(v, [1, 50, 99])
        frac = sig_acc.adj_big[i] / max(sig_acc.adj_n[i], 1) if i > 0 else float("nan")
        print(f"{BANDS[i]:>5} p1={p1:>7.2f} p50={p50:>8.2f} p99={p99:>10.1f} adj>10x={100*frac:>6.2f}%")
        val_rows.append(dict(band=BANDS[i], p1=p1, p50=p50, p99=p99,
                             adj_gt10x=None if i == 0 else frac, n=int(v.size)))
    out["values"] = val_rows

    fan = np.concatenate(sig_acc.fanout) if sig_acc.fanout else np.zeros(1)
    cone = np.concatenate(sig_acc.cone) if sig_acc.cone else np.zeros(1)
    occ = sig_acc.cells_union / max(sig_acc.cells_valid, 1)
    print(f"\n=== Coarse columns (1024-tick cells, detail-union D10..D2) ===")
    print(f"occupancy {100*occ:.1f}% | fan-out mean {fan.mean():.1f} p95 "
          f"{np.percentile(fan,95):.0f} max {fan.max():.0f} | cone mean {cone.mean():.2f}")
    out["cells"] = dict(occupancy=occ, fanout_mean=float(fan.mean()),
                        fanout_p95=float(np.percentile(fan, 95)),
                        fanout_max=int(fan.max()), cone_mean=float(cone.mean()))

    pct = np.percentile(ev_total, [5, 50, 95])
    print(f"\nper-event survivors: min {ev_total.min()} p5 {pct[0]:.0f} "
          f"p50 {pct[1]:.0f} p95 {pct[2]:.0f} max {ev_total.max()}")
    out["event_totals"] = dict(min=int(ev_total.min()), p5=float(pct[0]),
                               p50=float(pct[1]), p95=float(pct[2]),
                               max=int(ev_total.max()))

    # typical-event dump (median total) — the M2/M3 substrate
    mi = int(np.argsort(ev_total)[len(ev_total) // 2])
    path, ek = ev_keys[mi]
    ec = dop.read_event_chunks(path, ek)
    rows_d = {k: [] for k in ("chunk_id", "pmt_id", "label", "pe", "band_id", "idx", "value")}
    for ci, c in enumerate(ec.chunks):
        coeffs, acts, valids, _ = process_chunk(c, args.sigma, args.kappa)
        for i in range(NB):
            a = acts[i] & valids[i]
            idx = np.nonzero(a)[0]
            rows_d["chunk_id"].append(np.full(len(idx), ci, np.int32))
            rows_d["pmt_id"].append(np.full(len(idx), ec.pmt_id[ci], np.int16))
            rows_d["label"].append(np.full(len(idx), ec.label[ci], np.int16))
            rows_d["pe"].append(np.full(len(idx), ec.pe[ci], np.int32))
            rows_d["band_id"].append(np.full(len(idx), i, np.int8))
            rows_d["idx"].append(idx.astype(np.int32))
            rows_d["value"].append(coeffs[i][idx].astype(np.float32))
    npz = {k: np.concatenate(v) for k, v in rows_d.items()}
    npz["chunk_len"] = ec.lengths
    npz["t0_ns"] = ec.t0_ns
    dump = os.path.join(HERE, "artifacts", "typical_event_coeffs_doraemon.npz")
    np.savez_compressed(dump, **npz)
    print(f"typical event = {os.path.basename(path)}:{ek} "
          f"(total {ev_total[mi]}), {len(npz['value'])} rows -> {dump}")
    out["typical_event"] = dict(file=os.path.basename(path), key=ek,
                                total=int(ev_total[mi]))

    jp = os.path.join(HERE, "artifacts", "optical_tree_stats_doraemon.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"stats -> {jp}")


if __name__ == "__main__":
    main()
