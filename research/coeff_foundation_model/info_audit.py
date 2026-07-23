#!/usr/bin/env python
"""M2 — data-only information audit, in bits (EXECUTION_PLAN.md T2.1-T2.6).

Runs on the dumped typical-event npz files (no training, CPU):
  TPC      artifacts/typical_event_coeffs_smart.npz      (noisy, smart-removed)
  optical  artifacts/typical_event_coeffs_optical.npz    (light_output, noisy)
  doraemon artifacts/typical_event_coeffs_doraemon.npz   (clean support)

Computes, per child band:
  T2.1 exact activity (C)MI from contingency tables, in bits:
       I(child; parent), I(child; grand | parent),
       I(child; parent | L,R same-band neighbors)   <- the envelope control
       I(child; L,R) for comparison; shifted-non-parent MI vs shift;
       per activity-stratum (tercile of unit activity) to kill Simpson pooling
  T2.2 value-level Gaussian-copula MI among co-active pairs/triples:
       I(v_child; v_parent), partial I(v_child; v_grand | v_parent)
  T2.3 orphan census: P(parent inactive | child active), rescue by grandparent
  T2.4 coif3 group delay per level (impulse probe, exact) + value-level
       corr(|child|,|parent|) vs shift (the alignment closure)
  T2.5 optical anchor sweep: cells/event, detail-union occupancy, fan-out at
       cell = 1024/512/256 ticks
  T2.6 bit-count predictions (registered BEFORE M3)

Bits/slot AND bits/event (population-weighted) are reported — pointwise lift
arguments are retired (simplifier P5).

Run from this folder:  python info_audit.py
"""
import sys, os, json

sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

# ---------------------------------------------------------------- helpers ---

def entropy_bits(p):
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mi_from_table(T, x_ax, y_ax, z_ax=()):
    """I(X;Y|Z) in bits from a joint count table T (axes = binary vars)."""
    T = np.asarray(T, float)
    tot = T.sum()
    if tot == 0:
        return 0.0
    keep = tuple(sorted(set(x_ax) | set(y_ax) | set(z_ax)))
    drop = tuple(i for i in range(T.ndim) if i not in keep)
    if drop:
        T = T.sum(axis=drop)
    # re-index axes after drop
    remap = {old: new for new, old in enumerate(keep)}
    x_ax = tuple(remap[a] for a in x_ax)
    y_ax = tuple(remap[a] for a in y_ax)
    z_ax = tuple(remap[a] for a in z_ax)
    P = T / tot
    def marg(axes):
        d = tuple(i for i in range(P.ndim) if i not in axes)
        return P.sum(axis=d, keepdims=True) if d else P
    Pxyz = P
    Pxz = marg(x_ax + z_ax)
    Pyz = marg(y_ax + z_ax)
    Pz = marg(z_ax) if z_ax else np.array(1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = Pxyz * Pz / (Pxz * Pyz)
        term = np.where(Pxyz > 0, Pxyz * np.log2(np.where(ratio > 0, ratio, 1.0)), 0.0)
    return float(term.sum())


def normal_scores(x):
    from scipy.stats import rankdata
    from scipy.special import erfinv
    u = (rankdata(x) - 0.5) / len(x)
    return np.sqrt(2.0) * erfinv(2.0 * u - 1.0)


def copula_mi(x, y):
    if len(x) < 50:
        return float("nan")
    r = float(np.corrcoef(normal_scores(x), normal_scores(y))[0, 1])
    r = min(max(r, -0.999999), 0.999999)
    return -0.5 * np.log2(1.0 - r * r), r


def copula_partial_mi(x, y, z):
    """I(X;Y|Z) under Gaussian copula via partial correlation."""
    if len(x) < 50:
        return float("nan")
    xs, ys, zs = normal_scores(x), normal_scores(y), normal_scores(z)
    rxy = np.corrcoef(xs, ys)[0, 1]
    rxz = np.corrcoef(xs, zs)[0, 1]
    ryz = np.corrcoef(ys, zs)[0, 1]
    den = np.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12))
    rp = (rxy - rxz * ryz) / den
    rp = min(max(rp, -0.999999), 0.999999)
    return -0.5 * np.log2(1.0 - rp * rp), rp


# ------------------------------------------------------------ band setups ---

TPC_BANDS = ["A4", "D4", "D3", "D2", "D1"]
TPC_LEVELS = [4, 4, 3, 2, 1]
TPC_LENS = [271, 271, 542, 1084, 2168]
TPC_NTICKS = 4321
TPC_NWIRES = {0: 1969, 1: 1969, 2: 1443, 3: 1969, 4: 1969, 5: 1443}

OPT_BANDS = ["A10"] + [f"D{j}" for j in range(10, 0, -1)]
OPT_LEVELS = [10] + list(range(10, 0, -1))


def load_tpc(path):
    """-> list over gid of dict band_i -> (mask (n_w, len), val (n_w, len))."""
    d = np.load(path)
    gids, bands = d["plane_gid"], d["band_id"]
    wires, times, vals = d["wire"], d["time_index"], d["value"]
    units = []
    for g in range(6):
        nw = TPC_NWIRES[g]
        sel_g = gids == g
        per = {}
        for b in range(5):
            sel = sel_g & (bands == b)
            m = np.zeros((nw, TPC_LENS[b]), bool)
            v = np.zeros((nw, TPC_LENS[b]), np.float32)
            m[wires[sel], times[sel]] = True
            v[wires[sel], times[sel]] = vals[sel]
            valid = (np.arange(TPC_LENS[b]) << TPC_LEVELS[b]) < TPC_NTICKS
            per[b] = (m & valid[None, :], v)
        units.append(per)
    return units


def load_opt(path):
    """-> list over chunk of dict band_i -> (mask (len,), val (len,))."""
    d = np.load(path)
    cl = d["chunk_len"]
    chunks, bands = d["chunk_id"], d["band_id"]
    idxs, vals = d["idx"], d["value"]
    order = np.argsort(chunks, kind="stable")
    chunks, bands, idxs, vals = chunks[order], bands[order], idxs[order], vals[order]
    bounds = np.searchsorted(chunks, np.arange(len(cl) + 1))
    out = []
    for ci in range(len(cl)):
        s, e = bounds[ci], bounds[ci + 1]
        L = int(cl[ci]); Lp = int(np.ceil(L / 1024) * 1024)
        per = {}
        for i in range(11):
            j = OPT_LEVELS[i]
            n = Lp >> j
            m = np.zeros(n, bool); v = np.zeros(n, np.float32)
            sel = slice(s, e)
            bs = bands[sel] == i
            ii = idxs[sel][bs]
            m[ii] = True; v[ii] = vals[sel][bs]
            valid = (np.arange(n) << j) < L
            per[i] = (m & valid, v)
        out.append(per)
    return out


# ----------------------------------------------------------- accumulation ---

class BandAudit:
    """Per child band: contingency tables + value pair collections."""

    def __init__(self):
        self.Tcpg = np.zeros((2, 2, 2), np.int64)        # (child,parent,grand)
        self.Tcplr = np.zeros((2, 2, 2, 2), np.int64)    # (child,parent,L,R)
        self.Tcpg_strat = np.zeros((3, 2, 2, 2), np.int64)
        self.Tcplr_strat = np.zeros((3, 2, 2, 2, 2), np.int64)
        self.Tshift = {s: np.zeros((2, 2), np.int64) for s in range(-3, 4)}
        self.vp_pairs = []                                # (vc, vp) co-active
        self.vpg_trip = []                                # (vc, vp, vg)
        self.vpl_trip = []                                # (vc, vp, vLeft) co-active
        self.vshift = {s: [] for s in range(-3, 4)}       # (|vc|, |vp@s|)
        self.n_valid = 0

    def add(self, child, parent, grand, stratum):
        """child/parent/grand: (mask, val) arrays, 1D or 2D (..., len)."""
        mc, vc = child
        mp, vp = parent
        n = mc.shape[-1]
        idx = np.arange(n)
        pm = mp[..., idx >> 1]; pv = vp[..., idx >> 1]
        gm = gv = None
        if grand is not None:
            gm = grand[0][..., idx >> 2]; gv = grand[1][..., idx >> 2]
        # interior slots (need both neighbors)
        sl = slice(1, n - 1)
        c = mc[..., sl]; p = pm[..., sl]
        L = mc[..., 0:n - 2]; R = mc[..., 2:n]
        g = gm[..., sl] if gm is not None else np.zeros_like(c)
        self.n_valid += int(c.size)
        code3 = (c.astype(np.int64) << 2) | (p.astype(np.int64) << 1) | g.astype(np.int64)
        code4 = (c.astype(np.int64) << 3) | (p.astype(np.int64) << 2) \
            | (L.astype(np.int64) << 1) | R.astype(np.int64)
        self.Tcpg += np.bincount(code3.ravel(), minlength=8).reshape(2, 2, 2)
        self.Tcplr += np.bincount(code4.ravel(), minlength=16).reshape(2, 2, 2, 2)
        st = np.broadcast_to(np.asarray(stratum)[..., None], c.shape)
        for k in range(3):
            m = st == k
            if m.any():
                self.Tcpg_strat[k] += np.bincount(code3[m], minlength=8).reshape(2, 2, 2)
                self.Tcplr_strat[k] += np.bincount(code4[m], minlength=16).reshape(2, 2, 2, 2)
        # shifts (full range, not just interior)
        npar = mp.shape[-1]
        for s in range(-3, 4):
            a = (idx + s) >> 1
            ok = (a >= 0) & (a < npar)
            ps = np.zeros_like(mc)
            ps[..., ok] = mp[..., a[ok]]
            T = np.bincount(((mc.astype(np.int64) << 1) | ps.astype(np.int64)).ravel(),
                            minlength=4).reshape(2, 2)
            self.Tshift[s] += T
            both = mc & ps
            if both.any():
                vsv = np.zeros_like(vc)
                vsv[..., ok] = vp[..., a[ok]]
                self.vshift[s].append(np.stack(
                    [np.abs(vc[both]), np.abs(vsv[both])], 1))
        # value pairs
        both = mc & pm
        if both.any():
            self.vp_pairs.append(np.stack([vc[both], pv[both]], 1))
        if gm is not None:
            tri = mc & pm & gm
            if tri.any():
                self.vpg_trip.append(np.stack([vc[tri], pv[tri], gv[tri]], 1))
        Lm = np.zeros_like(mc)
        Lm[..., 1:] = mc[..., :-1]
        Lv = np.zeros_like(vc)
        Lv[..., 1:] = vc[..., :-1]
        tl = mc & pm & Lm
        if tl.any():
            self.vpl_trip.append(np.stack([vc[tl], pv[tl], Lv[tl]], 1))


def audit_units(units, band_ids, levels, child_range, unit_masks_for_strata):
    """units: list of per-band (mask,val) dicts. Returns dict child_band -> BandAudit."""
    # stratify units by total activity (terciles)
    tot = np.array([sum(int(u[b][0].sum()) for b in u) for u in units])
    qs = np.quantile(tot, [1 / 3, 2 / 3]) if len(tot) > 2 else [tot.max() + 1] * 2
    strat = np.digitize(tot, qs)
    audits = {}
    for ci in child_range:
        a = BandAudit()
        for u, st in zip(units, strat):
            child = u[ci]
            parent = u[ci - 1]
            grand = u[ci - 2] if ci - 2 >= 1 else None  # detail grandparent only
            if child[0].ndim == 2:                       # TPC: per-wire strata
                wtot = sum(u[b][0].sum(axis=1) for b in u)
                wq = np.quantile(wtot, [1 / 3, 2 / 3])
                wst = np.digitize(wtot, wq)
                a.add(child, parent, grand, wst)
            else:
                a.add(child, parent, grand, st)
        audits[ci] = a
    return audits


# --------------------------------------------------------------- reporting --

def report_audits(name, audits, band_names, n_units, out):
    res = []
    print(f"\n===== {name}: activity information (bits) =====")
    print(f"{'child':>5} {'H(c)':>7} | {'I(c;p)':>8} {'I(c;g|p)':>9} | "
          f"{'I(c;LR)':>8} {'I(c;p|LR)':>10} | {'orphan%':>8} {'rescue%':>8} | "
          f"{'bits/ev I(c;p|LR)':>16}")
    for ci, a in audits.items():
        T = a.Tcpg
        pc = T[1].sum() / max(T.sum(), 1)
        Hc = entropy_bits([pc, 1 - pc])
        i_cp = mi_from_table(a.Tcplr, (0,), (1,))
        i_cg_p = mi_from_table(a.Tcpg, (0,), (2,), (1,))
        i_clr = mi_from_table(a.Tcplr, (0,), (2, 3))
        i_cp_lr = mi_from_table(a.Tcplr, (0,), (1,), (2, 3))
        # orphan census
        n_child = T[1].sum()
        orphan = T[1, 0].sum() / max(n_child, 1)
        rescue = T[1, 0, 1] / max(T[1, 0].sum(), 1)
        bits_ev = i_cp_lr * a.n_valid
        # stratified (Simpson check): activity-weighted mean of per-stratum CMI
        cs = []
        for k in range(3):
            w = a.Tcplr_strat[k].sum()
            if w > 0:
                cs.append((mi_from_table(a.Tcplr_strat[k], (0,), (1,), (2, 3)), w))
        i_cp_lr_strat = sum(v * w for v, w in cs) / max(sum(w for _, w in cs), 1)
        print(f"{band_names[ci]:>5} {Hc:>7.4f} | {i_cp:>8.5f} {i_cg_p:>9.5f} | "
              f"{i_clr:>8.5f} {i_cp_lr:>10.5f} | {100*orphan:>7.1f}% {100*rescue:>7.1f}% | "
              f"{bits_ev:>14.0f}b")
        res.append(dict(child=band_names[ci], H_child=Hc, I_cp=i_cp,
                        I_cg_given_p=i_cg_p, I_cLR=i_clr, I_cp_given_LR=i_cp_lr,
                        I_cp_given_LR_stratified=i_cp_lr_strat,
                        orphan_frac=orphan, grand_rescue=rescue,
                        bits_per_event_cp_given_LR=bits_ev,
                        n_valid_slots=a.n_valid))
    print("  (orphan% = P(parent inactive | child active); rescue% = P(grand active | child active, parent inactive))")
    print("\n--- stratified vs pooled I(c;p|LR) (Simpson check) + shifted-non-parent control ---")
    for ci, a in audits.items():
        r = res[list(audits).index(ci)]
        mis = {s: mi_from_table(a.Tshift[s], (0,), (1,)) for s in range(-3, 4)}
        peak = max(mis, key=mis.get)
        print(f"{band_names[ci]:>5} pooled {r['I_cp_given_LR']:.5f} strat {r['I_cp_given_LR_stratified']:.5f}"
              f" | I(c;p@s) s=0 {mis[0]:.5f} peak s={peak:+d} {mis[peak]:.5f}"
              f" s=+2 {mis[2]:.5f}")
        r["I_shift"] = {str(s): mis[s] for s in mis}
    # value level
    print("\n--- value-level (Gaussian copula, co-active subpopulation) ---")
    for ci, a in audits.items():
        r = res[list(audits).index(ci)]
        if a.vp_pairs:
            P = np.concatenate(a.vp_pairs)
            if len(P) > 400000:
                P = P[np.random.default_rng(0).choice(len(P), 400000, replace=False)]
            mi_v, r_v = copula_mi(np.abs(P[:, 0]), np.abs(P[:, 1]))
            r["value_I_cp"], r["value_r_cp"], r["n_pairs"] = mi_v, r_v, int(len(P))
            line = f"{band_names[ci]:>5} I_v(c;p)={mi_v:.4f}b (r={r_v:+.3f}, n={len(P)})"
            if a.vpg_trip:
                Q = np.concatenate(a.vpg_trip)
                if len(Q) > 400000:
                    Q = Q[np.random.default_rng(0).choice(len(Q), 400000, replace=False)]
                mi_g, r_g = copula_partial_mi(np.abs(Q[:, 0]), np.abs(Q[:, 2]), np.abs(Q[:, 1]))
                r["value_I_cg_given_p"], r["value_rp_cg"] = mi_g, r_g
                line += f"  I_v(c;g|p)={mi_g:.4f}b (rp={r_g:+.3f})"
            if a.vpl_trip:
                Q = np.concatenate(a.vpl_trip)
                if len(Q) > 400000:
                    Q = Q[np.random.default_rng(0).choice(len(Q), 400000, replace=False)]
                mi_pl, r_pl = copula_partial_mi(np.abs(Q[:, 0]), np.abs(Q[:, 1]), np.abs(Q[:, 2]))
                r["value_I_cp_given_L"], r["value_rp_cp_L"] = mi_pl, r_pl
                line += f"  I_v(c;p|L)={mi_pl:.4f}b (rp={r_pl:+.3f})"
            print(line)
        # value-level alignment: corr vs shift
        rr = {}
        for s in range(-3, 4):
            if a.vshift[s]:
                V = np.concatenate(a.vshift[s])
                if len(V) > 200000:
                    V = V[np.random.default_rng(0).choice(len(V), 200000, replace=False)]
                _, rs = copula_mi(V[:, 0], V[:, 1])
                rr[s] = rs
        if rr:
            peak = max(rr, key=lambda s: rr[s])
            r["value_r_vs_shift"] = {str(s): rr[s] for s in rr}
            print(f"      value r(|c|,|p@s|): s=0 {rr.get(0, float('nan')):+.3f}"
                  f"  peak s={peak:+d} {rr[peak]:+.3f}")
    out[name] = res


def group_delay():
    import pywt
    print("\n===== T2.4 coif3 per-level group delay (impulse probe, periodization) =====")
    N, L = 32768, 10
    deltas = {}
    for p in (12000, 20000):
        x = np.zeros(N); x[p] = 1.0
        coeffs = pywt.wavedec(x, "coif3", level=L, mode="periodization")
        for i, c in enumerate(coeffs):
            j = L if i == 0 else L - i + 1
            name = ("A%d" % L) if i == 0 else "D%d" % j
            tau = int(np.argmax(np.abs(c)))
            deltas.setdefault(name, []).append(p / (1 << j) - tau)
    res = {k: float(np.mean(v)) for k, v in deltas.items()}
    for k, v in res.items():
        print(f"  {k:>4}: delta = {v:+.2f} coeffs (impulse maps to tau = t/2^j - delta)")
    return res


def anchor_sweep(units, name, out):
    print(f"\n===== T2.5 anchor sweep ({name}) =====")
    rows = []
    for cell in (1024, 512, 256):
        tot_cells = act_cells = 0
        fans = []
        for u in units:
            Lp = u[0][0].shape[-1] << 10
            nc = Lp // cell
            cnt = np.zeros(nc, np.int64)
            for i in range(1, 10):                       # D10..D2
                j = OPT_LEVELS[i]
                m, _ = u[i]
                act = np.nonzero(m)[0]
                if len(act):
                    cnt += np.bincount((act << j) // cell, minlength=nc)[:nc]
            # valid cells = unpadded chunk length / cell (A10 valid count * 1024/cell)
            valid_cells = max(1, (int(u[0][0].sum()) << 10) // cell)
            tot_cells += valid_cells
            occ = cnt[:valid_cells] > 0
            act_cells += int(occ.sum())
            if occ.any():
                fans.append(cnt[:valid_cells][occ])
        fan = np.concatenate(fans) if fans else np.zeros(1)
        rows.append(dict(cell=cell, cells_per_event=tot_cells,
                         occupancy=act_cells / max(tot_cells, 1),
                         fanout_mean=float(fan.mean()),
                         fanout_p95=float(np.percentile(fan, 95)),
                         fanout_max=int(fan.max())))
        print(f"  cell={cell:>5}: cells/ev {tot_cells:>6}  det-union occ {100*rows[-1]['occupancy']:>5.1f}%"
              f"  fan-out mean {fan.mean():>6.1f} p95 {np.percentile(fan,95):>5.0f} max {fan.max():>5.0f}")
    out[f"anchor_sweep_{name}"] = rows


def bit_count(out):
    print("\n===== T2.6 bit-count predictions (REGISTERED before M3) =====")
    preds = {}
    # TPC typical event (handoff: ~315k survivors, 25.4k tokens at 8x4)
    src_tpc = 315e3 * (12 + 3) / 1e6
    preds["tpc_source_Mbit"] = src_tpc
    for d in (256, 512):
        for be in (4, 8):
            preds[f"tpc_token_capacity_Mbit_d{d}_b{be}"] = 25.4e3 * d * be / 1e6
    print(f"  TPC source ~{src_tpc:.1f} Mbit/event vs token capacity "
          f"{preds['tpc_token_capacity_Mbit_d256_b4']:.0f}-"
          f"{preds['tpc_token_capacity_Mbit_d512_b8']:.0f} Mbit -> over-provisioned ~"
          f"{preds['tpc_token_capacity_Mbit_d256_b4']/src_tpc:.0f}-"
          f"{preds['tpc_token_capacity_Mbit_d512_b8']/src_tpc:.0f}x globally")
    print("  PREDICTION P1: bottleneck recon is flat in N, knees in d at 256-512,")
    print("                 residual loss concentrated in top core-load percentile.")
    print("  PREDICTION P2: d_s_min(anchor) ~ fanout_p95 * 12 / bits_per_dim;")
    print("                 at optical anchor-1024 (p95~200): d_s>=300 @8b/dim ->")
    print("                 anchor 512/256 or accept fine-band floor at d_s=64.")
    print("  PREDICTION P3: orphan-stratum recon gap drives any C0-vs-C2 difference;")
    print("                 parented-stratum gap ~0 at TPC depth and small at optical depth.")
    out["bit_count_predictions"] = preds


# -------------------------------------------------------------------- main --

def main():
    out = {}
    rng_note = "single typical events; event-level CIs deferred to multi-event rescan"
    out["note"] = rng_note

    tpc_path = os.path.join(ART, "typical_event_coeffs_smart.npz")
    if os.path.exists(tpc_path):
        units = load_tpc(tpc_path)
        audits = audit_units(units, range(5), TPC_LEVELS, range(2, 5), None)
        report_audits("tpc_smart", audits, TPC_BANDS, 6, out)

    for tag, fn in (("optical_lightout", "typical_event_coeffs_optical.npz"),
                    ("optical_doraemon", "typical_event_coeffs_doraemon.npz")):
        p = os.path.join(ART, fn)
        if not os.path.exists(p):
            print(f"[skip] {fn} not found")
            continue
        units = load_opt(p)
        audits = audit_units(units, range(11), OPT_LEVELS, range(2, 11), None)
        report_audits(tag, audits, OPT_BANDS, len(units), out)
        anchor_sweep(units, tag, out)

    out["group_delay_coif3"] = group_delay()
    bit_count(out)

    jp = os.path.join(ART, "info_audit.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\n-> {jp}")


if __name__ == "__main__":
    main()
