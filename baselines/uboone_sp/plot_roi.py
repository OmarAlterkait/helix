#!/usr/bin/env python3
"""ROI figures: ROC curves (classical variants + model operating points) and event
displays of found vs oracle ROIs over the noisy input.

    scripts/helix_run.sh python baselines/uboone_sp/plot_roi.py --results baselines/uboone_sp/results \
        --out baselines/uboone_sp/figs [--events 300,321,350]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))


def roc_figs(res, out, plt):
    sets = res["_meta"]["settings"]
    fams = {"ex": [s for s in sets if s.startswith("ex_s")], "wc": [s for s in sets if s.startswith("wc_s") and "nofake" not in s]}
    extra = [s for s in sets if s.startswith("wc_") and s not in fams["wc"] and s != "wc_oracle"]
    col = {"ex": "tab:blue", "wc": "tab:red"}
    for side in ("far", "near"):
        fig, ax = plt.subplots(1, 3, figsize=(19, 6), sharey=True)
        for x, eb in zip(ax, ("E0-0.1", "E0.1-0.2", "E0.2-0.5")):
            ebk = {"E0-0.1": "0-0.1", "E0.1-0.2": "0.1-0.2", "E0.2-0.5": "0.2-0.5"}[eb]
            for fam, ss in fams.items():
                pts = np.array([[res[s]["metrics"][f"roi_fpr_{side}"], res[s]["metrics"].get(f"roi_eff_{ebk}", np.nan)] for s in ss])
                x.plot(pts[:, 0], pts[:, 1], "o-", c=col[fam], label=f"{'example' if fam == 'ex' else 'Wire-Cell port'}: ROI operating points (threshold scale)")
                for s, (fx, fy) in zip(ss, pts):
                    x.annotate(s.split("_s")[1], (fx, fy), fontsize=7, color=col[fam], xytext=(3, 3), textcoords="offset points")
                d = "ex_s1" if fam == "ex" else "wc_s1"
                r = np.array(res[d]["metrics"][f"roc_{side}_{eb}"])
                x.plot(r[:, 0], r[:, 1], "-", c=col[fam], alpha=0.5, lw=1,
                       label=f"{d}: score-threshold ROC")
            for s in extra:
                m = res[s]["metrics"]
                x.plot(m[f"roi_fpr_{side}"], m.get(f"roi_eff_{ebk}", np.nan), "s", ms=5, alpha=0.7, label=s)
            for h in [k for k in res if k.startswith("hyb_")]:
                m = res[h]["metrics"]
                x.plot(m[f"roi_fpr_{side}"], m.get(f"roi_eff_{ebk}", np.nan), "*", ms=12, c="k", label=f"{h} (post hoc)")
            r = np.array(res["model"]["metrics"][f"roc_{side}_{eb}"])
            x.plot(r[:, 0], r[:, 1], "-", c="tab:green", lw=2, label="model M4n_ft_full_near (presence) ROC")
            x.axvline(0.01, c="k", ls=":", lw=1)
            x.set_xscale("symlog", linthresh=1e-3); x.set_xlim(0, 1); x.set_ylim(0, 1.02)
            x.set_xlabel(f"fraction of {side}-activity noise windows above threshold (FPR)")
            x.set_title(f"isolated deposits {ebk} MeV")
            x.grid(alpha=0.3)
        ax[0].set_ylabel("efficiency (window score > threshold)")
        ax[0].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"ROI chains vs model, {side}-activity noise windows; dotted: 1% FPR")
        fig.tight_layout(); fig.savefig(os.path.join(out, f"roi_roc_{side}.png"), dpi=100); plt.close(fig)
        print("wrote", f"roi_roc_{side}.png", flush=True)


def displays(events, out, plt):
    from sp.data import Event, registry, PLANES
    from sp.noise_filter import gate_clean
    from sp.pipeline import Kernel, deconvolve_2d_dual, apply_lf_filter_to_wiener, PLANE_PARAMS, form_rois_true
    from sp.response import load_kernel
    from sp import wirecell_sp as WC
    from uboone_pipeline import roi as RO
    from helix.tpc.noise import digitize, generate_noise
    reg = registry()
    for ev in events:
        E = Event(ev)
        rng = np.random.default_rng(1000 + ev)
        planes = {}
        for g in range(6):
            nw, nt = reg[g]["n_wires"], reg[g]["n_ticks"]
            n_inc = generate_noise((nw, nt), rng=rng, wire_lengths_m=np.asarray(reg[g]["wire_lengths"]), incoherent=True, coherent=False)
            n_coh = generate_noise((nw, nt), rng=rng, incoherent=False, coherent=True, group_size=64)
            planes[g] = (n_inc, n_coh)
        # one volume: the one holding the first faint isolated deposit
        iso = [m for m in E.meta if m["kind"] == "iso" and 0.1 <= float(np.sum(m["E"])) < 0.5]
        vol = iso[0]["g"] // 3 if iso else 0
        fig, ax = plt.subplots(1, 3, figsize=(21, 7))
        for p in range(3):
            g = vol * 3 + p
            pl = PLANES[p]
            C = E.clean(g, reg); T = E.truth(g, C.shape)
            M = gate_clean(digitize(C + planes[g][0] + planes[g][1], reg[g]["pedestal"]))
            K = Kernel(*load_kernel(pl, "fit"))
            cw, cg = deconvolve_2d_dual(M, pl, K, noiseless=False)
            pe = PLANE_PARAMS[pl]
            if pl == "Y":
                rex = RO.form_rois_collection(cw, pe, noiseless=False)
            else:
                rex = RO.form_rois_induction(cw, apply_lf_filter_to_wiener(cw, pl, which="loose"),
                                             apply_lf_filter_to_wiener(cw, pl, which="tight"), pe, noiseless=False)
            mex = np.zeros(C.shape, bool)
            for w, lst in rex.items():
                for s, e in lst:
                    mex[w, s:e] = True
            d = WC.decon_outputs(M, pl, K)
            rwc, _, _ = WC.roi_chain(d, pl)
            mwc = WC.rois_mask(rwc, C.shape)
            mor = np.zeros(C.shape, bool)
            for w, lst in form_rois_true(T, pad_ticks=10).items():
                for s, e in lst:
                    mor[w, s:e] = True
            mine = [m for m in E.meta if m["kind"] == "iso" and m["g"] == g and 0.05 <= float(np.sum(m["E"])) < 0.5]
            c = mine[0] if mine else dict(w0=C.shape[0] // 2, t0=2000, E=[0])
            w0, w1 = max(c["w0"] - 64, 0), min(c["w0"] + 96, C.shape[0])
            t0, t1 = max(c["t0"] - 400, 0), min(c["t0"] + 528, C.shape[1])
            x = ax[p]
            x.imshow(M[w0:w1, t0:t1], aspect="auto", cmap="gray_r", vmin=-8, vmax=8, interpolation="nearest",
                     extent=[t0, t1, w1, w0])
            for mk, cc, lab in ((mor, "lime", "oracle ROI (truth > 25 e-, pad 10)"), (mex, "tab:blue", "example ROI chain"),
                                (mwc, "red", "Wire-Cell port (MicroBooNE config)")):
                sub = mk[w0:w1, t0:t1].astype(float)
                if sub.any():
                    x.contour(np.arange(t0, t1) + 0.5, np.arange(w0, w1) + 0.5, sub, levels=[0.5], colors=[cc], linewidths=1.0)
                x.plot([], [], c=cc, label=lab)
            for m in E.meta:
                if m["g"] == g and w0 - 32 < m["w0"] < w1 and t0 - 128 < m["t0"] < t1:
                    cc = {"iso": "orange", "bg": "cyan", "pair": "magenta"}[m["kind"]]
                    x.add_patch(plt.Rectangle((m["t0"], m["w0"]), 128, 32, fill=False, ec=cc, lw=1.2, ls="--"))
            x.set_xlim(t0, t1); x.set_ylim(w1, w0)
            x.set_title(f"plane {g} ({pl}); dashed: iso (orange) / bg (cyan) windows; deposit {float(np.sum(c['E'])):.2f} MeV")
            x.set_xlabel("tick"); x.set_ylabel("wire")
        ax[0].legend(fontsize=8, loc="upper right")
        fig.suptitle(f"ev{ev}: noisy input after coherent removal (ADC, grey +-8) with ROIs")
        fig.tight_layout(); fig.savefig(os.path.join(out, f"roi_display_ev{ev}.png"), dpi=100); plt.close(fig)
        print("wrote", f"roi_display_ev{ev}.png", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "figs"))
    ap.add_argument("--events", default="300,321,350")
    ap.add_argument("--no-roc", action="store_true")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(a.out, exist_ok=True)
    if not a.no_roc:
        roc_figs(json.load(open(os.path.join(a.results, "roi_sweep.json"))), a.out, plt)
    displays([int(e) for e in a.events.split(",") if e], a.out, plt)


if __name__ == "__main__":
    main()
