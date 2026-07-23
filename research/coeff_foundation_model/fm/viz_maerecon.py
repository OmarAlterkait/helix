"""MAE reconstruction figures: full input | masked | reconstructed | true-clean, in ORIGINAL wire
space (inverse-DWT of coeffs, coif3 -- the HELIX viz_recon convention). Varies planes / events /
mask type (whole-plane vs random). Self-contained on the fm cache (no pimm/geom dep)."""
import sys, os, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
import numpy as np, torch, pywt, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import data as D, star_tpc as stp, vit_tpc as vtp
from data import DEV
from train import make_mask
from probe_3d_ridge import build
from probe_3d_mlp import build_serial

SIGMA = 2.6; LENS = [271, 271, 542, 1084]; APPROX = 2168
PLAB = lambda g: f"{['U','V','Y'][g % 3]}{g // 3}"           # U/V/Y per volume


def load_event(ev_id):
    cf = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../artifacts/fm_cache_tpc/ev_{ev_id}.npz")
    if not os.path.exists(cf):
        cf = cf.replace(f"ev_{ev_id}.npz", f"ev_{int(ev_id):05d}.npz")
    d = np.load(cf)
    cat = dict(band=d["band"].astype(np.int64), idx=d["idx"].astype(np.int64), gid=d["gid"].astype(np.int64),
               wire=d["wire"].astype(np.int64), val=d["val"], val_clean=d["val_clean"], unit=d["gid"].astype(np.int64))
    ev = stp.rows_to_struct(cat)
    B = vtp.assemble_tpc_band(ev, list(range(ev["n_chunks"])), device=DEV)
    return ev, D._to_fm(B)


def waverec_plane(ev, gid, per_coeff_val):
    """place per-coeff values into 4 DWT bands (cD1 zeroed) -> inverse coif3 -> (nw, nt) image."""
    egid, eband, ewire, eidx = ev["gid"], ev["band"], ev["wire"], ev["idx"]
    sel_g = egid == gid
    nw = int(ewire[sel_g].max()) + 1 if sel_g.any() else 1
    bands = [np.zeros((nw, LENS[b]), np.float64) for b in range(4)]
    for b in range(4):
        m = sel_g & (eband == b)
        bands[b][ewire[m], eidx[m] % LENS[b]] = per_coeff_val[m]
    cs = bands + [np.zeros((nw, APPROX))]                    # cD1 (finest) set to 0 (noise band)
    return pywt.waverec(cs, "coif3", mode="periodization", axis=1), nw


@torch.no_grad()
def make_images(model, Bfm, ev, gid, msk):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, mu, _ = model(Bfm, msk)
    pr = (torch.sinh(mu[Bfm["cell"], Bfm["slot"]].float()) * SIGMA).cpu().numpy()   # model out -> raw coeff
    mc = msk[Bfm["cell"]].cpu().numpy()                                             # per-coeff masked?
    vin, vcl = ev["val"], ev["val_clean"]
    full, nw = waverec_plane(ev, gid, vin)
    masked, _ = waverec_plane(ev, gid, np.where(mc, 0.0, vin))
    recon, _ = waverec_plane(ev, gid, np.where(mc, pr, vin))
    clean, _ = waverec_plane(ev, gid, vcl)
    return dict(full=full, masked=masked, recon=recon, clean=clean, diff=recon - full, nw=nw)


def zoom_of(clean, nw, margin_w=70, margin_t=180):
    a = np.abs(clean)
    w = int(np.argmax(a.sum(1))); tc = int(np.argmax(a[w]))
    return (max(0, w - margin_w), min(nw, w + margin_w), max(0, tc - margin_t), min(clean.shape[1], tc + margin_t))


def show(ax, img, title, ref, wl, wh, tl, th, lt=2.0, diff=False):
    win = img[wl:wh, tl:th]
    vmax = (np.abs(win).max() if diff else np.abs(ref[wl:wh, tl:th]).max()) + 1e-6
    ax.imshow(win, aspect="auto", origin="lower", extent=[tl, th, wl, wh],
              cmap=("PuOr_r" if diff else "RdBu_r"), norm=SymLogNorm(linthresh=lt, vmin=-vmax, vmax=vmax))
    ax.set_title(title, fontsize=10); ax.set_xlabel("drift tick", fontsize=8); ax.set_ylabel("wire", fontsize=8)


COLS = [("full", "input (full)"), ("masked", "masked (model sees)"),
        ("recon", "reconstructed"), ("diff", "recon − input")]


def fig_grid(rows, title, fname, zoom=True):
    """rows: list of (rowlabel, images_dict, gid). one row per, 4 cols. zoom=False -> whole plane."""
    n = len(rows)
    fig, ax = plt.subplots(n, 4, figsize=(19, 4.2 * n), squeeze=False)
    for r, (rlab, im, gid) in enumerate(rows):
        if zoom:
            wl, wh, tl, th = zoom_of(im["clean"], im["nw"])
        else:
            wl, wh, tl, th = 0, im["nw"], 0, im["full"].shape[1]
        for c, (key, ctitle) in enumerate(COLS):
            show(ax[r, c], im[key], f"{rlab} | {ctitle}", im["clean"], wl, wh, tl, th, diff=(key == "diff"))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.985]); fig.savefig(fname, dpi=105); plt.close(fig)
    print("saved", fname, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_dscale600_20k.pt")
    ap.add_argument("--serial", type=int, default=1); ap.add_argument("--rope_split", type=int, default=0)
    ap.add_argument("--tag", default="d20k")
    ap.add_argument("--events", default="")                  # comma ids; else auto-pick active
    a = ap.parse_args()
    D.init_pipeline_cpu()
    model = build_serial(a.ckpt, False, a.rope_split, 1024, 2048) if a.serial else build(a.ckpt, randinit=False)
    T = a.tag
    print("model:", a.ckpt, "serial=", a.serial, flush=True)

    # pick 3 active events (by total clean coeff energy) from a candidate range
    if a.events:
        evids = a.events.split(",")
    else:
        cand, en = [], []
        for e in range(0, 120):
            try:
                ev, _ = load_event(e)
            except Exception:
                continue
            cand.append(e); en.append(float(np.abs(ev["val_clean"]).sum()))
            if len(cand) >= 40: break
        order = np.argsort(en)[::-1]
        evids = [str(cand[i]) for i in order[:3]]
    print("events:", evids, flush=True)
    loaded = {e: load_event(e) for e in evids}

    def gmsk(Bfm, gid):                                       # hide exactly plane `gid`
        return (Bfm["plane_id"] == gid)

    e0 = evids[0]; ev0, B0 = loaded[e0]

    # FIG 1 — different PLANES (event e0, random mask), 3 planes  [FULL plane, not zoomed]
    torch.manual_seed(0)
    mrand = make_mask(B0, "random", 0.75, 1)
    rows = [(f"ev{e0} {PLAB(g)}", make_images(model, B0, ev0, g, mrand), g) for g in [0, 1, 2]]
    fig_grid(rows, f"Random-0.75 across planes — WHOLE PLANE (event {e0}, {T})", f"viz_planes_{T}_full.png", zoom=False)
    fig_grid(rows, f"Random-0.75 across planes — zoomed (event {e0}, {T})", f"viz_planes_{T}_zoom.png", zoom=True)

    # FIG 2 — different EVENTS (plane Y=gid2, random mask)  [zoomed]
    rows = []
    for e in evids:
        ev, B = loaded[e]; torch.manual_seed(0)
        m = make_mask(B, "random", 0.75, 1)
        rows.append((f"ev{e} {PLAB(2)}", make_images(model, B, ev, 2, m), 2))
    fig_grid(rows, f"Random-0.75 across events — zoomed (plane {PLAB(2)}, {T})", f"viz_events_{T}_zoom.png", zoom=True)

    # FIG 3 — different MASKING (event e0, plane Y=gid2): whole-plane vs random  [both zoom + full]
    rows = [("plane-mask (hide Y)", make_images(model, B0, ev0, 2, gmsk(B0, 2)), 2),
            ("random-0.75", make_images(model, B0, ev0, 2, make_mask(B0, "random", 0.75, 1)), 2)]
    fig_grid(rows, f"Mask type: plane-triangulation vs random — zoomed (ev {e0} {PLAB(2)}, {T})", f"viz_masking_{T}_zoom.png", zoom=True)
    fig_grid(rows, f"Mask type: plane-triangulation vs random — WHOLE PLANE (ev {e0} {PLAB(2)}, {T})", f"viz_masking_{T}_full.png", zoom=False)


if __name__ == "__main__":
    main()
