"""Visualize model outputs in ORIGINAL space (inverse-DWT of predicted coeffs).
  MAE model  -> predicted CLEAN WIRE  (response space)   vs true clean wire
  deconv     -> predicted CHARGE       (charge space)     vs true charge Q
Style ~ viz_2x2_jaxtpc (SymLogNorm linthresh=2, RdBu_r, 64-wire group lines).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, "/sdf/group/neutrino/omara/particle-imaging-models/libs/pimm-data/src")
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/research/coeff_foundation_model")
import hdf5plugin, numpy as np, torch, pywt, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import data as D, star_tpc as stp, vit_tpc as vtp, measure_coeffs as M
from data import DEV
from model import FMModel
import deconv_analysis as DA, charge_cache as CC
from pimm_data import JAXTPCDataset

SIGMA = 2.6; LENS = [271, 271, 542, 1084]; EV = 0
geom, nts = M.load_geom(); pad = (-nts) % 16


def load_model(ck):
    c = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), ck), map_location=DEV)
    m = FMModel(128, 4, 6, n_wirefeat=1, d=c["d"], blocks=c["blocks"], dec_blocks=c.get("dec_blocks", 4)).to(DEV)
    m.load_state_dict(c["model"]); m.eval()
    return m


@torch.no_grad()
def predict_image(model, Bfm, ev, gid, nw, scale_fn):
    """run model (all visible) -> per-row pred -> dense coeffs -> waverec image for one plane."""
    msk = torch.zeros(int(Bfm["n_cells"]), dtype=torch.bool, device=DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, mu, _ = model(Bfm, msk)
    pr = mu[Bfm["cell"], Bfm["slot"]].float().cpu().numpy()
    bands = [np.zeros((nw, LENS[b]), np.float64) for b in range(4)]
    egid, eband, ewire, eidx = ev["gid"], ev["band"], ev["wire"], ev["idx"]
    for b in range(4):
        m = (egid == gid) & (eband == b)
        bands[b][ewire[m], eidx[m] % LENS[b]] = scale_fn(pr[m], b)
    cs = bands + [np.zeros((nw, 2168))]
    return pywt.waverec(cs, "coif3", mode="periodization", axis=1)[:, :nts]


def main():
    D.init_pipeline_cpu()                                    # populate nw (avoid GPU pipeline import)
    ds = JAXTPCDataset(data_root=M.DATA_ROOT, split=M.SPLIT, modalities=("sensor",), dataset_name=M.DATASET_NAME)
    s = ds.get_data(EV)["sensor"]; name = s["name"]
    fidx = name.split("sensor_")[1].split(".h5")[0]; evn = int(name.split("evt")[1])
    hf = f"{DA.HITS_DIR}/sim_wire_hits_{fidx}.h5"

    # build ev + B (with charge), aligned
    cf = f"../artifacts/fm_cache_tpc/ev_{EV:05d}.npz"; qf = f"../artifacts/fm_charge_tpc/ev_charge_{EV:05d}.npz"
    d = np.load(cf)
    cat = dict(band=d["band"].astype(np.int64), idx=d["idx"].astype(np.int64), gid=d["gid"].astype(np.int64),
               wire=d["wire"].astype(np.int64), val=d["val"], val_clean=d["val_clean"],
               val_charge=np.load(qf)["val_charge"], unit=d["gid"].astype(np.int64))
    ev = stp.rows_to_struct(cat)
    B = vtp.assemble_tpc_band(ev, list(range(ev["n_chunks"])), device=DEV); Bfm = D._to_fm(B)

    mae = load_model("ckpt_dual44M.pt"); dec = load_model("ckpt_deconv_mae6k.pt")
    # per-band charge scale (match deconv_train)
    Sb = np.array([5772.9, 4029.7, 3666.5, 2976.5])
    wire_scale = lambda pr, b: np.sinh(pr) * SIGMA            # asinh(val/SIGMA) -> val
    chg_scale = lambda pr, b: np.sinh(pr) * Sb[b]             # asinh(charge/Sb) -> charge

    fig, ax = plt.subplots(2, 4, figsize=(20, 8))
    for r, gid in enumerate([2, 0]):                          # Y, U
        lab = geom[gid]["label"]; nw = geom[gid]["n_wires"]
        # truths
        Strue = DA.wire_image(s, gid, nw, nts)
        Qtrue = DA.charge_image(hf, evn, lab, nw, nts)
        # preds
        Spred = predict_image(mae, Bfm, ev, gid, nw, wire_scale)
        Qpred = predict_image(dec, Bfm, ev, gid, nw, chg_scale)
        # zoom around most active wire/time of charge
        w = int(np.argmax(np.abs(Qtrue).sum(1))); tc = int(np.argmax(np.abs(Qtrue[w])))
        wl, wh = max(0, w-60), w+60; tl, th = max(0, tc-150), tc+150
        ext = [tl, th, wl, wh]
        def show(a, img, title, cmap, ref, lt):                  # lt = linthresh above the ringing floor
            vmax = np.abs(ref[wl:wh, tl:th]).max() + 1e-6
            a.imshow(img[wl:wh, tl:th], aspect="auto", origin="lower", extent=ext, cmap=cmap,
                     norm=SymLogNorm(linthresh=lt, vmin=-vmax, vmax=vmax))
            a.set_title(title); a.set_xlabel("tick"); a.set_ylabel(f"{lab} wire")
        show(ax[r, 0], Strue, f"{lab}: TRUE clean wire", "RdBu_r", Strue, 2)       # wire ADC
        show(ax[r, 1], Spred, f"{lab}: MAE recon (wire)", "RdBu_r", Strue, 2)
        show(ax[r, 2], Qtrue, f"{lab}: TRUE charge", "RdBu_r", Qtrue, 100)         # charge e- (>ringing ~15)
        show(ax[r, 3], Qpred, f"{lab}: deconv recon (charge)", "RdBu_r", Qtrue, 100)
    fig.suptitle("Original-space reconstruction: MAE->clean wire (response) | deconv->charge | event 0")
    fig.tight_layout(); fig.savefig("viz_recon.png", dpi=105)
    print("saved viz_recon.png")


if __name__ == "__main__":
    main()
