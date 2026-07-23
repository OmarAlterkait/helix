"""Value analysis of the recon images: what are the 'low-value fluctuations'?
For wire (ADC) and charge (e-), true vs recon: value percentiles + the 'ringing floor'
(|recon| where |true|~0) -> informs the plotting linthresh per panel.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import data as D, star_tpc as stp, vit_tpc as vtp, measure_coeffs as M
import deconv_analysis as DA
from viz_recon import load_model, predict_image, SIGMA, LENS
from pimm_data import JAXTPCDataset

D.init_pipeline_cpu()
geom, nts = M.load_geom(); EV = 0
ds = JAXTPCDataset(data_root=M.DATA_ROOT, split=M.SPLIT, modalities=("sensor",), dataset_name=M.DATASET_NAME)
s = ds.get_data(EV)["sensor"]; name = s["name"]
fidx = name.split("sensor_")[1].split(".h5")[0]; evn = int(name.split("evt")[1])
hf = f"{DA.HITS_DIR}/sim_wire_hits_{fidx}.h5"
cf = f"../artifacts/fm_cache_tpc/ev_{EV:05d}.npz"; qf = f"../artifacts/fm_charge_tpc/ev_charge_{EV:05d}.npz"
d = np.load(cf)
cat = dict(band=d["band"].astype(np.int64), idx=d["idx"].astype(np.int64), gid=d["gid"].astype(np.int64),
           wire=d["wire"].astype(np.int64), val=d["val"], val_clean=d["val_clean"],
           val_charge=np.load(qf)["val_charge"], unit=d["gid"].astype(np.int64))
ev = stp.rows_to_struct(cat); B = vtp.assemble_tpc_band(ev, list(range(ev["n_chunks"])), device=D.DEV); Bfm = D._to_fm(B)
mae = load_model("ckpt_dual44M.pt"); dec = load_model("ckpt_deconv_mae_240.pt")
Sb = np.array([5772.9, 4029.7, 3666.5, 2976.5])


def stats(name, img, unit):
    a = np.abs(img); nz = a[a > 0]
    pct = np.percentile(a, [50, 90, 99, 99.9, 100])
    print(f" {name:22s} [{unit}]  max={pct[4]:9.1f}  p50={np.percentile(a,50):8.2f} p90={pct[1]:8.2f} "
          f"p99={pct[2]:8.2f} p99.9={pct[3]:8.2f}  nonzero%={100*len(nz)/a.size:.2f}")


def ringing(true, pred, unit, sig_cut):
    """|pred| in QUIET pixels (|true|<sig_cut) = the fluctuation floor; signal pixels for ref."""
    quiet = np.abs(true) < sig_cut
    rms_q = np.sqrt((pred[quiet] ** 2).mean()); p99_q = np.percentile(np.abs(pred[quiet]), 99)
    sig = np.abs(true) >= sig_cut
    print(f"    ringing floor [{unit}]: quiet-pixel |recon| RMS={rms_q:.2f} p99={p99_q:.2f}  "
          f"| signal-pixel |true| median={np.median(np.abs(true[sig])):.1f}  -> suggest linthresh ~{max(p99_q,rms_q*3):.0f}")


for gid in [2, 0]:
    lab = geom[gid]["label"]; nw = geom[gid]["n_wires"]
    Strue = DA.wire_image(s, gid, nw, nts); Qtrue = DA.charge_image(hf, evn, lab, nw, nts)
    Spred = predict_image(mae, Bfm, ev, gid, nw, lambda pr, b: np.sinh(pr) * SIGMA)
    Qpred = predict_image(dec, Bfm, ev, gid, nw, lambda pr, b: np.sinh(pr) * Sb[b])
    print(f"\n=== {lab} ===")
    stats("WIRE true (clean)", Strue, "ADC"); stats("WIRE recon (MAE)", Spred, "ADC")
    ringing(Strue, Spred, "ADC", sig_cut=5)
    stats("CHARGE true", Qtrue, "e-"); stats("CHARGE recon (deconv)", Qpred, "e-")
    ringing(Qtrue, Qpred, "e-", sig_cut=50)
