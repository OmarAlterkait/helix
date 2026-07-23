"""Visualize: true charge Q vs wire signal S (the response), and their coif3-L4
per-band coefficient energy. event 0, planes Y (collection) and U (induction)."""
import sys
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, "/sdf/group/neutrino/omara/particle-imaging-models/libs/pimm-data/src")
import hdf5plugin, numpy as np, pywt, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import measure_coeffs as M
from pimm_data import JAXTPCDataset
import deconv_analysis as DA

geom, nts = M.load_geom()
ds = JAXTPCDataset(data_root=M.DATA_ROOT, split=M.SPLIT, modalities=("sensor",), dataset_name=M.DATASET_NAME)
s = ds.get_data(0)["sensor"]; name = s["name"]
fidx = name.split("sensor_")[1].split(".h5")[0]; evn = int(name.split("evt")[1])
hf = f"{DA.HITS_DIR}/sim_wire_hits_{fidx}.h5"
bn = ["A4", "D4", "D3", "D2", "D1"]

fig, ax = plt.subplots(2, 3, figsize=(15, 7))
for r, gid in enumerate([2, 0]):
    lab = geom[gid]["label"]; nw = geom[gid]["n_wires"]
    Q = DA.charge_image(hf, evn, lab, nw, nts); S = DA.wire_image(s, gid, nw, nts)
    w = int(np.argmax(np.abs(Q).sum(1)))                       # most active wire
    tc = int(np.argmax(np.abs(Q[w]))); lo, hi = max(0, tc-120), tc+120
    t = np.arange(lo, hi)
    # (col0) single-wire response: charge vs wire signal
    a = ax[r, 0]; a.plot(t, Q[w, lo:hi], "C2", lw=1.3, label="charge Q (eâ»)")
    a.set_ylabel("charge eâ»", color="C2"); a.tick_params(axis="y", labelcolor="C2")
    a2 = a.twinx(); a2.plot(t, S[w, lo:hi], "C3", lw=1.0, label="wire S (ADC)")
    a2.set_ylabel("wire ADC", color="C3"); a2.tick_params(axis="y", labelcolor="C3")
    a2.axhline(0, color="k", lw=.4, ls=":")
    a.set_title(f"{lab} wire {w}: charge vs wire signal"); a.set_xlabel("tick")
    # (col1) 2D zoom of wire signal around the wire/tick
    wlo, whi = max(0, w-40), w+40
    a = ax[r, 1]; im = a.imshow(S[wlo:whi, lo:hi], aspect="auto", origin="lower",
                                extent=[lo, hi, wlo, whi], cmap="seismic",
                                vmin=-np.abs(S[wlo:whi, lo:hi]).max(), vmax=np.abs(S[wlo:whi, lo:hi]).max())
    a.set_title(f"{lab} wire SIGNAL S (ADC)"); a.set_xlabel("tick"); a.set_ylabel("wire")
    # (col2) per-band coeff energy: charge vs wire
    beq, bes = DA.band_energies(Q), DA.band_energies(S)
    tq = sum(e for _, e, _ in beq); tsv = sum(e for _, e, _ in bes)
    x = np.arange(5)
    a = ax[r, 2]; a.bar(x-0.2, [beq[b][1]/tq*100 for b in range(5)], 0.4, color="C2", label="charge")
    a.bar(x+0.2, [bes[b][1]/tsv*100 for b in range(5)], 0.4, color="C3", label="wire")
    a.set_xticks(x); a.set_xticklabels(bn); a.set_ylabel("% band energy")
    a.set_title(f"{lab} coeff energy (response = coarse-ward shift)"); a.legend()
fig.suptitle("Deconvolution parts: true charge (green) vs response-convolved wire signal (red) â€” event 0")
fig.tight_layout(); fig.savefig("deconv_parts.png", dpi=110)
print("saved deconv_parts.png")
