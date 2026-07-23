"""Deconvolution analysis: true charge Q (hits) vs wire signal S (sensor) vs their
coif3-L4 coefficients. NOT training — just understanding the parts.

event evtNNN of sim_wire_sensor_FFFF.h5 <-> sim_wire_hits_FFFF.h5 / event_NNN (same run).
"""
import sys
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, "/sdf/group/neutrino/omara/particle-imaging-models/libs/pimm-data/src")
import hdf5plugin, h5py, numpy as np, pywt
import measure_coeffs as M
from pimm_data import JAXTPCDataset

HITS_DIR = M.DATA_ROOT + "/hits/" + M.SPLIT       # derive from DATA_ROOT (no drift)
WAVELET, LEVEL = "coif3", 4


def charge_image(hf, ev, gid_label, n_wires, n_ticks):
    """Dense true-charge Q(wire,tick) for one plane from hits."""
    vol, pl = gid_label.rsplit("_", 1)               # volume_0_Y -> volume_0 / Y
    g = h5py.File(hf, "r")[f"event_{ev:03d}/{vol}/{pl}"]
    cw = g["center_wires"][:].astype(np.int64); ct = g["center_times"][:].astype(np.int64)
    gs = g["group_sizes"][:].astype(np.int64); pk = g["peak_charges"][:].astype(np.float64)
    cu = g["charges_u16"][:].astype(np.float64); dw = g["delta_wires"][:].astype(np.int64)
    dt = g["delta_times"][:].astype(np.int64)
    grp = np.repeat(np.arange(len(gs)), gs)
    q = cu / 65535.0 * pk[grp]                       # absolute charge per deposit
    W = cw[grp] + dw; T = ct[grp] + dt
    ok = (W >= 0) & (W < n_wires) & (T >= 0) & (T < n_ticks)
    Q = np.zeros((n_wires, n_ticks), np.float64)
    np.add.at(Q, (W[ok], T[ok]), q[ok])
    return Q


def wire_image(s, gid, n_wires, n_ticks):
    """Dense clean wire signal S(wire,tick) (pedestal-subtracted ADC)."""
    m = s["plane_gid"] == gid
    S = np.zeros((n_wires, n_ticks), np.float32)
    S[s["wire"][m].astype(int), s["time"][m].astype(int)] = s["value"][m]
    return S


def band_energies(img):
    """Per-wire coif3-L4 DWT; return per-band energy + survivor fraction (|c|>3*MAD)."""
    rows = img[np.abs(img).sum(1) > 0]                # active wires only
    if len(rows) == 0:
        return None
    coeffs = pywt.wavedec(rows, WAVELET, mode="periodization", level=LEVEL, axis=1)
    out = []
    for b, c in enumerate(coeffs):                   # cA4, cD4, cD3, cD2, cD1
        mad = np.median(np.abs(c - np.median(c))) / 0.6745 + 1e-9
        out.append((c.size, float((c**2).sum()), float((np.abs(c) > 3*mad).mean())))
    return out


def main():
    geom, nts = M.load_geom()
    ds = JAXTPCDataset(data_root=M.DATA_ROOT, split=M.SPLIT,
                       modalities=("sensor",), dataset_name=M.DATASET_NAME)
    EV = 0
    s = ds.get_data(EV)["sensor"]
    name = s["name"]                                  # sim_wire_sensor_0000.h5_evt000
    fidx = name.split("sensor_")[1].split(".h5")[0]
    evn = int(name.split("evt")[1])
    hf = f"{HITS_DIR}/sim_wire_hits_{fidx}.h5"
    print(f"event {EV}: {name} -> hits {hf.split('/')[-1]} event_{evn:03d}")

    bandnames = ["A4", "D4", "D3", "D2", "D1"]
    for gid in [2, 0]:                                # Y (collection), U (induction)
        lab = geom[gid]["label"]; nw = geom[gid]["n_wires"]
        Q = charge_image(hf, evn, lab, nw, nts)
        S = wire_image(s, gid, nw, nts)
        print(f"\n=== {lab} (gid {gid}) ===")
        print(f" charge Q: occ={ (Q!=0).mean()*100:.3f}%  tot={Q.sum():.3e} e-  max={Q.max():.0f}")
        print(f" wire  S: occ={ (S!=0).mean()*100:.3f}%  range=[{S.min():.0f},{S.max():.0f}] ADC"
              f"  (induction bipolar if min<<0)")
        # response footprint: active ticks per active wire
        qa = (np.abs(Q) > 0).sum(1); sa = (np.abs(S) > 0).sum(1)
        print(f" active ticks/active-wire: charge {qa[qa>0].mean():.1f}  wire {sa[sa>0].mean():.1f}"
              f"  -> response spreads {sa[sa>0].mean()/max(qa[qa>0].mean(),1):.1f}x")
        be_q, be_s = band_energies(Q), band_energies(S)
        print(" band   chargeE%   wireE%    charge_surv%  wire_surv%")
        tq = sum(e for _, e, _ in be_q); tsv = sum(e for _, e, _ in be_s)
        for b in range(5):
            print(f"  {bandnames[b]:>3}  {be_q[b][1]/tq*100:7.1f}   {be_s[b][1]/tsv*100:7.1f}"
                  f"     {be_q[b][2]*100:7.2f}    {be_s[b][2]*100:7.2f}")


if __name__ == "__main__":
    main()
