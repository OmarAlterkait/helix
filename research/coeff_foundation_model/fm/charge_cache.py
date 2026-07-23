"""Build the deconvolution-probe TARGET: the true-charge coif3-L4 coefficient at each
existing wire-token position (same band/gid/wire/tau as the cached wire coeffs).

For cached event i (== dataset event i): map i -> hits file/event via ds.get_data(i) name,
build Q(wire,tick) per plane, per-wire DWT, gather charge coeff at each cached row.
Saves ev_charge_NNNNN.npz {val_charge (same order/len as ev_NNNNN.npz rows)}.

Run:  python charge_cache.py --events 400
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, "/sdf/group/neutrino/omara/particle-imaging-models/libs/pimm-data/src")
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/research/coeff_foundation_model")
import hdf5plugin, h5py, numpy as np, pywt
import measure_coeffs as M
from pimm_data import JAXTPCDataset

LENS_T = [271, 271, 542, 1084]                       # A4,D4,D3,D2 (== star_tpc.LENS_T)
WAVELET, LEVEL = "coif3", 4
HITS_DIR = M.DATA_ROOT + "/hits/" + M.SPLIT       # derive from DATA_ROOT (no drift)


def charge_bands(hgrp, n_wires, n_ticks, pad):
    """true charge Q(wire,tick) -> per-wire coif3-L4 bands [A4,D4,D3,D2] (n_wires, Lb)."""
    cw = hgrp["center_wires"][:].astype(np.int64); ct = hgrp["center_times"][:].astype(np.int64)
    gs = hgrp["group_sizes"][:].astype(np.int64); pk = hgrp["peak_charges"][:].astype(np.float64)
    cu = hgrp["charges_u16"][:].astype(np.float64); dw = hgrp["delta_wires"][:].astype(np.int64)
    dt = hgrp["delta_times"][:].astype(np.int64)
    grp = np.repeat(np.arange(len(gs)), gs)
    q = cu / 65535.0 * pk[grp]; W = cw[grp] + dw; T = ct[grp] + dt
    ok = (W >= 0) & (W < n_wires) & (T >= 0) & (T < n_ticks)
    Q = np.zeros((n_wires, n_ticks), np.float64)
    np.add.at(Q, (W[ok], T[ok]), q[ok])
    Qp = np.pad(Q, ((0, 0), (0, pad)))
    cs = pywt.wavedec(Qp, WAVELET, mode="periodization", level=LEVEL, axis=1)
    return [cs[b].astype(np.float32) for b in range(4)]           # A4,D4,D3,D2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=400)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    cdir = args.cache_dir or os.path.join(here, "..", "artifacts", "fm_cache_tpc")
    out = args.out or os.path.join(here, "..", "artifacts", "fm_charge_tpc")
    cdir = os.path.abspath(cdir); out = os.path.abspath(out); os.makedirs(out, exist_ok=True)
    geom, nts = M.load_geom(); pad = (-nts) % 16
    labels = {g: geom[g]["label"] for g in geom}; nwof = {g: geom[g]["n_wires"] for g in geom}
    ds = JAXTPCDataset(data_root=M.DATA_ROOT, split=M.SPLIT, modalities=("sensor",), dataset_name=M.DATASET_NAME)

    t0 = time.time(); cov = []
    for i in range(args.events):
        cf = os.path.join(cdir, f"ev_{i:05d}.npz"); of = os.path.join(out, f"ev_charge_{i:05d}.npz")
        if not os.path.exists(cf) or os.path.exists(of):
            continue
        d = np.load(cf); band = d["band"].astype(np.int64); gid = d["gid"].astype(np.int64)
        wire = d["wire"].astype(np.int64); idx = d["idx"].astype(np.int64)
        name = ds.get_data(i)["sensor"]["name"]                  # sim_wire_sensor_FFFF.h5_evtNNN
        fidx = name.split("sensor_")[1].split(".h5")[0]; evn = int(name.split("evt")[1])
        h = h5py.File(f"{HITS_DIR}/sim_wire_hits_{fidx}.h5", "r")
        cc = {}
        for g in range(6):
            vol, pl = labels[g].rsplit("_", 1)
            cc[g] = charge_bands(h[f"event_{evn:03d}/{vol}/{pl}"], nwof[g], nts, pad)
        h.close()
        tau = idx % np.array(LENS_T)[band]
        vc = np.zeros(len(band), np.float32)
        for g in range(6):
            for b in range(4):
                m = (gid == g) & (band == b)
                if m.any():
                    vc[m] = cc[g][b][wire[m], tau[m]]
        np.savez_compressed(of, val_charge=vc)
        cov.append(float((vc != 0).mean()))
        if (i + 1) % 25 == 0:
            dt = time.time() - t0
            print(f"  {i+1}/{args.events}  ({dt/(i+1)*1000:.0f} ms/ev)  charge-cover={np.mean(cov):.2%}", flush=True)
    print(f"done: {len(cov)} events -> {out}  mean charge-coverage of wire-token slots = {np.mean(cov):.2%}")


if __name__ == "__main__":
    main()
