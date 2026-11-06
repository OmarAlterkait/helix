"""How much deposit charge is missing from the per-wire hits truth (25 e- CSR threshold
per group x wire x tick, u16 quantisation)? hits sum vs sum(q * attenuation) over deposits
with recovered t0 inside the readout window, per plane, ev260-263."""
import os, sys, json
import numpy as np
sys.path.insert(0, "baselines/uboone_sp/v2"); sys.path.insert(0, "baselines/uboone_sp")
import h5py, hdf5plugin
import resim as RS
from sp.data import Event, registry
sys.path.insert(0, RS.JAXTPC)
import yaml
from tools.config import create_sim_config
cfg = create_sim_config(yaml.safe_load(open(os.path.join(RS.JAXTPC, "config", "cubic_wireplane_config.yaml"))), total_pad=1000, include_digitize=True, num_s=16)
reg = registry(); out = {}
for ev in range(260, 264):
    E = Event(ev)
    fs = h5py.File(os.path.join(RS.SRC, "step", E.run, E.source_file.replace("_sensor_", "_step_")), "r")
    fh = h5py.File(os.path.join(RS.SRC, "hits", E.run, E.source_file.replace("_sensor_", "_hits_")), "r")
    for v in (0, 1):
        vg = fs[f"event_{E.event:03d}"][f"volume_{v}"]; hv = fh[f"event_{E.event:03d}"][f"volume_{v}"]
        org = np.array([vg.attrs["pos_origin_x"], vg.attrs["pos_origin_y"], vg.attrs["pos_origin_z"]])
        pos = vg["positions"][:].astype(np.float64) * float(vg.attrs["pos_step_mm"]) + org
        q = vg["charge"][:].astype(np.float64)
        t0d, _, _ = RS.recover_t0(cfg, hv, pos, q, v)
        for p in range(3):
            g = 3 * v + p
            tk, dd = RS.plane_ticks(cfg, pos, v, p)
            tick = tk + t0d
            ok = np.isfinite(tick) & (tick >= 0) & (tick < 4321 * 0.5)
            att = np.exp(-(dd / 0.16) / 10000.0)
            C = E.clean(g, reg)
            H = E.truth(g, C.shape)
            out[f"ev{ev}_g{g}"] = float(H.sum() / (q[ok] * att[ok]).sum())
for P, gs in (("U", (0, 3)), ("V", (1, 4)), ("Y", (2, 5))):
    vals = [v for k, v in out.items() if int(k.split("_g")[1]) in gs]
    print(P, "hits / sum(q att):", round(float(np.mean(vals)), 4), [round(x, 4) for x in vals])
json.dump(out, open("baselines/uboone_sp/v2/results/hits_completeness.json", "w"), indent=1)
