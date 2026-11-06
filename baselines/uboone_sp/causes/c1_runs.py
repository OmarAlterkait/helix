"""C1: does the simulation carry run-to-run (or event-to-event) differences in the amplitude-to-charge scale?
Prints, for one shard per corpus run, the /config arrays and the ident fields that could set a global scale."""
import os, glob, h5py, numpy as np
C = "/global/cfs/cdirs/m5238/users/oalter/ost61/corpus"
for run in sorted(r for r in os.listdir(C) if r.startswith("run_")):
    sh = sorted(glob.glob(f"{C}/{run}/sim_wire_coeff_[0-9]*.h5"))
    with h5py.File(sh[0], "r") as f:
        cfg = f["config"]; idn = f["ident"]
        if run == sorted(os.listdir(C))[0] or True:
            keys = list(cfg.keys())
        ns = cfg["norm_sigma"][:]
        attrs = {k: (v if np.ndim(v) == 0 else np.asarray(v).ravel()[:6]) for k, v in cfg.attrs.items()}
        print(run, len(sh), "shards | config keys", keys)
        print("   norm_sigma[:, :4] =", np.round(ns[:, :4], 3).tolist())
        print("   config attrs:", attrs)
        print("   ident fields:", list(idn.keys()), "| file attrs:", {k: str(v)[:80] for k, v in f.attrs.items()})
