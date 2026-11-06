"""C7: charge-scaling augmentation on a real training event. On bright coefficients (signal-dominated) the augmented
/ original value must equal the coefficient's region gain; with gains off the tokens are unchanged."""
import sys
import numpy as np
sys.path.insert(0, "/global/u1/o/oalter/nu/helix")
from helix.core.coeff_io import read_coeff_event
from helix.data.denoise import DenoiseEvents, split_events
C = "/global/cfs/cdirs/m5238/users/oalter/ost61/corpus"; CT = "/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete"
CL = "/global/cfs/cdirs/m5238/users/oalter/coeff_tpc_r1"
run, path, pos, ev = split_events(C, ["run_0027587651"], "train")[0]
b = DenoiseEvents(C, [], CT, items=[], gain_sigma_plane=0.1, gain_sigma_block=0.05, clean_root=CL)
gain = b.draw_gain(np.random.default_rng(1))
ce = read_coeff_event(path, pos)
cc = read_coeff_event(path.replace(C, CL).replace("sim_wire_coeff_", "sim_wire_coeff_clean_"), pos, coords_from=ce)
g = b.region_gain(gain, ce.plane_gid, ce.wire)
v_aug = ce.value + (g - 1) * cc.value
big = np.abs(cc.value) > 20 * np.abs(ce.value - cc.value).std()          # signal >> noise
r = v_aug[big] / ce.value[big]
print("bright coefficients:", big.sum(), "| (aug/orig) / region gain: median %.4f, 16-84%% %.4f-%.4f" % tuple(np.percentile(r / g[big], [50, 16, 84])))
print("noise (value - clean) unchanged:", np.allclose(v_aug - g * cc.value, ce.value - cc.value, atol=1e-4))
_, B0, _ = b.tokens(path, pos); _, B1, _ = DenoiseEvents(C, [], CT, items=[]).tokens(path, pos)
print("gains off: tokens identical:", np.array_equal(B0["inp"], B1["inp"]))
