"""Shape comparison of response kernels (fit / example / exact / exact0 / exact_s<k>):
relative L2 difference after the best integer time shift (|dt| <= 3) and best gain."""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sp.response import load_kernel

def grid(src, P):
    if src.startswith("exact_s"):
        d = np.load(f"baselines/uboone_sp/response/exact_{P}_s{src[7:]}.npz")
        R, wo, tz = d["R_avg"], d["wire_offsets"], int(d["time_zero_bin"])
    else:
        R, wo, tz = load_kernel(P, src)
    K = np.zeros((41, 400))                      # wires -20..20, ticks -200..199
    for j, w in enumerate(wo):
        if -20 <= w <= 20:
            t = np.arange(R.shape[0]) - tz
            ok = (t >= -200) & (t < 200)
            K[w + 20, t[ok] + 200] += R[ok, j]
    return K

out = {}
for P in "UVY":
    ref = grid("exact", P)
    row = {}
    for src in ("fit", "example", "exact0", "exact_s0", "exact_s4", "exact_s8"):
        K = grid(src, P)
        best = (9, 0, 1.0)
        for dt in range(-3, 4):
            Ks = np.roll(K, dt, 1)
            g = (Ks * ref).sum() / (Ks * Ks).sum()
            r = np.linalg.norm(g * Ks - ref) / np.linalg.norm(ref)
            if r < best[0]:
                best = (r, dt, g)
        row[src] = dict(rel_diff_raw=float(np.linalg.norm(K - ref) / np.linalg.norm(ref)), rel_diff_after_shift_gain=float(best[0]),
                        shift=int(best[1]), gain=float(best[2]), integral=float(K.sum()), ref_integral=float(ref.sum()),
                        neighbour_sum=float(K[[19, 21]].sum()), ref_neighbour_sum=float(ref[[19, 21]].sum()))
    out[P] = row
    print(P, {k: (round(v['rel_diff_raw'], 3), round(v['rel_diff_after_shift_gain'], 3), v['shift'], round(v['gain'], 3)) for k, v in row.items()})
json.dump(out, open("baselines/uboone_sp/v2/results/kernel_shapes.json", "w"), indent=1)

# example vs the no-diffusion pitch average (the hypothesis that the example IS exact0)
for P in "UVY":
    ref, K = grid("exact0", P), grid("example", P)
    best = min((np.linalg.norm((Ks * ref).sum() / (Ks * Ks).sum() * Ks - ref) / np.linalg.norm(ref), dt)
               for dt in range(-3, 4) for Ks in [np.roll(K, dt, 1)])
    out[P]["example_vs_exact0"] = dict(rel_diff_after_shift_gain=float(best[0]), shift=int(best[1]))
    print(P, "example vs exact0", round(best[0], 3), best[1])
json.dump(out, open("baselines/uboone_sp/v2/results/kernel_shapes.json", "w"), indent=1)

# was the example built from the same NPZ? Re-run its recipe (build_avg_response.py) on the
# production-commit NPZ, and on a wire-mirrored version (JAXTPC reads column center - 10*w + 10*u)
JX = "/pscratch/sd/o/oalter/helix_work/uboone_sp/jaxtpc_1ea2282/tools/responses"
for P in "UVY":
    d = np.load(f"{JX}/{P}_plane_kernel.npz", allow_pickle=True)
    K = d["kernel"].astype(np.float64); wzb = int(d["wire_zero_bin"])
    ex = np.load(f"baselines/uboone_sp/reference/uboone_deconv/avg_response_{P}.npz")["R_avg"].astype(np.float64)
    w8 = np.r_[0.5, np.ones(9), 0.5] / 10
    mine = np.stack([(K[:, wzb + 10 * w - 5: wzb + 10 * w + 6] * w8).sum(1) for w in range(-9, 10)], 1)
    exs = ex[:, 1:-1] if ex.shape[1] == 21 else ex
    out[P]["example_recipe_reproduced"] = float(np.abs(mine - exs).max() / np.abs(exs).max())
    out[P]["kernel_wire_asymmetry"] = float(np.linalg.norm(K - K[:, ::-1]) / np.linalg.norm(K))
    print(P, "example recipe on 1ea2282 NPZ: max rel diff", out[P]["example_recipe_reproduced"], "ex shape", ex.shape,
          "NPZ wire asymmetry", round(out[P]["kernel_wire_asymmetry"], 4))
json.dump(out, open("baselines/uboone_sp/v2/results/kernel_shapes.json", "w"), indent=1)

# the example with JAXTPC's time interpolation applied (mean of adjacent bins = average over t_offset)
for P in "UVY":
    ref, K = grid("exact0", P), grid("example", P)
    Kh = 0.5 * (K + np.roll(K, 1, 1))
    best = min((np.linalg.norm((Ks * ref).sum() / (Ks * Ks).sum() * Ks - ref) / np.linalg.norm(ref), dt)
               for dt in range(-3, 4) for Ks in [np.roll(Kh, dt, 1)])
    out[P]["example_halftick_vs_exact0"] = dict(rel_diff_after_shift_gain=float(best[0]), shift=int(best[1]))
    print(P, "example + half-tick interpolation vs exact0", round(best[0], 4), best[1])
json.dump(out, open("baselines/uboone_sp/v2/results/kernel_shapes.json", "w"), indent=1)
