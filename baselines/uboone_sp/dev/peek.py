import json, sys, numpy as np
z = np.load(sys.argv[1]); P = json.loads(str(z["planes"]))
for g, info in P.items():
    print(f"g{g} hits {info['hits']:.3g} noise inc {info['noise_rms_inc']:.2f} coh {info['noise_rms_coh']:.2f} resid gate {info['resid_rms_gate']:.2f} mp {info['resid_rms_mp']:.2f} truth-in-ROI R0 {info['R0_truth_in_roi']:.3f} R3 {info['R3_truth_in_roi']:.3f}")
    for r in ["R0","R0ex","R1","R2","R2mp","R3","R3inc","R1nr","R2nr"]:
        d = info[r]; print(f"   {r:6s} charge/hits {d['sum_charge']/info['hits']:.3f} cells/hits {d['sum_cells']/info['hits']:.3f} roi_frac {d['roi_frac']:.4f} F0_f {d['F0_f']:.3f} corr_f {d['corr_f']:.3f} alpha {d['alpha_f']:.3f}")
