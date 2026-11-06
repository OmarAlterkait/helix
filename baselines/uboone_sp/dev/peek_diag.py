import json, sys, numpy as np
r = json.load(open(sys.argv[1]))
for g, P in r["planes"].items():
    print("g", g, {k: {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in P["rms"].items()})
    for s, f in P["flags"].items():
        print("   ", s, {k: (round(float(np.mean(v)), 3) if isinstance(v, list) and v else v) for k, v in f.items()})
