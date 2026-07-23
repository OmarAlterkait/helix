"""Run the OAT knob sweep for ONE plane capturing ALL 4 metrics (F0, kept, nz_in, nz_out),
dump JSON. Usage: knob_consolidate.py <plane> [n_ev] [gpu]. Consolidate with consolidate_plot.py."""
import os
import sys
import json

PLANE = sys.argv[1]
N_EV = int(sys.argv[2]) if len(sys.argv) > 2 else 12
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[3] if len(sys.argv) > 3 else '0'

import numpy as np  # noqa: E402
import cc_common as cc  # noqa: E402
import sweep_knob as sk  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wire_denoise'))
import common as wd  # noqa: E402

events = list(range(0, N_EV * 9, 9))
data = []
for e in events:
    s, c, i = cc.components(PLANE, e)
    noisy = wd.digitize(s + c + i, cc.PLANES[PLANE]['pedestal'])
    data.append((s, c, noisy, sk.smc_k(noisy, 4.0)))

out = {'plane': PLANE, 'n_ev': N_EV, 'def': None, 'knobs': {}}
rows = [sk.metrics(sk.build(no, dict(sk.DEF), smc), no, s) for (s, c, no, smc) in data]
out['def'] = [float(x) for x in np.array(rows).mean(0)]
print(f"[{PLANE}] DEF  F0={out['def'][0]:.4f} kept={out['def'][1]:.0f} "
      f"nz_in={out['def'][2]:.3f} nz_out={out['def'][3]:.3f}", flush=True)

for knob, vals in sk.SWEEP.items():
    out['knobs'][knob] = []
    for v in vals:
        cfg = dict(sk.DEF); cfg[knob] = v
        rows = [sk.metrics(sk.build(no, cfg, smc), no, s) for (s, c, no, smc) in data]
        a = np.array(rows).mean(0)
        out['knobs'][knob].append([v, float(a[0]), float(a[1]), float(a[2]), float(a[3])])
        print(f"[{PLANE}] {knob:>9}={str(v):>7}  F0={a[0]:.4f} kept={a[1]:.0f} "
              f"nz_in={a[2]:.3f} nz_out={a[3]:.3f}", flush=True)

json.dump(out, open(f'knob_{PLANE}.json', 'w'), indent=1)
print(f"[{PLANE}] wrote knob_{PLANE}.json", flush=True)
