"""Many-event validation (coherent + helix removal) of the Pareto-best bior4.4 L8
vs the production coif3 L4, per plane. Reports F0/compression/bias/noise mean+-std
over many events x seeds. Helix removal on jax/GPU (sweep is CPU-only now)."""
import os, sys, json
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import numpy as np, pywt
sys.path.insert(0, '/sdf/group/neutrino/omara/helix')
import common as C
from helix.core.backend import set_backend; set_backend('jax')
from helix.tpc.coherent import remove_coherent
from helix.tpc.config import DetectorConfig

N_EVAL = 16
SEEDS = [31, 32, 33]
CFG = DetectorConfig(group_size=64, mask_threshold_nsigma=3.0, num_time_steps=C.N_TICKS,
                     temporal_dilation_ticks=11, n_passes=3)
PROD = {'Y': ('coif3', 4, 0.75), 'U': ('coif3', 4, 2.0), 'V': ('coif3', 4, 1.0)}
BEST = {'Y': ('bior4.4', 8, 1.25), 'U': ('bior4.4', 8, 2.0), 'V': ('bior4.4', 8, 1.5)}


def helix_remove(noisy):
    return np.stack([np.asarray(remove_coherent(noisy[i], CFG, sigma_per_wire=None))
                     for i in range(noisy.shape[0])])


def eval_cfg(removed, clean, w, lv, k):
    rec = np.empty_like(removed); nk = nt = 0
    for i in range(removed.shape[0]):
        co = pywt.wavedec(removed[i], w, mode='periodization', level=lv, axis=-1)
        sg = np.median(np.abs(co[-1]), axis=-1, keepdims=True) / 0.6745
        nk += co[0].size; nt += co[0].size; cc = [co[0]]
        for b in range(1, len(co)):
            t = k * sg * np.sqrt(2 * np.log(max(co[b].shape[-1], 2)))
            bb = np.where(np.abs(co[b]) >= t, co[b], 0.0); cc.append(bb)
            nk += int(np.count_nonzero(bb)); nt += bb.size
        rec[i] = pywt.waverec(cc, w, mode='periodization', axis=-1)[:, :C.N_TICKS]
    f0, nr = C.aggregate(clean, rec); sig = np.abs(clean) > 0
    return f0, nr, float((rec - clean)[sig].mean()), nt / max(nk, 1)


def run():
    test = C.train_test_events(n_train=24, n_test=24)[1]
    ev = test[4:4 + N_EVAL]                              # held-out, disjoint from sweep's test[:4]
    out = {}
    for pt in ['Y', 'U', 'V']:
        clean = C.load_clean(ev, pt)
        rows = {'prod': [], 'best': []}
        for s in SEEDS:
            removed = helix_remove(C.make_noisy(clean, pt, seed=s, coherent=True))
            rows['prod'].append(eval_cfg(removed, clean, *PROD[pt]))
            rows['best'].append(eval_cfg(removed, clean, *BEST[pt]))
        out[pt] = {}
        for tag, cfg in [('prod', PROD[pt]), ('best', BEST[pt])]:
            a = np.array(rows[tag])
            out[pt][tag] = dict(cfg=f"{cfg[0]} L{cfg[1]} k{cfg[2]:g}",
                                f0=a[:, 0].mean(), f0_std=a[:, 0].std(), nrms=a[:, 1].mean(),
                                bias=a[:, 2].mean(), comp=a[:, 3].mean())
        p, b = out[pt]['prod'], out[pt]['best']
        print(f"\n[{pt}]  ({N_EVAL} events x {len(SEEDS)} seeds, coherent + helix removal)")
        print(f"   production {p['cfg']:14s}: comp={p['comp']:5.1f}x  F0={p['f0']:.4f}+-{p['f0_std']:.4f}  bias={p['bias']:+.3f}  nrms={p['nrms']:.2f}")
        print(f"   BEST       {b['cfg']:14s}: comp={b['comp']:5.1f}x  F0={b['f0']:.4f}+-{b['f0_std']:.4f}  bias={b['bias']:+.3f}  nrms={b['nrms']:.2f}")
        print(f"   -> {b['comp']/p['comp']:.1f}x more compression, dF0={b['f0']-p['f0']:+.4f}", flush=True)
    json.dump(out, open('artifacts/validate_bior.json', 'w'), indent=1, default=float)
    print('\nsaved artifacts/validate_bior.json')


if __name__ == '__main__':
    run()
