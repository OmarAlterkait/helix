"""Validate the Pareto-selected config(s) per plane on a LARGER held-out set
(more events x seeds -> error bars), Stage A (intrinsic) and Stage B (coherent +
learned group-aware removal front-end). Reads the recommendation from pareto.json."""
import sys, json
import numpy as np, pywt, torch
sys.path.insert(0, '/sdf/group/neutrino/omara/helix')
import common as C
from group_removal import CoherentNet, learned_remove

N_EVAL = 12
SEEDS = [21, 22, 23]


def eval_config(noisy, clean, w, lv, k):
    rec = np.empty_like(noisy); nk = nt = 0
    for i in range(noisy.shape[0]):
        co = pywt.wavedec(noisy[i], w, mode='periodization', level=lv, axis=-1)
        sg = np.median(np.abs(co[-1]), axis=-1, keepdims=True) / 0.6745
        nk += co[0].size; nt += co[0].size
        cc = [co[0]]
        for b in range(1, len(co)):
            t = k * sg * np.sqrt(2 * np.log(max(co[b].shape[-1], 2)))
            bb = np.where(np.abs(co[b]) >= t, co[b], 0.0); cc.append(bb)
            nk += int(np.count_nonzero(bb)); nt += bb.size
        rec[i] = pywt.waverec(cc, w, mode='periodization', axis=-1)[:, :C.N_TICKS]
    f0, nr = C.aggregate(clean, rec); sig = np.abs(clean) > 0
    return f0, nr, float((rec - clean)[sig].mean()), nt / max(nk, 1)


def run():
    R = json.load(open('artifacts/pareto.json'))
    _, test = C.train_test_events(n_train=24, n_test=20)
    ev = test[:N_EVAL]
    out = {}
    for pt in ['Y', 'U', 'V']:
        rc = R[pt]['select']['recommended']
        w, lv, k = rc['w'], rc['lv'], rc['k']
        clean = C.load_clean(ev, pt)
        net = CoherentNet().to('cuda'); net.load_state_dict(torch.load(f'artifacts/grpnet_{pt}.pt', weights_only=True))
        A, Bc = [], []
        for s in SEEDS:
            nA = C.make_noisy(clean, pt, seed=s, coherent=False)
            A.append(eval_config(nA, clean, w, lv, k))
            nB = C.make_noisy(clean, pt, seed=s, coherent=True)
            rmv = learned_remove(nB, net)
            Bc.append(eval_config(rmv, clean, w, lv, k))
        A = np.array(A); Bc = np.array(Bc)  # (seeds, [f0,nr,bias,comp])
        out[pt] = dict(config=f"{w} L{lv} k{k:g}",
                       stageA=dict(f0=A[:, 0].mean(), f0_std=A[:, 0].std(), nrms=A[:, 1].mean(),
                                   bias=A[:, 2].mean(), comp=A[:, 3].mean()),
                       stageB=dict(f0=Bc[:, 0].mean(), f0_std=Bc[:, 0].std(), nrms=Bc[:, 1].mean(),
                                   bias=Bc[:, 2].mean(), comp=Bc[:, 3].mean()))
        a, b = out[pt]['stageA'], out[pt]['stageB']
        print(f"[{pt}] {out[pt]['config']}  ({N_EVAL} ev x {len(SEEDS)} seeds)")
        print(f"    Stage A (intrinsic):        F0={a['f0']:.4f}±{a['f0_std']:.4f} comp={a['comp']:.1f}x bias={a['bias']:+.3f} nrms={a['nrms']:.2f}")
        print(f"    Stage B (coh+learned rm):   F0={b['f0']:.4f}±{b['f0_std']:.4f} comp={b['comp']:.1f}x bias={b['bias']:+.3f} nrms={b['nrms']:.2f}", flush=True)
    json.dump(out, open('artifacts/validate_best.json', 'w'), indent=1)
    print('saved artifacts/validate_best.json')


if __name__ == '__main__':
    run()
