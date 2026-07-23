"""P9b: coefficient-SELECTION strategies on the DWT (the transform that gives high
F0). All hard-keep (unbiased). Compare F0-vs-compression for:
  visu  : VisuShrink-hard (noise-relative, production)
  topk  : keep largest-|coeff| fraction per wire (R-D optimal for fidelity)
  energy: keep smallest set holding energy fraction per wire
  oracle: keep noisy coeffs where the CLEAN coeff is significant (detection ceiling)
Reports bias = mean(recon-clean) on signal pixels (want ~0)."""
import json
import numpy as np
import pywt
import common as C

BEST = {'Y': ('db8', 8), 'U': ('coif3', 8), 'V': ('db8', 8)}
N_TEST = 6
SEED = 1


def bands(img, w, lv):
    co = pywt.wavedec(img, w, mode='periodization', level=lv, axis=-1)
    sizes = [c.shape[-1] for c in co]
    arr = np.concatenate(co, axis=-1)            # (nw, Ttot)
    return arr, sizes


def unbands(arr, sizes, w):
    co, i = [], 0
    for s in sizes:
        co.append(arr[:, i:i + s]); i += s
    return pywt.waverec(co, w, mode='periodization', axis=-1)[:, :C.N_TICKS]


def recon_metrics(arr_kept, sizes, w, clean):
    rec = unbands(arr_kept, sizes, w)
    f0, nr = C.aggregate(clean, rec)
    sig = np.abs(clean) > 0
    bias = float((rec - clean)[sig].mean())
    nk = int(np.count_nonzero(arr_kept)); nt = arr_kept.size
    return dict(f0=f0, noise_rms=nr, bias=bias, compression=nt / max(nk, 1), n_kept=nk)


def run():
    _, test = C.train_test_events()
    out = {}
    for pt in ['Y', 'U', 'V']:
        w, lv = BEST[pt]
        clean = C.load_clean(test[:N_TEST], pt)
        noisy = C.make_noisy(clean, pt, seed=SEED, coherent=False)
        pts = []
        for i in range(noisy.shape[0]):
            pass
        # build per-image arrays once
        A = [bands(noisy[i], w, lv) for i in range(noisy.shape[0])]
        Acl = [bands(clean[i], w, lv) for i in range(noisy.shape[0])]
        sizes = A[0][1]

        def evalstrat(keepfn, label, param):
            kept = np.empty((noisy.shape[0], *A[0][0].shape), np.float32)
            for i in range(noisy.shape[0]):
                kept[i] = keepfn(A[i][0], Acl[i][0])
            # metrics pooled
            recs = np.stack([unbands(kept[i], sizes, w) for i in range(noisy.shape[0])])
            f0, nr = C.aggregate(clean, recs)
            sig = np.abs(clean) > 0
            bias = float((recs - clean)[sig].mean())
            nk = int(np.count_nonzero(kept)); nt = kept.size
            return dict(method=label, param=param, f0=f0, noise_rms=nr, bias=bias,
                        compression=nt / max(nk, 1), n_kept=nk)

        # visu (per-band kappa)
        def visu(kappa):
            def fn(arr, _):
                out = arr.copy()
                sig = np.median(np.abs(arr[:, -sizes[-1]:]), axis=1, keepdims=True) / 0.6745
                i = sizes[0]
                for s in sizes[1:]:
                    t = kappa * sig * np.sqrt(2 * np.log(max(s, 2)))
                    band = out[:, i:i + s]
                    out[:, i:i + s] = np.where(np.abs(band) >= t, band, 0.0)
                    i += s
                return out
            return fn
        for k in [0.5, 1, 1.5, 2, 3, 4, 6]:
            pts.append(evalstrat(visu(k), 'visu', k))

        # topk per wire (keep largest fraction of ALL coeffs)
        def topk(frac):
            def fn(arr, _):
                n = arr.shape[1]; kk = max(1, int(frac * n))
                thr = np.partition(np.abs(arr), n - kk, axis=1)[:, n - kk][:, None]
                return np.where(np.abs(arr) >= thr, arr, 0.0)
            return fn
        for f in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
            pts.append(evalstrat(topk(f), 'topk', f))

        # energy per wire
        def energy(frac):
            def fn(arr, _):
                a2 = arr ** 2
                order = np.argsort(-a2, axis=1)
                cs = np.take_along_axis(a2, order, axis=1).cumsum(1)
                tot = cs[:, -1:].clip(1e-30)
                kc = (cs < frac * tot).sum(1)
                out = np.zeros_like(arr)
                for r in range(arr.shape[0]):
                    idx = order[r, :kc[r] + 1]
                    out[r, idx] = arr[r, idx]
                return out
            return fn
        for f in [0.999, 0.995, 0.99, 0.98, 0.95, 0.9]:
            pts.append(evalstrat(energy(f), 'energy', f))

        # oracle: keep noisy coeff where |clean coeff| significant
        def oracle(thr):
            def fn(arr, arrcl):
                return np.where(np.abs(arrcl) >= thr, arr, 0.0)
            return fn
        for t in [0.5, 1, 2, 5, 10]:
            pts.append(evalstrat(oracle(t), 'oracle', t))

        out[pt] = dict(points=pts)
        # report best F0 at ~20x and ~100x per strategy
        for comp_t in [20, 100]:
            line = f"[{pt}] ~{comp_t}x: "
            for m in ['visu', 'topk', 'energy', 'oracle']:
                cand = [p for p in pts if p['method'] == m]
                best = min(cand, key=lambda p: abs(p['compression'] - comp_t))
                line += f"{m} {best['compression']:.0f}x/F0{best['f0']:.3f}/b{best['bias']:+.2f}  "
            print(line, flush=True)
    json.dump(out, open('artifacts/rd_strategies.json', 'w'), indent=1)
    print('saved artifacts/rd_strategies.json')


if __name__ == '__main__':
    run()
