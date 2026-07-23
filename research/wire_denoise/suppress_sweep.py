"""Suppression-method sweep on the REAL TPC mechanism: per-band sigma (MAD), threshold
ALL bands incl. approx, on the coherent+helix-removed residual. Sweeps decomposition
LEVEL x wavelet x suppression-method x strength. Streams to JSONL (resumable, shardable).

Methods (all per-band sigma):
  uni_hard  : t = k*sigma_b*sqrt(2 ln N_b), hard            (original)
  garrote   : same t, non-negative garrote shrink           (less amplitude bias)
  fixed     : t = k*sigma_b (no length penalty), hard
  snr_topk  : keep top fraction by per-band SNR |c|/sigma_b  (R-D optimal selection)
  block     : block-RMS hard (block size 4) -> keep clustered signal, drop isolated noise
  xscale    : keep coeff if it OR its coarser parent is significant (cross-scale persistence)
"""
import os, sys, json, time
import numpy as np, pywt
sys.path.insert(0, '/sdf/group/neutrino/omara/helix')
import common as C
from helix.core.backend import set_backend; set_backend('jax')
from helix.tpc.coherent import remove_coherent
from helix.tpc.config import DetectorConfig

CFG = DetectorConfig(group_size=64, mask_threshold_nsigma=3.0, num_time_steps=C.N_TICKS,
                     temporal_dilation_ticks=11, n_passes=3)
WAVELETS = ['haar', 'db2', 'db4', 'sym4', 'sym8', 'coif1', 'coif3', 'bior2.2', 'bior4.4']
N_TEST = 4
SEED = 1
PLANES = ['Y', 'U', 'V']
JSONL = 'artifacts/suppress_sweep.jsonl'
B = 4   # block size

METHODS = {
    'uni_hard':  [0.5, 1.0, 1.5, 2.0, 3.0],
    'garrote':   [0.5, 1.0, 1.5, 2.0, 3.0],
    'fixed':     [2.0, 3.0, 4.0, 5.0],
    'snr_topk':  [0.05, 0.02, 0.01, 0.005, 0.002],
    'block':     [0.5, 1.0, 1.5, 2.0],
    'xscale':    [1.0, 1.5, 2.0],
}


def levels_for(w):
    mx = pywt.dwt_max_level(C.N_TICKS, pywt.Wavelet(w).dec_len)
    return list(range(2, mx + 1))


def per_band_sigma(coeffs):
    return [float(np.median(np.abs(c)) / 0.6745) for c in coeffs]


def _lf(N):
    return np.sqrt(2.0 * np.log(max(N, 2)))


def apply_method(coeffs, sig, method, param):
    out = []
    if method == 'snr_topk':
        zs = [np.abs(c) / max(s, 1e-9) for c, s in zip(coeffs, sig)]
        allz = np.concatenate([z.ravel() for z in zs])
        k = max(1, int(param * allz.size))
        thr = np.partition(allz, allz.size - k)[allz.size - k]
        return [np.where(z >= thr, c, 0.0) for c, z in zip(coeffs, zs)]
    if method == 'xscale':
        keep_prev = None
        for i, (c, s) in enumerate(zip(coeffs, sig)):
            t = param * s * _lf(c.shape[-1]); a = np.abs(c)
            kp = a >= t
            if keep_prev is not None:
                pu = np.repeat(keep_prev, 2, axis=-1)
                if pu.shape[-1] < c.shape[-1]:
                    pu = np.pad(pu, ((0, 0), (0, c.shape[-1] - pu.shape[-1])))
                pu = pu[:, :c.shape[-1]]
                kp = kp | (pu & (a >= 0.5 * t))
            out.append(np.where(kp, c, 0.0)); keep_prev = kp
        return out
    for c, s in zip(coeffs, sig):
        N = c.shape[-1]; a = np.abs(c)
        if method == 'uni_hard':
            t = param * s * _lf(N); out.append(np.where(a >= t, c, 0.0))
        elif method == 'fixed':
            t = param * s; out.append(np.where(a >= t, c, 0.0))
        elif method == 'garrote':
            t = param * s * _lf(N)
            out.append(np.where(a >= t, c - t * t / np.where(c == 0, 1.0, c), 0.0))
        elif method == 'block':
            nw, n = c.shape; nb = n // B
            if nb:
                blk = c[:, :nb * B].reshape(nw, nb, B)
                keep = (np.sqrt((blk ** 2).mean(-1)) >= param * s)[..., None]
                head = (blk * keep).reshape(nw, nb * B)
            else:
                head = np.zeros((nw, 0))
            tail = np.where(a[:, nb * B:] >= param * s, c[:, nb * B:], 0.0)
            out.append(np.concatenate([head, tail], axis=1))
    return out


def helix_remove(noisy):
    return np.stack([np.asarray(remove_coherent(noisy[i], CFG, sigma_per_wire=None))
                     for i in range(noisy.shape[0])])


def _shard():
    i, n = (int(x) for x in os.environ.get('SHARD', '0/1').split('/'))
    return [w for j, w in enumerate(WAVELETS) if j % n == i]


def load_done():
    d = set()
    if os.path.exists(JSONL):
        for line in open(JSONL):
            try:
                p = json.loads(line); d.add((p['plane'], p['w'], p['lv'], p['method'], p['param']))
            except Exception:
                pass
    return d


def run():
    mine = _shard(); done = load_done()
    fh = open(JSONL, 'a', buffering=1)
    for pt in PLANES:
        clean = C.load_clean(C.train_test_events(n_train=24, n_test=24)[1][4:4 + N_TEST], pt)
        removed = helix_remove(C.make_noisy(clean, pt, seed=SEED, coherent=True))
        sig = np.abs(clean) > 0
        for w in mine:
            for lv in levels_for(w):
                cos = [pywt.wavedec(removed[i], w, level=lv, axis=1) for i in range(N_TEST)]
                sigs = [per_band_sigma(co) for co in cos]
                t0 = time.perf_counter()
                for method, params in METHODS.items():
                    for pm in params:
                        if (pt, w, lv, method, pm) in done:
                            continue
                        recs = []; nk = nt = 0
                        for i in range(N_TEST):
                            kept = apply_method(cos[i], sigs[i], method, pm)
                            for c in kept:
                                nk += int(np.count_nonzero(c)); nt += c.size
                            recs.append(pywt.waverec(kept, w, axis=1)[:, :C.N_TICKS])
                        R = np.stack(recs); f0, nr = C.aggregate(clean, R)
                        rec = dict(plane=pt, w=w, lv=lv, method=method, param=pm,
                                   f0=round(f0, 5), noise_rms=round(nr, 4),
                                   bias=round(float((R - clean)[sig].mean()), 4),
                                   n_kept=int(nk // N_TEST),
                                   compression=round(nt / max(nk, 1), 2))
                        fh.write(json.dumps(rec) + '\n'); fh.flush()
                print(f"  [{pt}] {w} L{lv} done ({time.perf_counter()-t0:.1f}s)", flush=True)
    fh.close()


if __name__ == '__main__':
    run()
