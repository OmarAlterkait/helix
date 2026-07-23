"""Headroom diagnostic: how close is smart removal to the BEST POSSIBLE?

  raw       sparsify(noisy)                          no removal
  helix     sparsify(helix-removed)
  smart     sparsify(smart-removed, k=4)
  idealmask sparsify(coeff-removed using the TRUE signal location as the mask) -- the best
            a coefficient-space common-mode method can do with perfect signal detection
  oracle    sparsify(signal + intrinsic, coherent NEVER added) -- the absolute target

If smart ~ oracle: no smarter removal can reduce the count. If idealmask << smart:
better signal detection is the lever. If idealmask ~ smart >> oracle: coherent in
dense-signal regions is fundamentally unrecoverable in coefficient space.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import sys
import numpy as np
import pywt
import cc_common as cc
import smart as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wire_denoise'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import common as wd  # noqa: E402
from helix.core import backend as _be  # noqa: E402
_be.set_backend('jax')
from helix.core import wavelet as cw  # noqa: E402
from helix.tpc.config import DetectorConfig  # noqa: E402
from helix.tpc.coherent import remove_coherent  # noqa: E402
GS = cc.GROUP_SIZE


def _dilate_t(mask, ticks):
    k = np.ones(ticks)
    return np.array([np.convolve(r.astype(np.float32), k, 'same') > 0.5 for r in mask])


def ideal_removal(noisy, signal, minkeep=6, sig_adc=2.0, dilate=11):
    """Coeff-space common-mode removal using the TRUE signal location as the wire mask."""
    nw, nt = noisy.shape
    smask = _dilate_t(np.abs(signal) > sig_adc, dilate)               # perfect wire x time mask
    bands = pywt.wavedec(noisy.astype(np.float32), cc.WAVELET, level=cc.LEVEL, mode=cc.MODE, axis=-1)
    nblk = cc.n_groups(nw); ngf = cc.full_groups(nw)
    est = []
    for b in bands:
        Lj = b.shape[-1]
        pot = (np.arange(nt) * Lj) // nt; bnd = np.searchsorted(pot, np.arange(Lj))
        wmask = np.logical_or.reduceat(smask, bnd, axis=1)           # (nw,Lj)
        M = np.zeros((nblk, Lj), np.float32)
        if ngf:
            blk = b[:ngf * GS].reshape(ngf, GS, Lj)
            uf = ~wmask[:ngf * GS].reshape(ngf, GS, Lj)
            nuf = uf.sum(1)
            M[:ngf] = np.where(nuf >= minkeep, (blk * uf).sum(1) / np.maximum(nuf, 1), 0.0)
        if ngf * GS < nw:
            t = b[ngf * GS:]; ut = ~wmask[ngf * GS:]; nuf = ut.sum(0)
            M[ngf] = np.where(nuf >= minkeep, (t * ut).sum(0) / np.maximum(nuf, 1), 0.0)
        est.append(sm.broadcast_blocks(M, nw))
    return pywt.waverec([b - e for b, e in zip(bands, est)], cc.WAVELET, mode=cc.MODE, axis=-1)[..., :nt].astype(np.float32)


def spcount(cleaned, signal, cfg):
    res = cw.sparsify(cleaned, wavelet=cfg.wavelet, level=cfg.dwt_level,
                      mode=cfg.dwt_mode, threshold=cfg.threshold_spec())
    recon = np.asarray(cw.reconstruct(res, cleaned.shape[-1]))
    sig = np.abs(signal) > 0
    f0 = 1.0 - float(np.abs(recon - signal)[sig].sum()) / max(float(np.abs(signal)[sig].sum()), 1e-9)
    return res.n_kept, res.compression, f0


def main():
    n_ev = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    events = list(range(0, n_ev * 7, 7))
    cfg = DetectorConfig(group_size=64)
    methods = ['raw', 'helix', 'smart', 'idealmask', 'oracle']
    for p in ['Y', 'U', 'V']:
        agg = {m: [] for m in methods}
        for e in events:
            s, c, i = cc.components(p, e)
            ped = cc.PLANES[p]['pedestal']
            noisy = wd.digitize(s + c + i, ped)
            imgs = {'raw': noisy,
                    'helix': np.asarray(remove_coherent(noisy, cfg)),
                    'smart': sm.smart_removal(noisy, kgate=4.0)[0],
                    'idealmask': ideal_removal(noisy, s),
                    'oracle': wd.digitize(s + i, ped)}
            for m in methods:
                agg[m].append(spcount(imgs[m], s, cfg))
        print(f"\n  plane {p}:")
        print(f"   {'method':>11}  {'kept':>8}  {'comp':>8}  {'F0':>8}")
        for m in methods:
            k, comp, f0 = np.array(agg[m]).mean(0)
            print(f"   {m:>11}  {k:>8.0f}  {comp:>7.1f}x  {f0:>8.4f}")


if __name__ == '__main__':
    main()
